"""
Training-Free Delta-Weighted Localizability Scorer
====================================================
Uses Cliff's delta effect sizes (recomputed per CV fold) as feature weights
to score bug reports — no model fitting on outcomes.

Two modes:
  1. INTERNAL  — cross-validated evaluation on your 835-bug dataset
                 (deltas recomputed per fold to avoid leakage)
  2. EXTERNAL  — score a new dataset using deltas from the full 835-bug set
                 (run: python delta_score_predictor.py --mode external
                        --new_features /path/to/new_features.csv
                        --new_tools    /path/to/new_tool_comparison.csv)

Usage:
    # Internal CV evaluation
    python delta_score_predictor.py

    # External dataset
    python delta_score_predictor.py --mode external \\
        --new_features new_features.csv \\
        --new_tools    new_tool_comparison.csv

Expects in working directory (or pass via --features / --tools):
    - final_feature_set.csv
    - tool_comparison_summary.csv

Outputs (written to ./delta_score_outputs/):
    - cv_auc_summary.csv                       — per-fold and mean AUC per tool/task
    - full_dataset_deltas.csv                  — Cliff's delta weights from full 835-bug set
    - full_dataset_scaling.csv                 — mean/std per feature (for external use)
    - score_vs_success_top5.csv                — per-bug scores + outcomes (internal)
    - cascade_routing_topk_sweep.csv           — cascade sweep results (all K)
    - external_scores.csv                      — scores for new dataset (external mode)
    - plots/                                   — ROC, score distributions, stability,
                                                 cascade routing curves
"""

import argparse
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import mannwhitneyu, spearmanr

warnings.filterwarnings('ignore')

ROOT_DIR = Path(__file__).resolve().parent.parent

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--mode',         default='internal',
                    choices=['internal', 'external'])
parser.add_argument('--features',     default=str(ROOT_DIR / 'full_feature_preproccessed_fixed' / 'final_feature_set_bug_reports_analysis.csv'))
parser.add_argument('--tools',        default=str(ROOT_DIR / 'tool_feature_analysis' / 'tool_comparison_summary_main.csv'))
parser.add_argument('--new_features', default=None,
                    help='Feature CSV for external dataset')
parser.add_argument('--new_tools',    default=None,
                    help='Tool comparison CSV for external dataset')
args = parser.parse_args()

# ── Paths ─────────────────────────────────────────────────────────────────────
OUT_DIR  = ROOT_DIR / 'delta_score_outputs'
PLOT_DIR = OUT_DIR / 'plots'
OUT_DIR.mkdir(exist_ok=True)
PLOT_DIR.mkdir(exist_ok=True)

# ── Config ────────────────────────────────────────────────────────────────────
# boostnsift excluded (per this session's decision), bluir added (real, complete data).
# blia held back until its CleanBaselines run produces non-empty output for more than
# Chart + 2 Math bugs.
TOOLS = ['FlexFL', 'BRaIn', 'bluir', 'buglocator', 'locus']
DISPLAY = {'FlexFL': 'FlexFL', 'BRaIn': 'BRaIn', 'bluir': 'BLUiR',
           'buglocator': 'BugLocator', 'locus': 'Locus'}
THRESHOLDS   = [1, 5]
N_SPLITS     = 5
RANDOM_STATE = 42

# TOOL_FEATURES rebuilt from the UNION of significant+practically-significant features
# across tool_vs_rest_top1/5/10.csv (not just top5 -- verified empirically that different
# K thresholds surface materially different features per tool, e.g. buglocator/locus have
# small unique-bug counts so which features clear Holm-corrected significance is sensitive
# to K; using only top5 silently dropped real signal). Per tool: union across all three K's,
# each feature scored by its max |Cliff's delta| across the K's where it qualified, ranked
# descending, capped at 10. Tools with fewer than 4 union features (bluir, buglocator) are
# backfilled from top5's full ranking (by |delta|, regardless of significance) up to 4.
# Regenerated after the tool_comparison_summary.csv 835-per-tool fix, which shifted these
# lists again (e.g. FlexFL went from 0 qualifying features needing a backfill, to 22 in the
# raw union / 10 after capping).
TOOL_FEATURES = {
    'FlexFL':     ['project_num_java_files', 'project_java_bytes', 'description_length',
                   'embedding_cluster_distance', 'txt_description_line_count',
                   'code_vocab_overlap_count', 'code_vocab_jaccard', 'ari',
                   'embedding_cluster_size', 'txt_description_avg_sentence_len'],
    'BRaIn':      ['code_vocab_jaccard', 'z_repair_readiness', 'z_hidden_reproducibility',
                   'quality_composite', 'project_num_java_files', 'txt_description_line_count',
                   'project_java_bytes', 'description_length'],
    'bluir':      ['project_num_java_files', 'z_clarity', 'z_specificity',
                   'specificity_repro_gap'],
    'buglocator': ['project_num_java_files', 'project_java_bytes', 'quality_std',
                   'z_impact_scope'],
    'locus':      ['txt_description_line_count', 'description_length',
                   'txt_title_avg_sentence_len', 'txt_description_avg_sentence_len',
                   'txt_title_digit_density'],
}

# All-vs-none: from all_vs_none_top5.csv, significant AND practically significant, by |delta|.
# k=5-only stays justified here (unlike TOOL_FEATURES): checked union across k=1/5/10 for
# all_vs_none too, and k=5 alone already captures effectively everything (all of k=1's
# qualifying features are a subset of k=5's, k=10 adds only 1 more) -- the "all tools found
# it" vs "no tool found it" contrast is a large, stable split (94 vs 203 bugs) that doesn't
# move much with the threshold, unlike the small per-tool "unique bugs" groups that motivated
# the union rebuild above. Regenerated post the tool_comparison_summary.csv 835-per-tool fix.
AVN_FEATURES = [
    'project_num_java_files', 'project_java_bytes', 'txt_description_line_count',
    'txt_description_avg_sentence_len', 'txt_title_avg_sentence_len',
    'embedding_cluster_distance', 'txt_title_digit_density', 'embedding_cluster_size',
    'ari', 'description_length', 'z_ambiguity', 'flesch', 'z_clarity', 'z_actionability',
    'kincaid', 'code_vocab_overlap_count', 'z_specificity', 'z_repair_readiness',
    'technical_completeness', 'z_hidden_reproducibility', 'z_reproducibility',
    'num_versions', 'code_vocab_jaccard', 'z_root_cause_evidence', 'buggy_vocab_size',
    'num_causal_markers',
]

INTERACTION_FEATURES = [
    'complexity_reasoning',
    'complexity_description',
    'complexity_readability',
    'reasoning_clarity',
    'structure_actionability',
    'readability_combo',
    'ambiguity_reasoning',
]


# ── Helpers ───────────────────────────────────────────────────────────────────
def cliffs_delta_fast(group1: np.ndarray, group2: np.ndarray) -> float:
    """Vectorised Cliff's delta via Mann-Whitney U."""
    g1 = np.asarray(group1, dtype=float)
    g2 = np.asarray(group2, dtype=float)
    if len(g1) == 0 or len(g2) == 0:
        return 0.0
    try:
        u_stat, _ = mannwhitneyu(g1, g2, alternative='two-sided')
    except ValueError:
        return 0.0
    n1, n2 = len(g1), len(g2)
    delta = (2 * u_stat / (n1 * n2)) - 1
    return float(delta)


def compute_deltas(df_train: pd.DataFrame,
                   features: list,
                   target_col: str,
                   comparison: str = 'binary') -> dict:
    deltas = {}

    if comparison == 'binary':
        pos = df_train[df_train[target_col] == 1]
        neg = df_train[df_train[target_col] == 0]

        for f in features:
            if f not in df_train.columns:
                deltas[f] = 0.0
                continue
            d = cliffs_delta_fast(
                pos[f].dropna().values,
                neg[f].dropna().values
            )
            if f in INTERACTION_FEATURES:
                d *= 1 / len(TOOLS)
            deltas[f] = d

    elif comparison == 'all_vs_none':
        all_bugs  = df_train[df_train[target_col] == len(TOOLS)]
        none_bugs = df_train[df_train[target_col] == 0]

        for f in features:
            if f not in df_train.columns:
                deltas[f] = 0.0
                continue
            d = cliffs_delta_fast(
                all_bugs[f].dropna().values,
                none_bugs[f].dropna().values
            )
            if f in INTERACTION_FEATURES:
                d *= 1 / len(TOOLS)
            deltas[f] = d

    return deltas


def delta_score(df: pd.DataFrame,
                features: list,
                deltas: dict,
                feature_means: dict,
                feature_stds: dict) -> np.ndarray:
    """
    Compute delta-weighted score for each bug.
    score_i = Σ_f  delta_f * z_f(i)
    where z_f = (x_f - mean_f) / std_f
    feature_means/stds are from the TRAINING set only.
    """
    scores = np.zeros(len(df))
    for f in features:
        if f not in df.columns or f not in deltas:
            continue
        delta = deltas[f]
        if abs(delta) < 1e-9:
            continue
        mu  = feature_means.get(f, 0.0)
        std = feature_stds.get(f, 1.0)
        if std < 1e-9:
            std = 1.0
        z = (df[f].fillna(mu).values - mu) / std
        scores += delta * z
    return scores


def evaluate_scores(y_true: np.ndarray,
                    scores: np.ndarray) -> dict:
    """AUC, Spearman rho, and optimal-threshold F1."""
    from sklearn.metrics import roc_auc_score, f1_score, roc_curve
    if len(np.unique(y_true)) < 2:
        return {'auc': np.nan, 'spearman': np.nan, 'f1_opt': np.nan}

    auc = roc_auc_score(y_true, scores)
    rho, _ = spearmanr(scores, y_true)

    fpr, tpr, thresholds = roc_curve(y_true, scores)
    f1s = []
    for thr in thresholds:
        pred = (scores >= thr).astype(int)
        f1s.append(f1_score(y_true, pred, zero_division=0))
    f1_opt = max(f1s) if f1s else np.nan

    return {'auc': auc, 'spearman': rho, 'f1_opt': f1_opt}


# ── Load data ─────────────────────────────────────────────────────────────────
print("Loading data...")
df = pd.read_csv(args.features)


# ── Interaction features ──────────────────────────────────────────────────────
def add_interaction_features(df):
    # NOTE: substituted z_-scored / currently-available equivalents for the original
    # repair_difficulty/reasoning_composite/actionability/clarity/coleman_liau/
    # ambiguity_type_count -- none of those survive in the current schema (redundancy-
    # pruned or from the old single-Gemini-judge schema), so every guard below used to
    # silently fail and this function was a no-op.
    if {'z_repair_readiness', 'z_reasoning_quality'}.issubset(df.columns):
        df['complexity_reasoning'] = (
            df['z_repair_readiness'] * df['z_reasoning_quality'])
    if {'z_repair_readiness', 'txt_description_line_count'}.issubset(df.columns):
        df['complexity_description'] = (
            df['z_repair_readiness'] * df['txt_description_line_count'])
    if {'z_repair_readiness', 'ari'}.issubset(df.columns):
        df['complexity_readability'] = (
            df['z_repair_readiness'] * df['ari'])
    if {'z_reasoning_quality', 'z_clarity'}.issubset(df.columns):
        df['reasoning_clarity'] = (
            df['z_reasoning_quality'] * df['z_clarity'])
    if {'txt_description_line_count', 'z_actionability'}.issubset(df.columns):
        df['structure_actionability'] = (
            df['txt_description_line_count'] * df['z_actionability'])
    # coleman_liau dropped by redundancy pass, substituted with fog (same "higher = harder"
    # direction as ari, unlike flesch)
    if {'ari', 'fog', 'flesch'}.issubset(df.columns):
        df['readability_combo'] = (
            df['ari'] + df['fog'] - df['flesch'])
    # ambiguity_type_count never existed in this schema, substituted with z_ambiguity
    if {'z_ambiguity', 'z_reasoning_quality'}.issubset(df.columns):
        df['ambiguity_reasoning'] = (
            df['z_ambiguity'] * df['z_reasoning_quality'])
    return df


df = add_interaction_features(df)

if Path(args.tools).exists():
    df_tools = pd.read_csv(args.tools)
    brain_mask = df_tools['tool'] == 'BRaIn'
    df_tools['bug_id'] = df_tools['bug_id'].astype(str)
    df_tools.loc[brain_mask, 'bug_id'] = (
        df_tools.loc[brain_mask, 'bug_id']
        .str.split('-').str[-1]
    )
    feat_keys = df[['project', 'bug_id']].copy().astype(str)
    df_tools[['project', 'bug_id']] = df_tools[['project', 'bug_id']].astype(str)
    df_filtered = df_tools.merge(feat_keys, on=['project', 'bug_id'], how='inner')
    pivot = df_filtered.pivot_table(
        index=['project', 'bug_id'], columns='tool',
        values=['top@1', 'top@5', 'top@10', 'rank'], aggfunc='max'
    ).fillna(0)
    pivot.columns = ['_'.join(str(c) for c in col) for col in pivot.columns]
    pivot = pivot.reset_index().astype({'bug_id': str})
    df['bug_id'] = df['bug_id'].astype(str)
    df = df.merge(pivot, on=['project', 'bug_id'], how='left', suffixes=('_old', ''))
    for t in TOOLS:
        for k in [1, 5, 10]:
            col = f'top@{k}_{t}'
            old = f'top@{k}_{t}_old'
            if col in df.columns:
                df[col] = df[col].fillna(0)
            elif old in df.columns:
                df.rename(columns={old: col}, inplace=True)
        # rank_<tool>: same override-from-tools-CSV pattern as top@k above, so the
        # cascade routing block's ground truth also tracks --tools instead of always
        # reading the value baked into --features.
        rcol = f'rank_{t}'
        rold = f'rank_{t}_old'
        if rcol in df.columns:
            df[rcol] = df[rcol].fillna(0)
        elif rold in df.columns:
            df.rename(columns={rold: rcol}, inplace=True)

N = len(df)
print(f"  N = {N} bugs")

# Exclude leakage / metadata columns (prefix-based: also catches mrr@k_/map_/map@k_
# columns, not just mrr_/rank_/top@ -- those didn't exist when this filter was written)
leakage = [c for c in df.columns
           if c.startswith(('mrr', 'rank_', 'map')) or 'top@' in c]
exclude  = ['project', 'bug_id'] + leakage
ALL_FEAT = (df.drop(columns=[c for c in exclude if c in df.columns])
              .select_dtypes(include=[np.number]).columns.tolist())

# Filter feature lists to available columns
for tool in TOOLS:
    TOOL_FEATURES[tool] += [f for f in INTERACTION_FEATURES if f in df.columns]
AVN_FEATURES += [f for f in INTERACTION_FEATURES if f in df.columns]
AVN_FEATURES = [f for f in AVN_FEATURES if f in ALL_FEAT]

# n_success columns (top-1 and top-5 for existing internal CV)
for thresh in THRESHOLDS:
    df[f'n_success_{thresh}'] = sum(
        df[f'top@{thresh}_{t}'].fillna(0) for t in TOOLS
    )


# ════════════════════════════════════════════════════════════════════════════
# MODE 1: INTERNAL CROSS-VALIDATED EVALUATION
# ════════════════════════════════════════════════════════════════════════════
if args.mode == 'internal':
    print("\n" + "="*70)
    print("INTERNAL CROSS-VALIDATED DELTA SCORING")
    print("="*70)

    from sklearn.model_selection import StratifiedKFold

    cv_rows          = []
    all_scores_store = {}

    # ── Per-tool prediction ──────────────────────────────────────────────────
    for thresh in THRESHOLDS:
        print(f"\n── Top-{thresh} per-tool ──")
        for tool in TOOLS:
            target_col = f'top@{thresh}_{tool}'
            y = df[target_col].fillna(0).astype(int).values
            feats = TOOL_FEATURES[tool]

            skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True,
                                  random_state=RANDOM_STATE)
            fold_aucs, fold_rhos, fold_f1s = [], [], []
            oof_scores = np.zeros(N)
            oof_y      = np.zeros(N, dtype=int)

            for fold, (train_idx, test_idx) in enumerate(skf.split(df, y)):
                df_train = df.iloc[train_idx]
                df_test  = df.iloc[test_idx]
                y_test   = y[test_idx]

                deltas = compute_deltas(df_train, feats, target_col,
                                        comparison='binary')
                means = df_train[feats].mean().to_dict()
                stds  = df_train[feats].std().to_dict()

                scores = delta_score(df_test, feats, deltas, means, stds)
                oof_scores[test_idx] = scores
                oof_y[test_idx]      = y_test

                res = evaluate_scores(y_test, scores)
                fold_aucs.append(res['auc'])
                fold_rhos.append(res['spearman'])
                fold_f1s.append(res['f1_opt'])

                cv_rows.append({
                    'thresh': thresh, 'task': f'tool_{tool}',
                    'fold': fold + 1,
                    'auc': round(res['auc'], 3),
                    'spearman': round(res['spearman'], 3),
                    'f1_opt': round(res['f1_opt'], 3),
                })

            mean_auc = np.nanmean(fold_aucs)
            mean_rho = np.nanmean(fold_rhos)
            all_scores_store[(thresh, tool)] = (oof_scores, oof_y)

            print(f"  {DISPLAY[tool]:<12}  "
                  f"AUC={mean_auc:.3f}±{np.nanstd(fold_aucs):.3f}  "
                  f"Spearman={mean_rho:.3f}±{np.nanstd(fold_rhos):.3f}  "
                  f"F1_opt={np.nanmean(fold_f1s):.3f}")

    # ── All-vs-None localizability ───────────────────────────────────────────
    print(f"\n── All-vs-None localizability ──")
    for thresh in THRESHOLDS:
        n_succ = df[f'n_success_{thresh}'].values
        y_any  = (n_succ >= 1).astype(int)
        y_all  = (n_succ == len(TOOLS)).astype(int)

        for task_name, y_bin in [
            (f'any_vs_none_top{thresh}',   y_any),
            (f'all_vs_notall_top{thresh}',  y_all),
        ]:
            skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True,
                                  random_state=RANDOM_STATE)
            fold_aucs, fold_rhos, fold_f1s = [], [], []
            oof_scores = np.zeros(N)
            oof_y      = np.zeros(N, dtype=int)

            for fold, (train_idx, test_idx) in enumerate(skf.split(df, y_bin)):
                df_train = df.iloc[train_idx].copy()
                df_test  = df.iloc[test_idx]
                y_test   = y_bin[test_idx]

                df_train['_target'] = y_bin[train_idx]
                deltas = compute_deltas(df_train, AVN_FEATURES, '_target',
                                        comparison='binary')
                means  = df_train[AVN_FEATURES].mean().to_dict()
                stds   = df_train[AVN_FEATURES].std().to_dict()

                scores = delta_score(df_test, AVN_FEATURES, deltas, means, stds)
                oof_scores[test_idx] = scores
                oof_y[test_idx]      = y_test

                res = evaluate_scores(y_test, scores)
                fold_aucs.append(res['auc'])
                fold_rhos.append(res['spearman'])
                fold_f1s.append(res['f1_opt'])

                cv_rows.append({
                    'thresh': thresh, 'task': task_name,
                    'fold': fold + 1,
                    'auc': round(res['auc'], 3),
                    'spearman': round(res['spearman'], 3),
                    'f1_opt': round(res['f1_opt'], 3),
                })

            mean_auc = np.nanmean(fold_aucs)
            mean_rho = np.nanmean(fold_rhos)
            all_scores_store[(thresh, task_name)] = (oof_scores, oof_y)

            print(f"  {task_name:<30}  "
                  f"AUC={mean_auc:.3f}±{np.nanstd(fold_aucs):.3f}  "
                  f"Spearman={mean_rho:.3f}±{np.nanstd(fold_rhos):.3f}  "
                  f"F1_opt={np.nanmean(fold_f1s):.3f}")

    # ── Save CV results ──────────────────────────────────────────────────────
    cv_df = pd.DataFrame(cv_rows)
    cv_df.to_csv(OUT_DIR / 'cv_fold_results.csv', index=False)

    summary = (cv_df.groupby(['thresh', 'task'])
               [['auc', 'spearman', 'f1_opt']]
               .agg(['mean', 'std'])
               .round(3))
    summary.to_csv(OUT_DIR / 'cv_auc_summary.csv')
    print(f"\nSummary saved to {OUT_DIR / 'cv_auc_summary.csv'}")

    # ── Delta stability plots ────────────────────────────────────────────────
    print("\nPlotting delta stability...")
    for tool in TOOLS:
        feats      = TOOL_FEATURES[tool]
        target_col = f'top@5_{tool}'
        y = df[target_col].fillna(0).astype(int).values
        skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True,
                              random_state=RANDOM_STATE)

        fold_deltas = {f: [] for f in feats}
        for train_idx, _ in skf.split(df, y):
            d = compute_deltas(df.iloc[train_idx], feats, target_col, 'binary')
            for f in feats:
                fold_deltas[f].append(d.get(f, 0.0))

        full_deltas = compute_deltas(df, feats, target_col, 'binary')

        fig, ax = plt.subplots(figsize=(7, max(3, len(feats) * 0.35)))
        ys = np.arange(len(feats))
        full_vals  = [full_deltas.get(f, 0) for f in feats]
        fold_means = [np.mean(fold_deltas[f]) for f in feats]
        fold_stds  = [np.std(fold_deltas[f])  for f in feats]

        ax.barh(ys, full_vals, height=0.4, color='steelblue',
                alpha=0.6, label='Full dataset')
        ax.errorbar(fold_means, ys, xerr=fold_stds, fmt='o',
                    color='darkorange', ms=5, lw=1.5,
                    label='CV folds (mean±std)')
        ax.set_yticks(ys)
        ax.set_yticklabels(feats, fontsize=8)
        ax.axvline(0, color='black', lw=0.8)
        ax.set_xlabel("Cliff's delta")
        ax.set_title(f"Delta stability — {DISPLAY[tool]} (Top-5)")
        ax.legend(fontsize=8)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.tight_layout()
        for ext in ['pdf', 'png']:
            plt.savefig(PLOT_DIR / f'delta_stability_{tool}.{ext}', dpi=150)
        plt.close()

    # ── ROC curves (OOF scores) ──────────────────────────────────────────────
    print("Plotting OOF ROC curves...")
    from sklearn.metrics import roc_curve, roc_auc_score

    for thresh in THRESHOLDS:
        fig, ax = plt.subplots(figsize=(6, 5))
        colors = plt.cm.tab10(np.linspace(0, 0.5, len(TOOLS)))
        for i, tool in enumerate(TOOLS):
            key = (thresh, tool)
            if key not in all_scores_store:
                continue
            scores, y_true = all_scores_store[key]
            if len(np.unique(y_true)) < 2:
                continue
            fpr, tpr, _ = roc_curve(y_true, scores)
            auc = roc_auc_score(y_true, scores)
            ax.plot(fpr, tpr, color=colors[i], lw=1.8,
                    label=f"{DISPLAY[tool]} (AUC={auc:.3f})")
        ax.plot([0, 1], [0, 1], 'k--', lw=0.8)
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title(f'ROC — Delta Scorer, Top-{thresh} (CV OOF)')
        ax.legend(fontsize=8, loc='lower right')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.tight_layout()
        for ext in ['pdf', 'png']:
            plt.savefig(PLOT_DIR / f'delta_roc_top{thresh}.{ext}', dpi=150)
        plt.close()

    # ── Score distribution by outcome ───────────────────────────────────────
    print("Plotting score distributions...")
    for thresh in THRESHOLDS:
        fig, axes = plt.subplots(1, len(TOOLS), figsize=(16, 3.5), sharey=False)
        for ax, tool in zip(axes, TOOLS):
            key = (thresh, tool)
            if key not in all_scores_store:
                continue
            scores, y_true = all_scores_store[key]
            ax.hist(scores[y_true == 0], bins=30, alpha=0.6,
                    color='#d73027', label='Fail', density=True)
            ax.hist(scores[y_true == 1], bins=30, alpha=0.6,
                    color='#1a9850', label='Success', density=True)
            ax.set_title(DISPLAY[tool], fontsize=9)
            ax.set_xlabel('Delta score', fontsize=8)
            ax.legend(fontsize=7)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
        plt.suptitle(f'Score Distribution by Outcome — Top-{thresh}',
                     fontweight='bold')
        plt.tight_layout()
        for ext in ['pdf', 'png']:
            plt.savefig(PLOT_DIR / f'delta_score_dist_top{thresh}.{ext}', dpi=150)
        plt.close()

    # ── Save per-bug OOF scores for Top-5 ───────────────────────────────────
    score_rows = df[['project', 'bug_id']].copy()
    for tool in TOOLS:
        key = (5, tool)
        if key in all_scores_store:
            scores, y_true = all_scores_store[key]
            score_rows[f'score_{tool}'] = scores
            score_rows[f'success_{tool}'] = y_true
    score_rows['score_any'] = all_scores_store.get(
        (5, 'any_vs_none_top5'), (np.zeros(N), None))[0]
    score_rows['y_any'] = (df['n_success_5'] >= 1).astype(int)
    score_rows.to_csv(OUT_DIR / 'score_vs_success_top5.csv', index=False)
    print(f"\nPer-bug OOF scores saved to {OUT_DIR / 'score_vs_success_top5.csv'}")


# ════════════════════════════════════════════════════════════════════════════
# COMPUTE & SAVE FULL-DATASET DELTAS + SCALING (for external use)
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("COMPUTING FULL-DATASET DELTAS AND SCALING PARAMETERS")
print("="*70)

delta_rows   = []
scaling_rows = []

for thresh in THRESHOLDS:
    for tool in TOOLS:
        target_col = f'top@{thresh}_{tool}'
        feats = TOOL_FEATURES[tool]
        deltas = compute_deltas(df, feats, target_col, 'binary')
        for f, d in deltas.items():
            delta_rows.append({
                'task': f'tool_{tool}', 'threshold': thresh,
                'feature': f, 'delta': round(d, 4),
            })

for thresh in THRESHOLDS:
    n_succ = df[f'n_success_{thresh}'].values
    for task, y_bin in [('any_vs_none',   (n_succ >= 1).astype(int)),
                        ('all_vs_notall', (n_succ == len(TOOLS)).astype(int))]:
        df['_avn_target'] = y_bin
        deltas = compute_deltas(df, AVN_FEATURES, '_avn_target', 'binary')
        for f, d in deltas.items():
            delta_rows.append({
                'task': task, 'threshold': thresh,
                'feature': f, 'delta': round(d, 4),
            })

all_feats_used = list(set(
    f for feats in TOOL_FEATURES.values() for f in feats
) | set(AVN_FEATURES))

for f in sorted(all_feats_used):
    if f not in df.columns:
        continue
    scaling_rows.append({
        'feature': f,
        'mean': round(df[f].mean(), 6),
        'std':  round(df[f].std(),  6),
    })

delta_df   = pd.DataFrame(delta_rows)
scaling_df = pd.DataFrame(scaling_rows)
delta_df.to_csv(OUT_DIR / 'full_dataset_deltas.csv', index=False)
scaling_df.to_csv(OUT_DIR / 'full_dataset_scaling.csv', index=False)

print(f"  Deltas:  {OUT_DIR / 'full_dataset_deltas.csv'}")
print(f"  Scaling: {OUT_DIR / 'full_dataset_scaling.csv'}")


# ════════════════════════════════════════════════════════════════════════════
# MODE 2: EXTERNAL DATASET SCORING
# ════════════════════════════════════════════════════════════════════════════
if args.mode == 'external':
    print("\n" + "="*70)
    print("EXTERNAL DATASET SCORING")
    print("="*70)

    if not args.new_features:
        raise ValueError("--new_features required for external mode")

    df_new = pd.read_csv(args.new_features)
    print(f"  New dataset: {len(df_new)} bugs")

    scaling = pd.read_csv(OUT_DIR / 'full_dataset_scaling.csv')
    means   = dict(zip(scaling['feature'], scaling['mean']))
    stds    = dict(zip(scaling['feature'], scaling['std']))
    deltas_full = pd.read_csv(OUT_DIR / 'full_dataset_deltas.csv')

    score_out = df_new[['project', 'bug_id']].copy() if 'project' in df_new.columns \
                else pd.DataFrame({'idx': range(len(df_new))})

    for thresh in THRESHOLDS:
        for tool in TOOLS:
            task_key = f'tool_{tool}'
            feats = TOOL_FEATURES[tool]
            d_sub = deltas_full[
                (deltas_full['task'] == task_key) &
                (deltas_full['threshold'] == thresh)
            ]
            deltas = dict(zip(d_sub['feature'], d_sub['delta']))
            scores = delta_score(df_new, feats, deltas, means, stds)
            score_out[f'score_{tool}_top{thresh}'] = scores.round(4)

        for task in ['any_vs_none', 'all_vs_notall']:
            d_sub = deltas_full[
                (deltas_full['task'] == task) &
                (deltas_full['threshold'] == thresh)
            ]
            deltas = dict(zip(d_sub['feature'], d_sub['delta']))
            scores = delta_score(df_new, AVN_FEATURES, deltas, means, stds)
            score_out[f'score_{task}_top{thresh}'] = scores.round(4)

    score_out.to_csv(OUT_DIR / 'external_scores.csv', index=False)
    print(f"  Scores saved to {OUT_DIR / 'external_scores.csv'}")

    if args.new_tools:
        print("\n  Ground truth provided — evaluating predictions...")
        df_new_tools = pd.read_csv(args.new_tools)

        brain_mask = df_new_tools['tool'] == 'BRaIn'
        df_new_tools.loc[brain_mask, 'bug_id'] = \
            df_new_tools.loc[brain_mask, 'bug_id'].str.split('-').str[-1]

        pivot_new = df_new_tools.pivot_table(
            index=['project', 'bug_id'], columns='tool',
            values=['top@1', 'top@5', 'top@10'], aggfunc='max'
        ).fillna(0)
        pivot_new.columns = ['_'.join(str(c) for c in col)
                             for col in pivot_new.columns]
        pivot_new = pivot_new.reset_index()

        eval_df = score_out.merge(pivot_new, on=['project', 'bug_id'], how='inner')
        print(f"  Matched {len(eval_df)} bugs with ground truth")

        from sklearn.metrics import roc_auc_score
        print(f"\n  {'Task':<30} {'AUC':>8}")
        print("  " + "-"*40)
        for thresh in THRESHOLDS:
            for tool in TOOLS:
                col_score = f'score_{tool}_top{thresh}'
                col_truth = f'top@{thresh}_{tool}'
                if col_score not in eval_df or col_truth not in eval_df:
                    continue
                y_true = eval_df[col_truth].fillna(0).astype(int)
                if len(np.unique(y_true)) < 2:
                    continue
                auc = roc_auc_score(y_true, eval_df[col_score])
                print(f"  {DISPLAY[tool]+' Top-'+str(thresh):<30} {auc:>8.3f}")

            for task in ['any_vs_none', 'all_vs_notall']:
                col_score = f'score_{task}_top{thresh}'
                n_succ = sum(
                    eval_df.get(f'top@{thresh}_{t}', pd.Series(0)).fillna(0)
                    for t in TOOLS
                )
                if task == 'any_vs_none':
                    y_true = (n_succ >= 1).astype(int)
                else:
                    y_true = (n_succ == len(TOOLS)).astype(int)
                if len(np.unique(y_true)) < 2 or col_score not in eval_df:
                    continue
                auc = roc_auc_score(y_true, eval_df[col_score])
                print(f"  {task+' Top-'+str(thresh):<30} {auc:>8.3f}")


"""
THREE-TIER CASCADE ROUTING — 5-FOLD CROSS-VALIDATED
Top-K Hit Rate + MRR + MAP Version (K = 1, 5, 10)
======================================================

Metric definitions:
  hit_rate@K  = fraction of bugs where min(rank across tier tools) <= K
  MRR         = mean(1 / best_rank_i) where best_rank_i = min rank across
                tier tools; bugs with no rank contribute 0
  MAP         = mean(1 / best_rank_i) for localised bugs only
                (i.e. MAP@cutoff used in FL literature: average precision
                 over the single ground-truth file per bug)

  "Best rank" = min rank across the tools available in the assigned tier:
    Tier 1 (IR only):   min(rank_bluir, rank_buglocator, rank_locus)
    Tier 2 (BRaIn):     min(rank_bluir, rank_buglocator, rank_locus,
                             rank_BRaIn)
    Tier 3 (FlexFL):    min(rank_bluir, rank_buglocator, rank_locus,
                             rank_BRaIn, rank_FlexFL)

Drop this block in place of the old THREE-TIER CASCADE ROUTING section.
All upstream variables must already be defined:
    df, AVN_FEATURES, compute_deltas, delta_score,
    N_SPLITS, RANDOM_STATE, OUT_DIR, PLOT_DIR
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

print("\n" + "="*70)
print("THREE-TIER CASCADE ROUTING (IR → BRaIn → FlexFL), 5-FOLD CV")
print("Hit Rate@K  |  MRR  |  MAP   —   K ∈ {1, 5, 10}")
print("="*70)

# ── Tier definitions ──────────────────────────────────────────────────────────
# boostnsift -> bluir: both are pure IR-based FL tools, same role in Tier 1.
IR_TOOLS_CASCADE = ['bluir', 'buglocator', 'locus']
K_VALUES         = [1, 5, 10]
SCORER_K         = 5   # K used to define IR success when training the scorer

# ── Load rank arrays ──────────────────────────────────────────────────────────
# rank_* columns: integer rank of the correct file, NaN / 0 if not localised.
# We convert to float and treat 0 as NaN (unlocalized).

def load_rank(tool_name: str) -> np.ndarray:
    """Return float rank array; unlocalized bugs → NaN."""
    col = f'rank_{tool_name}'
    if col not in df.columns:
        return np.full(len(df), np.nan)
    ranks = df[col].copy().astype(float)
    ranks[ranks <= 0] = np.nan   # 0 means not found in some FL outputs
    return ranks.values

rank_arrays = {
    'bluir':       load_rank('bluir'),
    'buglocator':  load_rank('buglocator'),
    'locus':       load_rank('locus'),
    'BRaIn':       load_rank('BRaIn'),
    'FlexFL':      load_rank('FlexFL'),
}

# Tier rank matrices: shape (N,) — best (min) rank across available tools
# np.nanmin ignores NaN; if ALL tools have NaN, result is NaN (unlocalized)
def tier_best_rank(tool_names: list) -> np.ndarray:
    stack = np.vstack([rank_arrays[t] for t in tool_names])   # (n_tools, N)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        best = np.nanmin(stack, axis=0)   # NaN only if all tools are NaN
    return best

tier_rank = {
    1: tier_best_rank(IR_TOOLS_CASCADE),
    2: tier_best_rank(IR_TOOLS_CASCADE + ['BRaIn']),
    3: tier_best_rank(IR_TOOLS_CASCADE + ['BRaIn', 'FlexFL']),
}

# ── Per-K binary success arrays (from best rank) ──────────────────────────────
ir_success    = {}
brain_success = {}   # cumulative: IR or BRaIn
flex_success  = {}   # cumulative: IR or BRaIn or FlexFL

for k in K_VALUES:
    ir_success[k]    = (tier_rank[1] <= k).astype(int)
    brain_success[k] = (tier_rank[2] <= k).astype(int)
    flex_success[k]  = (tier_rank[3] <= k).astype(int)

# Write scorer target to df for compute_deltas slicing
df['_ir_success'] = ir_success[SCORER_K]

# ── Helper: compute MRR and MAP from a rank array ─────────────────────────────
def mrr_from_ranks(ranks: np.ndarray, cutoff: int = 5) -> float:
    """MRR@cutoff: best rank > cutoff contributes 0."""
    capped = np.where((np.isnan(ranks)) | (ranks > cutoff), np.nan, ranks)
    rr = np.where(np.isnan(capped), 0.0, 1.0 / capped)
    return float(rr.mean())

def map_from_ranks(ranks: np.ndarray, cutoff: int = 5) -> float:
    """MAP@cutoff: mean(1/rank) over bugs localised within cutoff only."""
    localised = ranks[(~np.isnan(ranks)) & (ranks <= cutoff)]
    if len(localised) == 0:
        return float('nan')
    return float((1.0 / localised).mean())

# ── Baselines ─────────────────────────────────────────────────────────────────
# Two different kinds of baseline, kept visually separate so they don't get read
# as the same thing: "standalone" is one tool running alone (no escalation);
# "cascade" is cumulative -- Tier 2 counts a hit if IR OR BRaIn finds it, Tier 3
# if IR OR BRaIn OR FlexFL finds it. A "100% BRaIn" cascade row is therefore
# IR+BRaIn combined, not BRaIn in isolation -- compare against "Standalone: BRaIn
# only" below to see IR's actual contribution.
print(f"\n{'Baseline':<25}", end="")
for k in K_VALUES:
    print(f"  HR@{k:>2}", end="")
print(f"    MRR    MAP")
print("-" * (25 + 9 * len(K_VALUES) + 16))

standalone_hr = {
    'Standalone: BRaIn only':  {k: (rank_arrays['BRaIn']  <= k).mean() for k in K_VALUES},
    'Standalone: FlexFL only': {k: (rank_arrays['FlexFL'] <= k).mean() for k in K_VALUES},
}
standalone_mrr = {
    'Standalone: BRaIn only':  mrr_from_ranks(rank_arrays['BRaIn'],  cutoff=5),
    'Standalone: FlexFL only': mrr_from_ranks(rank_arrays['FlexFL'], cutoff=5),
}
standalone_map = {
    'Standalone: BRaIn only':  map_from_ranks(rank_arrays['BRaIn'],  cutoff=5),
    'Standalone: FlexFL only': map_from_ranks(rank_arrays['FlexFL'], cutoff=5),
}
for name in standalone_hr:
    print(f"{name:<25}", end="")
    for k in K_VALUES:
        print(f"  {standalone_hr[name][k]:.3f}", end="")
    print(f"  {standalone_mrr[name]:.3f}  {standalone_map[name]:.3f}")
print()

baselines_hr = {
    'Cascade: IR only':              {k: ir_success[k].mean()    for k in K_VALUES},
    'Cascade: IR+BRaIn':             {k: brain_success[k].mean() for k in K_VALUES},
    'Cascade: IR+BRaIn+FlexFL (Oracle)': {k: flex_success[k].mean()  for k in K_VALUES},
}
baselines_mrr = {
    'Cascade: IR only':              mrr_from_ranks(tier_rank[1], cutoff=5),
    'Cascade: IR+BRaIn':             mrr_from_ranks(tier_rank[2], cutoff=5),
    'Cascade: IR+BRaIn+FlexFL (Oracle)': mrr_from_ranks(tier_rank[3], cutoff=5),
}
baselines_map = {
    'Cascade: IR only':              map_from_ranks(tier_rank[1], cutoff=5),
    'Cascade: IR+BRaIn':             map_from_ranks(tier_rank[2], cutoff=5),
    'Cascade: IR+BRaIn+FlexFL (Oracle)': map_from_ranks(tier_rank[3], cutoff=5),
}

for name in baselines_hr:
    print(f"{name:<25}", end="")
    for k in K_VALUES:
        print(f"  {baselines_hr[name][k]:.3f}", end="")
    print(f"  {baselines_mrr[name]:.3f}  {baselines_map[name]:.3f}")

# ── Energy cost per tier ────────────────────────────────────────────────────────
# Per-bug energy (kWh) and CO2 (kg) for each tier, built from the measured energy
# logs. IR and FlexFL logs are per-bug (mean of the per-bug sums); the BRaIn log
# only has pipeline-stage totals across the whole run, so its marginal per-bug
# cost is approximated as (sum of successful-stage energy) / N bugs.
ENERGY_IR_PATH    = ROOT_DIR / 'energy_consumption' / 'IR' / 'energy_per_bug.csv'
ENERGY_BRAIN_PATH = ROOT_DIR / 'energy_consumption' / 'BRaIn' / 'Energy_Logs' / 'energy_summary.csv'
ENERGY_FLEXFL_PATH = ROOT_DIR / 'energy_consumption' / 'FlexFL' / 'energy_log.csv'

tier_energy_kwh = None
tier_co2_kg = None

if ENERGY_IR_PATH.exists() and ENERGY_BRAIN_PATH.exists() and ENERGY_FLEXFL_PATH.exists():
    energy_ir = pd.read_csv(ENERGY_IR_PATH)
    ir_per_bug = (energy_ir[energy_ir['tool'].isin(IR_TOOLS_CASCADE)]
                  .groupby('bug_id')[['energy_consumed_kwh', 'emissions_kg_co2']].sum())
    e1_kwh, e1_co2 = ir_per_bug['energy_consumed_kwh'].mean(), ir_per_bug['emissions_kg_co2'].mean()

    energy_brain = pd.read_csv(ENERGY_BRAIN_PATH)
    success = energy_brain[energy_brain['status'] == 'success']
    e2_kwh = success['codecarbon_energy_kwh'].sum() / N
    e2_co2 = success['codecarbon_co2eq_kg'].sum() / N

    energy_flex = pd.read_csv(ENERGY_FLEXFL_PATH)
    flex_per_bug = energy_flex.groupby('bug')[['codecarbon_energy_kwh', 'codecarbon_co2_kg']].sum()
    e3_kwh, e3_co2 = flex_per_bug['codecarbon_energy_kwh'].mean(), flex_per_bug['codecarbon_co2_kg'].mean()

    # Cumulative cost of the tier a bug is routed to (cascade: IR, then +BRaIn, then +FlexFL)
    tier_energy_kwh = {1: e1_kwh, 2: e1_kwh + e2_kwh, 3: e1_kwh + e2_kwh + e3_kwh}
    tier_co2_kg     = {1: e1_co2, 2: e1_co2 + e2_co2, 3: e1_co2 + e2_co2 + e3_co2}

    print(f"\n{'Per-bug energy cost':<25}  {'kWh':>10}  {'CO2 (kg)':>10}")
    print(f"{'  IR (Tier 1)':<25}  {e1_kwh:>10.6f}  {e1_co2:>10.6f}")
    print(f"{'  + BRaIn (Tier 2)':<25}  {e2_kwh:>10.6f}  {e2_co2:>10.6f}")
    print(f"{'  + FlexFL (Tier 3)':<25}  {e3_kwh:>10.6f}  {e3_co2:>10.6f}")
    print(f"{'Full cascade (everyone → Tier 3)':<25} {tier_energy_kwh[3]:>10.6f}  {tier_co2_kg[3]:>10.6f}")

    # Per-bug *realized* IR/FlexFL energy, aligned to df's row order, so the energy
    # cost of a routing policy can be computed from the actual bugs it assigned to
    # each tier -- not just (tier average) x (bug count). This is what lets the
    # random-routing baseline's energy come out different from the scorer's: same
    # tier sizes, but if the scorer systematically escalates bugs that happen to be
    # more/less expensive to run than a random bug would be, the totals diverge.
    # BRaIn has no per-bug log, so its marginal cost still uses the tier average.
    bug_key = df['project'].astype(str) + '-' + df['bug_id'].astype(str)
    bug_ir_kwh   = bug_key.map(ir_per_bug['energy_consumed_kwh']).fillna(e1_kwh).values
    bug_ir_co2   = bug_key.map(ir_per_bug['emissions_kg_co2']).fillna(e1_co2).values
    bug_flex_kwh = bug_key.map(flex_per_bug['codecarbon_energy_kwh']).fillna(e3_kwh).values
    bug_flex_co2 = bug_key.map(flex_per_bug['codecarbon_co2_kg']).fillna(e3_co2).values
else:
    print("\n[energy] one or more energy log files not found — skipping energy estimate")

# ── Sweep configuration ───────────────────────────────────────────────────────
SWEEP = [
    (100,  0,  0),
    ( 70, 20, 10),
    ( 60, 25, 15),
    ( 50, 30, 20),
    ( 50, 40, 10),
    ( 40, 40, 20),
    ( 35, 40, 25),
    ( 35, 35, 30),
    ( 30, 40, 30),
    ( 20, 50, 30),
    ( 10, 60, 30),
    (  0, 70, 30),
    (  0, 100,  0),
    (  0,   0, 100),
]

# ── 5-fold CV ─────────────────────────────────────────────────────────────────
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True,
                      random_state=RANDOM_STATE)

fold_hit_rate  = {s: {k: [] for k in K_VALUES} for s in SWEEP}
fold_mrr       = {s: [] for s in SWEEP}
fold_map       = {s: [] for s in SWEEP}
fold_tier1_frac = {s: [] for s in SWEEP}
fold_tier2_frac = {s: [] for s in SWEEP}
fold_flex_frac = {s: [] for s in SWEEP}
fold_auc_casc  = []

# Realized per-bug energy/CO2 for the actual bugs each policy assigns to each tier
# (only populated when the energy logs were found -- see tier_energy_kwh above).
fold_energy_kwh      = {s: [] for s in SWEEP}
fold_co2_kg          = {s: [] for s in SWEEP}
fold_energy_kwh_rand = {s: [] for s in SWEEP}
fold_co2_kg_rand     = {s: [] for s in SWEEP}

# Random-routing baseline: same tier sizes as the scorer-based routing (matched budget),
# just randomly assigned instead of score-based -- tests whether the IR-success scorer is
# actually doing anything, vs. blindly sending the same fraction of bugs to each tier.
rand_rng = np.random.RandomState(RANDOM_STATE)
fold_hit_rate_rand = {s: {k: [] for k in K_VALUES} for s in SWEEP}
fold_mrr_rand      = {s: [] for s in SWEEP}
fold_map_rand      = {s: [] for s in SWEEP}

for fold, (tr_idx, te_idx) in enumerate(
        skf.split(df, df['_ir_success'])):

    df_tr = df.iloc[tr_idx]
    df_te = df.iloc[te_idx]
    n_te  = len(te_idx)

    # Train scorer on IR top-SCORER_K success within this fold only
    deltas = compute_deltas(df_tr, AVN_FEATURES, '_ir_success',
                            comparison='binary')
    means  = df_tr[AVN_FEATURES].mean().to_dict()
    stds   = df_tr[AVN_FEATURES].std().to_dict()

    te_scores = delta_score(df_te, AVN_FEATURES, deltas, means, stds)
    tr_scores = delta_score(df_tr, AVN_FEATURES, deltas, means, stds)

    if len(np.unique(ir_success[SCORER_K][te_idx])) == 2:
        fold_auc_casc.append(
            roc_auc_score(ir_success[SCORER_K][te_idx], te_scores)
        )

    for s in SWEEP:
        pct_ir, pct_brain, pct_flex = s
        assert pct_ir + pct_brain + pct_flex == 100

        # Thresholds from training score distribution (no leakage)
        t_high = (np.percentile(tr_scores, 100 - pct_ir)
                  if pct_ir < 100 else -np.inf)
        t_low  = (np.percentile(tr_scores, pct_flex)
                  if pct_flex > 0 else -np.inf)

        # Mutually exclusive tier assignment
        tier1 = te_scores >= t_high
        tier3 = te_scores <  t_low
        tier2 = ~tier1 & ~tier3

        fold_tier1_frac[s].append(tier1.sum() / n_te)
        fold_tier2_frac[s].append(tier2.sum() / n_te)
        fold_flex_frac[s].append(tier3.sum() / n_te)

        if tier_energy_kwh:
            ir_te_kwh, ir_te_co2     = bug_ir_kwh[te_idx],   bug_ir_co2[te_idx]
            flex_te_kwh, flex_te_co2 = bug_flex_kwh[te_idx], bug_flex_co2[te_idx]
            # Every bug runs IR regardless of tier. Tier2/Tier3 bugs additionally pay
            # BRaIn's marginal cost (tier-average, no per-bug log); Tier3 bugs also
            # pay FlexFL's actual per-bug cost.
            n_escalated = tier2.sum() + tier3.sum()
            energy_total = (ir_te_kwh.sum() + n_escalated * e2_kwh
                            + flex_te_kwh[tier3].sum())
            co2_total = (ir_te_co2.sum() + n_escalated * e2_co2
                        + flex_te_co2[tier3].sum())
            fold_energy_kwh[s].append(energy_total / n_te)
            fold_co2_kg[s].append(co2_total / n_te)

        # ── Hit rate per K ────────────────────────────────────────────────
        for k in K_VALUES:
            ir_te    = ir_success[k][te_idx]
            brain_te = brain_success[k][te_idx]
            flex_te  = flex_success[k][te_idx]

            hits1 = tier1 & (ir_te == 1)
            hits2 = tier2 & (brain_te == 1)   # cumulative: IR or BRaIn
            hits3 = tier3 & (flex_te == 1)    # cumulative: IR or BRaIn or FlexFL

            hit_rate = (hits1.sum() + hits2.sum() + hits3.sum()) / n_te
            fold_hit_rate[s][k].append(hit_rate)

        # ── MRR and MAP ───────────────────────────────────────────────────
        # For each bug, pick the rank from its assigned tier's best rank array
        routed_ranks = np.full(n_te, np.nan)
        routed_ranks[tier1] = tier_rank[1][te_idx][tier1]
        routed_ranks[tier2] = tier_rank[2][te_idx][tier2]
        routed_ranks[tier3] = tier_rank[3][te_idx][tier3]

        fold_mrr[s].append(mrr_from_ranks(routed_ranks))
        fold_map[s].append(map_from_ranks(routed_ranks))

        # ── Random-routing baseline (same tier sizes, blind assignment) ────
        n1, n2, n3 = int(tier1.sum()), int(tier2.sum()), int(tier3.sum())
        perm = rand_rng.permutation(n_te)
        rtier1 = np.zeros(n_te, dtype=bool)
        rtier2 = np.zeros(n_te, dtype=bool)
        rtier3 = np.zeros(n_te, dtype=bool)
        rtier1[perm[:n1]] = True
        rtier2[perm[n1:n1 + n2]] = True
        rtier3[perm[n1 + n2:n1 + n2 + n3]] = True

        for k in K_VALUES:
            ir_te    = ir_success[k][te_idx]
            brain_te = brain_success[k][te_idx]
            flex_te  = flex_success[k][te_idx]

            rhits1 = rtier1 & (ir_te == 1)
            rhits2 = rtier2 & (brain_te == 1)
            rhits3 = rtier3 & (flex_te == 1)

            rand_hit_rate = (rhits1.sum() + rhits2.sum() + rhits3.sum()) / n_te
            fold_hit_rate_rand[s][k].append(rand_hit_rate)

        rrouted_ranks = np.full(n_te, np.nan)
        rrouted_ranks[rtier1] = tier_rank[1][te_idx][rtier1]
        rrouted_ranks[rtier2] = tier_rank[2][te_idx][rtier2]
        rrouted_ranks[rtier3] = tier_rank[3][te_idx][rtier3]

        fold_mrr_rand[s].append(mrr_from_ranks(rrouted_ranks))
        fold_map_rand[s].append(map_from_ranks(rrouted_ranks))

        if tier_energy_kwh:
            # Same bugs, same per-bug costs -- only which bugs land in which tier
            # differs (random permutation vs. the scorer's choice), so any gap here
            # vs. fold_energy_kwh[s] is entirely down to which bugs got escalated.
            n_escalated_r = rtier2.sum() + rtier3.sum()
            energy_total_r = (ir_te_kwh.sum() + n_escalated_r * e2_kwh
                              + flex_te_kwh[rtier3].sum())
            co2_total_r = (ir_te_co2.sum() + n_escalated_r * e2_co2
                          + flex_te_co2[rtier3].sum())
            fold_energy_kwh_rand[s].append(energy_total_r / n_te)
            fold_co2_kg_rand[s].append(co2_total_r / n_te)

# ── Results tables ────────────────────────────────────────────────────────────
mean_auc_c = np.mean(fold_auc_casc) if fold_auc_casc else float('nan')
std_auc_c  = np.std(fold_auc_casc)  if fold_auc_casc else float('nan')
print(f"\nIR-success scorer (Top-{SCORER_K}) AUC: "
      f"{mean_auc_c:.3f} ± {std_auc_c:.3f}")

routing_rows = []

# Print combined table (scorer-routed vs. random-routed at the same tier sizes).
# Column headers spell out that tiers are cumulative -- "Tier2%" is the % of bugs
# getting IR+BRaIn, not BRaIn alone -- so this can't be misread as a per-tool split.
print("\n(tier %'s are cumulative: Tier1=IR only, Tier2=IR+BRaIn, Tier3=IR+BRaIn+FlexFL)")
print(f"\n{'Tier1%':>6} {'Tier2%':>7} {'Tier3%':>8} | "
      f"{'HR@1':>6} {'HR@5':>6} {'HR@10':>6} | "
      f"{'MRR':>6} {'MAP':>6} | "
      f"{'rndHR@5':>8} {'rndMRR':>7} {'rndMAP':>7} | "
      f"{'Tier3frac':>9}" +
      ("  |  energy_kWh  rndEnergy  saved_%" if tier_energy_kwh else ""))
print("-" * (100 + (37 if tier_energy_kwh else 0)))

for s in SWEEP:
    pct_ir, pct_brain, pct_flex = s
    hr1   = np.mean(fold_hit_rate[s][1])
    hr5   = np.mean(fold_hit_rate[s][5])
    hr10  = np.mean(fold_hit_rate[s][10])
    mrr   = np.mean(fold_mrr[s])
    mapp  = np.mean(fold_map[s])
    t1_frac = np.mean(fold_tier1_frac[s])
    t2_frac = np.mean(fold_tier2_frac[s])
    flex  = np.mean(fold_flex_frac[s])

    rhr1  = np.mean(fold_hit_rate_rand[s][1])
    rhr5  = np.mean(fold_hit_rate_rand[s][5])
    rhr10 = np.mean(fold_hit_rate_rand[s][10])
    rmrr  = np.mean(fold_mrr_rand[s])
    rmapp = np.mean(fold_map_rand[s])

    line = (f"{pct_ir:>5}% {pct_brain:>6}% {pct_flex:>7}% | "
            f"{hr1:>6.3f} {hr5:>6.3f} {hr10:>6.3f} | "
            f"{mrr:>6.3f} {mapp:>6.3f} | "
            f"{rhr5:>8.3f} {rmrr:>7.3f} {rmapp:>7.3f} | "
            f"{flex:>9.2f}")

    energy_kwh = energy_co2 = energy_saved_pct = co2_saved_pct = None
    energy_kwh_rand = energy_co2_rand = energy_lift_vs_random = None
    if tier_energy_kwh:
        # Realized energy: actual per-bug IR/FlexFL cost for the bugs each policy
        # assigned to each tier (see fold loop), not (tier average) x (bug count).
        energy_kwh = np.mean(fold_energy_kwh[s])
        energy_co2 = np.mean(fold_co2_kg[s])
        energy_saved_pct = (1 - energy_kwh / tier_energy_kwh[3]) * 100
        co2_saved_pct    = (1 - energy_co2 / tier_co2_kg[3]) * 100

        # Same tier sizes as the real routing (matched budget) -- any gap from
        # energy_kwh here is purely down to *which* bugs the scorer escalates vs.
        # a random pick of the same count.
        energy_kwh_rand = np.mean(fold_energy_kwh_rand[s])
        energy_co2_rand = np.mean(fold_co2_kg_rand[s])
        energy_lift_vs_random = energy_kwh_rand - energy_kwh

        line += f"  |  {energy_kwh:>10.6f}  {energy_kwh_rand:>9.6f}  {energy_saved_pct:>6.1f}%"

    print(line)

    routing_rows.append({
        # Target composition of each cumulative tier (see header note above):
        # tier1_pct_ir_only is the % of bugs assigned to IR-only; tier2 and tier3
        # are the % assigned to (IR+BRaIn) and (IR+BRaIn+FlexFL) respectively --
        # NOT the % of bugs that see BRaIn/FlexFL in isolation.
        'tier1_pct_ir_only':                 pct_ir,
        'tier2_pct_ir_plus_brain':           pct_brain,
        'tier3_pct_ir_plus_brain_plus_flexfl': pct_flex,
        'hr@1_mean':        round(hr1,  4),
        'hr@5_mean':        round(hr5,  4),
        'hr@10_mean':       round(hr10, 4),
        'hr@1_std':         round(np.std(fold_hit_rate[s][1]),  4),
        'hr@5_std':         round(np.std(fold_hit_rate[s][5]),  4),
        'hr@10_std':        round(np.std(fold_hit_rate[s][10]), 4),
        'mrr_mean':         round(mrr,  4),
        'mrr_std':          round(np.std(fold_mrr[s]),          4),
        'map_mean':         round(mapp, 4),
        'map_std':          round(np.std(fold_map[s]),          4),
        'tier1_frac_mean':  round(t1_frac, 4),
        'tier2_frac_mean':  round(t2_frac, 4),
        'tier3_frac_mean':  round(flex, 4),
        # Random-routing baseline, same tier sizes each fold (matched budget)
        'random_hr@1_mean':  round(rhr1,  4),
        'random_hr@5_mean':  round(rhr5,  4),
        'random_hr@10_mean': round(rhr10, 4),
        'random_mrr_mean':   round(rmrr,  4),
        'random_map_mean':   round(rmapp, 4),
        'lift_hr@5_vs_random': round(hr5 - rhr5, 4),
        'lift_mrr_vs_random':  round(mrr - rmrr, 4),
        # Realized energy cost (actual per-bug IR/FlexFL cost for the bugs each
        # policy assigned to each tier) vs. running the full cascade on every bug
        'energy_kwh_mean':      round(energy_kwh, 6) if energy_kwh is not None else None,
        'co2_kg_mean':          round(energy_co2, 6) if energy_co2 is not None else None,
        'energy_saved_pct_vs_full_cascade': round(energy_saved_pct, 2) if energy_saved_pct is not None else None,
        'co2_saved_pct_vs_full_cascade':    round(co2_saved_pct, 2) if co2_saved_pct is not None else None,
        # Random-routing baseline's energy at the *same* tier sizes -- isolates
        # whether the scorer's bug selection (not just tier sizes) affects cost.
        'random_energy_kwh_mean': round(energy_kwh_rand, 6) if energy_kwh_rand is not None else None,
        'random_co2_kg_mean':     round(energy_co2_rand, 6) if energy_co2_rand is not None else None,
        'energy_lift_vs_random_kwh': round(energy_lift_vs_random, 6) if energy_lift_vs_random is not None else None,
    })

# ── Best operating points ─────────────────────────────────────────────────────
print(f"\n=== Best results by FlexFL budget ===")
for budget in [0.0, 0.10, 0.20, 0.30]:
    candidates = [
        (s, np.mean(fold_hit_rate[s][5]),   # rank by HR@5
             np.mean(fold_mrr[s]),
             np.mean(fold_map[s]))
        for s in SWEEP
        if np.mean(fold_flex_frac[s]) <= budget + 0.02
    ]
    if not candidates:
        continue
    best = max(candidates, key=lambda x: x[1])
    best_s = best[0]
    energy_note = ""
    if tier_energy_kwh:
        t1 = np.mean(fold_tier1_frac[best_s])
        t2 = np.mean(fold_tier2_frac[best_s])
        t3 = np.mean(fold_flex_frac[best_s])
        kwh = t1 * tier_energy_kwh[1] + t2 * tier_energy_kwh[2] + t3 * tier_energy_kwh[3]
        saved_pct = (1 - kwh / tier_energy_kwh[3]) * 100
        energy_note = f"  energy_saved={saved_pct:.1f}%"
    print(f"  FlexFL ≤{budget:.0%}:  "
          f"HR@5={best[1]:.3f}  MRR={best[2]:.3f}  MAP={best[3]:.3f}  "
          f"[IR={best_s[0]}%, BRaIn={best_s[1]}%, FlexFL={best_s[2]}%]" + energy_note)

# ── Save CSV ───────────────────────────────────────────────────────────────────
routing_df = pd.DataFrame(routing_rows)
routing_df.to_csv(OUT_DIR / 'cascade_routing_topk_sweep.csv', index=False)
print(f"\nSweep results saved to {OUT_DIR / 'cascade_routing_topk_sweep.csv'}")

# ── Plots ─────────────────────────────────────────────────────────────────────
bl_colors = {
    'always_ir':    '#4dac26',
    'always_brain': '#f4a582',
    'always_flex':  '#d73027',
}

# Hit rate curves (one per K)
for k in K_VALUES:
    fig, ax = plt.subplots(figsize=(7, 5))

    xs = [np.mean(fold_flex_frac[s])   for s in SWEEP]
    ys = [np.mean(fold_hit_rate[s][k]) for s in SWEEP]
    es = [np.std(fold_hit_rate[s][k])  for s in SWEEP]
    order = np.argsort(xs)
    xs_s, ys_s, es_s = [xs[i] for i in order], [ys[i] for i in order], [es[i] for i in order]

    ax.plot(xs_s, ys_s, 'o-', color='#2166ac', lw=2, ms=6,
            label='Three-tier cascade')
    ax.fill_between(xs_s, [y-e for y,e in zip(ys_s,es_s)],
                          [y+e for y,e in zip(ys_s,es_s)],
                    alpha=0.15, color='#2166ac')
    ax.axhline(baselines_hr['Cascade: IR+BRaIn'][k],
               color=bl_colors['always_brain'], lw=1.5, ls='-.',
               label=f"Always-IR+BRaIn: {baselines_hr['Cascade: IR+BRaIn'][k]:.3f}")
    ax.axhline(baselines_hr['Cascade: IR only'][k],
               color=bl_colors['always_ir'], lw=1.5, ls=':',
               label=f"Always-IR: {baselines_hr['Cascade: IR only'][k]:.3f}")

    ax.set_xlabel('Fraction of bugs invoking FlexFL', fontsize=10)
    ax.set_ylabel(f'Top-{k} Hit Rate', fontsize=10)
    ax.set_title(f'Three-Tier Cascade — Top-{k} Hit Rate\n(IR → BRaIn → FlexFL, 5-fold CV)', fontsize=11)
    ax.set_xlim(-0.03, 1.05)
    ax.legend(fontsize=8, loc='lower right')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    for ext in ['pdf', 'png']:
        plt.savefig(PLOT_DIR / f'cascade_routing_top{k}.{ext}', dpi=150, bbox_inches='tight')
    plt.close()

# MRR and MAP combined curve
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, metric, fold_data, bl_data, ylabel in [
    (axes[0], 'MRR', fold_mrr, baselines_mrr, 'MRR'),
    (axes[1], 'MAP', fold_map, baselines_map, 'MAP'),
]:
    xs = [np.mean(fold_flex_frac[s]) for s in SWEEP]
    ys = [np.mean(fold_data[s])      for s in SWEEP]
    es = [np.std(fold_data[s])       for s in SWEEP]
    order = np.argsort(xs)
    xs_s = [xs[i] for i in order]
    ys_s = [ys[i] for i in order]
    es_s = [es[i] for i in order]

    ax.plot(xs_s, ys_s, 'o-', color='#2166ac', lw=2, ms=6,
            label='Three-tier cascade')
    ax.fill_between(xs_s, [y-e for y,e in zip(ys_s,es_s)],
                          [y+e for y,e in zip(ys_s,es_s)],
                    alpha=0.15, color='#2166ac')
    ax.axhline(bl_data['Cascade: IR+BRaIn'],
               color=bl_colors['always_brain'], lw=1.5, ls='-.',
               label=f"Always-IR+BRaIn: {bl_data['Cascade: IR+BRaIn']:.3f}")
    ax.axhline(bl_data['Cascade: IR only'],
               color=bl_colors['always_ir'], lw=1.5, ls=':',
               label=f"Always-IR: {bl_data['Cascade: IR only']:.3f}")
    ax.set_xlabel('Fraction of bugs invoking FlexFL', fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(f'Three-Tier Cascade — {metric}\n(IR → BRaIn → FlexFL, 5-fold CV)', fontsize=11)
    ax.set_xlim(-0.03, 1.05)
    ax.legend(fontsize=8, loc='lower right')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

plt.tight_layout()
for ext in ['pdf', 'png']:
    plt.savefig(PLOT_DIR / f'cascade_mrr_map.{ext}', dpi=150, bbox_inches='tight')
plt.close()
print(f"Plots saved to {PLOT_DIR}/")