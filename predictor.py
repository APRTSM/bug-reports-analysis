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
    - cv_auc_summary.csv              — per-fold and mean AUC per tool/task
    - full_dataset_deltas.csv         — Cliff's delta weights from full 835-bug set
    - full_dataset_scaling.csv        — mean/std per feature (for external use)
    - score_vs_success_top5.csv       — per-bug scores + outcomes (internal)
    - external_scores.csv             — scores for new dataset (external mode)
    - plots/                          — ROC, score distributions, stability plots
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

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--mode',         default='internal',
                    choices=['internal','external'])
parser.add_argument('--features',     default='final_feature_set.csv')
parser.add_argument('--tools',        default='tool_comparison_summary.csv')
parser.add_argument('--new_features', default=None,
                    help='Feature CSV for external dataset')
parser.add_argument('--new_tools',    default=None,
                    help='Tool comparison CSV for external dataset')
args = parser.parse_args()

# ── Paths ─────────────────────────────────────────────────────────────────────
OUT_DIR  = Path('delta_score_outputs')
PLOT_DIR = OUT_DIR / 'plots'
OUT_DIR.mkdir(exist_ok=True)
PLOT_DIR.mkdir(exist_ok=True)

# ── Config ────────────────────────────────────────────────────────────────────
TOOLS = ['FlexFL', 'BRaIn', 'boostnsift', 'buglocator', 'locus']
DISPLAY = {'FlexFL':'FlexFL', 'BRaIn':'BRaIn', 'boostnsift':'BoostNSift',
           'buglocator':'BugLocator', 'locus':'Locus'}
THRESHOLDS = [1, 5]
N_SPLITS   = 5
RANDOM_STATE = 42

# Features to use per task
# Tool-specific: from tool-vs-rest analysis
TOOL_FEATURES = {
    'FlexFL':     ['repair_difficulty','txt_description_line_count',
                   'reasoning_composite','actionability','clarity',
                   'ari','coleman_liau','txt_title_digit_density'],
    'BRaIn':      ['ari','txt_description_line_count','technical_completeness','clarity'],
    'boostnsift': ['repair_difficulty','expected_observed_alignment',
                   'ari','coleman_liau'],
    'buglocator': ['repair_difficulty','txt_description_line_count',
                   'txt_title_avg_sentence_len'],
    'locus':      ['repair_difficulty','txt_title_digit_density',
                   'txt_title_avg_sentence_len','expected_observed_alignment',
                   'concept_network_concept_breadth','embedding_cluster_size'],
}
# All-vs-none: from all-vs-none Cliff's delta analysis
AVN_FEATURES = [
    'reasoning_composite','actionability','txt_description_line_count',
    'txt_title_digit_density','txt_title_avg_sentence_len','clarity',
    'expected_observed_alignment','technical_completeness',
    'embedding_cluster_distance','description_length','num_versions',
    'repair_difficulty','coleman_liau','ari','ambiguity_type_count',
    'flesch','kincaid','embedding_cluster_size',
]

INTERACTION_FEATURES = [
    'complexity_reasoning',
    'complexity_description',
    'complexity_readability',
    'reasoning_clarity',
    'structure_actionability',
    'readability_combo',
    'ambiguity_reasoning'
]


# ── Helpers ───────────────────────────────────────────────────────────────────
def cliffs_delta(group1: np.ndarray, group2: np.ndarray) -> float:
    """
    Cliff's delta: P(X > Y) - P(X < Y) for X in group1, Y in group2.
    Positive = group1 tends to be larger.
    """
    n1, n2 = len(group1), len(group2)
    if n1 == 0 or n2 == 0:
        return 0.0
    dominance = sum(
        (1 if x > y else -1 if x < y else 0)
        for x in group1 for y in group2
    )
    return dominance / (n1 * n2)


def cliffs_delta_fast(group1: np.ndarray, group2: np.ndarray) -> float:
    """Vectorised Cliff's delta (faster for large groups)."""
    g1 = np.asarray(group1, dtype=float)
    g2 = np.asarray(group2, dtype=float)
    if len(g1) == 0 or len(g2) == 0:
        return 0.0
    # U statistic gives concordant pairs
    try:
        u_stat, _ = mannwhitneyu(g1, g2, alternative='two-sided')
    except ValueError:
        return 0.0
    n1, n2 = len(g1), len(g2)
    # Convert U to delta: delta = (2U / n1*n2) - 1
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
                d *= 1.5

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
                d *= 1.5

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

    # Find threshold maximising F1 on this fold
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

# ── Interaction features derived from significant feature patterns ───────────

def add_interaction_features(df):

    # Core interaction: complexity × reasoning
    if {'repair_difficulty','reasoning_composite'}.issubset(df.columns):
        df['complexity_reasoning'] = (
            df['repair_difficulty'] * df['reasoning_composite']
        )

    # Complexity × description structure
    if {'repair_difficulty','txt_description_line_count'}.issubset(df.columns):
        df['complexity_description'] = (
            df['repair_difficulty'] * df['txt_description_line_count']
        )

    # Complexity × readability (FlexFL pattern)
    if {'repair_difficulty','ari'}.issubset(df.columns):
        df['complexity_readability'] = (
            df['repair_difficulty'] * df['ari']
        )

    # Reasoning × clarity
    if {'reasoning_composite','clarity'}.issubset(df.columns):
        df['reasoning_clarity'] = (
            df['reasoning_composite'] * df['clarity']
        )

    # Structure × actionability
    if {'txt_description_line_count','actionability'}.issubset(df.columns):
        df['structure_actionability'] = (
            df['txt_description_line_count'] * df['actionability']
        )

    # Readability cluster (BoostNSift pattern)
    if {'ari','coleman_liau','flesch'}.issubset(df.columns):
        df['readability_combo'] = (
            df['ari'] + df['coleman_liau'] - df['flesch']
        )

    # Semantic ambiguity × reasoning
    if {'ambiguity_type_count','reasoning_composite'}.issubset(df.columns):
        df['ambiguity_reasoning'] = (
            df['ambiguity_type_count'] * df['reasoning_composite']
        )

    return df


df = add_interaction_features(df)

if Path(args.tools).exists():
    df_tools = pd.read_csv(args.tools)
    brain_mask = df_tools['tool'] == 'BRaIn'

    df_tools['bug_id'] = df_tools['bug_id'].astype(str)

    df_tools.loc[brain_mask, 'bug_id'] = (
        df_tools.loc[brain_mask, 'bug_id']
        .str.split('-')
        .str[-1]
    )
    feat_keys = df[['project','bug_id']].copy().astype(str)
    df_tools[['project','bug_id']] = df_tools[['project','bug_id']].astype(str)
    df_filtered = df_tools.merge(feat_keys, on=['project','bug_id'], how='inner')
    pivot = df_filtered.pivot_table(
        index=['project','bug_id'], columns='tool',
        values=['top@1','top@5'], aggfunc='max'
    ).fillna(0)
    pivot.columns = ['_'.join(str(c) for c in col) for col in pivot.columns]
    pivot = pivot.reset_index().astype({'bug_id': str})
    df['bug_id'] = df['bug_id'].astype(str)
    df = df.merge(pivot, on=['project','bug_id'], how='left', suffixes=('_old',''))
    for t in TOOLS:
        for k in [1, 5]:
            col = f'top@{k}_{t}'
            old = f'top@{k}_{t}_old'
            if col in df.columns:
                df[col] = df[col].fillna(0)
            elif old in df.columns:
                df.rename(columns={old: col}, inplace=True)

N = len(df)
print(f"  N = {N} bugs")

# Exclude leakage / metadata columns
leakage = [c for c in df.columns
           if c.startswith('mrr_') or c.startswith('rank_') or 'top@' in c]
exclude  = ['project','bug_id'] + leakage
ALL_FEAT = (df.drop(columns=[c for c in exclude if c in df.columns])
              .select_dtypes(include=[np.number]).columns.tolist())

# Filter feature lists to available columns
for tool in TOOLS:
    TOOL_FEATURES[tool] += [f for f in INTERACTION_FEATURES if f in df.columns]
AVN_FEATURES = [f for f in AVN_FEATURES if f in ALL_FEAT]



# Add n_success column
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

    cv_rows   = []    # per-fold results
    all_scores_store = {}  # (thresh, tool_or_task) -> (scores, y_true) across folds

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

                # Recompute deltas on training fold only
                deltas = compute_deltas(df_train, feats, target_col,
                                        comparison='binary')

                # Scaling from training fold only
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
                    'fold': fold+1,
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

        for task_name, y_bin, comparison, feats in [
            (f'any_vs_none_top{thresh}',  y_any, 'binary',      AVN_FEATURES),
            (f'all_vs_notall_top{thresh}', y_all, 'binary',      AVN_FEATURES),
        ]:
            skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True,
                                  random_state=RANDOM_STATE)
            fold_aucs, fold_rhos, fold_f1s = [], [], []
            oof_scores = np.zeros(N)
            oof_y      = np.zeros(N, dtype=int)

            target_col_avn = f'n_success_{thresh}'

            for fold, (train_idx, test_idx) in enumerate(skf.split(df, y_bin)):
                df_train = df.iloc[train_idx].copy()
                df_test  = df.iloc[test_idx]
                y_test   = y_bin[test_idx]

                # For any-vs-none: delta between success>=1 and success==0
                # Reuse binary comparison on the actual binary label
                df_train['_target'] = y_bin[train_idx]
                deltas = compute_deltas(df_train, feats, '_target',
                                        comparison='binary')
                means  = df_train[feats].mean().to_dict()
                stds   = df_train[feats].std().to_dict()

                scores = delta_score(df_test, feats, deltas, means, stds)
                oof_scores[test_idx] = scores
                oof_y[test_idx]      = y_test

                res = evaluate_scores(y_test, scores)
                fold_aucs.append(res['auc'])
                fold_rhos.append(res['spearman'])
                fold_f1s.append(res['f1_opt'])

                cv_rows.append({
                    'thresh': thresh, 'task': task_name,
                    'fold': fold+1,
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

    # Summary: mean ± std across folds
    summary = (cv_df.groupby(['thresh','task'])
               [['auc','spearman','f1_opt']]
               .agg(['mean','std'])
               .round(3))
    summary.to_csv(OUT_DIR / 'cv_auc_summary.csv')
    print(f"\nSummary saved to {OUT_DIR / 'cv_auc_summary.csv'}")

    # ── Delta stability plot ─────────────────────────────────────────────────
    # Shows how stable the per-fold deltas are vs full-dataset deltas
    print("\nPlotting delta stability...")
    for tool in TOOLS:
        feats = TOOL_FEATURES[tool]
        target_col = f'top@5_{tool}'
        y = df[target_col].fillna(0).astype(int).values
        skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True,
                              random_state=RANDOM_STATE)

        fold_deltas = {f: [] for f in feats}
        for train_idx, _ in skf.split(df, y):
            d = compute_deltas(df.iloc[train_idx], feats, target_col, 'binary')
            for f in feats:
                fold_deltas[f].append(d.get(f, 0.0))

        # Full-dataset deltas
        full_deltas = compute_deltas(df, feats, target_col, 'binary')

        fig, ax = plt.subplots(figsize=(7, max(3, len(feats)*0.35)))
        ys = np.arange(len(feats))
        full_vals  = [full_deltas.get(f, 0) for f in feats]
        fold_means = [np.mean(fold_deltas[f]) for f in feats]
        fold_stds  = [np.std(fold_deltas[f])  for f in feats]

        ax.barh(ys, full_vals, height=0.4, color='steelblue',
                alpha=0.6, label='Full dataset')
        ax.errorbar(fold_means, ys, xerr=fold_stds, fmt='o',
                    color='darkorange', ms=5, lw=1.5, label='CV folds (mean±std)')
        ax.set_yticks(ys); ax.set_yticklabels(feats, fontsize=8)
        ax.axvline(0, color='black', lw=0.8)
        ax.set_xlabel("Cliff's delta")
        ax.set_title(f"Delta stability — {DISPLAY[tool]} (Top-5)")
        ax.legend(fontsize=8)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.tight_layout()
        for ext in ['pdf','png']:
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
        ax.plot([0,1],[0,1],'k--',lw=0.8)
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title(f'ROC — Delta Scorer, Top-{thresh} (CV OOF)')
        ax.legend(fontsize=8, loc='lower right')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.tight_layout()
        for ext in ['pdf','png']:
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
            ax.hist(scores[y_true==0], bins=30, alpha=0.6,
                    color='#d73027', label='Fail', density=True)
            ax.hist(scores[y_true==1], bins=30, alpha=0.6,
                    color='#1a9850', label='Success', density=True)
            ax.set_title(DISPLAY[tool], fontsize=9)
            ax.set_xlabel('Delta score', fontsize=8)
            ax.legend(fontsize=7)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
        plt.suptitle(f'Score Distribution by Outcome — Top-{thresh}',
                     fontweight='bold')
        plt.tight_layout()
        for ext in ['pdf','png']:
            plt.savefig(PLOT_DIR / f'delta_score_dist_top{thresh}.{ext}', dpi=150)
        plt.close()

    # ── Save per-bug OOF scores for Top-5 ───────────────────────────────────
    score_rows = df[['project','bug_id']].copy()
    for tool in TOOLS:
        key = (5, tool)
        if key in all_scores_store:
            scores, y_true = all_scores_store[key]
            score_rows[f'score_{tool}'] = scores
            score_rows[f'success_{tool}'] = y_true
    score_rows['score_any'] = all_scores_store.get(
        (5, f'any_vs_none_top5'), (np.zeros(N), None))[0]
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

# Per-tool deltas
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

# All-vs-none deltas
for thresh in THRESHOLDS:
    n_succ = df[f'n_success_{thresh}'].values
    for task, y_bin in [('any_vs_none', (n_succ>=1).astype(int)),
                        ('all_vs_notall', (n_succ==len(TOOLS)).astype(int))]:
        df['_avn_target'] = y_bin
        deltas = compute_deltas(df, AVN_FEATURES, '_avn_target', 'binary')
        for f, d in deltas.items():
            delta_rows.append({
                'task': task, 'threshold': thresh,
                'feature': f, 'delta': round(d, 4),
            })

# Scaling parameters (mean + std from full dataset)
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

    # Load scaling and deltas from full training set
    scaling = pd.read_csv(OUT_DIR / 'full_dataset_scaling.csv')
    means   = dict(zip(scaling['feature'], scaling['mean']))
    stds    = dict(zip(scaling['feature'], scaling['std']))
    deltas_full = pd.read_csv(OUT_DIR / 'full_dataset_deltas.csv')

    # Score each bug for each task
    score_out = df_new[['project','bug_id']].copy() if 'project' in df_new.columns \
                else pd.DataFrame({'idx': range(len(df_new))})

    for thresh in THRESHOLDS:
        for tool in TOOLS:
            task_key = f'tool_{tool}'
            feats = TOOL_FEATURES[tool]
            d_sub = deltas_full[
                (deltas_full['task']==task_key) &
                (deltas_full['threshold']==thresh)
            ]
            deltas = dict(zip(d_sub['feature'], d_sub['delta']))
            scores = delta_score(df_new, feats, deltas, means, stds)
            score_out[f'score_{tool}_top{thresh}'] = scores.round(4)

        # All-vs-none
        for task in ['any_vs_none', 'all_vs_notall']:
            d_sub = deltas_full[
                (deltas_full['task']==task) &
                (deltas_full['threshold']==thresh)
            ]
            deltas = dict(zip(d_sub['feature'], d_sub['delta']))
            scores = delta_score(df_new, AVN_FEATURES, deltas, means, stds)
            score_out[f'score_{task}_top{thresh}'] = scores.round(4)

    score_out.to_csv(OUT_DIR / 'external_scores.csv', index=False)
    print(f"  Scores saved to {OUT_DIR / 'external_scores.csv'}")

    # If ground truth outcomes are provided, evaluate
    if args.new_tools:
        print("\n  Ground truth provided — evaluating predictions...")
        df_new_tools = pd.read_csv(args.new_tools)

        # Fix BRaIn bug_id if needed
        brain_mask = df_new_tools['tool'] == 'BRaIn'
        df_new_tools.loc[brain_mask, 'bug_id'] = \
            df_new_tools.loc[brain_mask, 'bug_id'].str.split('-').str[-1]

        pivot_new = df_new_tools.pivot_table(
            index=['project','bug_id'], columns='tool',
            values=['top@1','top@5'], aggfunc='max'
        ).fillna(0)
        pivot_new.columns = ['_'.join(str(c) for c in col) for col in pivot_new.columns]
        pivot_new = pivot_new.reset_index()

        eval_df = score_out.merge(pivot_new, on=['project','bug_id'], how='inner')
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
                n_succ = sum(eval_df.get(f'top@{thresh}_{t}', pd.Series(0)).fillna(0)
                             for t in TOOLS)
                if task == 'any_vs_none':
                    y_true = (n_succ >= 1).astype(int)
                else:
                    y_true = (n_succ == len(TOOLS)).astype(int)
                if len(np.unique(y_true)) < 2 or col_score not in eval_df:
                    continue
                auc = roc_auc_score(y_true, eval_df[col_score])
                print(f"  {task+' Top-'+str(thresh):<30} {auc:>8.3f}")

# ════════════════════════════════════════════════════════════════════════════
# DONE
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("DONE. Outputs written to:", OUT_DIR.resolve())
if args.mode == 'internal':
    print("  cv_fold_results.csv        — per-fold AUC/Spearman/F1")
    print("  cv_auc_summary.csv         — mean ± std across folds")
    print("  score_vs_success_top5.csv  — per-bug OOF scores")
    print("  plots/delta_stability_*.{pdf,png}   — delta stability per tool")
    print("  plots/delta_roc_top*.{pdf,png}      — OOF ROC curves")
    print("  plots/delta_score_dist_top*.{pdf,png} — score distributions")
print("  full_dataset_deltas.csv    — weights for external use")
print("  full_dataset_scaling.csv   — mean/std for external use")
if args.mode == 'external':
    print("  external_scores.csv        — scores for new dataset")
print("="*70)
print("""
To run on a new dataset:
    python delta_score_predictor.py --mode external \\
        --new_features /path/to/new_features.csv \\
        --new_tools    /path/to/new_tool_comparison.csv
""")

# ════════════════════════════════════════════════════════════════════════════
# ════════════════════════════════════════════════════════════════════════════

# ════════════════════════════════════════════════════════════════════════════
# THREE-TIER CASCADE ROUTING — 5-FOLD CROSS-VALIDATED
# ════════════════════════════════════════════════════════════════════════════
#
# Motivation (from empirical findings):
#   Bug report features predict general localizability (AUC ~0.79) but
#   cannot distinguish which individual tool will succeed on a given hard
#   bug (RQ3 null result). Rather than selecting between tools, we exploit
#   the architectural cost hierarchy of the tools themselves:
#
#   Tier 1 — IR only (cheapest):
#       Pure lexical retrieval, no LLM calls, local execution.
#       Deployed for high-quality reports where lexical signal is strong.
#
#   Tier 2 — BRaIn (intermediate):
#       IR retrieval + LLM relevance feedback for query expansion/rerank.
#       Uses local open-source model (Mistral 7B), one LLM pass per bug.
#       Deployed for medium-quality reports needing contextual augmentation.
#
#   Tier 3 — FlexFL (most expensive):
#       Full agentic LLM pipeline with codebase navigation.
#       Uses commercial LLM API. Deployed only for low-quality,
#       high-difficulty reports where lighter approaches fall short.
#
# The localizability scorer (delta-weighted, trained on IR success) assigns
# a quality score to each bug. Two thresholds partition bugs into the three
# tiers. We sweep threshold pairs to produce a cost-coverage curve.
#
# References:
#   BRaIn architecture — Samir & Rahman (2025), arXiv:2501.10542
#   FlexFL architecture — Xu et al. (2025), IEEE TSE
# ════════════════════════════════════════════════════════════════════════════

print("\n" + "="*70)
print("THREE-TIER CASCADE ROUTING (IR → BRaIn → FlexFL), 5-FOLD CV")
print("="*70)

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── Tier definitions ──────────────────────────────────────────────────────
IR_TOOLS  = ['boostnsift', 'buglocator', 'locus']
BRAIN_COL = 'top@5_BRaIn'
FLEX_COL  = 'top@5_FlexFL'

ir_cols = [f'top@5_{t}' for t in IR_TOOLS if f'top@5_{t}' in df.columns]

df['_ir_success']    = (df[ir_cols].max(axis=1) == 1).astype(int) if ir_cols else 0
df['_brain_success'] = df[BRAIN_COL].fillna(0).astype(int) if BRAIN_COL in df.columns else 0
df['_flex_success']  = df[FLEX_COL].fillna(0).astype(int)  if FLEX_COL  in df.columns else 0

N = len(df)
ir_y    = df['_ir_success'].values
brain_y = df['_brain_success'].values
flex_y  = df['_flex_success'].values

# ── Baselines ─────────────────────────────────────────────────────────────
always_ir    = ir_y.mean()
always_brain = ((ir_y == 1) | (brain_y == 1)).mean()  # IR + BRaIn on all bugs
always_flex  = flex_y.mean()
oracle       = ((ir_y == 1) | (brain_y == 1) | (flex_y == 1)).mean()

print(f"  Always-IR only:              {always_ir:.3f}")
print(f"  Always-IR+BRaIn:             {always_brain:.3f}")
print(f"  Always-FlexFL:               {always_flex:.3f}")
print(f"  Oracle (all three tools):    {oracle:.3f}")
print(f"  N = {N} bugs\n")

# ── Sweep configuration ───────────────────────────────────────────────────
# ir_pct   = % of bugs routed to Tier 1 (IR only, top scorers)
# brain_pct = % of bugs routed to Tier 2 (IR + BRaIn, middle scorers)
# flex_pct  = remaining % routed to Tier 3 (IR + BRaIn + FlexFL)
# Constraint: ir_pct + brain_pct + flex_pct = 100

SWEEP = [
    # (ir_pct, brain_pct, flex_pct)
    (100,  0,  0),   # Always IR (baseline)
    ( 70, 20, 10),
    ( 60, 25, 15),
    ( 50, 30, 20),
    ( 50, 40, 10),
    ( 40, 40, 20),
    ( 35, 35, 30),
    ( 30, 40, 30),
    ( 20, 50, 30),
    (  0,100,  0),   # Always IR+BRaIn (no FlexFL)
    (  0,  0,100),   # Always full cascade (oracle routing)
]

# ── 5-fold CV ─────────────────────────────────────────────────────────────
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

fold_coverage  = {k: [] for k in SWEEP}
fold_flex_frac = {k: [] for k in SWEEP}
fold_auc       = []
oof_scores     = np.zeros(N)

for fold, (tr_idx, te_idx) in enumerate(skf.split(df, ir_y)):
    df_tr = df.iloc[tr_idx]
    df_te = df.iloc[te_idx]
    n_te  = len(te_idx)

    # Train scorer on IR success within this fold
    deltas = compute_deltas(df_tr, AVN_FEATURES, '_ir_success',
                            comparison='binary')
    means  = df_tr[AVN_FEATURES].mean().to_dict()
    stds   = df_tr[AVN_FEATURES].std().to_dict()

    te_scores = delta_score(df_te, AVN_FEATURES, deltas, means, stds)
    tr_scores = delta_score(df_tr, AVN_FEATURES, deltas, means, stds)
    oof_scores[te_idx] = te_scores

    if len(np.unique(ir_y[te_idx])) == 2:
        fold_auc.append(roc_auc_score(ir_y[te_idx], te_scores))

    ir_te    = ir_y[te_idx]
    brain_te = brain_y[te_idx]
    flex_te  = flex_y[te_idx]

    for (pct_ir, pct_brain, pct_flex) in SWEEP:
        assert pct_ir + pct_brain + pct_flex == 100, \
            f"Tier percentages must sum to 100: {pct_ir}+{pct_brain}+{pct_flex}"

        # Thresholds derived from TRAINING score distribution
        # Top pct_ir% → Tier 1; bottom pct_flex% → Tier 3; middle → Tier 2
        t_high = np.percentile(tr_scores, 100 - pct_ir)   if pct_ir   < 100 else  np.inf
        t_low  = np.percentile(tr_scores, pct_flex)        if pct_flex > 0   else -np.inf

        # Mutually exclusive tier assignment
        tier1 = te_scores >= t_high                        # IR only
        tier3 = te_scores <  t_low                        # Full cascade
        tier2 = ~tier1 & ~tier3                            # IR + BRaIn

        # Coverage per tier (best available tool within tier)
        cov1 = (tier1 & (ir_te    == 1)).sum()
        cov2 = (tier2 & ((ir_te == 1) | (brain_te == 1))).sum()
        cov3 = (tier3 & ((ir_te == 1) | (brain_te == 1) | (flex_te == 1))).sum()
        coverage = (cov1 + cov2 + cov3) / n_te

        # FlexFL invocation fraction (cost proxy for commercial API calls)
        flex_fraction = tier3.sum() / n_te

        fold_coverage[k].append(coverage) if (k := (pct_ir, pct_brain, pct_flex)) else None
        fold_coverage[(pct_ir, pct_brain, pct_flex)][-1]  # already appended above
        fold_flex_frac[(pct_ir, pct_brain, pct_flex)].append(flex_fraction)

    # Fix the double-append issue above with a cleaner loop
    # (re-doing cleanly below — the walrus operator caused a duplicate)

# Re-run cleanly
fold_coverage  = {k: [] for k in SWEEP}
fold_flex_frac = {k: [] for k in SWEEP}
fold_auc       = []

for fold, (tr_idx, te_idx) in enumerate(skf.split(df, ir_y)):
    df_tr = df.iloc[tr_idx]
    df_te = df.iloc[te_idx]
    n_te  = len(te_idx)

    deltas = compute_deltas(df_tr, AVN_FEATURES, '_ir_success',
                            comparison='binary')
    means  = df_tr[AVN_FEATURES].mean().to_dict()
    stds   = df_tr[AVN_FEATURES].std().to_dict()

    te_scores = delta_score(df_te, AVN_FEATURES, deltas, means, stds)
    tr_scores = delta_score(df_tr, AVN_FEATURES, deltas, means, stds)

    if len(np.unique(ir_y[te_idx])) == 2:
        fold_auc.append(roc_auc_score(ir_y[te_idx], te_scores))

    ir_te    = ir_y[te_idx]
    brain_te = brain_y[te_idx]
    flex_te  = flex_y[te_idx]

    for (pct_ir, pct_brain, pct_flex) in SWEEP:
        t_high = np.percentile(tr_scores, 100 - pct_ir)   if pct_ir   < 100 else  np.inf
        t_low  = np.percentile(tr_scores, pct_flex)        if pct_flex > 0   else -np.inf

        tier1 = te_scores >= t_high
        tier3 = te_scores <  t_low
        tier2 = ~tier1 & ~tier3

        cov1 = (tier1 & (ir_te    == 1)).sum()
        cov2 = (tier2 & ((ir_te == 1) | (brain_te == 1))).sum()
        cov3 = (tier3 & ((ir_te == 1) | (brain_te == 1) | (flex_te == 1))).sum()
        coverage = (cov1 + cov2 + cov3) / n_te

        flex_fraction = tier3.sum() / n_te

        fold_coverage[(pct_ir, pct_brain, pct_flex)].append(coverage)
        fold_flex_frac[(pct_ir, pct_brain, pct_flex)].append(flex_fraction)

# ── Results table ─────────────────────────────────────────────────────────
mean_auc = np.mean(fold_auc) if fold_auc else float('nan')
std_auc  = np.std(fold_auc)  if fold_auc else float('nan')
print(f"IR-success scorer AUC: {mean_auc:.3f} ± {std_auc:.3f}")

print(f"\n{'IR%':>5} {'BRaIn%':>7} {'FlexFL%':>8} | "
      f"{'Coverage':>14} {'FlexFL calls':>13} {'vs FlexFL':>11}")
print("-"*65)

routing_rows = []
for k in SWEEP:
    pct_ir, pct_brain, pct_flex = k
    cov_mean  = np.mean(fold_coverage[k])
    cov_std   = np.std(fold_coverage[k])
    flex_mean = np.mean(fold_flex_frac[k])
    diff      = cov_mean - always_flex
    print(f"{pct_ir:>4}%  {pct_brain:>6}%  {pct_flex:>7}% | "
          f"{cov_mean:.3f}±{cov_std:.3f}  "
          f"{flex_mean:>12.2f}  {diff:>+.3f}")
    routing_rows.append({
        'tier1_ir_pct':     pct_ir,
        'tier2_brain_pct':  pct_brain,
        'tier3_flex_pct':   pct_flex,
        'coverage_mean':    round(cov_mean,  4),
        'coverage_std':     round(cov_std,   4),
        'flexfl_frac_mean': round(flex_mean, 4),
        'vs_always_flexfl': round(diff,      4),
    })

# ── Key operating points ──────────────────────────────────────────────────
print(f"\n=== Baselines ===")
print(f"  Always-IR only:           {always_ir:.3f}  (0% BRaIn, 0% FlexFL)")
print(f"  Always-IR+BRaIn:          {always_brain:.3f}  (100% BRaIn, 0% FlexFL)")
print(f"  Always-FlexFL:            {always_flex:.3f}  (100% FlexFL)")
print(f"  Oracle full cascade:      {oracle:.3f}")

# Best coverage at each FlexFL budget
print(f"\n=== Best coverage by FlexFL invocation budget ===")
for budget in [0.0, 0.10, 0.20, 0.30, 0.50]:
    candidates = [(k, np.mean(fold_coverage[k]))
                  for k in SWEEP
                  if np.mean(fold_flex_frac[k]) <= budget + 0.02]
    if not candidates:
        continue
    best_k, best_cov = max(candidates, key=lambda x: x[1])
    best_flex = np.mean(fold_flex_frac[best_k])
    best_std  = np.std(fold_coverage[best_k])
    print(f"  FlexFL ≤{budget:.0%}: coverage {best_cov:.3f}±{best_std:.3f}  "
          f"[IR={best_k[0]}%, BRaIn={best_k[1]}%, FlexFL={best_k[2]}%]  "
          f"(actual FlexFL={best_flex:.2f})")

# ── Save CSV ──────────────────────────────────────────────────────────────
routing_df = pd.DataFrame(routing_rows)
routing_df.to_csv(OUT_DIR / 'cascade_routing_sweep.csv', index=False)
print(f"\nSweep results saved to {OUT_DIR / 'cascade_routing_sweep.csv'}")

# ── Plot: cost-coverage curve ─────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 5))

# Plot the sweep curve (x = FlexFL fraction, y = coverage)
xs = [np.mean(fold_flex_frac[k]) for k in SWEEP]
ys = [np.mean(fold_coverage[k])  for k in SWEEP]
es = [np.std(fold_coverage[k])   for k in SWEEP]

# Sort by x for clean line
order = np.argsort(xs)
xs_s  = [xs[i] for i in order]
ys_s  = [ys[i] for i in order]
es_s  = [es[i] for i in order]

ax.plot(xs_s, ys_s, 'o-', color='#2166ac', lw=2, ms=6,
        label='Three-tier cascade (IR → BRaIn → FlexFL)')
ax.fill_between(xs_s,
                [y-e for y,e in zip(ys_s, es_s)],
                [y+e for y,e in zip(ys_s, es_s)],
                alpha=0.15, color='#2166ac')

# Baselines
ax.axhline(always_flex,  color='#d73027', lw=1.5, ls='--',
           label=f'Always-FlexFL: {always_flex:.3f}')
ax.axhline(always_brain, color='#f4a582', lw=1.5, ls='-.',
           label=f'Always-IR+BRaIn: {always_brain:.3f}')
ax.axhline(always_ir,    color='#4dac26', lw=1.5, ls=':',
           label=f'Always-IR: {always_ir:.3f}')

# Annotate best point (highest coverage at ≤30% FlexFL)
candidates_30 = [(k, np.mean(fold_coverage[k]))
                 for k in SWEEP
                 if np.mean(fold_flex_frac[k]) <= 0.32]
if candidates_30:
    best_k30, best_cov30 = max(candidates_30, key=lambda x: x[1])
    bx = np.mean(fold_flex_frac[best_k30])
    by = best_cov30
    ax.annotate(
        f'IR={best_k30[0]}%, BRaIn={best_k30[1]}%\nFlexFL={best_k30[2]}%\n'
        f'Coverage={by:.3f}',
        xy=(bx, by), xytext=(bx + 0.08, by - 0.02),
        fontsize=7.5, color='#2166ac',
        arrowprops=dict(arrowstyle='->', color='#2166ac', lw=1)
    )

ax.set_xlabel('Fraction of bugs invoking FlexFL (commercial LLM)', fontsize=10)
ax.set_ylabel('Top-5 Coverage', fontsize=10)
ax.set_title('Cost–Coverage Tradeoff:\nThree-Tier Cascade (IR → BRaIn → FlexFL)',
             fontsize=11)
ax.set_xlim(-0.03, 1.05)
ax.legend(fontsize=8, loc='lower right')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()

for ext in ['pdf', 'png']:
    plt.savefig(PLOT_DIR / f'cascade_routing_curve.{ext}',
                dpi=150, bbox_inches='tight')
plt.close()
print(f"Plot saved to {PLOT_DIR / 'cascade_routing_curve.png'}")