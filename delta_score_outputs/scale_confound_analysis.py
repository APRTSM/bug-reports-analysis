"""
Does the report-feature signal survive conditioning on project scale (num Java files)?

Scoped to exactly the two analyses actually reported so far:
  1. All-vs-none: AVN_FEATURES vs any_vs_none / all_vs_notall @ top1, top5
  2. Tool-vs-rest: each tool's own TOOL_FEATURES vs that tool's own top@k success
     @ top1, top5, top10

For each (feature, outcome) pair, fits:
  raw:      outcome ~ z(feature)
  adjusted: outcome ~ z(feature) + z(project_num_java_files)
and reports the standardized logistic coefficient + p-value in both cases, plus
the feature's own correlation with scale and a stratified (tertile) Cliff's
delta cross-check.
"""
import warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import spearmanr, mannwhitneyu

warnings.filterwarnings('ignore')

df = pd.read_csv('full_feature_preproccessed_fixed/final_feature_set_bug_reports_analysis.csv')

TOOLS = ['FlexFL', 'BRaIn', 'bluir', 'buglocator', 'locus']
SCALE_COL = 'project_num_java_files'
SCALE_FEATURES = {'project_num_java_files', 'project_java_bytes'}

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

def cliffs_delta(g1, g2):
    g1, g2 = np.asarray(g1, float), np.asarray(g2, float)
    if len(g1) == 0 or len(g2) == 0:
        return np.nan
    u, _ = mannwhitneyu(g1, g2, alternative='two-sided')
    return (2 * u / (len(g1) * len(g2))) - 1

def zscore(s):
    return (s - s.mean()) / s.std(ddof=0)

def fit_logit(y, X):
    X = sm.add_constant(X, has_constant='add')
    return sm.Logit(y, X).fit(disp=0, method='newton', maxiter=100)

def test_feature_outcome(analysis, outcome_name, feat, y_full):
    if feat not in df.columns:
        return None
    sub = df[[feat, SCALE_COL]].copy()
    sub['y'] = y_full
    sub = sub.dropna()
    if sub['y'].nunique() < 2 or len(sub) < 30:
        return None

    f_z = zscore(sub[feat])
    s_z = zscore(sub[SCALE_COL])
    y = sub['y'].values

    pos = sub.loc[sub['y'] == 1, feat]
    neg = sub.loc[sub['y'] == 0, feat]
    raw_delta = cliffs_delta(pos, neg)

    try:
        raw_res = fit_logit(y, f_z.to_frame('f'))
        raw_coef, raw_p = raw_res.params['f'], raw_res.pvalues['f']
    except Exception:
        raw_coef, raw_p = np.nan, np.nan

    try:
        adj_res = fit_logit(y, pd.DataFrame({'f': f_z, 'scale': s_z}))
        adj_coef, adj_p = adj_res.params['f'], adj_res.pvalues['f']
    except Exception:
        adj_coef, adj_p = np.nan, np.nan

    rho_scale, _ = spearmanr(sub[feat], sub[SCALE_COL])

    try:
        terts = pd.qcut(sub[SCALE_COL], 3, labels=['low', 'mid', 'high'], duplicates='drop')
        strat_deltas = {}
        for band in terts.cat.categories:
            bsub = sub[terts == band]
            p = bsub.loc[bsub['y'] == 1, feat]
            n = bsub.loc[bsub['y'] == 0, feat]
            strat_deltas[band] = cliffs_delta(p, n) if len(p) and len(n) else np.nan
    except Exception:
        strat_deltas = {}

    attenuation = np.nan
    if raw_coef and not np.isnan(raw_coef) and abs(raw_coef) > 1e-9:
        attenuation = 1 - (adj_coef / raw_coef)

    return {
        'analysis': analysis,
        'outcome': outcome_name,
        'feature': feat,
        'n': len(sub),
        'raw_cliffs_delta': round(raw_delta, 4),
        'raw_coef': round(raw_coef, 4) if not np.isnan(raw_coef) else np.nan,
        'raw_p': round(raw_p, 4) if not np.isnan(raw_p) else np.nan,
        'corr_with_scale': round(rho_scale, 4),
        'adj_coef': round(adj_coef, 4) if not np.isnan(adj_coef) else np.nan,
        'adj_p': round(adj_p, 4) if not np.isnan(adj_p) else np.nan,
        'attenuation_pct': round(attenuation * 100, 1) if not np.isnan(attenuation) else np.nan,
        'strat_low': round(strat_deltas.get('low', np.nan), 3) if strat_deltas else np.nan,
        'strat_mid': round(strat_deltas.get('mid', np.nan), 3) if strat_deltas else np.nan,
        'strat_high': round(strat_deltas.get('high', np.nan), 3) if strat_deltas else np.nan,
    }

rows = []

# ── 1. All-vs-none ────────────────────────────────────────────────────────────
avn_report_feats = [f for f in AVN_FEATURES if f not in SCALE_FEATURES]
for thresh in [1, 5]:
    cols = [f'top@{thresh}_{t}' for t in TOOLS]
    n_succ = df[cols].fillna(0).sum(axis=1)
    outcomes = {
        f'any_vs_none_top{thresh}': (n_succ >= 1).astype(int),
        f'all_vs_notall_top{thresh}': (n_succ == len(TOOLS)).astype(int),
    }
    for outcome_name, y in outcomes.items():
        for feat in avn_report_feats:
            r = test_feature_outcome('all_vs_none', outcome_name, feat, y)
            if r:
                rows.append(r)

# ── 2. Tool-vs-rest ───────────────────────────────────────────────────────────
for tool in TOOLS:
    tool_report_feats = [f for f in TOOL_FEATURES[tool] if f not in SCALE_FEATURES]
    for thresh in [1, 5, 10]:
        col = f'top@{thresh}_{tool}'
        if col not in df.columns:
            continue
        y = df[col].fillna(0).astype(int)
        outcome_name = f'{tool}_top{thresh}'
        for feat in tool_report_feats:
            r = test_feature_outcome('tool_vs_rest', outcome_name, feat, y)
            if r:
                rows.append(r)

result = pd.DataFrame(rows)

def verdict(row):
    if pd.isna(row['adj_p']):
        return 'model_failed'
    raw_sig = row['raw_p'] < 0.05
    adj_sig = row['adj_p'] < 0.05
    same_sign = np.sign(row['raw_coef']) == np.sign(row['adj_coef'])
    if not raw_sig:
        return 'not_significant_raw'
    if adj_sig and same_sign and row['attenuation_pct'] < 40:
        return 'survives'
    if adj_sig and same_sign:
        return 'attenuated_but_significant'
    if not same_sign:
        return 'sign_flip'
    return 'vanishes_after_adjustment'

result['verdict'] = result.apply(verdict, axis=1)

out_path = 'delta_score_outputs/scale_confound_analysis.csv'
result.to_csv(out_path, index=False)

pd.set_option('display.width', 200)
for analysis in ['all_vs_none', 'tool_vs_rest']:
    sub = result[result['analysis'] == analysis]
    print(f"\n{'='*70}\n{analysis}  (n tests = {len(sub)})\n{'='*70}")
    print(sub['verdict'].value_counts())
    print()
    print("by outcome:")
    print(sub.groupby(['outcome', 'verdict']).size().unstack(fill_value=0))

print(f"\nSaved full table to {out_path}")

print("\n=== all_vs_none: vanish/flip/attenuated detail ===")
bad_avn = result[(result['analysis']=='all_vs_none') & result['verdict'].isin(
    ['vanishes_after_adjustment','sign_flip','attenuated_but_significant'])]
print(bad_avn[['outcome','feature','raw_p','corr_with_scale','adj_p','attenuation_pct','verdict']].to_string(index=False))

print("\n=== tool_vs_rest: vanish/flip/attenuated detail ===")
bad_tvr = result[(result['analysis']=='tool_vs_rest') & result['verdict'].isin(
    ['vanishes_after_adjustment','sign_flip','attenuated_but_significant'])]
print(bad_tvr[['outcome','feature','raw_p','corr_with_scale','adj_p','attenuation_pct','verdict']].to_string(index=False))
