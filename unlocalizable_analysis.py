"""
Analysis of 117 Unlocalizable Bugs
====================================
Three angles:
  1. Hard vs. Soft failure classification
  2. Characterization: feature comparison (unlocalizable vs. localizable)
  3. Tool comparison: rank distributions and closest-tool analysis
"""

import pandas as pd
import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# ── Load data ──────────────────────────────────────────────────────────────────
summary = pd.read_csv('tool_comparison_summary.csv')
features = pd.read_csv('final_feature_set.csv')

TOOLS = ['BRaIn', 'FlexFL', 'boostnsift', 'buglocator', 'locus']

# ── Identify 117 unlocalizable bugs (top@10 == 0 for all tools in summary) ───
# final_feature_set only has top@1 and top@5; use tool_comparison_summary for top@10
top10_per_bug = (
    summary.groupby(['project', 'bug_id'])['top@10']
    .apply(lambda x: (x == 0).all())
)
unloc_ids = top10_per_bug[top10_per_bug].reset_index()[['project', 'bug_id']]

assert len(unloc_ids) == 117, f"Expected 117, got {len(unloc_ids)}"

# Build a boolean mask aligned to features dataframe
features['_key'] = features['project'] + '-' + features['bug_id'].astype(str)
unloc_keys = set(unloc_ids['project'] + '-' + unloc_ids['bug_id'].astype(str))
unloc_mask = features['_key'].isin(unloc_keys)

# ═══════════════════════════════════════════════════════════════════════════════
# 1. HARD vs. SOFT FAILURE CLASSIFICATION
# ═══════════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("1. HARD vs. SOFT FAILURE CLASSIFICATION")
print("=" * 70)

# Hard = all tools detected=No  |  Soft = at least 1 tool detected=Yes but rank>10
detected_per_bug = (
    summary.groupby(['project', 'bug_id'])['detected']
    .apply(lambda x: (x == 'Yes').any())
    .reset_index()
    .rename(columns={'detected': 'any_detected'})
)

unloc_ids = unloc_ids.merge(detected_per_bug, on=['project', 'bug_id'], how='left')
unloc_ids['failure_type'] = unloc_ids['any_detected'].map({True: 'Soft', False: 'Hard'})

hard = unloc_ids[unloc_ids['failure_type'] == 'Hard']
soft = unloc_ids[unloc_ids['failure_type'] == 'Soft']

print(f"\nTotal unlocalizable: {len(unloc_ids)}")
print(f"  Hard failures (no tool detected):         {len(hard):3d}  ({100*len(hard)/len(unloc_ids):.1f}%)")
print(f"  Soft failures (detected but rank > 10):   {len(soft):3d}  ({100*len(soft)/len(unloc_ids):.1f}%)")

print("\n--- Hard failures by project ---")
print(hard.groupby('project')['bug_id'].apply(lambda x: sorted(x.tolist())).to_string())

print("\n--- Soft failures by project ---")
print(soft.groupby('project')['bug_id'].apply(lambda x: sorted(x.tolist())).to_string())

# For soft failures: what rank DID the tools achieve?
soft_keys = soft[['project', 'bug_id']].copy()
soft_detail = summary.merge(soft_keys, on=['project', 'bug_id'])
soft_detail = soft_detail[soft_detail['detected'] == 'Yes']

print("\n--- Rank distribution for soft failures (detected=Yes, rank>10) ---")
rank_stats = soft_detail['rank'].describe()
print(rank_stats.to_string())

print("\n--- Per-tool rank stats for soft failures ---")
print(
    soft_detail.groupby('tool')['rank']
    .describe()[['count', 'mean', '50%', 'min', 'max']]
    .rename(columns={'50%': 'median'})
    .to_string()
)

# ═══════════════════════════════════════════════════════════════════════════════
# 2. CHARACTERIZATION: UNLOCALIZABLE vs. LOCALIZABLE
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("2. CHARACTERIZATION: UNLOCALIZABLE vs. LOCALIZABLE BUGS")
print("=" * 70)

TOOL_COLS   = [c for c in features.columns if any(t in c for t in ['BRaIn','FlexFL','boostnsift','buglocator','locus','mrr_','rank_','top@'])]
FEAT_COLS   = [c for c in features.columns if c not in TOOL_COLS + ['project', 'bug_id', 'id']]
BOOL_FEATS  = ['has_stacktrace', 'has_code', 'has_patch', 'has_enumeration',
               'hidden_s2r_present', 'contradiction_present',
               'has_OB', 'has_EB', 'has_S2R', 'missing_OB', 'missing_EB', 'missing_S2R']
CAT_FEATS   = [c for c in FEAT_COLS if c.startswith('fg_cat_') or c.startswith('cat_')]
NUM_FEATS   = [c for c in FEAT_COLS if c not in BOOL_FEATS + CAT_FEATS]

features['is_unlocalizable'] = unloc_mask

unloc_feat = features[features['is_unlocalizable']]
local_feat  = features[~features['is_unlocalizable']]

print(f"\nGroups: {len(unloc_feat)} unlocalizable  |  {len(local_feat)} localizable\n")

# ── Numeric feature comparison ─────────────────────────────────────────────────
results = []
for col in NUM_FEATS:
    u = pd.to_numeric(unloc_feat[col], errors='coerce').dropna()
    l = pd.to_numeric(local_feat[col],  errors='coerce').dropna()
    if len(u) < 5 or len(l) < 5:
        continue
    stat, p = stats.mannwhitneyu(u, l, alternative='two-sided')
    results.append({
        'feature':       col,
        'unloc_mean':    u.mean(),
        'local_mean':    l.mean(),
        'unloc_median':  u.median(),
        'local_median':  l.median(),
        'p_value':       p,
        'significant':   p < 0.05,
    })

res_df = pd.DataFrame(results).sort_values('p_value')
sig_df  = res_df[res_df['significant']]

print("--- Numeric features with significant difference (p < 0.05, Mann-Whitney U) ---")
print(f"  {len(sig_df)} / {len(res_df)} features are significantly different\n")
pd.set_option('display.float_format', '{:.4f}'.format)
pd.set_option('display.max_colwidth', 40)
pd.set_option('display.width', 120)
print(sig_df[['feature', 'unloc_mean', 'local_mean', 'unloc_median', 'local_median', 'p_value']].to_string(index=False))

# ── Boolean feature comparison ─────────────────────────────────────────────────
print("\n--- Boolean / structural feature comparison ---")
bool_rows = []
for col in BOOL_FEATS:
    u_rate = unloc_feat[col].mean() if col in unloc_feat else np.nan
    l_rate = local_feat[col].mean() if col in local_feat else np.nan
    bool_rows.append({'feature': col, 'unloc_%': 100*u_rate, 'local_%': 100*l_rate, 'diff_pp': 100*(u_rate - l_rate)})
bool_df = pd.DataFrame(bool_rows).sort_values('diff_pp')
print(bool_df.to_string(index=False))

# ── Category feature comparison ────────────────────────────────────────────────
print("\n--- Fault-category distribution ---")
cat_rows = []
for col in CAT_FEATS:
    u_rate = unloc_feat[col].mean() if col in unloc_feat else np.nan
    l_rate = local_feat[col].mean() if col in local_feat else np.nan
    cat_rows.append({'category': col, 'unloc_%': 100*u_rate, 'local_%': 100*l_rate, 'diff_pp': 100*(u_rate - l_rate)})
cat_df = pd.DataFrame(cat_rows).sort_values('diff_pp')
print(cat_df.to_string(index=False))

# ── Top distinguishing features summary ───────────────────────────────────────
print("\n--- TOP 10 most distinguishing features (by p-value) ---")
top10 = sig_df.head(10).copy()
top10['direction'] = top10.apply(
    lambda r: '↓ lower in unloc' if r['unloc_mean'] < r['local_mean'] else '↑ higher in unloc', axis=1
)
print(top10[['feature', 'unloc_median', 'local_median', 'p_value', 'direction']].to_string(index=False))

# ═══════════════════════════════════════════════════════════════════════════════
# 3. TOOL COMPARISON ON UNLOCALIZABLE BUGS
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("3. TOOL COMPARISON ON THE 117 UNLOCALIZABLE BUGS")
print("=" * 70)

unloc_summary = summary.merge(unloc_ids[['project', 'bug_id', 'failure_type']], on=['project', 'bug_id'])

print("\n--- Detection rate per tool (detected=Yes) ---")
det_rate = (
    unloc_summary.groupby('tool')['detected']
    .apply(lambda x: f"{(x=='Yes').sum():3d} / {len(x):3d}  ({100*(x=='Yes').mean():.1f}%)")
)
print(det_rate.to_string())

print("\n--- Average MRR per tool on unlocalizable bugs ---")
mrr_stats = unloc_summary.groupby('tool')['mrr'].agg(['mean', 'median', 'max'])
mrr_stats.columns = ['mean_MRR', 'median_MRR', 'max_MRR']
print(mrr_stats.to_string())

print("\n--- Per tool: how close did they get? (rank distribution when detected=Yes) ---")
detected_unloc = unloc_summary[unloc_summary['detected'] == 'Yes']
rank_buckets = detected_unloc.copy()
rank_buckets['rank_bucket'] = pd.cut(
    rank_buckets['rank'],
    bins=[0, 10, 20, 50, 100, np.inf],
    labels=['≤10 (would be loc.)', '11-20', '21-50', '51-100', '>100']
)
print(
    rank_buckets.groupby(['tool', 'rank_bucket'], observed=True)['bug_id']
    .count()
    .unstack(fill_value=0)
    .to_string()
)

print("\n--- For soft failures: which tool gets closest? (lowest rank) ---")
soft_det = unloc_summary[
    (unloc_summary['failure_type'] == 'Soft') & (unloc_summary['detected'] == 'Yes')
]
# For each soft-failure bug, find the tool with lowest rank
best_tool_per_bug = (
    soft_det.loc[soft_det.groupby(['project', 'bug_id'])['rank'].idxmin()]
    [['project', 'bug_id', 'tool', 'rank']]
    .rename(columns={'tool': 'closest_tool'})
)
print("Best tool (lowest rank) per soft-failure bug:")
print(best_tool_per_bug['closest_tool'].value_counts().to_string())

# ── BRaIn on Closure/JacksonDatabind deep-dive ────────────────────────────────
print("\n--- BRaIn systematic failure on Closure & JacksonDatabind ---")
brain_focus = unloc_summary[
    (unloc_summary['tool'] == 'BRaIn') &
    (unloc_summary['project'].isin(['Closure', 'JacksonDatabind']))
]
print(brain_focus.groupby('project')['detected'].value_counts().to_string())

# ── Project-level breakdown for unlocalizable bugs ────────────────────────────
print("\n--- Project-level breakdown of unlocalizable bugs ---")
proj_breakdown = (
    unloc_summary.groupby(['project', 'failure_type'])['bug_id']
    .nunique()
    .unstack(fill_value=0)
)
# add total column
proj_breakdown['Total'] = proj_breakdown.sum(axis=1)
print(proj_breakdown.sort_values('Total', ascending=False).to_string())

# ═══════════════════════════════════════════════════════════════════════════════
# SAVE OUTPUTS
# ═══════════════════════════════════════════════════════════════════════════════
unloc_ids.to_csv('unlocalizable_bugs_classified.csv', index=False)
res_df.to_csv('unlocalizable_feature_comparison.csv', index=False)
print("\n\nSaved:")
print("  unlocalizable_bugs_classified.csv   – 117 bugs with Hard/Soft label")
print("  unlocalizable_feature_comparison.csv – feature comparison results")
print("\nDone.")
