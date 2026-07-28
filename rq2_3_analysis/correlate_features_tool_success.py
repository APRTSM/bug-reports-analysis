"""
Correlate bug report features with repair tool success.

toolDict.csv   – repair tools; cell non-empty ⟹ tool fixed the bug
final_feature_set.csv – preprocessed (standardized) bug report features

Output:
  correlation_features_vs_tools.csv  – Spearman ρ + p-value per (feature, tool)
  correlation_features_vs_num_tools.csv – correlation with #tools that fixed the bug
  top_correlations_report.txt        – human-readable summary
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats

ROOT_DIR = Path(__file__).resolve().parent.parent
OUT_DIR = Path(__file__).resolve().parent

# ── 1. Load toolDict and build binary success matrix ──────────────────────────

tool_raw = pd.read_csv(ROOT_DIR / "tool_feature_analysis" / "toolDict.csv", index_col=0)

# Column layout: Identifier, Active, Deprecated, <tool_1>, ..., <tool_n>
meta_cols = ["Identifier", "Active", "Deprecated"]
tool_cols = [c for c in tool_raw.columns if c not in meta_cols]

# Only keep active, non-deprecated bugs
active = tool_raw[(tool_raw["Active"] == True) & (tool_raw["Deprecated"] == False)].copy()

# Binary: 1 if cell is non-empty (tool fixed the bug), else 0
binary = active[tool_cols].notna() & (active[tool_cols].apply(lambda col: col.astype(str).str.strip()) != "")
binary = binary.astype(int)
binary.index.name = "id"

# Aggregate metric: how many tools fixed this bug
binary["num_tools_fixed"] = binary[tool_cols].sum(axis=1)
binary["any_tool_fixed"]  = (binary["num_tools_fixed"] > 0).astype(int)

print(f"Repair tool matrix: {binary.shape[0]} bugs × {len(tool_cols)} tools")
print(f"Bugs fixed by at least one tool: {binary['any_tool_fixed'].sum()}")
print(f"Median tools per fixable bug: {binary.loc[binary['any_tool_fixed']==1,'num_tools_fixed'].median():.1f}")

# ── 2. Load features ──────────────────────────────────────────────────────────

feat = pd.read_csv(ROOT_DIR / "full_feature_preproccessed_fixed" / "final_feature_set_bug_reports.csv")
feat = feat.set_index("id")
print(f"\nFeature matrix: {feat.shape[0]} bugs × {feat.shape[1]} features")

# Identify numeric feature columns (exclude fault-localization performance cols
# and identifier columns -- exact-match, not prefix-match, so features like
# project_num_java_files/project_java_bytes aren't accidentally swept up)
exclude_prefixes = ("mrr_", "rank_", "top@")
exclude_exact = {"project", "bug_id"}
feature_cols = [
    c for c in feat.columns
    if c not in exclude_exact
    and not any(c.startswith(p) for p in exclude_prefixes)
    and feat[c].dtype in [np.float64, np.int64, np.float32, np.int32, bool]
]
# Coerce boolean cols to int
for c in feature_cols:
    if feat[c].dtype == bool:
        feat[c] = feat[c].astype(int)

print(f"Feature columns selected for analysis: {len(feature_cols)}")

# ── 3. Merge ──────────────────────────────────────────────────────────────────

merged = feat[feature_cols].join(binary, how="inner")
print(f"\nMerged dataset: {merged.shape[0]} bugs")

overlap_ids = set(feat.index) & set(binary.index)
only_feat   = set(feat.index) - set(binary.index)
only_tool   = set(binary.index) - set(feat.index)
print(f"  Overlap: {len(overlap_ids)}  |  only in features: {len(only_feat)}  |  only in toolDict: {len(only_tool)}")

# ── 4. Spearman correlation: features vs each repair tool ─────────────────────

target_cols = tool_cols + ["num_tools_fixed", "any_tool_fixed"]

rows = []
for feat_col in feature_cols:
    x = merged[feat_col].fillna(0).values
    for target in target_cols:
        y = merged[target].values
        if x.std() == 0 or y.std() == 0:
            continue
        rho, pval = stats.spearmanr(x, y)
        rows.append({
            "feature": feat_col,
            "target": target,
            "spearman_rho": round(rho, 4),
            "p_value": round(pval, 5),
            "significant": pval < 0.05,
        })

corr_df = pd.DataFrame(rows)
corr_df.to_csv(OUT_DIR / "correlation_features_vs_tools.csv", index=False)
print(f"\nSaved {len(corr_df)} (feature, tool) pairs → correlation_features_vs_tools.csv")

# ── 5. Summary: top correlations with num_tools_fixed ─────────────────────────

ntf = (
    corr_df[corr_df["target"] == "num_tools_fixed"]
    .sort_values("spearman_rho", key=abs, ascending=False)
    .reset_index(drop=True)
)
ntf.to_csv(OUT_DIR / "correlation_features_vs_num_tools.csv", index=False)

# ── 6. Human-readable report ──────────────────────────────────────────────────

lines = []
lines.append("=" * 70)
lines.append("FEATURE ↔ REPAIR-TOOL SUCCESS CORRELATION REPORT")
lines.append("=" * 70)
lines.append(f"\nDataset: {merged.shape[0]} bugs, {len(tool_cols)} repair tools, {len(feature_cols)} features")
lines.append(f"Method: Spearman rank correlation (α = 0.05)\n")

# -- Top positively correlated features with num_tools_fixed
lines.append("── Top 20 features positively correlated with #tools that fixed the bug ──")
top_pos = ntf[ntf["significant"]].head(20)
for _, r in top_pos.iterrows():
    lines.append(f"  {r['feature']:<55} ρ = {r['spearman_rho']:+.3f}  p = {r['p_value']:.4f}")

lines.append("\n── Top 20 features negatively correlated with #tools that fixed the bug ──")
top_neg = ntf[ntf["significant"]].tail(20)[::-1]
for _, r in top_neg.iterrows():
    lines.append(f"  {r['feature']:<55} ρ = {r['spearman_rho']:+.3f}  p = {r['p_value']:.4f}")

# -- Per-tool top predictors (significant only)
lines.append("\n\n── Most predictive feature per repair tool (|ρ| max, p<0.05) ──")
for tool in tool_cols:
    sub = corr_df[(corr_df["target"] == tool) & corr_df["significant"]].copy()
    if sub.empty:
        lines.append(f"  {tool:<20}  no significant correlations")
        continue
    best = sub.loc[sub["spearman_rho"].abs().idxmax()]
    lines.append(f"  {tool:<20}  {best['feature']:<50} ρ = {best['spearman_rho']:+.3f}  p = {best['p_value']:.4f}")

# -- How many features are significant for each tool
lines.append("\n\n── Significant feature count per tool (p<0.05) ──")
sig_counts = (
    corr_df[corr_df["significant"] & corr_df["target"].isin(tool_cols)]
    .groupby("target")
    .size()
    .sort_values(ascending=False)
)
for tool, cnt in sig_counts.items():
    lines.append(f"  {tool:<25} {cnt:>3} significant features")

report = "\n".join(lines)
with open(OUT_DIR / "top_correlations_report.txt", "w") as f:
    f.write(report)
print("\n" + report)
print("\nSaved report → top_correlations_report.txt")
