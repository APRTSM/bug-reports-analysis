import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path
from scipy.stats import spearmanr
from statsmodels.stats.multitest import multipletests

# ======================================
# CONFIG
# ======================================
DATA_DIR = Path(".")
IN_FILE  = DATA_DIR / "experimentA_preprocessed_rich.csv"

OUT_DIR  = DATA_DIR / "experimentA_gap_corr_results"
OUT_DIR.mkdir(exist_ok=True, parents=True)

ALPHA = 0.05
MAX_FEATURES_HEATMAP = 80     # for readability
INCLUDE_MISSINGNESS_FEATURES = True
TOP_N_FEATURES_HEATMAP = 15

# Choose which base performance metric to use for gaps
# "mrr" is usually best; you can also do "top@1" or "top@5"
BASE_METRIC_PREFIXES = ["mrr"]  # e.g., ["mrr", "top@1", "top@5"]

# ======================================
# 1. LOAD DATA
# ======================================
df = pd.read_csv(IN_FILE)
print("Loaded:", df.shape, "from", IN_FILE)

id_cols = [c for c in ["project", "bug_id", "id"] if c in df.columns]

# All perf columns
perf_cols = [c for c in df.columns if c.startswith("mrr_") or c.startswith("top@")]
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

# Candidate features: numeric, not perf, not IDs
feature_cols = [c for c in numeric_cols if c not in perf_cols + id_cols]
if not INCLUDE_MISSINGNESS_FEATURES:
    feature_cols = [c for c in feature_cols if not (c.endswith("_is_missing") or "__is_missing" in c)]

print("Feature columns:", len(feature_cols))

def tools_for_prefix(prefix: str):
    cols = [c for c in df.columns if c.startswith(prefix + "_")]
    tools = sorted({c.split("_", 1)[1] for c in cols})
    return tools, cols

# ======================================
# 2. BUILD GAP + ADVANTAGE TARGETS
# ======================================

gap_targets = {}      # name -> series
adv_targets = {}      # name -> series  (tool vs best-other)

for prefix in BASE_METRIC_PREFIXES:
    tools, cols = tools_for_prefix(prefix)
    if len(tools) < 2:
        print(f"[WARN] Not enough tools found for prefix {prefix}. Skipping.")
        continue

    # Pairwise gaps: metric_A - metric_B
    for a, b in itertools.combinations(tools, 2):
        col_a = f"{prefix}_{a}"
        col_b = f"{prefix}_{b}"
        name_ab = f"gap_{prefix}_{a}_minus_{b}"
        name_ba = f"gap_{prefix}_{b}_minus_{a}"
        gap_targets[name_ab] = df[col_a] - df[col_b]
        gap_targets[name_ba] = df[col_b] - df[col_a]  # optional symmetric target

    # Advantage: tool - max(other tools)
    mat = np.column_stack([df[f"{prefix}_{t}"].to_numpy(dtype=float) for t in tools])
    for j, t in enumerate(tools):
        others = np.delete(mat, j, axis=1)
        best_other = np.max(others, axis=1) if others.shape[1] > 0 else np.zeros(len(df))
        adv_targets[f"adv_{prefix}_{t}"] = mat[:, j] - best_other

gap_df = pd.DataFrame(gap_targets)
adv_df = pd.DataFrame(adv_targets)

print("Gap targets:", gap_df.shape[1])
print("Adv targets:", adv_df.shape[1])

# Combine into one working DF (keeps df untouched)
df_gap = pd.concat([df, gap_df, adv_df], axis=1)

def shorten_target(t):
    t = str(t)

    # Tool abbreviations
    t = t.replace("boostnsift", "BNS")
    t = t.replace("buglocator", "BL")
    t = t.replace("locus", "LOC")

    # Advantage
    if t.startswith("adv_mrr_"):
        core = t.replace("adv_mrr_", "")
        if core.endswith("_is_missing"):
            core = core.replace("_is_missing", "")
            return f"Adv({core.upper()})*"
        return f"Adv({core.upper()})"

    # Gap
    if t.startswith("gap_mrr_"):
        core = t.replace("gap_mrr_", "")
        core = core.replace("_minus_", "−")
        return f"Δ({core.upper()})"

    return t


# ======================================
# 3. CORRELATION (Spearman) + HOLM
# ======================================

def compute_spearman_table(df, feature_cols, target_cols):
    records = []
    for target in target_cols:
        y = df[target].to_numpy(dtype=float)
        for feat in feature_cols:
            x = df[feat].to_numpy(dtype=float)

            mask = ~np.isnan(x) & ~np.isnan(y)
            if mask.sum() < 3:
                records.append({"target": target, "feature": feat, "corr": np.nan, "pval": np.nan, "n": int(mask.sum())})
                continue

            x_clean = x[mask]
            y_clean = y[mask]

            if np.nanstd(x_clean) == 0 or np.nanstd(y_clean) == 0:
                records.append({"target": target, "feature": feat, "corr": np.nan, "pval": np.nan, "n": int(mask.sum())})
                continue

            try:
                rho, p = spearmanr(x_clean, y_clean)
            except Exception:
                rho, p = np.nan, np.nan

            records.append({"target": target, "feature": feat, "corr": rho, "pval": p, "n": int(mask.sum())})

    return pd.DataFrame(records)

def apply_holm(df_corr, alpha=0.05):
    df_corr = df_corr.copy()
    mask = df_corr["pval"].notna()
    pvals = df_corr.loc[mask, "pval"].values
    if len(pvals) == 0:
        df_corr["pval_adj"] = np.nan
        df_corr["reject"] = False
        return df_corr

    reject, p_adj, _, _ = multipletests(pvals, alpha=alpha, method="holm")
    df_corr.loc[mask, "pval_adj"] = p_adj
    df_corr.loc[mask, "reject"] = reject
    df_corr["reject"] = df_corr["reject"].fillna(False)
    return df_corr

gap_corr = compute_spearman_table(df_gap, feature_cols, gap_df.columns.tolist())
gap_corr = apply_holm(gap_corr, alpha=ALPHA)
gap_corr["target"] = gap_corr["target"].apply(shorten_target)

gap_corr.to_csv(OUT_DIR / "gap_corr_spearman.csv", index=False)
print("Saved:", OUT_DIR / "gap_corr_spearman.csv")

adv_corr = compute_spearman_table(df_gap, feature_cols, adv_df.columns.tolist())
adv_corr = apply_holm(adv_corr, alpha=ALPHA)
adv_corr["target"] = adv_corr["target"].apply(shorten_target)
adv_corr.to_csv(OUT_DIR / "adv_corr_spearman.csv", index=False)
print("Saved:", OUT_DIR / "adv_corr_spearman.csv")


def select_top_bottom_features(df_corr, feature_col="feature",
                               value_col="corr", target_col="target",
                               top_n=15):
    """
    Select top-N and bottom-N features by max absolute effect
    across all targets.
    """
    d = df_corr.dropna(subset=[value_col]).copy()

    # max |effect| per feature across all targets
    scores = (
        d.groupby(feature_col)[value_col]
        .apply(lambda s: s.abs().max())
        .sort_values(ascending=False)
    )

    top_feats = scores.head(top_n).index.tolist()
    bottom_feats = scores.tail(top_n).index.tolist()

    return top_feats + bottom_feats

# ======================================
# 4. HEATMAPS (top features only)
# ======================================

def top_features_for_heatmap(df_corr, max_feats=80):
    # keep features that have strongest absolute correlation across any target
    d = df_corr.dropna(subset=["corr"]).copy()
    if d.empty:
        return []
    scores = d.groupby("feature")["corr"].apply(lambda s: s.abs().max()).sort_values(ascending=False)
    return scores.head(max_feats).index.tolist()

def make_heatmap(df_corr, title, filename, top_n=15):
    feats = select_top_bottom_features(df_corr, top_n=top_n)
    d = df_corr[df_corr["feature"].isin(feats)].copy()
    if d.empty:
        print("[WARN] No data for heatmap:", filename)
        return

    pivot = d.pivot(index="feature", columns="target", values="corr")

    plt.figure(figsize=(max(10, pivot.shape[1] * 0.9), max(10, pivot.shape[0] * 0.25)))
    sns.heatmap(pivot, cmap="coolwarm", center=0, annot=False)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(OUT_DIR / filename, dpi=300)
    plt.close()
    print("Saved heatmap:", OUT_DIR / filename)

make_heatmap(
    gap_corr,
    title="Feature correlations with tool–tool gaps (Spearman)",
    filename="heatmap_gap_corr_compressed.png",
    top_n=TOP_N_FEATURES_HEATMAP
)

make_heatmap(
    adv_corr,
    title="Feature correlations with tool advantage vs best alternative (Spearman)",
    filename="heatmap_adv_corr_compressed.png",
    top_n=TOP_N_FEATURES_HEATMAP
)

print("Done. Outputs in:", OUT_DIR)
