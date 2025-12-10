import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import pearsonr, spearmanr
from statsmodels.stats.multitest import multipletests
from pathlib import Path

# ======================================
# CONFIG
# ======================================

DATA_DIR = Path(".")
IN_FILE  = DATA_DIR / "experimentA_preprocessed.csv"

OUT_DIR  = DATA_DIR / "experimentA_corr_results"
OUT_DIR.mkdir(exist_ok=True, parents=True)

TOP_K_PLOTS = 5  # how many strongest features per metric to visualize with scatter plots


# ======================================
# 1. LOAD DATA
# ======================================

df = pd.read_csv(IN_FILE)
print("Loaded:", df.shape, "from", IN_FILE)

# Identify key columns
id_cols = [c for c in ["bug_id", "project"] if c in df.columns]

# Performance metrics: any column starting with mrr_ or top@
perf_cols = [c for c in df.columns if c.startswith("mrr_") or c.startswith("top@")]
print("Performance columns:", perf_cols)

# Candidate feature columns: numeric, not perf, not IDs
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
feature_cols = [c for c in numeric_cols if c not in perf_cols + id_cols]

print("Number of feature columns:", len(feature_cols))


# ======================================
# 2. CORRELATION FUNCTIONS
# ======================================

def compute_corr_table(df, feature_cols, perf_cols, method="pearson"):
    """
    Compute feature × metric correlations with p-values.
    method: "pearson" or "spearman"
    Returns tidy DataFrame with columns:
        metric, feature, corr, pval
    """
    records = []
    for metric in perf_cols:
        y = df[metric].values
        for feat in feature_cols:
            x = df[feat].values

            # drop rows where either is NaN
            mask = ~np.isnan(x) & ~np.isnan(y)
            if mask.sum() < 3:
                corr = np.nan
                pval = np.nan
            else:
                x_clean = x[mask]
                y_clean = y[mask]
                
                # Check if either array is constant (all values the same)
                if np.std(x_clean) == 0 or np.std(y_clean) == 0:
                    corr = np.nan
                    pval = np.nan
                else:
                    try:
                        if method == "pearson":
                            corr, pval = pearsonr(x_clean, y_clean)
                        elif method == "spearman":
                            corr, pval = spearmanr(x_clean, y_clean)
                        else:
                            raise ValueError("Unknown method: " + method)
                    except (ValueError, RuntimeWarning):
                        # Handle any edge cases that might cause errors
                        corr = np.nan
                        pval = np.nan

            records.append({
                "metric": metric,
                "feature": feat,
                "corr": corr,
                "pval": pval
            })
    return pd.DataFrame(records)


def apply_holm_bonferroni(df_corr, alpha=0.05):
    """
    Apply Holm–Bonferroni correction over all correlations in df_corr.
    Adds columns: pval_adj, reject
    """
    mask_valid = df_corr["pval"].notna()
    pvals = df_corr.loc[mask_valid, "pval"].values

    # Handle case where there are no valid p-values to adjust
    if len(pvals) == 0:
        df_corr["pval_adj"] = np.nan
        df_corr["reject"] = False
        return df_corr

    reject, pval_adj, _, _ = multipletests(pvals, alpha=alpha, method="holm")

    df_corr.loc[mask_valid, "pval_adj"] = pval_adj
    df_corr.loc[mask_valid, "reject"] = reject

    return df_corr


# ======================================
# 3. COMPUTE PEARSON & SPEARMAN
# ======================================

pearson_df = compute_corr_table(df, feature_cols, perf_cols, method="pearson")
spearman_df = compute_corr_table(df, feature_cols, perf_cols, method="spearman")

pearson_df = apply_holm_bonferroni(pearson_df, alpha=0.05)
spearman_df = apply_holm_bonferroni(spearman_df, alpha=0.05)

# Save tables
pearson_df.to_csv(OUT_DIR / "correlations_pearson.csv", index=False)
spearman_df.to_csv(OUT_DIR / "correlations_spearman.csv", index=False)

print("Saved correlation tables to", OUT_DIR)


# ======================================
# 4. HEATMAP VISUALIZATIONS
# ======================================

def make_heatmap(df_corr, title, filename, value_col="corr"):
    """
    Create a heatmap with features on y-axis and metrics on x-axis.
    """
    # Pivot to metrics (columns) × features (rows)
    pivot = df_corr.pivot(index="feature", columns="metric", values=value_col)

    plt.figure(figsize=(max(8, len(perf_cols) * 1.2), max(10, len(feature_cols) * 0.2)))
    sns.heatmap(pivot, cmap="coolwarm", center=0, annot=False)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(OUT_DIR / filename, dpi=300)
    plt.close()


# Pearson heatmap
make_heatmap(
    pearson_df,
    title="Feature–Performance Correlations (Pearson)",
    filename="heatmap_pearson.png",
    value_col="corr"
)

# Spearman heatmap
make_heatmap(
    spearman_df,
    title="Feature–Performance Correlations (Spearman)",
    filename="heatmap_spearman.png",
    value_col="corr"
)

print("Saved heatmaps to", OUT_DIR)


# ======================================
# 5. SCATTER PLOTS FOR STRONGEST CORRELATIONS
# ======================================

def plot_top_k_scatter(df, df_corr, metric, k=5, method_name="pearson"):
    """
    For a given metric, plot scatter + regression for top-k |corr| features.
    """
    sub = df_corr[df_corr["metric"] == metric].dropna(subset=["corr"])
    sub = sub.reindex(sub["corr"].abs().sort_values(ascending=False).index)  # sort by |corr|
    top = sub.head(k)

    for _, row in top.iterrows():
        feat = row["feature"]
        corr_val = row["corr"]
        p_adj = row.get("pval_adj", np.nan)

        plt.figure(figsize=(6, 4))
        sns.regplot(x=df[feat], y=df[metric], scatter_kws={"alpha": 0.5}, line_kws={"linewidth": 2})
        plt.xlabel(feat)
        plt.ylabel(metric)
        plt.title(f"{method_name} corr={corr_val:.3f}, p_adj={p_adj:.3g}")
        plt.tight_layout()

        fname = f"scatter_{method_name}_{metric}_{feat}.png".replace("/", "_")
        plt.savefig(OUT_DIR / fname, dpi=300)
        plt.close()


# Example: use Spearman for scatter (often nicer for monotonic relationships)
for metric in perf_cols:
    print(f"Generating scatter plots for metric: {metric}")
    plot_top_k_scatter(df, spearman_df, metric, k=TOP_K_PLOTS, method_name="spearman")

print("Scatter plots saved to", OUT_DIR)
print("Done.")
