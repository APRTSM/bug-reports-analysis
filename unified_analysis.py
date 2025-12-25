"""
Unified Analysis Script for Bug Report Feature Analysis

This script combines four analysis types:
1. Correlation Analysis: Spearman correlations between features and tool performance gaps/advantages
2. Success/Failure Analysis: Mann-Whitney U tests comparing features between success/failure groups
3. Clustered Heatmaps: Feature-clustered heatmaps from correlation results
4. Venn/UpSet Diagrams: Tool intersection analysis

Each analysis can be enabled/disabled via configuration flags.
"""

import itertools
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.stats import spearmanr, mannwhitneyu
from statsmodels.stats.multitest import multipletests

# ======================================
# CONFIGURATION
# ======================================
DATA_DIR = Path(".")

# Input files
IN_FILE_PREPROCESSED = DATA_DIR / "experimentA_preprocessed_rich.csv"
IN_FILE_TOOL_COMPARISON = DATA_DIR / "tool_comparison_summary.csv"

# Output directories
OUT_DIR_CORR = DATA_DIR / "experimentA_gap_corr_results"
OUT_DIR_SUCCESS = DATA_DIR / "experimentA_success_failure"
OUT_DIR_CLUSTERED = DATA_DIR / "clustered_heatmaps_gap"
OUT_DIR_VENN = DATA_DIR / "tool_intersections"

# Analysis flags - set to True/False to enable/disable each analysis
RUN_CORRELATION_ANALYSIS = True
RUN_SUCCESS_FAILURE_ANALYSIS = True
RUN_CLUSTERED_HEATMAPS = True
RUN_VENN_DIAGRAMS = True

# Shared settings
ALPHA = 0.05
INCLUDE_MISSINGNESS_FEATURES = True

# Correlation Analysis settings
BASE_METRIC_PREFIXES = ["mrr"]  # e.g., ["mrr", "top@1", "top@5"]
TOP_N_FEATURES_HEATMAP = 15

# Success/Failure Analysis settings
BASE_PREFIX = "mrr"  # "mrr" or "top@5" or "top@1"
LABEL_MODE = "winner"  # "success", "winner", or "advantage"
ADV_EPS = 1e-12
TOP_K_PLOTS = 6
MIN_GROUP_N = 8

# Clustered Heatmaps settings
MAX_ROWS_PER_CLUSTER = 25
TOP_BOTTOM_SPLIT = True
# Note: Uses correlation results from RUN_CORRELATION_ANALYSIS
# If correlation analysis is disabled, set IN_CORR_FILE manually
IN_CORR_FILE = OUT_DIR_CORR / "gap_corr_spearman.csv"

# Venn Diagrams settings
FOUND_DEF = "rank"  # "rank" or "mrr"

# ======================================
# SHARED HELPER FUNCTIONS
# ======================================

def safe_name(s: str) -> str:
    """Convert string to safe filename."""
    s = re.sub(r"[^a-zA-Z0-9_]+", "_", s)
    return s.strip("_").lower()

def get_tools(df, prefix):
    """Extract tool names from columns with given prefix."""
    cols = [c for c in df.columns if c.startswith(prefix + "_")]
    tools = sorted({c.split("_", 1)[1] for c in cols})
    return tools

def tools_for_prefix(df, prefix: str):
    """Get tools and columns for a given prefix."""
    cols = [c for c in df.columns if c.startswith(prefix + "_")]
    tools = sorted({c.split("_", 1)[1] for c in cols})
    return tools, cols

def apply_holm(df_corr, alpha=0.05, pval_col="pval"):
    """Apply Holm-Bonferroni correction for multiple testing."""
    df_corr = df_corr.copy()
    mask = df_corr[pval_col].notna()
    pvals = df_corr.loc[mask, pval_col].values
    if len(pvals) == 0:
        df_corr["pval_adj"] = np.nan
        df_corr["reject"] = False
        return df_corr

    reject, p_adj, _, _ = multipletests(pvals, alpha=alpha, method="holm")
    df_corr.loc[mask, "pval_adj"] = p_adj
    df_corr.loc[mask, "reject"] = reject
    df_corr["reject"] = df_corr["reject"].fillna(False)
    return df_corr

def load_data_and_features(in_file):
    """Load data and extract feature columns."""
    df = pd.read_csv(in_file)
    print(f"Loaded: {df.shape} from {in_file}")

    id_cols = [c for c in ["project", "bug_id", "id"] if c in df.columns]
    perf_cols = [c for c in df.columns if c.startswith("mrr_") or c.startswith("top@")]
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [c for c in numeric_cols if c not in perf_cols + id_cols]

    if not INCLUDE_MISSINGNESS_FEATURES:
        feature_cols = [c for c in feature_cols if not (c.endswith("_is_missing") or "__is_missing" in c)]

    print(f"Feature columns: {len(feature_cols)}")
    return df, feature_cols, id_cols, perf_cols

# ======================================
# 1. CORRELATION ANALYSIS
# ======================================

def shorten_target(t):
    """Shorten target names for display."""
    t = str(t)
    t = t.replace("boostnsift", "BNS")
    t = t.replace("buglocator", "BL")
    t = t.replace("locus", "LOC")

    if t.startswith("adv_mrr_"):
        core = t.replace("adv_mrr_", "")
        if core.endswith("_is_missing"):
            core = core.replace("_is_missing", "")
            return f"Adv({core.upper()})*"
        return f"Adv({core.upper()})"

    if t.startswith("gap_mrr_"):
        core = t.replace("gap_mrr_", "")
        core = core.replace("_minus_", "−")
        return f"Δ({core.upper()})"

    return t

def compute_spearman_table(df, feature_cols, target_cols):
    """Compute Spearman correlations between features and targets."""
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

def select_top_bottom_features(df_corr, feature_col="feature", value_col="corr", top_n=15):
    """Select top-N and bottom-N features by max absolute effect."""
    d = df_corr.dropna(subset=[value_col]).copy()
    scores = (
        d.groupby(feature_col)[value_col]
        .apply(lambda s: s.abs().max())
        .sort_values(ascending=False)
    )
    top_feats = scores.head(top_n).index.tolist()
    bottom_feats = scores.tail(top_n).index.tolist()
    return top_feats + bottom_feats

def make_heatmap(df_corr, title, filename, out_dir, top_n=15):
    """Create heatmap from correlation results."""
    feats = select_top_bottom_features(df_corr, top_n=top_n)
    d = df_corr[df_corr["feature"].isin(feats)].copy()
    if d.empty:
        print(f"[WARN] No data for heatmap: {filename}")
        return

    pivot = d.pivot(index="feature", columns="target", values="corr")
    plt.figure(figsize=(max(10, pivot.shape[1] * 0.9), max(10, pivot.shape[0] * 0.25)))
    sns.heatmap(pivot, cmap="coolwarm", center=0, annot=False)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_dir / filename, dpi=300)
    plt.close()
    print(f"Saved heatmap: {out_dir / filename}")

def run_correlation_analysis():
    """Run correlation analysis between features and tool performance gaps/advantages."""
    print("\n" + "=" * 60)
    print("CORRELATION ANALYSIS")
    print("=" * 60)

    OUT_DIR_CORR.mkdir(exist_ok=True, parents=True)

    df, feature_cols, id_cols, perf_cols = load_data_and_features(IN_FILE_PREPROCESSED)

    # Build gap and advantage targets
    gap_targets = {}
    adv_targets = {}

    for prefix in BASE_METRIC_PREFIXES:
        tools, cols = tools_for_prefix(df, prefix)
        if len(tools) < 2:
            print(f"[WARN] Not enough tools found for prefix {prefix}. Skipping.")
            continue

        # Pairwise gaps
        for a, b in itertools.combinations(tools, 2):
            col_a = f"{prefix}_{a}"
            col_b = f"{prefix}_{b}"
            name_ab = f"gap_{prefix}_{a}_minus_{b}"
            name_ba = f"gap_{prefix}_{b}_minus_{a}"
            gap_targets[name_ab] = df[col_a] - df[col_b]
            gap_targets[name_ba] = df[col_b] - df[col_a]

        # Advantage: tool - max(other tools)
        mat = np.column_stack([df[f"{prefix}_{t}"].to_numpy(dtype=float) for t in tools])
        for j, t in enumerate(tools):
            others = np.delete(mat, j, axis=1)
            best_other = np.max(others, axis=1) if others.shape[1] > 0 else np.zeros(len(df))
            adv_targets[f"adv_{prefix}_{t}"] = mat[:, j] - best_other

    gap_df = pd.DataFrame(gap_targets)
    adv_df = pd.DataFrame(adv_targets)

    print(f"Gap targets: {gap_df.shape[1]}")
    print(f"Adv targets: {adv_df.shape[1]}")

    df_gap = pd.concat([df, gap_df, adv_df], axis=1)

    # Compute correlations
    gap_corr = compute_spearman_table(df_gap, feature_cols, gap_df.columns.tolist())
    gap_corr = apply_holm(gap_corr, alpha=ALPHA)
    gap_corr["target"] = gap_corr["target"].apply(shorten_target)

    adv_corr = compute_spearman_table(df_gap, feature_cols, adv_df.columns.tolist())
    adv_corr = apply_holm(adv_corr, alpha=ALPHA)
    adv_corr["target"] = adv_corr["target"].apply(shorten_target)

    # Save results
    gap_corr.to_csv(OUT_DIR_CORR / "gap_corr_spearman.csv", index=False)
    print(f"Saved: {OUT_DIR_CORR / 'gap_corr_spearman.csv'}")

    adv_corr.to_csv(OUT_DIR_CORR / "adv_corr_spearman.csv", index=False)
    print(f"Saved: {OUT_DIR_CORR / 'adv_corr_spearman.csv'}")

    # Create heatmaps
    make_heatmap(
        gap_corr,
        title="Feature correlations with tool–tool gaps (Spearman)",
        filename="heatmap_gap_corr_compressed.png",
        out_dir=OUT_DIR_CORR,
        top_n=TOP_N_FEATURES_HEATMAP
    )

    make_heatmap(
        adv_corr,
        title="Feature correlations with tool advantage vs best alternative (Spearman)",
        filename="heatmap_adv_corr_compressed.png",
        out_dir=OUT_DIR_CORR,
        top_n=TOP_N_FEATURES_HEATMAP
    )

    print(f"Done. Outputs in: {OUT_DIR_CORR}")
    return gap_corr, adv_corr

# ======================================
# 2. SUCCESS/FAILURE ANALYSIS
# ======================================

def cliffs_delta(x, y):
    """Compute Cliff's delta effect size."""
    x = np.asarray(x)
    y = np.asarray(y)
    nx, ny = len(x), len(y)
    if nx == 0 or ny == 0:
        return np.nan
    more = 0
    less = 0
    for xi in x:
        more += np.sum(xi > y)
        less += np.sum(xi < y)
    return (more - less) / (nx * ny)

def build_labels(df, tools, prefix, mode, adv_eps=1e-12):
    """Build binary labels for success/failure analysis."""
    mat = np.column_stack([df[f"{prefix}_{t}"].to_numpy(dtype=float) for t in tools])
    best = np.max(mat, axis=1)
    ties = (np.abs(mat - best[:, None]) <= adv_eps).sum(axis=1)

    labels = {}
    for j, t in enumerate(tools):
        col = f"{prefix}_{t}"
        if mode == "success":
            if prefix == "mrr":
                labels[t] = (df[col] > 0).astype(int)
            else:
                labels[t] = (df[col] == 1).astype(int)
        elif mode == "winner":
            is_best = np.abs(mat[:, j] - best) <= adv_eps
            labels[t] = (is_best & (ties == 1)).astype(int)
        elif mode == "advantage":
            others = np.delete(mat, j, axis=1)
            best_other = np.max(others, axis=1) if others.shape[1] > 0 else np.zeros(len(df))
            labels[t] = ((mat[:, j] - best_other) > adv_eps).astype(int)
        else:
            raise ValueError("Unknown LABEL_MODE: " + mode)

    return labels

def run_success_failure_analysis():
    """Run success/failure analysis using Mann-Whitney U tests."""
    print("\n" + "=" * 60)
    print("SUCCESS/FAILURE ANALYSIS")
    print("=" * 60)

    OUT_DIR_SUCCESS.mkdir(exist_ok=True, parents=True)

    df, feature_cols, id_cols, perf_cols = load_data_and_features(IN_FILE_PREPROCESSED)

    tools = get_tools(df, BASE_PREFIX)
    print(f"Tools: {tools}")

    if len(tools) < 2:
        raise RuntimeError(f"Need at least 2 tools for BASE_PREFIX={BASE_PREFIX}")

    labels = build_labels(df, tools, BASE_PREFIX, LABEL_MODE, adv_eps=ADV_EPS)

    all_tool_stats = []

    for tool in tools:
        label = labels[tool]
        label_name = f"{LABEL_MODE}_{BASE_PREFIX}_{tool}"
        df[label_name] = label

        n_pos = int(label.sum())
        n_neg = int((label == 0).sum())
        print(f"\n=== {tool} | label={label_name} ===")
        print(f"Pos: {n_pos}, Neg: {n_neg}")

        if n_pos < MIN_GROUP_N or n_neg < MIN_GROUP_N:
            print(f"Skipping {tool}: insufficient group sizes (min={MIN_GROUP_N})")
            continue

        records = []
        for feat in feature_cols:
            x = df.loc[df[label_name] == 1, feat].dropna()
            y = df.loc[df[label_name] == 0, feat].dropna()

            if len(x) < MIN_GROUP_N or len(y) < MIN_GROUP_N:
                records.append({
                    "tool": tool,
                    "label_mode": LABEL_MODE,
                    "base_prefix": BASE_PREFIX,
                    "feature": feat,
                    "n_pos": len(x),
                    "n_neg": len(y),
                    "median_pos": x.median() if len(x) else np.nan,
                    "median_neg": y.median() if len(y) else np.nan,
                    "u_stat": np.nan,
                    "p_value": np.nan,
                    "cliffs_delta": np.nan
                })
                continue

            u_stat, p_val = mannwhitneyu(x, y, alternative="two-sided", method="asymptotic")
            delta = cliffs_delta(x.values, y.values)

            records.append({
                "tool": tool,
                "label_mode": LABEL_MODE,
                "base_prefix": BASE_PREFIX,
                "feature": feat,
                "n_pos": len(x),
                "n_neg": len(y),
                "median_pos": x.median(),
                "median_neg": y.median(),
                "u_stat": u_stat,
                "p_value": p_val,
                "cliffs_delta": delta
            })

        stats_df = pd.DataFrame(records)
        stats_df = apply_holm(stats_df, alpha=ALPHA, pval_col="p_value")
        stats_df["abs_delta"] = stats_df["cliffs_delta"].abs()
        stats_df.sort_values(["reject", "abs_delta", "p_adj"], ascending=[False, False, True], inplace=True)

        out_stats = OUT_DIR_SUCCESS / f"stats_{safe_name(label_name)}.csv"
        stats_df.to_csv(out_stats, index=False)
        print(f"Saved: {out_stats}")

        all_tool_stats.append(stats_df)

        # Plots
        sig = stats_df[stats_df["reject"] == True]
        top_feats = (sig.head(TOP_K_PLOTS)["feature"].tolist()
                     if len(sig) >= 1 else stats_df.head(TOP_K_PLOTS)["feature"].tolist())

        for feat in top_feats:
            clean = df[[feat, label_name]].dropna()
            if clean.empty:
                continue

            plt.figure(figsize=(5, 4))
            sns.boxplot(data=clean, x=label_name, y=feat)
            plt.xticks([0, 1], ["neg", "pos"])
            plt.title(f"{tool} | {LABEL_MODE} | {BASE_PREFIX} | {feat}")
            plt.tight_layout()
            plt.savefig(OUT_DIR_SUCCESS / f"box_{safe_name(label_name)}_{safe_name(feat)}.png", dpi=300)
            plt.close()

    if len(all_tool_stats) == 0:
        raise RuntimeError("No tool produced enough positives/negatives to analyze.")

    global_df = pd.concat(all_tool_stats, ignore_index=True)
    out_global = OUT_DIR_SUCCESS / f"GLOBAL_{safe_name(LABEL_MODE)}_{safe_name(BASE_PREFIX)}.csv"
    global_df.to_csv(out_global, index=False)
    print(f"\nSaved GLOBAL: {out_global}")
    print(f"Global shape: {global_df.shape}")

    # Global heatmap
    pivot = global_df.pivot_table(index="feature", columns="tool", values="cliffs_delta", aggfunc="mean")
    top_feats = pivot.abs().max(axis=1).sort_values(ascending=False).head(50).index
    pivot = pivot.loc[top_feats]

    plt.figure(figsize=(max(8, pivot.shape[1] * 1.2), max(10, pivot.shape[0] * 0.3)))
    sns.heatmap(pivot, cmap="coolwarm", center=0)
    plt.title(f"Feature separation by {LABEL_MODE} ({BASE_PREFIX}) using Cliff's delta")
    plt.tight_layout()
    plt.savefig(OUT_DIR_SUCCESS / f"HEATMAP_{safe_name(LABEL_MODE)}_{safe_name(BASE_PREFIX)}.png", dpi=300)
    plt.close()

    print(f"Done. Outputs in: {OUT_DIR_SUCCESS}")

# ======================================
# 3. CLUSTERED HEATMAPS
# ======================================

CLUSTER_RULES = [
    ("Concept signals", [r"^concept_", r"^concept__"]),
    ("Text: title", [r"^txt_title_"]),
    ("Text: reasoning", [r"^txt_reasoning_"]),
    ("Text: impacted concepts", [r"^txt_likely_impacted_code_concepts_"]),
    ("Verbosity & length", [
        r"(?:^n_words$|^n_tokens$|^n_sentences$)",
        r"(?:description_length|description_chars|summary_chars)",
        r"(?:word_count|char_len|avg_word_len|avg_sentence_len|avg_words_per_line)",
    ]),
    ("Steps & reproducibility", [
        r"(?:num_steps|avg_step_length|steps_with_|potential_.*steps)",
        r"(?:temporal_|num_temporal_markers)",
    ]),
    ("Readability indices", [
        r"(?:flesch|gunning_fog|smog|coleman_liau|automated_readability_index)",
    ]),
    ("Modality & hedging", [
        r"(?:modal_|num_modal_|hedge_|exclaim_|question_density|num_negative_modals)",
    ]),
    ("Environment & platform", [
        r"(?:num_env_mentions|num_os_mentions|num_versions|num_browser_mentions)",
    ]),
    ("Exceptions & stack traces", [r"(?:num_exception_types|stacktrace_)"]),
    ("Reasoning quality (LLM)", [
        r"(?:clarity|confidence|actionability|ambiguity|specificity|completeness_score|causal_reasoning_quality|expected_observed_alignment|repair_difficulty)",
        r"(?:causal_density|num_causal_markers|num_caused_by)",
    ]),
    ("Other", [r".*"]),
]

def assign_cluster(feature_name: str) -> str:
    """Assign feature to cluster based on naming patterns."""
    for cluster, patterns in CLUSTER_RULES:
        for pat in patterns:
            if re.search(pat, feature_name):
                return cluster
    return "Other"

def select_rows_for_cluster(dsub: pd.DataFrame, max_rows: int) -> pd.DataFrame:
    """Select top/bottom features within a cluster."""
    scores = (
        dsub.dropna(subset=["corr"])
            .groupby("feature")["corr"]
            .apply(lambda s: np.max(np.abs(s.values)))
            .sort_values(ascending=False)
    )
    if len(scores) <= max_rows:
        return dsub[dsub["feature"].isin(scores.index.tolist())]

    if TOP_BOTTOM_SPLIT:
        k = max_rows // 2
        top = scores.head(k).index.tolist()
        bottom = scores.tail(max_rows - k).index.tolist()
        keep = top + bottom
    else:
        keep = scores.head(max_rows).index.tolist()

    return dsub[dsub["feature"].isin(keep)]

def plot_cluster_heatmap(dsub: pd.DataFrame, cluster_name: str, out_dir: Path):
    """Plot heatmap for a feature cluster."""
    pivot = dsub.pivot_table(index="feature", columns="target", values="corr", aggfunc="first")
    strength = pivot.abs().max(axis=1).sort_values(ascending=False)
    pivot = pivot.loc[strength.index]

    mat = pivot.values
    fig_w = max(7, 0.9 * pivot.shape[1] + 2)
    fig_h = max(4, 0.28 * pivot.shape[0] + 2)

    plt.figure(figsize=(fig_w, fig_h))
    im = plt.imshow(mat, aspect="auto", vmin=-0.5, vmax=0.5)
    plt.colorbar(im, fraction=0.03, pad=0.02)
    plt.title(f"{cluster_name} (Spearman corr)", fontsize=14)
    plt.yticks(range(pivot.shape[0]), pivot.index, fontsize=9)
    plt.xticks(range(pivot.shape[1]), pivot.columns, rotation=90, fontsize=9)
    plt.xlabel("target")
    plt.ylabel("feature")
    plt.tight_layout()

    out_path = out_dir / f"heatmap_{cluster_name.replace(' ','_').replace('/','_')}.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved: {out_path}")

def run_clustered_heatmaps():
    """Run clustered heatmap analysis from correlation results."""
    print("\n" + "=" * 60)
    print("CLUSTERED HEATMAPS")
    print("=" * 60)

    OUT_DIR_CLUSTERED.mkdir(exist_ok=True, parents=True)

    if not IN_CORR_FILE.exists():
        print(f"[WARN] Correlation file not found: {IN_CORR_FILE}")
        print("Skipping clustered heatmaps. Run correlation analysis first.")
        return

    df = pd.read_csv(IN_CORR_FILE)
    print(f"Loaded correlations: {df.shape} from {IN_CORR_FILE}")

    if "target" not in df.columns and "metric" in df.columns:
        df = df.rename(columns={"metric": "target"})
    if "corr" not in df.columns:
        raise ValueError("Expected a 'corr' column in the correlation table.")

    df["cluster"] = df["feature"].astype(str).apply(assign_cluster)

    print("\nCluster counts:")
    print(df["cluster"].value_counts())

    for cluster_name in sorted(df["cluster"].unique()):
        dsub = df[df["cluster"] == cluster_name].copy()
        if dsub.empty:
            continue

        dsub = select_rows_for_cluster(dsub, MAX_ROWS_PER_CLUSTER)

        if dsub["feature"].nunique() < 2:
            continue

        plot_cluster_heatmap(dsub, cluster_name, OUT_DIR_CLUSTERED)

    print(f"\nDone. Clustered heatmaps in: {OUT_DIR_CLUSTERED}")

# ======================================
# 4. VENN/UpSet DIAGRAMS
# ======================================

def pivot_success_long(perf_long: pd.DataFrame) -> pd.DataFrame:
    """Convert long-format performance data to wide boolean format."""
    df = perf_long.copy()

    if "project" not in df.columns or "bug_id" not in df.columns:
        raise ValueError("Expected columns project and bug_id in IN_FILE.")
    if "tool" not in df.columns:
        raise ValueError("Expected a 'tool' column (long format) in IN_FILE.")

    if FOUND_DEF == "rank":
        if "rank" not in df.columns:
            raise ValueError("FOUND_DEF='rank' but 'rank' column not found.")
        df["found"] = df["rank"].notna()
    elif FOUND_DEF == "mrr":
        if "mrr" in df.columns:
            df["found"] = df["mrr"].fillna(0.0) > 0
        elif "rank" in df.columns:
            df["found"] = df["rank"].notna()
        else:
            raise ValueError("FOUND_DEF='mrr' but neither 'mrr' nor 'rank' columns exist.")
    else:
        raise ValueError("FOUND_DEF must be 'rank' or 'mrr'.")

    wide = df.pivot_table(
        index=["project", "bug_id"],
        columns="tool",
        values="found",
        aggfunc="max",
        fill_value=False
    )

    wide.columns = [f"found_{c}" for c in wide.columns]
    wide = wide.reset_index()
    return wide

def compute_intersections(wide: pd.DataFrame, tool_cols: list) -> pd.DataFrame:
    """Compute intersection patterns and counts."""
    M = wide[tool_cols].astype(int)
    pattern = M.astype(str).agg("".join, axis=1)
    out = pattern.value_counts().rename_axis("pattern").reset_index(name="count")

    tool_names = [c.replace("found_", "") for c in tool_cols]
    def label_from_pattern(p):
        yes = [tool_names[i] for i, ch in enumerate(p) if ch == "1"]
        return " & ".join(yes) if yes else "None"

    out["label"] = out["pattern"].apply(label_from_pattern)
    return out

def save_basic_summary(wide: pd.DataFrame, tool_cols: list) -> pd.DataFrame:
    """Save summary statistics for tool intersections."""
    tool_names = [c.replace("found_", "") for c in tool_cols]
    M = wide[tool_cols].astype(bool).to_numpy()

    counts = {}
    for j, t in enumerate(tool_names):
        counts[f"found_{t}"] = int(M[:, j].sum())

    for j, t in enumerate(tool_names):
        others = np.delete(M, j, axis=1)
        unique = M[:, j] & (~others.any(axis=1) if others.shape[1] else True)
        counts[f"unique_{t}"] = int(unique.sum())

    counts["found_all_tools"] = int(M.all(axis=1).sum())
    counts["found_none"] = int((~M.any(axis=1)).sum())

    summary = pd.DataFrame([counts])
    summary.to_csv(OUT_DIR_VENN / "intersection_summary.csv", index=False)
    return summary

def plot_venn(wide: pd.DataFrame, tool_cols: list):
    """Plot Venn diagram (2 or 3 tools only)."""
    tool_names = [c.replace("found_", "") for c in tool_cols]
    sets = [set(wide.loc[wide[c], ["project", "bug_id"]].apply(tuple, axis=1)) for c in tool_cols]

    if len(tool_cols) == 2:
        from matplotlib_venn import venn2
        plt.figure(figsize=(6, 5))
        venn2(subsets=sets, set_labels=tool_names)
        plt.title("Bug intersections (found by tool)")
        plt.tight_layout()
        plt.savefig(OUT_DIR_VENN / "venn2_tools.png", dpi=300)
        plt.close()
    elif len(tool_cols) == 3:
        from matplotlib_venn import venn3
        plt.figure(figsize=(7, 6))
        venn3(subsets=sets, set_labels=tool_names)
        plt.title("Bug intersections (found by tool)")
        plt.tight_layout()
        plt.savefig(OUT_DIR_VENN / "venn3_tools.png", dpi=300)
        plt.close()
    else:
        raise ValueError("Venn plotting supports only 2 or 3 tools.")

def plot_upset(wide: pd.DataFrame, tool_cols: list):
    """Plot UpSet diagram (for 4+ tools)."""
    try:
        from upsetplot import UpSet, from_indicators
    except ImportError:
        raise ImportError(
            "upsetplot is not installed. Install with: pip install upsetplot\n"
            "Or reduce to 3 tools for a Venn diagram."
        )

    tool_names = [c.replace("found_", "") for c in tool_cols]
    data = wide[tool_cols].copy()
    data.columns = tool_names

    upset_data = from_indicators(tool_names, data=data)
    plt.figure(figsize=(10, 6))
    UpSet(upset_data, show_counts=True, sort_by="cardinality").plot()
    plt.suptitle("Bug intersections (found by tool)")
    plt.tight_layout()
    plt.savefig(OUT_DIR_VENN / "upset_tools.png", dpi=300)
    plt.close()

def run_venn_diagrams():
    """Run Venn/UpSet diagram analysis."""
    print("\n" + "=" * 60)
    print("VENN/UPSET DIAGRAMS")
    print("=" * 60)

    OUT_DIR_VENN.mkdir(exist_ok=True, parents=True)

    if not IN_FILE_TOOL_COMPARISON.exists():
        print(f"[WARN] Tool comparison file not found: {IN_FILE_TOOL_COMPARISON}")
        print("Skipping Venn diagrams.")
        return

    perf = pd.read_csv(IN_FILE_TOOL_COMPARISON)
    print(f"Loaded: {perf.shape} from {IN_FILE_TOOL_COMPARISON}")

    wide = pivot_success_long(perf)
    tool_cols = [c for c in wide.columns if c.startswith("found_")]
    if len(tool_cols) < 2:
        raise RuntimeError("Need at least 2 tools to compute intersections.")

    print(f"Tools: {[c.replace('found_', '') for c in tool_cols]}")

    intersections = compute_intersections(wide, tool_cols)
    intersections.to_csv(OUT_DIR_VENN / "intersection_patterns.csv", index=False)
    print(f"Saved: {OUT_DIR_VENN / 'intersection_patterns.csv'}")

    summary = save_basic_summary(wide, tool_cols)
    print(f"Saved: {OUT_DIR_VENN / 'intersection_summary.csv'}")

    # Plot
    if len(tool_cols) <= 3:
        plot_venn(wide, tool_cols)
    else:
        plot_upset(wide, tool_cols)

    print(f"Done. Outputs in: {OUT_DIR_VENN}")

# ======================================
# MAIN EXECUTION
# ======================================

if __name__ == "__main__":
    print("=" * 60)
    print("UNIFIED ANALYSIS SCRIPT")
    print("=" * 60)
    print(f"Correlation Analysis: {RUN_CORRELATION_ANALYSIS}")
    print(f"Success/Failure Analysis: {RUN_SUCCESS_FAILURE_ANALYSIS}")
    print(f"Clustered Heatmaps: {RUN_CLUSTERED_HEATMAPS}")
    print(f"Venn Diagrams: {RUN_VENN_DIAGRAMS}")
    print("=" * 60)

    if RUN_CORRELATION_ANALYSIS:
        run_correlation_analysis()

    if RUN_SUCCESS_FAILURE_ANALYSIS:
        run_success_failure_analysis()

    if RUN_CLUSTERED_HEATMAPS:
        run_clustered_heatmaps()

    if RUN_VENN_DIAGRAMS:
        run_venn_diagrams()

    print("\n" + "=" * 60)
    print("ALL ANALYSES COMPLETE")
    print("=" * 60)


