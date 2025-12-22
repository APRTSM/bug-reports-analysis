import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# =========================
# CONFIG
# =========================
DATA_DIR = Path(".")
# Use your produced correlations table (adv or gap), e.g.:
IN_CORR = DATA_DIR / "experimentA_gap_corr_results" / "gap_corr_spearman.csv"   # change
# IN_CORR = DATA_DIR / "experimentA_corr_results" / "gap_correlations_spearman.csv" # change

OUT_DIR = DATA_DIR / "clustered_heatmaps_gap"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# How many rows max per cluster heatmap (keeps plots compact).
# If a cluster has more than this, we keep top/bottom by |corr| within that cluster.
MAX_ROWS_PER_CLUSTER = 25
TOP_BOTTOM_SPLIT = True  # if True: keep top K and bottom K by abs effect

# =========================
# 1) LOAD CORR TABLE
# =========================
df = pd.read_csv(IN_CORR)
print("Loaded correlations:", df.shape, "from", IN_CORR)

# Normalize column names
if "target" not in df.columns and "metric" in df.columns:
    df = df.rename(columns={"metric": "target"})
if "corr" not in df.columns:
    raise ValueError("Expected a 'corr' column in the correlation table.")
if "feature" not in df.columns or "target" not in df.columns:
    raise ValueError("Expected columns: feature, target, corr (and optionally p-values).")

# =========================
# 2) FEATURE -> CLUSTER RULES
# =========================
# You can adjust these rules to match *your* feature naming conventions.
# The first matching rule wins.
CLUSTER_RULES = [
    ("Concept signals", [
        r"^concept_",                 # concept_*
        r"^concept__",                # just in case
    ]),
    ("Text: title", [
        r"^txt_title_",               # txt_title_*
    ]),
    ("Text: reasoning", [
        r"^txt_reasoning_",           # txt_reasoning_*
    ]),
    ("Text: impacted concepts", [
        r"^txt_likely_impacted_code_concepts_",  # txt_likely_*
    ]),
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
    ("Exceptions & stack traces", [
        r"(?:num_exception_types|stacktrace_)",
    ]),
    ("Reasoning quality (LLM)", [
        r"(?:clarity|confidence|actionability|ambiguity|specificity|completeness_score|causal_reasoning_quality|expected_observed_alignment|repair_difficulty)",
        r"(?:causal_density|num_causal_markers|num_caused_by)",
    ]),
    ("Other", [r".*"]),  # fallback
]

def assign_cluster(feature_name: str) -> str:
    for cluster, patterns in CLUSTER_RULES:
        for pat in patterns:
            if re.search(pat, feature_name):
                return cluster
    return "Other"

df["cluster"] = df["feature"].astype(str).apply(assign_cluster)

print("\nCluster counts:")
print(df["cluster"].value_counts())

# =========================
# 3) OPTIONAL: LIMIT ROWS PER CLUSTER (TOP/BOTTOM BY |corr|)
# =========================
def select_rows_for_cluster(dsub: pd.DataFrame, max_rows: int) -> pd.DataFrame:
    # score feature by max |corr| across targets (within this cluster)
    scores = (
        dsub.dropna(subset=["corr"])
            .groupby("feature")["corr"]
            .apply(lambda s: np.max(np.abs(s.values)))
            .sort_values(ascending=False)
    )
    if len(scores) <= max_rows:
        keep = scores.index.tolist()
        return dsub[dsub["feature"].isin(keep)]

    if TOP_BOTTOM_SPLIT:
        k = max_rows // 2
        top = scores.head(k).index.tolist()
        bottom = scores.tail(max_rows - k).index.tolist()
        keep = top + bottom
    else:
        keep = scores.head(max_rows).index.tolist()

    return dsub[dsub["feature"].isin(keep)]

# =========================
# 4) PLOT PER CLUSTER
# =========================
def plot_cluster_heatmap(dsub: pd.DataFrame, cluster_name: str):
    # Pivot to matrix feature x target
    pivot = dsub.pivot_table(index="feature", columns="target", values="corr", aggfunc="first")

    # Order features by "strength" for nicer visuals
    strength = pivot.abs().max(axis=1).sort_values(ascending=False)
    pivot = pivot.loc[strength.index]

    # Basic heatmap with matplotlib (no seaborn needed)
    mat = pivot.values
    fig_w = max(7, 0.9 * pivot.shape[1] + 2)
    fig_h = max(4, 0.28 * pivot.shape[0] + 2)

    plt.figure(figsize=(fig_w, fig_h))
    im = plt.imshow(mat, aspect="auto", vmin=-0.5, vmax=0.5)  # adjust bounds if needed

    plt.colorbar(im, fraction=0.03, pad=0.02)
    plt.title(f"{cluster_name} (Spearman corr)", fontsize=14)

    plt.yticks(range(pivot.shape[0]), pivot.index, fontsize=9)
    plt.xticks(range(pivot.shape[1]), pivot.columns, rotation=90, fontsize=9)

    plt.xlabel("target")
    plt.ylabel("feature")
    plt.tight_layout()

    out_path = OUT_DIR / f"heatmap_{cluster_name.replace(' ','_').replace('/','_')}.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print("Saved:", out_path)

# Make heatmaps
for cluster_name in sorted(df["cluster"].unique()):
    dsub = df[df["cluster"] == cluster_name].copy()
    if dsub.empty:
        continue

    # Optionally reduce rows per cluster
    dsub = select_rows_for_cluster(dsub, MAX_ROWS_PER_CLUSTER)

    # Avoid plotting clusters with too few features
    if dsub["feature"].nunique() < 2:
        continue

    plot_cluster_heatmap(dsub, cluster_name)

print("\nDone. Clustered heatmaps in:", OUT_DIR)
