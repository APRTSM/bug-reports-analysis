import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests
import matplotlib.pyplot as plt
import seaborn as sns
import re

# ===========================
# CONFIG
# ===========================
DATA_DIR = Path(".")
IN_FILE = DATA_DIR / "experimentA_preprocessed_rich.csv"
OUT_DIR = DATA_DIR / "experimentA_success_failure"
OUT_DIR.mkdir(parents=True, exist_ok=True)

ALPHA = 0.05
TOP_K_PLOTS = 6
MIN_GROUP_N = 8
INCLUDE_MISSINGNESS_FEATURES = True

# Choose performance base for labels
BASE_PREFIX = "mrr"   # "mrr" or "top@5" or "top@1"

# Label type:
#  - "success"   : tool succeeds vs fails (e.g., mrr>0 or top@k==1)
#  - "winner"    : tool is unique best vs not (based on BASE_PREFIX)
#  - "advantage" : tool has positive advantage vs best other tool (based on BASE_PREFIX)
LABEL_MODE = "winner"

# For advantage label: require margin > EPS (avoids tiny floating ties)
ADV_EPS = 1e-12


# ===========================
# HELPERS
# ===========================
def safe_name(s: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9_]+", "_", s)
    return s.strip("_").lower()

def cliffs_delta(x, y):
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

def get_tools(df, prefix):
    cols = [c for c in df.columns if c.startswith(prefix + "_")]
    tools = sorted({c.split("_", 1)[1] for c in cols})
    return tools

def apply_holm(stats_df, alpha=0.05):
    stats_df = stats_df.copy()
    mask = stats_df["p_value"].notna()
    pvals = stats_df.loc[mask, "p_value"].values
    if len(pvals) == 0:
        stats_df["p_adj"] = np.nan
        stats_df["reject"] = False
        return stats_df
    reject, p_adj, _, _ = multipletests(pvals, alpha=alpha, method="holm")
    stats_df.loc[mask, "p_adj"] = p_adj
    stats_df.loc[mask, "reject"] = reject
    stats_df["reject"] = stats_df["reject"].fillna(False)
    return stats_df

def build_labels(df, tools, prefix, mode, adv_eps=1e-12):
    """
    Returns a dict: tool -> binary label Series
    mode:
      - success: (mrr_tool > 0) or (top@k_tool == 1)
      - winner: tool is unique best for this bug
      - advantage: tool > best_other by eps
    """
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


# ===========================
# LOAD + FEATURES
# ===========================
df = pd.read_csv(IN_FILE)
print("Loaded:", df.shape, "from", IN_FILE)

id_cols = [c for c in ["project", "bug_id", "id"] if c in df.columns]
perf_cols = [c for c in df.columns if c.startswith("mrr_") or c.startswith("top@")]

numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
feature_cols = [c for c in numeric_cols if c not in perf_cols + id_cols]

if not INCLUDE_MISSINGNESS_FEATURES:
    feature_cols = [c for c in feature_cols if not (c.endswith("_is_missing") or "__is_missing" in c)]

tools = get_tools(df, BASE_PREFIX)
print("Tools:", tools)
print("Features:", len(feature_cols))

if len(tools) < 2:
    raise RuntimeError(f"Need at least 2 tools for BASE_PREFIX={BASE_PREFIX}")

labels = build_labels(df, tools, BASE_PREFIX, LABEL_MODE, adv_eps=ADV_EPS)

# Collect global stats
all_tool_stats = []

# ===========================
# MAIN LOOP: PER TOOL
# ===========================
for tool in tools:
    label = labels[tool]
    label_name = f"{LABEL_MODE}_{BASE_PREFIX}_{tool}"
    df[label_name] = label

    n_pos = int(label.sum())
    n_neg = int((label == 0).sum())
    print(f"\n=== {tool} | label={label_name} ===")
    print("Pos:", n_pos, "Neg:", n_neg)

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
    stats_df = apply_holm(stats_df, alpha=ALPHA)

    stats_df["abs_delta"] = stats_df["cliffs_delta"].abs()
    stats_df.sort_values(["reject", "abs_delta", "p_adj"], ascending=[False, False, True], inplace=True)

    out_stats = OUT_DIR / f"stats_{safe_name(label_name)}.csv"
    stats_df.to_csv(out_stats, index=False)
    print("Saved:", out_stats)

    all_tool_stats.append(stats_df)

    # Plots: pick significant first
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
        plt.savefig(OUT_DIR / f"box_{safe_name(label_name)}_{safe_name(feat)}.png", dpi=300)
        plt.close()

# ===========================
# GLOBAL CSV + GLOBAL HEATMAP
# ===========================
if len(all_tool_stats) == 0:
    raise RuntimeError("No tool produced enough positives/negatives to analyze. Lower MIN_GROUP_N or change LABEL_MODE.")

global_df = pd.concat(all_tool_stats, ignore_index=True)
out_global = OUT_DIR / f"GLOBAL_{safe_name(LABEL_MODE)}_{safe_name(BASE_PREFIX)}.csv"
global_df.to_csv(out_global, index=False)
print("\nSaved GLOBAL:", out_global)
print("Global shape:", global_df.shape)

# Heatmap of Cliff's delta (features x tools)
# Keep only top features for readability
pivot = global_df.pivot_table(index="feature", columns="tool", values="cliffs_delta", aggfunc="mean")
top_feats = pivot.abs().max(axis=1).sort_values(ascending=False).head(50).index
pivot = pivot.loc[top_feats]

plt.figure(figsize=(max(8, pivot.shape[1] * 1.2), max(10, pivot.shape[0] * 0.3)))
sns.heatmap(pivot, cmap="coolwarm", center=0)
plt.title(f"Feature separation by {LABEL_MODE} ({BASE_PREFIX}) using Cliff's delta")
plt.tight_layout()
plt.savefig(OUT_DIR / f"HEATMAP_{safe_name(LABEL_MODE)}_{safe_name(BASE_PREFIX)}.png", dpi=300)
plt.close()

print("Done. Outputs in:", OUT_DIR)
