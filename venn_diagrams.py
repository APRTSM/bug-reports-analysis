import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

# =========================
# CONFIG
# =========================
DATA_DIR = Path(".")
# Use whichever file has per-bug per-tool success info.
# If you have tool_comparison_summary.csv (long format), use it.
IN_FILE = DATA_DIR / "tool_comparison_summary.csv"

OUT_DIR = DATA_DIR / "tool_intersections"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Define "found" condition
# If you have rank column: found = rank is not null
# Otherwise: found = mrr > 0
FOUND_DEF = "rank"  # "rank" or "mrr"

# If FOUND_DEF == "mrr" and your file does not have "mrr", we compute from rank if present.
# =========================


def pivot_success_long(perf_long: pd.DataFrame) -> pd.DataFrame:
    """
    Input expected columns:
      - project, bug_id, tool
      - rank OR mrr OR top@k columns
    Output: wide boolean DataFrame indexed by (project, bug_id) with columns like found_<tool>
    """
    df = perf_long.copy()

    # Ensure IDs exist
    if "project" not in df.columns or "bug_id" not in df.columns:
        raise ValueError("Expected columns project and bug_id in IN_FILE.")

    if "tool" not in df.columns:
        raise ValueError("Expected a 'tool' column (long format) in IN_FILE.")

    # Build found flag
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

    # Pivot wide: one row per bug, one column per tool
    wide = df.pivot_table(
        index=["project", "bug_id"],
        columns="tool",
        values="found",
        aggfunc="max",   # if duplicates exist, treat found if any entry says found
        fill_value=False
    )

    # Normalize column names
    wide.columns = [f"found_{c}" for c in wide.columns]
    wide = wide.reset_index()
    return wide


def compute_intersections(wide: pd.DataFrame, tool_cols: list[str]) -> pd.DataFrame:
    """
    Returns a table of all intersection patterns and counts.
    Pattern is like '101' aligned to tool_cols order.
    """
    M = wide[tool_cols].astype(int)
    pattern = M.astype(str).agg("".join, axis=1)
    out = pattern.value_counts().rename_axis("pattern").reset_index(name="count")

    # Add readable label
    tool_names = [c.replace("found_", "") for c in tool_cols]
    def label_from_pattern(p):
        yes = [tool_names[i] for i, ch in enumerate(p) if ch == "1"]
        no  = [tool_names[i] for i, ch in enumerate(p) if ch == "0"]
        if len(yes) == 0:
            return "None"
        return " & ".join(yes)

    out["label"] = out["pattern"].apply(label_from_pattern)
    return out


def save_basic_summary(wide: pd.DataFrame, tool_cols: list[str]) -> pd.DataFrame:
    """
    Saves: per-tool count, unique count, and pairwise overlaps.
    """
    tool_names = [c.replace("found_", "") for c in tool_cols]
    M = wide[tool_cols].astype(bool).to_numpy()

    counts = {}
    # Per-tool found
    for j, t in enumerate(tool_names):
        counts[f"found_{t}"] = int(M[:, j].sum())

    # Unique per tool
    for j, t in enumerate(tool_names):
        others = np.delete(M, j, axis=1)
        unique = M[:, j] & (~others.any(axis=1) if others.shape[1] else True)
        counts[f"unique_{t}"] = int(unique.sum())

    # All-found overlap
    counts["found_all_tools"] = int(M.all(axis=1).sum())

    # None-found
    counts["found_none"] = int((~M.any(axis=1)).sum())

    summary = pd.DataFrame([counts])
    summary.to_csv(OUT_DIR / "intersection_summary.csv", index=False)
    return summary


def plot_venn(wide: pd.DataFrame, tool_cols: list[str]):
    """
    Venn only supports 2 or 3 sets cleanly.
    """
    tool_names = [c.replace("found_", "") for c in tool_cols]
    sets = [set(wide.loc[wide[c], ["project", "bug_id"]].apply(tuple, axis=1)) for c in tool_cols]

    if len(tool_cols) == 2:
        from matplotlib_venn import venn2
        plt.figure(figsize=(6, 5))
        venn2(subsets=sets, set_labels=tool_names)
        plt.title("Bug intersections (found by tool)")
        plt.tight_layout()
        plt.savefig(OUT_DIR / "venn2_tools.png", dpi=300)
        plt.close()

    elif len(tool_cols) == 3:
        from matplotlib_venn import venn3
        plt.figure(figsize=(7, 6))
        venn3(subsets=sets, set_labels=tool_names)
        plt.title("Bug intersections (found by tool)")
        plt.tight_layout()
        plt.savefig(OUT_DIR / "venn3_tools.png", dpi=300)
        plt.close()
    else:
        raise ValueError("Venn plotting supports only 2 or 3 tools.")


def plot_upset(wide: pd.DataFrame, tool_cols: list[str]):
    """
    UpSet is the right plot for 4+ tools.
    """
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
    plt.savefig(OUT_DIR / "upset_tools.png", dpi=300)
    plt.close()


def bar_from_patterns(pattern_df, tools=("boostnsift","buglocator","locus"),
                      out_path="intersections_bar.png"):
    # pattern_df has columns: pattern, count
    tool_order = list(tools)

    # helper to name patterns
    def name(p):
        on = [tool_order[i] for i,ch in enumerate(p) if ch=="1"]
        return " ∩ ".join(on) if on else "none"

    # keep only non-empty intersections
    d = pattern_df.copy()
    d = d[d["pattern"] != "000"].copy()
    d["label"] = d["pattern"].apply(name)

    # sort by count desc
    d = d.sort_values("count", ascending=False)

    plt.figure(figsize=(10, 5))
    plt.bar(d["label"], d["count"])
    plt.xticks(rotation=35, ha="right")
    plt.ylabel("#bugs")
    plt.title("Bug intersections (found by tool)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print("Saved:", out_path)


# =========================
# MAIN
# =========================
perf = pd.read_csv(IN_FILE)
print("Loaded:", perf.shape, "from", IN_FILE)

# Expect long format tool_comparison_summary.csv
wide = pivot_success_long(perf)

tool_cols = [c for c in wide.columns if c.startswith("found_")]
if len(tool_cols) < 2:
    raise RuntimeError("Need at least 2 tools to compute intersections.")

print("Tools:", [c.replace("found_", "") for c in tool_cols])

# Save intersections table
intersections = compute_intersections(wide, tool_cols)
intersections.to_csv(OUT_DIR / "intersection_patterns.csv", index=False)
print("Saved:", OUT_DIR / "intersection_patterns.csv")

# Save basic summary
summary = save_basic_summary(wide, tool_cols)
print("Saved:", OUT_DIR / "intersection_summary.csv")

# Plot
plot_upset(wide, tool_cols)
print("Saved UpSet plot in:", OUT_DIR)

