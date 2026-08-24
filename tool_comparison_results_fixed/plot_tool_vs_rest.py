#!/usr/bin/env python3
"""
Generate two figure families from tool-vs-rest discriminating-feature results:

  1. Per-tool lollipop small-multiples: one figure per K (Top-1/5/10),
     five panels (one per tool), each showing that tool's significant
     features as signed-delta lollipops. FlexFL renders as an empty panel
     labelled "no significant features".

  2. Per-K heatmap: one figure per K, features (rows) x tools (columns),
     cell = signed Cliff's delta, only where significant. Non-significant
     and untested cells are left blank. Effect-size class is shown by a
     marker overlaid on the cell.

Input CSVs (in --indir): tool_vs_rest_top{1,5,10}.csv
Columns used: tool, feature, cliffs_delta, effect_size, pval_adj, significant

Output: PDF + PNG for each figure in --outdir.
"""

import argparse
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D

# Bold + large everywhere by default (titles, axis labels, tick labels all inherit
# these unless overridden locally -- most explicit fontsize= calls below still set
# their own larger size, this just guarantees nothing renders thin/small by accident).
plt.rcParams.update({
    "font.weight": "bold",
    "axes.labelweight": "bold",
    "axes.titleweight": "bold",
    "font.size": 15,
})

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
KS = [1, 5, 10]
# Canonical tool order and display names. Keep FlexFL last so the empty
# panel/column reads as a deliberate "nothing here" rather than a gap.
TOOL_ORDER = ["BRaIn", "bluir", "buglocator", "locus"]
# Matching against the "tool" column must use the exact case tool_comparison_summary.csv
# uses (bluir/buglocator/locus are lowercase there) -- this is display-only.
DISPLAY = {"BRaIn": "BRaIn", "bluir": "BLUiR", "buglocator": "BugLocator",
           "locus": "Locus", "FlexFL": "FlexFL"}

# Fixed diverging scale so magnitudes are comparable across every panel,
# tool and K. Cliff's delta is bounded in [-1, 1]; observed |delta| tops
# out well under 0.9, so this keeps colour/length resolution high.
DELTA_LIMIT = 0.9

NEG_COLOR = "#2166ac"   # higher for the others (delta < 0)
POS_COLOR = "#b2182b"   # higher for this tool (delta > 0)

EFFECT_MARKER = {"large": "o", "medium": "s", "small": "^"}
EFFECT_FILLED = {"large": True, "medium": False, "small": False}


def prettify(feature: str) -> str:
    """Turn a snake_case feature id into a compact label.

    Drops a leading "z" token (z-scored-feature prefix) rather than showing it --
    e.g. z_clarity -> "Clarity", not "Z Clarity".
    """
    special = {
        "ari": "ARI",
        "txt": "Txt",
    }
    parts = feature.split("_")
    out = []
    for p in parts:
        if p == "z":
            continue
        out.append(special.get(p, p.capitalize()))
    return " ".join(out)


def load(indir: str) -> dict:
    data = {}
    for k in KS:
        path = os.path.join(indir, f"tool_vs_rest_top{k}.csv")
        if not os.path.exists(path):
            print(f"  [warn] missing {path}, skipping K={k}")
            continue
        df = pd.read_csv(path)
        # Normalise the significance flag to real booleans.
        df["significant"] = df["significant"].astype(str).str.lower().eq("true")
        data[k] = df
    return data


def sig_features(df: pd.DataFrame, tool: str) -> pd.DataFrame:
    sub = df[(df["tool"] == tool) & (df["significant"])].copy()
    sub = sub.sort_values("cliffs_delta")
    return sub


# ----------------------------------------------------------------------
# Figure 1: per-tool lollipop small-multiples (one figure per K)
# ----------------------------------------------------------------------
def lollipop_figure(df: pd.DataFrame, k: int, outdir: str):
    n_tools = len(TOOL_ORDER)
    fig, axes = plt.subplots(
        1, n_tools,
        figsize=(7.8 * n_tools, 11),
        squeeze=False,
    )
    axes = axes[0]

    for ax, tool in zip(axes, TOOL_ORDER):
        sub = sig_features(df, tool) if tool in df["tool"].unique() else df.iloc[0:0]

        if len(sub) == 0:
            ax.set_title(DISPLAY.get(tool, tool), fontsize=26, fontweight="bold")
            ax.text(0.5, 0.5, "no significant\nfeatures",
                    ha="center", va="center", fontsize=20, fontweight="bold",
                    color="0.35", transform=ax.transAxes)
            ax.set_xlim(-DELTA_LIMIT, DELTA_LIMIT)
            ax.set_xticks([-0.8, 0.0, 0.8])
            ax.set_yticks([])
            ax.axvline(0, color="0.7", lw=1.5, zorder=0)
            ax.set_xlabel(r"Cliff's $\delta$", fontsize=21, fontweight="bold")
            ax.tick_params(axis="x", labelsize=17)
            plt.setp(ax.get_xticklabels(), fontweight="bold")
            for s in ("top", "right", "left"):
                ax.spines[s].set_visible(False)
            continue

        labels = [prettify(f) for f in sub["feature"]]
        y = np.arange(len(sub))
        deltas = sub["cliffs_delta"].values
        colors = [POS_COLOR if d > 0 else NEG_COLOR for d in deltas]

        ax.axvline(0, color="0.7", lw=1.5, zorder=0)
        ax.hlines(y, 0, deltas, color=colors, lw=4.5, zorder=1)

        for yi, d, eff in zip(y, deltas, sub["effect_size"]):
            marker = EFFECT_MARKER.get(eff, "o")
            filled = EFFECT_FILLED.get(eff, True)
            c = POS_COLOR if d > 0 else NEG_COLOR
            ax.scatter(d, yi, marker=marker, s=260, zorder=2,
                       facecolor=(c if filled else "white"),
                       edgecolor=c, linewidths=3.0)

        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=19, fontweight="bold")
        ax.set_ylim(-0.6, len(sub) - 0.4)
        ax.set_xlim(-DELTA_LIMIT, DELTA_LIMIT)
        ax.set_xticks([-0.8, 0.0, 0.8])
        ax.set_title(f"{DISPLAY.get(tool, tool)}  (n={len(sub)})", fontsize=26, fontweight="bold")
        ax.set_xlabel(r"Cliff's $\delta$", fontsize=21, fontweight="bold")
        ax.tick_params(axis="x", labelsize=17)
        plt.setp(ax.get_xticklabels(), fontweight="bold")
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)

    # Shared legend: sign + effect-size class.
    legend_elems = [
        Line2D([0], [0], color=NEG_COLOR, lw=7, label=r"$\delta<0$ (higher for others)"),
        Line2D([0], [0], color=POS_COLOR, lw=7, label=r"$\delta>0$ (higher for this tool)"),
        Line2D([0], [0], marker="o", color="0.3", lw=0, markerfacecolor="0.3",
               markersize=19, label="large effect"),
        Line2D([0], [0], marker="s", color="0.3", lw=0, markerfacecolor="white",
               markeredgecolor="0.3", markersize=19, label="medium effect"),
    ]
    fig.legend(handles=legend_elems, loc="lower center", ncol=4,
               frameon=False, fontsize=19, bbox_to_anchor=(0.5, -0.05),
               prop={"weight": "bold", "size": 19})

    fig.tight_layout(rect=[0, 0.06, 1, 0.98])

    base = os.path.join(outdir, f"lollipop_top{k}")
    fig.savefig(base + ".pdf", bbox_inches="tight")
    fig.savefig(base + ".png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {base}.pdf / .png")


# ----------------------------------------------------------------------
# Figure 2: per-K heatmap (features x tools)
# ----------------------------------------------------------------------
def heatmap_figure(df: pd.DataFrame, k: int, outdir: str):
    sig = df[df["significant"]].copy()
    if len(sig) == 0:
        print(f"  [warn] no significant features at all for K={k}, skipping heatmap")
        return

    # Feature rows: only those significant for at least one tool.
    # Order by how many tools they discriminate (descending), then by
    # mean |delta|, so consistent cross-tool features rise to the top.
    feat_stats = (sig.groupby("feature")
                     .agg(n_tools=("tool", "nunique"),
                          mean_abs=("cliffs_delta", lambda s: s.abs().mean()))
                     .sort_values(["n_tools", "mean_abs"], ascending=[False, False]))
    features = list(feat_stats.index)

    # Build signed-delta matrix; NaN where not significant.
    mat = np.full((len(features), len(TOOL_ORDER)), np.nan)
    eff = np.empty((len(features), len(TOOL_ORDER)), dtype=object)
    for i, f in enumerate(features):
        for j, t in enumerate(TOOL_ORDER):
            row = sig[(sig["feature"] == f) & (sig["tool"] == t)]
            if len(row):
                mat[i, j] = row["cliffs_delta"].values[0]
                eff[i, j] = row["effect_size"].values[0]

    fig_h = max(8, 0.75 * len(features) + 3)
    fig, ax = plt.subplots(figsize=(14, fig_h))

    norm = TwoSlopeNorm(vmin=-DELTA_LIMIT, vcenter=0.0, vmax=DELTA_LIMIT)
    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad("white")

    im = ax.imshow(mat, aspect="auto", cmap=cmap, norm=norm)

    # Effect-size markers overlaid on significant cells.
    for i in range(len(features)):
        for j in range(len(TOOL_ORDER)):
            if not np.isnan(mat[i, j]):
                e = eff[i, j]
                marker = EFFECT_MARKER.get(e, "o")
                filled = EFFECT_FILLED.get(e, True)
                ax.scatter(j, i, marker=marker, s=170,
                           facecolor=("0.15" if filled else "none"),
                           edgecolor="0.15", linewidths=2.2, zorder=3)

    ax.set_xticks(np.arange(len(TOOL_ORDER)))
    ax.set_xticklabels([DISPLAY.get(t, t) for t in TOOL_ORDER], fontsize=21,
                        fontweight="bold", rotation=30, ha="right")
    ax.set_yticks(np.arange(len(features)))
    ax.set_yticklabels([prettify(f) for f in features], fontsize=18, fontweight="bold")
    ax.set_xticks(np.arange(-0.5, len(TOOL_ORDER)), minor=True)
    ax.set_yticks(np.arange(-0.5, len(features)), minor=True)
    ax.grid(which="minor", color="0.8", lw=0.9)
    ax.tick_params(which="minor", length=0)
    ax.set_title(f"Signed Cliff's $\\delta$ of significant features (Top-{k})",
                 fontsize=26, fontweight="bold", pad=16)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cbar.set_label(r"Cliff's $\delta$  ($-$ others  /  $+$ this tool)", fontsize=18, fontweight="bold")
    cbar.ax.tick_params(labelsize=15)
    plt.setp(cbar.ax.get_yticklabels(), fontweight="bold")

    legend_elems = [
        Line2D([0], [0], marker="o", color="0.15", lw=0, markerfacecolor="0.15",
               markersize=18, label="large"),
        Line2D([0], [0], marker="s", color="0.15", lw=0, markerfacecolor="white",
               markeredgecolor="0.15", markersize=18, label="medium"),
    ]
    leg = ax.legend(handles=legend_elems, title="effect size", loc="upper left",
                     bbox_to_anchor=(1.32, 1.0), frameon=False, fontsize=17,
                     title_fontsize=17, prop={"weight": "bold", "size": 17})
    leg.get_title().set_fontweight("bold")

    fig.tight_layout()
    base = os.path.join(outdir, f"heatmap_top{k}")
    fig.savefig(base + ".pdf", bbox_inches="tight")
    fig.savefig(base + ".png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {base}.pdf / .png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--indir", default="tool_comparison_results_fixed")
    ap.add_argument("--outdir", default="tool_comparison_results_fixed/figures")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    data = load(args.indir)

    print("Lollipop small-multiples:")
    for k in KS:
        if k in data:
            lollipop_figure(data[k], k, args.outdir)

    print("Heatmaps:")
    for k in KS:
        if k in data:
            heatmap_figure(data[k], k, args.outdir)


if __name__ == "__main__":
    main()