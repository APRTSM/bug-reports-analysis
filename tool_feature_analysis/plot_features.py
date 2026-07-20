from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

KEY_COLS = ("project", "bug_id")

# Map final_feature_set column -> column name in experimentA_full_dataset.csv (pre-standardization).
RAW_FULL_COLUMN_MAP: dict[str, str] = {
    "technical_completeness": "technical_depth",
}

# Columns that stay on --csv only (not in full dataset); in final_feature_set they are still z-scored.
STILL_STANDARDIZED_IN_FINAL: frozenset[str] = frozenset(
    {
        "reasoning_composite",
        "ambiguity_type_count",
        "txt_description_line_count",
        "txt_title_digit_density",
        "txt_title_avg_sentence_len",
        "concept_network_concept_breadth",
    }
)


@dataclass(frozen=True)
class FeatureStats:
    group: str
    name: str
    mean: float
    std: float
    min: float
    max: float
    n: int | None = None

    @property
    def is_boolean(self) -> bool:
        if self.name.strip().lower().startswith("has "):
            return True
        if self.min == 0.0 and self.max == 1.0 and 0.0 <= self.mean <= 1.0:
            return True
        return False


# Same grouping / ordering as results/feature_statistics.tex; values come from CSV columns.
FEATURE_TABLE_SPEC: list[tuple[str, str, str]] = [
    ("LLM-Assessed Semantic Quality", "Actionability", "actionability"),
    ("LLM-Assessed Semantic Quality", "Clarity", "clarity"),
    ("LLM-Assessed Semantic Quality", "Expected-Observed Alignment", "expected_observed_alignment"),
    ("LLM-Assessed Semantic Quality", "Technical Completeness", "technical_completeness"),
    ("LLM-Assessed Semantic Quality", "Reasoning Composite", "reasoning_composite"),
    ("LLM-Assessed Semantic Quality", "Repair Difficulty", "repair_difficulty"),
    ("Ambiguity", "Ambiguity", "ambiguity"),
    ("Ambiguity", "Ambiguity Type Count", "ambiguity_type_count"),
    ("Structural & Surface", "Description Length (chars)", "description_length"),
    ("Structural & Surface", "Description Line Count", "txt_description_line_count"),
    ("Structural & Surface", "Title Digit Density", "txt_title_digit_density"),
    ("Structural & Surface", "Title Avg Sentence Length", "txt_title_avg_sentence_len"),
    ("Structural & Surface", "Num Versions", "num_versions"),
    ("Structural & Surface", "Num Causal Markers", "num_causal_markers"),
    ("Structural & Surface", "Num Temporal Markers", "num_temporal_markers"),
    ("Structural & Surface", "Num Env Mentions", "num_env_mentions"),
    ("Structural & Surface", "Num Exception Types", "num_exception_types"),
    ("Structural & Surface", "Exception User Frames", "exception_user_frames"),
    ("Structural & Surface", "Stacktrace Depth", "stacktrace_depth"),
    ("Structural & Surface", "Has Stacktrace", "has_stacktrace"),
    ("Structural & Surface", "Has Code", "has_code"),
    ("Structural & Surface", "Has Patch", "has_patch"),
    ("Structural & Surface", "Num Observed Behaviors", "num_OB"),
    ("Structural & Surface", "Num Expected Behaviors", "num_EB"),
    ("Readability", "Flesch Reading Ease", "flesch"),
    ("Readability", "Gunning Fog", "fog"),
    ("Readability", "Coleman-Liau Index", "coleman_liau"),
    ("Readability", "ARI", "ari"),
    ("Readability", "Kincaid Grade", "kincaid"),
    ("Readability", "SMOG", "smog"),
    ("Readability", "LIX", "lix"),
    ("Embedding & Semantic", "Embedding Cluster Distance", "embedding_cluster_distance"),
    ("Embedding & Semantic", "Embedding Cluster Size", "embedding_cluster_size"),
    ("Embedding & Semantic", "Concept Network Breadth", "concept_network_concept_breadth"),
    ("Embedding & Semantic", "Semantic Coherence", "semantic_coherence"),
    ("Embedding & Semantic", "Semantic Entropy", "semantic_entropy"),
]

_GROUP_RE = re.compile(r"\\multicolumn\{5\}\{l\}\{\\textit\{(.+?)\}\}\s*\\\\")
_ROW_RE = re.compile(
    r"^\s*(?P<name>.+?)\s*&\s*\$(?P<mean>[-\d.]+)\$\s*&\s*\$(?P<std>[-\d.]+)\$\s*&\s*\$(?P<min>[-\d.]+)\$\s*&\s*\$(?P<max>[-\d.]+)\$\s*\\\\\s*$"
)


def _clean_feature_name(raw: str) -> str:
    s = raw.strip()
    s = s.replace("$", "")
    s = re.sub(r"\\dagger|\^\\dagger|\^\{\s*\\dagger\s*\}", "", s)
    s = s.replace(r"\&", "&")
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def _coerce_numeric_series(s: pd.Series) -> pd.Series:
    if s.dtype == bool or str(s.dtype) == "boolean":
        return s.astype(float)
    if s.dtype == object:
        non_na = s.dropna()
        if len(non_na) and non_na.map(lambda x: isinstance(x, bool)).all():
            return s.astype(float)
    return pd.to_numeric(s, errors="coerce")


def _raw_src_col(full_dataset_col: str) -> str:
    return f"__raw_src__{full_dataset_col}"


def merge_raw_full_dataset(final_df: pd.DataFrame, raw_csv: Path) -> pd.DataFrame:
    """Left-join pre-standardization columns from experimentA_full_dataset.csv."""
    full = pd.read_csv(raw_csv)
    missing_keys = [k for k in KEY_COLS if k not in final_df.columns or k not in full.columns]
    if missing_keys:
        raise ValueError(f"Missing merge keys {missing_keys} in final or raw CSV.")

    needed: set[str] = set()
    for _, _, col in FEATURE_TABLE_SPEC:
        src = RAW_FULL_COLUMN_MAP.get(col, col)
        if src in full.columns:
            needed.add(src)

    cols = list(KEY_COLS) + sorted(needed)
    sub = full[cols].copy()
    for c in needed:
        sub = sub.rename(columns={c: _raw_src_col(c)})
    return final_df.merge(sub, on=list(KEY_COLS), how="left")


def series_for_plot_column(merged_df: pd.DataFrame, plot_col: str, *, use_raw: bool) -> pd.Series:
    """Prefer merged raw column from full dataset when use_raw and values exist."""
    full_src = RAW_FULL_COLUMN_MAP.get(plot_col, plot_col)
    raw_name = _raw_src_col(full_src)
    if use_raw and raw_name in merged_df.columns and merged_df[raw_name].notna().any():
        return merged_df[raw_name]
    if plot_col not in merged_df.columns:
        raise KeyError(plot_col)
    return merged_df[plot_col]


def load_feature_arrays_by_group(
    merged_df: pd.DataFrame, *, use_raw: bool
) -> dict[str, list[tuple[str, np.ndarray]]]:
    """Group name -> ordered list of (display name, 1D float array) for box plots."""
    out: dict[str, list[tuple[str, np.ndarray]]] = {}
    for group, display_name, col in FEATURE_TABLE_SPEC:
        if col not in merged_df.columns:
            continue
        try:
            s = series_for_plot_column(merged_df, col, use_raw=use_raw)
        except KeyError:
            continue
        s = _coerce_numeric_series(s).dropna()
        if len(s) == 0:
            continue
        arr = s.to_numpy(dtype=float)
        out.setdefault(group, []).append((display_name, arr))
    if not out:
        raise ValueError("No feature columns matched in dataframe.")
    return out


def _is_binary_feature(name: str, arr: np.ndarray) -> bool:
    if name.strip().lower().startswith("has "):
        return True
    u = np.unique(arr)
    if len(u) > 2:
        return False
    return bool(np.all(np.isin(u, (0.0, 1.0))))


def parse_feature_statistics_tex(path: Path) -> list[FeatureStats]:
    text = path.read_text(encoding="utf-8")
    group = "Ungrouped"
    rows: list[FeatureStats] = []

    for line in text.splitlines():
        m_group = _GROUP_RE.search(line)
        if m_group:
            group = m_group.group(1).strip()
            continue

        m_row = _ROW_RE.match(line)
        if not m_row:
            continue

        rows.append(
            FeatureStats(
                group=group,
                name=_clean_feature_name(m_row.group("name")),
                mean=float(m_row.group("mean")),
                std=float(m_row.group("std")),
                min=float(m_row.group("min")),
                max=float(m_row.group("max")),
                n=None,
            )
        )

    if not rows:
        raise ValueError(f"No feature rows parsed from {path}")
    return rows


def _slug(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    return s.strip("_") or "plot"


def plot_box_features(
    items: list[tuple[str, np.ndarray]],
    title: str,
    out_prefix: Path,
    *,
    max_features: int | None = None,
    xlabel: str = "Value",
) -> None:
    if not items:
        return
    if max_features is not None:
        items = items[:max_features]

    labels = [name for name, _ in items]
    data = [arr for _, arr in items]
    n = len(data)
    fig_h = max(3.5, 0.38 * n + 1.8)
    fig, ax = plt.subplots(figsize=(10.5, fig_h))

    bp = ax.boxplot(
        data,
        vert=False,
        patch_artist=True,
        widths=0.55,
        showfliers=True,
        flierprops=dict(marker=".", markersize=3, alpha=0.35),
    )
    for patch in bp["boxes"]:
        patch.set_facecolor("0.88")
        patch.set_edgecolor("0.35")
        patch.set_linewidth(0.8)
    for el in ("whiskers", "caps", "medians"):
        for line in bp[el]:
            line.set_color("0.25")
            line.set_linewidth(0.9)

    ax.set_yticks(np.arange(1, n + 1))
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_title(title)
    ax.grid(axis="x", color="0.9", linewidth=1)
    ax.set_xlabel(xlabel)

    fig.tight_layout()
    fig.savefig(out_prefix.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out_prefix.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_interval_features(
    features: list[FeatureStats],
    title: str,
    out_prefix: Path,
    *,
    show_mean_std: bool = True,
    max_features: int | None = None,
) -> None:
    if max_features is not None:
        features = features[:max_features]

    labels = [f.name for f in features]
    y = np.arange(len(features))

    fig_h = max(3.5, 0.33 * len(features) + 1.6)
    fig, ax = plt.subplots(figsize=(10.5, fig_h))

    for i, f in enumerate(features):
        ax.hlines(i, f.min, f.max, color="0.80", linewidth=2, zorder=1)
        if show_mean_std:
            ax.hlines(i, f.mean - f.std, f.mean + f.std, color="C0", linewidth=5, zorder=2)
        ax.plot(f.mean, i, marker="o", color="black", markersize=4, zorder=3)

    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_title(title)
    ax.grid(axis="x", color="0.9", linewidth=1)
    ax.set_xlabel("Value (min–max, mean ± std)")

    fig.tight_layout()
    fig.savefig(out_prefix.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out_prefix.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_boolean_features(
    features: list[FeatureStats],
    title: str,
    out_prefix: Path,
) -> None:
    labels = [f.name for f in features]
    means = np.array([f.mean for f in features], dtype=float)
    y = np.arange(len(features))

    fig_h = max(3.0, 0.55 * len(features) + 1.4)
    fig, ax = plt.subplots(figsize=(8.5, fig_h))
    ax.barh(y, means, color="C0", alpha=0.85)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("Proportion (mean)")
    ax.set_title(title)
    ax.grid(axis="x", color="0.9", linewidth=1)

    fig.tight_layout()
    fig.savefig(out_prefix.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out_prefix.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def _grouped(rows: Iterable[FeatureStats]) -> dict[str, list[FeatureStats]]:
    out: dict[str, list[FeatureStats]] = {}
    for r in rows:
        out.setdefault(r.group, []).append(r)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Plot feature statistics from final_feature_set.csv (or optional LaTeX table)."
    )
    ap.add_argument(
        "--csv",
        type=Path,
        default=Path("final_feature_set.csv"),
        help="CSV with one row per bug (modeling features; may include z-scored columns).",
    )
    ap.add_argument(
        "--raw-csv",
        type=Path,
        default=Path("full_feature_preproccessed_fixed/experimentA_full_dataset.csv"),
        help="Pre-standardization merge file used to recover raw LLM scores, counts, etc.",
    )
    ap.add_argument(
        "--scaled",
        action="store_true",
        help="Plot values from --csv only (z-scored / preprocessed); do not merge --raw-csv.",
    )
    ap.add_argument(
        "--tex",
        type=Path,
        default=None,
        help="If set, use precomputed stats from this LaTeX file instead of --csv.",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=Path("delta_score_outputs/plots"),
        help="Output directory for plots.",
    )
    ap.add_argument(
        "--max-features",
        type=int,
        default=0,
        help="If >0, cap number of features per plot (useful for quick iteration).",
    )
    args = ap.parse_args()

    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    max_features = args.max_features if args.max_features and args.max_features > 0 else None

    if args.tex is not None:
        rows = parse_feature_statistics_tex(args.tex)
        for group_name, feats in _grouped(rows).items():
            bool_feats = [f for f in feats if f.is_boolean]
            cont_feats = [f for f in feats if not f.is_boolean]

            if cont_feats:
                plot_interval_features(
                    cont_feats,
                    title=f"Feature statistics — {group_name}",
                    out_prefix=out_dir / f"feature_stats_{_slug(group_name)}",
                    max_features=max_features,
                )
            if bool_feats:
                plot_boolean_features(
                    bool_feats,
                    title=f"Boolean feature proportions — {group_name}",
                    out_prefix=out_dir / f"feature_stats_boolean_{_slug(group_name)}",
                )
    else:
        use_raw = not args.scaled
        final_df = pd.read_csv(args.csv)
        if use_raw:
            if not args.raw_csv.is_file():
                print(
                    f"Warning: --raw-csv not found ({args.raw_csv}); plotting --csv columns only.",
                    file=sys.stderr,
                )
                merged_df = final_df
                use_raw = False
            else:
                merged_df = merge_raw_full_dataset(final_df, args.raw_csv)
                print(
                    f"Raw mode: merged pre-standardization columns from {args.raw_csv.name} "
                    "on (project, bug_id).",
                    file=sys.stderr,
                )
                still = sorted(c for c in STILL_STANDARDIZED_IN_FINAL if c in final_df.columns)
                if still:
                    print(
                        "Note: these features are not in the raw merge file; plots still use "
                        f"standardized values from --csv: {', '.join(still)}",
                        file=sys.stderr,
                    )
        else:
            merged_df = final_df

        xlabel = (
            "Value (original units where merged from full dataset)"
            if use_raw
            else "Value (from --csv, including z-scored features)"
        )

        by_group = load_feature_arrays_by_group(merged_df, use_raw=use_raw)
        for group_name, items in by_group.items():
            cont = [(n, a) for n, a in items if not _is_binary_feature(n, a)]
            binary = [(n, a) for n, a in items if _is_binary_feature(n, a)]

            if cont:
                plot_box_features(
                    cont,
                    title=f"Feature distribution — {group_name}",
                    out_prefix=out_dir / f"feature_stats_{_slug(group_name)}",
                    max_features=max_features,
                    xlabel=xlabel,
                )
            if binary:
                plot_box_features(
                    binary,
                    title=f"Binary features (0/1) — {group_name}",
                    out_prefix=out_dir / f"feature_stats_boolean_{_slug(group_name)}",
                    max_features=max_features,
                    xlabel=xlabel,
                )


if __name__ == "__main__":
    main()
