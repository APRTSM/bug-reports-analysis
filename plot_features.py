from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


@dataclass(frozen=True)
class FeatureStats:
    group: str
    name: str
    mean: float
    std: float
    min: float
    max: float

    @property
    def is_boolean(self) -> bool:
        if self.min == 0.0 and self.max == 1.0 and 0.0 <= self.mean <= 1.0:
            return True
        return self.name.strip().lower().startswith("has ")


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
            )
        )

    if not rows:
        raise ValueError(f"No feature rows parsed from {path}")
    return rows


def _slug(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    return s.strip("_") or "plot"


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
    ap = argparse.ArgumentParser(description="Plot feature statistics from LaTeX table.")
    ap.add_argument(
        "--tex",
        type=Path,
        default=Path("results/feature_statistics.tex"),
        help="Path to LaTeX table with feature stats.",
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

    rows = parse_feature_statistics_tex(args.tex)
    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    max_features = args.max_features if args.max_features and args.max_features > 0 else None

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


if __name__ == "__main__":
    main()