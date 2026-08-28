#!/usr/bin/env python3
"""
Run the full end-to-end analysis pipeline.

Execution order:
  1) All scripts in bug_feature_extraction/ (alphabetical order)
  2) merge_features_and_performance.py
  3) unified_analysis.py
  4) predictor.py
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def _run_python_script(script_path: Path, *, dry_run: bool) -> None:
    cmd = [sys.executable, str(script_path)]
    rel = script_path.relative_to(ROOT)
    print(f"\n=== Running: {rel} ===")
    if dry_run:
        print("DRY RUN:", " ".join(cmd))
        return
    subprocess.run(cmd, cwd=str(ROOT), check=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the commands that would run, but do not execute anything.",
    )
    args = parser.parse_args()

    extraction_dir = ROOT / "bug_feature_extraction"
    # NOT alphabetical: fine_grained_gemini_catg.py reads gemini_bug_categorization_overall.py's
    # output (FUNCTIONAL_ISSUES_CSV), so categorization must run before fine-grained categorization.
    extraction_order = [
        "extract_bug_features.py",
        "gemini_bug_categorization_overall.py",
        "fine_grained_gemini_catg.py",
        "gemini_bug_ratings.py",
    ]
    extraction_scripts = [extraction_dir / name for name in extraction_order]
    missing_extraction = [p for p in extraction_scripts if not p.exists()]
    if missing_extraction:
        missing_rel = ", ".join(str(p.relative_to(ROOT)) for p in missing_extraction)
        raise FileNotFoundError(f"Missing extraction scripts: {missing_rel}")

    fixed_steps = [
        ROOT / "tool_feature_analysis" / "merge_features_and_performance.py",
        ROOT / "tool_comparison_results_fixed" / "unified_analysis.py",
        ROOT / "delta_score_outputs" / "predictor.py",
    ]

    missing = [p for p in fixed_steps if not p.exists()]
    if missing:
        missing_rel = ", ".join(str(p.relative_to(ROOT)) for p in missing)
        raise FileNotFoundError(f"Missing required scripts: {missing_rel}")

    # 1) Feature extraction scripts
    for script in extraction_scripts:
        _run_python_script(script, dry_run=args.dry_run)

    # 2-4) Remaining pipeline steps
    for script in fixed_steps:
        _run_python_script(script, dry_run=args.dry_run)

    print("\nPipeline complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

