#!/usr/bin/env python3
"""
Calculate MRR@k and MAP@k metrics from tool results, for k in {1, 5, 10}.

MRR@k: Mean Reciprocal Rank considering only the top k results
       - If best rank <= k: MRR@k = 1/best_rank
       - Otherwise: MRR@k = 0

MAP@k: Mean Average Precision considering only the top k results
       - Only ground truth files within top k ranks are considered
       - Calculate average precision using only those files
"""

import sys
import pandas as pd
from pathlib import Path
from collections import defaultdict
import csv

# Import functions from compare_tools.py
sys.path.insert(0, str(Path(__file__).parent))
from compare_tools import (
    load_ground_truth,
    load_tool_results,
)

K_VALUES = [1, 5, 10]


def _available_bugs_from_issue_reports_html(issue_reports_html_dir: str | Path):
    """
    Return a set of (Project, bug_id_str) pairs that have an HTML issue report.

    Expected filenames: defects4j-<Project>-<BugId>.html (e.g., defects4j-Chart-1.html)
    """
    import re

    d = Path(issue_reports_html_dir)
    if not d.exists():
        raise FileNotFoundError(f"issue_reports_html directory not found: {d}")

    pat = re.compile(r"^defects4j-(?P<project>[^-]+)-(?P<bug>\d+)\.html$")
    available: set[tuple[str, str]] = set()

    for f in d.glob("defects4j-*.html"):
        m = pat.match(f.name)
        if not m:
            continue
        available.add((m.group("project"), m.group("bug")))

    return available


def _filter_df_to_available_issue_reports(df: pd.DataFrame, issue_reports_html_dir: str | Path):
    """Filter a tool summary df to only bugs that exist in issue_reports_html."""
    available = _available_bugs_from_issue_reports_html(issue_reports_html_dir)

    # Normalize bug_id to string integer (CSV sometimes stores floats like 5.0)
    def _bug_str(v):
        if pd.isna(v):
            return ""
        s = str(v).strip()
        if s.endswith(".0"):
            s = s[:-2]
        return s

    bug_ids = df["bug_id"].apply(_bug_str)
    keep_mask = [
        (proj, bug) in available
        for proj, bug in zip(df["project"].astype(str), bug_ids)
    ]

    filtered = df.loc[keep_mask].copy()
    filtered["bug_id"] = bug_ids.loc[filtered.index]
    return filtered


def calculate_mrr_at_k(best_rank, k):
    """
    Calculate MRR@k for a single bug.

    Args:
        best_rank: The best rank where a ground truth file was found (or None)
        k: Cutoff rank

    Returns:
        MRR@k value (0.0 if best_rank > k or None)
    """
    if best_rank is None:
        return 0.0
    if best_rank <= k:
        return 1.0 / best_rank
    return 0.0


def calculate_map_at_k(predictions, ground_truth_files, k):
    """
    Calculate MAP@k for a single bug.

    Args:
        predictions: List of (file_path, rank) tuples from tool results
        ground_truth_files: List of ground truth file paths
        k: Cutoff rank

    Returns:
        MAP@k value (0.0 if no ground truth files in top k)
    """
    if not predictions or not ground_truth_files:
        return 0.0

    # Import normalize function
    from compare_tools import normalize_file_path

    # Normalize ground truth files
    gt_normalized = {normalize_file_path(gt) for gt in ground_truth_files}

    # Find all ranks of ground truth files, but only consider those <= k
    gt_ranks = []
    for pred_file, rank in predictions:
        if rank > k:  # Skip ranks beyond k
            continue
        pred_normalized = normalize_file_path(pred_file)
        # Check if this prediction matches any ground truth file
        for gt in gt_normalized:
            if pred_normalized == gt or pred_normalized.endswith(gt) or gt.endswith(pred_normalized):
                gt_ranks.append(rank)
                break

    if not gt_ranks:
        return 0.0

    # Sort ranks
    gt_ranks.sort()

    # Calculate average precision: sum((i+1)/rank_i) / num_gt_files
    # But only for files within top k
    precision_sum = 0.0
    for i, rank in enumerate(gt_ranks):
        precision_sum += (i + 1) / rank

    # Normalize by total number of ground truth files (not just those in top k)
    # This is the standard MAP@k definition
    return precision_sum / len(ground_truth_files)


def _print_metric_table(title, tool_rows, metric_names, count_by_row):
    """
    Print a table of {metric_name: value} dicts keyed by row label (tool or project).

    Args:
        title: Table title printed above the header
        tool_rows: dict of row_label -> {metric_name: value}
        metric_names: ordered list of metric column names (e.g. ['mrr@1', ..., 'map@10'])
        count_by_row: dict of row_label -> sample count
    """
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)
    label_width = 20
    header = f"\n{'Tool':<{label_width}}" + "".join(f"{name.upper():<12}" for name in metric_names) + f"{'Count':<10}"
    print(header)
    print("-" * 80)
    for label in sorted(tool_rows.keys()):
        row = tool_rows[label]
        row_str = f"{label:<{label_width}}"
        for name in metric_names:
            row_str += f"{row[name]:<12.6f}"
        row_str += f"{count_by_row[label]:<10}"
        print(row_str)


def calculate_from_csv(csv_path, issue_reports_html_dir=None):
    """
    Calculate MRR@k (k in K_VALUES) from existing CSV (quick method).
    MAP@k requires recalculating from raw results.
    """
    print(f"📊 Loading results from {csv_path}...")
    df = pd.read_csv(csv_path)

    if issue_reports_html_dir:
        before = len(df)
        df = _filter_df_to_available_issue_reports(df, issue_reports_html_dir)
        after = len(df)
        print(f"  Issue report filter: {after}/{before} rows kept")

    def calc_mrr_row(row, k):
        if pd.isna(row['rank']) or row['rank'] == '':
            return 0.0
        try:
            rank = float(row['rank'])
        except (ValueError, TypeError):
            return 0.0
        return 1.0 / rank if rank <= k else 0.0

    metric_names = [f'mrr@{k}' for k in K_VALUES]
    for k in K_VALUES:
        df[f'mrr@{k}'] = df.apply(lambda row: calc_mrr_row(row, k), axis=1)

    tool_rows = {}
    count_by_row = {}
    for tool, group in df.groupby('tool'):
        tool_rows[tool] = {name: group[name].mean() for name in metric_names}
        count_by_row[tool] = len(group)

    _print_metric_table("MRR@k RESULTS (calculated from CSV)", tool_rows, metric_names, count_by_row)

    print("-" * 80)
    overall_row = f"{'Overall':<20}"
    for name in metric_names:
        overall_row += f"{df[name].mean():<12.6f}"
    overall_row += f"{len(df):<10}"
    print(overall_row)

    return df


def calculate_metrics_from_raw_results(csv_path, projects=None, tools=None, issue_reports_html_dir=None):
    """
    Calculate MRR@k and MAP@k (k in K_VALUES) by loading raw tool results.
    This is more accurate but slower than --quick.
    """
    print(f"\n📊 Calculating MRR@k and MAP@k from raw results...")
    print("=" * 80)

    # Load CSV to get list of project/bug/tool combinations
    df = pd.read_csv(csv_path)
    print(f"  Loaded {len(df)} rows from CSV")

    if issue_reports_html_dir:
        before = len(df)
        df = _filter_df_to_available_issue_reports(df, issue_reports_html_dir)
        after = len(df)
        print(f"  Issue report filter: {after}/{before} rows")

    if projects:
        df = df[df['project'].isin(projects)]
        print(f"  After project filter: {len(df)} rows")
    if tools:
        df = df[df['tool'].isin(tools)]
        print(f"  After tool filter: {len(df)} rows")

    mrr_names = [f'mrr@{k}' for k in K_VALUES]
    map_names = [f'map@{k}' for k in K_VALUES]
    metric_names = mrr_names + map_names

    def _empty_metrics():
        return {name: [] for name in metric_names}

    results = defaultdict(_empty_metrics)

    grouped = df.groupby(['project', 'bug_id', 'tool'])
    total = len(grouped)
    print(f"  Processing {total} unique project/bug/tool combinations...")

    processed = 0
    errors = 0
    for (project, bug_id, tool), group in grouped:
        processed += 1
        if processed % 50 == 0:
            print(f"  Processed {processed}/{total}...")

        try:
            # Load ground truth
            ground_truth = load_ground_truth(project)
            # Convert bug_id to string for comparison
            bug_id_str = str(bug_id)
            if bug_id_str not in ground_truth:
                continue

            gt_files = ground_truth[bug_id_str]

            # Load tool results
            predictions = load_tool_results(project, tool, bug_id_str)

            if not predictions:
                # No predictions - all metrics are 0
                for name in metric_names:
                    results[tool][name].append(0.0)
                continue
        except Exception as e:
            errors += 1
            if errors <= 5:  # Print first few errors
                print(f"    Error processing {project}-{bug_id} {tool}: {e}")
            continue

        # Find best rank (need to normalize file paths for comparison)
        from compare_tools import normalize_file_path

        best_rank = None
        gt_normalized = {normalize_file_path(gt) for gt in gt_files}

        for pred_file, rank in predictions:
            pred_normalized = normalize_file_path(pred_file)
            # Check if this prediction matches any ground truth file
            for gt in gt_normalized:
                if pred_normalized == gt or pred_normalized.endswith(gt) or gt.endswith(pred_normalized):
                    if best_rank is None or rank < best_rank:
                        best_rank = rank
                    break

        for k in K_VALUES:
            results[tool][f'mrr@{k}'].append(calculate_mrr_at_k(best_rank, k))
            results[tool][f'map@{k}'].append(calculate_map_at_k(predictions, gt_files, k))

    # Print results
    tool_rows = {}
    count_by_row = {}
    for tool, values in results.items():
        count = len(values[mrr_names[0]])
        if not count:
            continue
        tool_rows[tool] = {name: sum(values[name]) / count for name in metric_names}
        count_by_row[tool] = count

    _print_metric_table("MRR@k AND MAP@k RESULTS (calculated from raw results)", tool_rows, metric_names, count_by_row)

    # Overall averages
    combined = _empty_metrics()
    for tool in results.keys():
        for name in metric_names:
            combined[name].extend(results[tool][name])

    if combined[mrr_names[0]]:
        overall_count = len(combined[mrr_names[0]])
        print("-" * 80)
        overall_row = f"{'Overall':<20}"
        for name in metric_names:
            overall_row += f"{(sum(combined[name]) / overall_count):<12.6f}"
        overall_row += f"{overall_count:<10}"
        print(overall_row)

    return results


def update_csv_with_at_k_metrics(csv_path, output_path=None, issue_reports_html_dir=None):
    """
    Add MRR@k and MAP@k columns (k in K_VALUES) to the CSV file.
    """
    print(f"\n📝 Updating CSV with MRR@k and MAP@k metrics...")

    if output_path is None:
        output_path = csv_path.replace('.csv', '_with_at_k.csv')

    # Load CSV
    df = pd.read_csv(csv_path)

    if issue_reports_html_dir:
        before = len(df)
        df = _filter_df_to_available_issue_reports(df, issue_reports_html_dir)
        after = len(df)
        print(f"  Issue report filter: {after}/{before} rows kept")

    def calc_mrr_row(row, k):
        if pd.isna(row['rank']) or row['rank'] == '':
            return 0.0
        try:
            rank = float(row['rank'])
            return 1.0 / rank if rank <= k else 0.0
        except (ValueError, TypeError):
            return 0.0

    for k in K_VALUES:
        df[f'mrr@{k}'] = df.apply(lambda row: calc_mrr_row(row, k), axis=1)

    # For MAP@k, we need to recalculate from raw results
    # This is slower but more accurate
    print("  Calculating MAP@k from raw results (this may take a while)...")

    map_values = {k: [] for k in K_VALUES}
    processed = 0
    total = len(df)

    for idx, row in df.iterrows():
        processed += 1
        if processed % 100 == 0:
            print(f"  Processed {processed}/{total}...")

        project = row['project']
        bug_id = str(row['bug_id'])
        tool = row['tool']

        # Load ground truth
        ground_truth = load_ground_truth(project)
        if bug_id not in ground_truth:
            for k in K_VALUES:
                map_values[k].append(0.0)
            continue

        gt_files = ground_truth[bug_id]

        # Load tool results
        predictions = load_tool_results(project, tool, bug_id)

        if not predictions:
            for k in K_VALUES:
                map_values[k].append(0.0)
            continue

        for k in K_VALUES:
            map_values[k].append(calculate_map_at_k(predictions, gt_files, k))

    for k in K_VALUES:
        df[f'map@{k}'] = map_values[k]

    # Save updated CSV
    df.to_csv(output_path, index=False)
    print(f"\n✅ Updated CSV saved to: {output_path}")

    return df


def calculate_boostnsift_from_defects4j_dir(boostnsift_dir, projects=None, issue_reports_html_dir=None):
    """
    Calculate MRR@k and MAP@k (k in K_VALUES) directly from
    boostnsift_Defects4J/*_method-susps.csv. This bypasses results/ and
    tool_comparison_summary.csv completely.
    """
    from compare_tools import load_boostnsift_method_susps, normalize_file_path
    from collections import defaultdict
    import re

    boostnsift_dir = Path(boostnsift_dir)
    if not boostnsift_dir.exists():
        raise FileNotFoundError(f"BoostNSift directory not found: {boostnsift_dir}")

    # Expected filename pattern: Project-Bug_method-susps.csv (e.g., Chart-1_method-susps.csv)
    pattern = re.compile(r"^(?P<project>[^-]+)-(?P<bug_id>\d+)_method-susps\.csv$")

    files = sorted(boostnsift_dir.glob("*_method-susps.csv"))
    if not files:
        print(f"❌ No '*_method-susps.csv' files found in {boostnsift_dir}")
        return {}

    mrr_names = [f'mrr@{k}' for k in K_VALUES]
    map_names = [f'map@{k}' for k in K_VALUES]
    metric_names = mrr_names + map_names

    def _empty_metrics():
        return {name: [] for name in metric_names}

    per_tool = defaultdict(_empty_metrics)
    per_project = defaultdict(lambda: defaultdict(_empty_metrics))

    processed = 0
    skipped = 0

    available_issue_reports = None
    if issue_reports_html_dir:
        available_issue_reports = _available_bugs_from_issue_reports_html(issue_reports_html_dir)

    for f in files:
        m = pattern.match(f.name)
        if not m:
            skipped += 1
            continue

        project = m.group("project")
        bug_id = m.group("bug_id")

        if projects and project not in projects:
            continue

        if available_issue_reports is not None and (project, bug_id) not in available_issue_reports:
            skipped += 1
            continue

        ground_truth = load_ground_truth(project)
        if bug_id not in ground_truth:
            skipped += 1
            continue

        gt_files = ground_truth[bug_id]
        predictions = load_boostnsift_method_susps(f)

        # Find best rank
        best_rank = None
        gt_normalized = {normalize_file_path(gt) for gt in gt_files}
        for pred_file, rank in predictions:
            pred_normalized = normalize_file_path(pred_file)
            for gt in gt_normalized:
                if pred_normalized == gt or pred_normalized.endswith(gt) or gt.endswith(pred_normalized):
                    if best_rank is None or rank < best_rank:
                        best_rank = rank
                    break

        for k in K_VALUES:
            mrr_k = calculate_mrr_at_k(best_rank, k)
            map_k = calculate_map_at_k(predictions, gt_files, k)
            per_tool["boostnsift"][f'mrr@{k}'].append(mrr_k)
            per_tool["boostnsift"][f'map@{k}'].append(map_k)
            per_project[project]["boostnsift"][f'mrr@{k}'].append(mrr_k)
            per_project[project]["boostnsift"][f'map@{k}'].append(map_k)
        processed += 1

    print("\n" + "=" * 80)
    print("BOOSTNSIFT MRR@k AND MAP@k (from boostnsift_Defects4J)")
    print("=" * 80)
    print(f"  Directory: {boostnsift_dir}")
    if projects:
        print(f"  Project filter: {', '.join(projects)}")
    print(f"  Files processed: {processed}")
    if skipped:
        print(f"  Files skipped: {skipped}")

    count = len(per_tool["boostnsift"][mrr_names[0]])
    if count:
        print("\n" + "-" * 80)
        header = f"{'Tool':<20}" + "".join(f"{name.upper():<12}" for name in metric_names) + f"{'Count':<10}"
        print(header)
        print("-" * 80)
        row_str = f"{'boostnsift':<20}"
        for name in metric_names:
            row_str += f"{(sum(per_tool['boostnsift'][name]) / count):<12.6f}"
        row_str += f"{count:<10}"
        print(row_str)
    else:
        print("\n❌ No BoostNSift rows were evaluated (after filtering).")
        return {}

    # Per-project
    print("\n" + "-" * 80)
    print("Per-project:")
    header = f"{'Project':<15}" + "".join(f"{name.upper():<12}" for name in metric_names) + f"{'Count':<10}"
    print(header)
    print("-" * 80)
    for project in sorted(per_project.keys()):
        proj_metrics = per_project[project]["boostnsift"]
        n = len(proj_metrics[mrr_names[0]])
        if not n:
            continue
        row_str = f"{project:<15}"
        for name in metric_names:
            row_str += f"{(sum(proj_metrics[name]) / n):<12.6f}"
        row_str += f"{n:<10}"
        print(row_str)

    return {"boostnsift": {"overall": per_tool["boostnsift"], "per_project": per_project}}


def main():
    """Main function."""
    import argparse

    parser = argparse.ArgumentParser(
        description='Calculate MRR@k and MAP@k metrics (k=1,5,10) from tool results'
    )
    parser.add_argument(
        'csv_path',
        type=str,
        help='Path to tool_comparison_summary.csv'
    )
    parser.add_argument(
        '--quick',
        action='store_true',
        help='Only calculate MRR@k from CSV (faster, but no MAP@k)'
    )
    parser.add_argument(
        '--update-csv',
        action='store_true',
        help='Add MRR@k and MAP@k columns to CSV file'
    )
    parser.add_argument(
        '--output',
        type=str,
        help='Output CSV path (only used with --update-csv)'
    )
    parser.add_argument(
        '--projects',
        nargs='+',
        help='Filter by specific projects (e.g., Chart Math)'
    )
    parser.add_argument(
        '--tools',
        nargs='+',
        help='Filter by specific tools (e.g., buglocator locus)'
    )
    parser.add_argument(
        '--boostnsift-defects4j',
        type=str,
        default=None,
        help='Evaluate BoostNSift directly from boostnsift_Defects4J directory (expects *_method-susps.csv). '
             'If set, csv_path is ignored.'
    )
    parser.add_argument(
        '--issue-reports-html-dir',
        type=str,
        default=None,
        help='If set, only evaluate (project,bug_id) pairs that exist as '
             'issue_reports_html/defects4j-<Project>-<BugId>.html'
    )

    args = parser.parse_args()

    if args.boostnsift_defects4j:
        calculate_boostnsift_from_defects4j_dir(
            args.boostnsift_defects4j,
            args.projects,
            issue_reports_html_dir=args.issue_reports_html_dir,
        )
        return

    csv_path = Path(args.csv_path)
    if not csv_path.exists():
        print(f"❌ Error: CSV file not found: {csv_path}")
        sys.exit(1)

    if args.quick:
        # Quick calculation: only MRR@k from CSV
        calculate_from_csv(csv_path, issue_reports_html_dir=args.issue_reports_html_dir)
    elif args.update_csv:
        # Update CSV with both metrics
        update_csv_with_at_k_metrics(csv_path, args.output, issue_reports_html_dir=args.issue_reports_html_dir)
    else:
        # Full calculation from raw results
        calculate_metrics_from_raw_results(
            csv_path,
            args.projects,
            args.tools,
            issue_reports_html_dir=args.issue_reports_html_dir,
        )


if __name__ == "__main__":
    main()
