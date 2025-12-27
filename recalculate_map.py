"""
Recalculate MAP (Mean Average Precision) from tool_comparison_summary.csv

MAP calculation for bug localization:
- For single ground truth location: MAP = MRR = 1/rank (if found)
- For multiple ground truth locations: MAP = average of precisions at each relevant item found
  - precision@i = (number of relevant items found up to rank i) / i
  - Average Precision = (1/k) * sum(precision@i) where k = number of relevant items found

Note: Without detailed rank information for all relevant items, we use approximations.
"""

import pandas as pd
import numpy as np
from pathlib import Path

# ============================
# CONFIG
# ============================

DATA_DIR = Path(".")
IN_FILE = DATA_DIR / "tool_comparison_summary.csv"
FLEXFL_FILE = DATA_DIR / "FlexFL_br_results.csv"
OUT_FILE = DATA_DIR / "tool_comparison_summary_map_recalculated.csv"
COMPARISON_FILE = DATA_DIR / "map_calculation_comparison.csv"

# ============================
# HELPER FUNCTIONS
# ============================

def calculate_map_single_gt(rank):
    """
    Calculate MAP for single ground truth location.
    MAP = MRR = 1/rank if found, else 0.
    """
    if pd.isna(rank) or rank <= 0:
        return 0.0
    return 1.0 / rank

def calculate_map_approximate(rank, top1, top5, top10, num_gt=1):
    """
    Approximate MAP calculation when we don't have exact ranks for all relevant items.
    
    This is an approximation based on:
    - rank: rank of first relevant item found
    - top1, top5, top10: whether relevant items were found in top k
    - num_gt: number of ground truth locations
    
    For single GT: MAP = 1/rank
    For multiple GT: We approximate based on available information
    """
    if pd.isna(rank) or rank <= 0:
        return 0.0
    
    # If no ground truth locations, MAP is undefined (return 0)
    if num_gt <= 0:
        return 0.0
    
    # Single ground truth: MAP = MRR
    if num_gt == 1:
        return 1.0 / rank
    
    # Multiple ground truth: approximate calculation
    # We know at least one item was found at 'rank'
    # We need to estimate how many items were found and at what ranks
    
    # If found at rank 1 and top1=1, likely all items found early
    if rank == 1 and top1 == 1:
        # Optimistic: assume all items found at rank 1
        # AP = (1/num_gt) * sum(precision@1 for each item)
        # If all k items at rank 1: AP = (1/k) * k * (k/1) = k
        # But AP should be <= 1, so we cap it
        # Actually, if k items all at rank 1: precision@1 = k/1, but we normalize
        # More realistic: if k items found at rank 1, AP = 1.0
        return 1.0
    
    # If found at rank > 1, we have less information
    # Conservative approximation: assume only one item found at 'rank'
    # AP ≈ (1/num_gt) * (1/rank)
    # This is a lower bound
    base_ap = (1.0 / num_gt) * (1.0 / rank)
    
    # If top5=1 and rank <= 5, there might be more items found
    # Adjust upward if we have evidence of multiple hits
    if top5 == 1 and rank <= 5:
        # If we found something in top5, there might be more items
        # But we don't know exactly where, so we use a conservative estimate
        # Assume at least one more item might be in top5
        if num_gt > 1:
            # Estimate: one item at 'rank', potentially another at average of (rank+5)/2
            estimated_second_rank = (rank + 5) / 2
            ap_item1 = 1.0 / rank
            ap_item2 = 2.0 / estimated_second_rank if estimated_second_rank <= 5 else 0
            base_ap = (1.0 / min(num_gt, 2)) * (ap_item1 + ap_item2)
    
    return min(base_ap, 1.0)  # Cap at 1.0

def load_ground_truth_info():
    """Load ground truth information from FlexFL results if available."""
    gt_info = {}
    if FLEXFL_FILE.exists():
        try:
            flexfl = pd.read_csv(FLEXFL_FILE)
            if 'bug_id' in flexfl.columns and 'num_gt_files' in flexfl.columns:
                for _, row in flexfl.iterrows():
                    bug_id = str(row['bug_id'])
                    gt_info[bug_id] = {
                        'num_gt_files': int(row.get('num_gt_files', 1)),
                        'num_gt_methods': int(row.get('num_gt_methods', 1))
                    }
            print(f"Loaded ground truth info for {len(gt_info)} bugs from FlexFL results")
        except Exception as e:
            print(f"Warning: Could not load FlexFL results: {e}")
    return gt_info

# ============================
# MAIN CALCULATION
# ============================

print("=" * 60)
print("MAP RECALCULATION SCRIPT")
print("=" * 60)

# Load data
df = pd.read_csv(IN_FILE)
print(f"\nLoaded {len(df)} rows from {IN_FILE}")

# Load ground truth information
gt_info = load_ground_truth_info()

# Create bug_id string for matching
df['bug_id_str'] = df['project'] + '-' + df['bug_id'].astype(str)

# Calculate MAP using different methods
print("\nCalculating MAP...")

# Method 1: Assume single GT (MAP = MRR)
df['map_recalc_single_gt'] = df['rank'].apply(calculate_map_single_gt)

# Method 2: Use ground truth info when available
df['map_recalc_with_gt'] = df.apply(
    lambda row: calculate_map_approximate(
        row['rank'],
        row['top@1'],
        row['top@5'],
        row['top@10'],
        num_gt=gt_info.get(row['bug_id_str'], {}).get('num_gt_files', 1)
    ),
    axis=1
)

# Method 3: Conservative estimate (assume single GT unless we have evidence otherwise)
df['map_recalc_conservative'] = df.apply(
    lambda row: calculate_map_approximate(
        row['rank'],
        row['top@1'],
        row['top@5'],
        row['top@10'],
        num_gt=1  # Conservative: assume single GT
    ),
    axis=1
)

# Compare with original MAP
print("\nComparing with original MAP values...")
df['map_diff_single'] = (df['map'] - df['map_recalc_single_gt']).abs()
df['map_diff_with_gt'] = (df['map'] - df['map_recalc_with_gt']).abs()
df['map_diff_conservative'] = (df['map'] - df['map_recalc_conservative']).abs()

# Statistics
print("\n" + "=" * 60)
print("COMPARISON STATISTICS")
print("=" * 60)

print("\nMethod 1: MAP = MRR (single GT assumption)")
print(f"  Exact matches: {(df['map_diff_single'] < 0.001).sum()} / {len(df)}")
print(f"  Mean absolute difference: {df['map_diff_single'].mean():.6f}")
print(f"  Max difference: {df['map_diff_single'].max():.6f}")
print(f"  Cases with difference > 0.01: {(df['map_diff_single'] > 0.01).sum()}")

print("\nMethod 2: MAP with ground truth info (when available)")
print(f"  Exact matches: {(df['map_diff_with_gt'] < 0.001).sum()} / {len(df)}")
print(f"  Mean absolute difference: {df['map_diff_with_gt'].mean():.6f}")
print(f"  Max difference: {df['map_diff_with_gt'].max():.6f}")
print(f"  Cases with difference > 0.01: {(df['map_diff_with_gt'] > 0.01).sum()}")

print("\nMethod 3: MAP conservative (single GT)")
print(f"  Exact matches: {(df['map_diff_conservative'] < 0.001).sum()} / {len(df)}")
print(f"  Mean absolute difference: {df['map_diff_conservative'].mean():.6f}")
print(f"  Max difference: {df['map_diff_conservative'].max():.6f}")
print(f"  Cases with difference > 0.01: {(df['map_diff_conservative'] > 0.01).sum()}")

# Show cases where original MAP differs significantly
print("\n" + "=" * 60)
print("CASES WHERE ORIGINAL MAP DIFFERS FROM RECALCULATED")
print("=" * 60)

significant_diff = df[df['map_diff_single'] > 0.01].copy()
if len(significant_diff) > 0:
    print(f"\nFound {len(significant_diff)} cases where original MAP differs from MRR:")
    print("\nTop 20 cases:")
    cols_to_show = ['project', 'bug_id', 'tool', 'rank', 'mrr', 'map', 
                     'map_recalc_single_gt', 'map_diff_single', 'top@1', 'top@5']
    print(significant_diff[cols_to_show].head(20).to_string())
    
    # Group by tool
    print("\nDifferences by tool:")
    print(significant_diff.groupby('tool')['map_diff_single'].agg(['count', 'mean', 'max']))
else:
    print("\nAll MAP values match MRR (single GT assumption)")

# Save comparison
comparison_cols = ['project', 'bug_id', 'tool', 'rank', 'mrr', 'map',
                   'map_recalc_single_gt', 'map_recalc_with_gt', 'map_recalc_conservative',
                   'map_diff_single', 'map_diff_with_gt', 'map_diff_conservative',
                   'top@1', 'top@5', 'top@10']
comparison_df = df[comparison_cols].copy()
comparison_df.to_csv(COMPARISON_FILE, index=False)
print(f"\nSaved comparison to: {COMPARISON_FILE}")

# Option to update the original file
print("\n" + "=" * 60)
print("RECOMMENDATION")
print("=" * 60)
print("\nBased on the analysis:")
if (df['map_diff_single'] < 0.001).sum() / len(df) > 0.95:
    print("✓ Most MAP values match MRR (single GT assumption)")
    print("  The original MAP column appears correct for single GT cases.")
    print("  Differences may be due to multiple GT locations.")
else:
    print("⚠ Significant differences found between original MAP and recalculated values.")
    print("  Review the comparison file to understand the discrepancies.")

print("\nTo update tool_comparison_summary.csv with recalculated MAP:")
print("  Option 1: Use 'map_recalc_single_gt' (MAP = MRR for all cases)")
print("  Option 2: Use 'map_recalc_with_gt' (uses GT info when available)")
print("  Option 3: Keep original 'map' if it was calculated with more detailed information")

print("\nDone!")

