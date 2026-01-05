"""
Create heatmaps for tool comparison analysis results.

Generates heatmaps for:
1. ALL vs NONE comparisons
2. Pairwise tool comparisons
3. Tool vs REST comparisons
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Optional

# Configuration
INPUT_DIR = Path("tool_comparison_results_fixed")
OUTPUT_DIR = INPUT_DIR
THRESHOLDS = [1, 5, 10]

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300

def create_all_vs_none_heatmap(df: pd.DataFrame, threshold: int, top_n: int = 30):
    """Create heatmap for ALL vs NONE comparison."""
    if df is None or len(df) == 0:
        return
    
    # Filter for practically significant features
    df_sig = df[df['practically_significant'] == True].copy()
    if len(df_sig) == 0:
        # Fall back to just significant features
        df_sig = df[df['significant'] == True].copy()
    
    if len(df_sig) == 0:
        print(f"  No significant features for all_vs_none_top{threshold}")
        return
    
    # Sort by absolute delta and take top N
    df_sig = df_sig.sort_values('abs_delta', ascending=False).head(top_n)
    
    # Create matrix: features x metrics
    features = df_sig['feature'].values
    metrics = {
        'Cliff\'s Delta': df_sig['cliffs_delta'].values,
        'Median Diff': df_sig['median_diff'].values,
        'ALL Median': df_sig['all_median'].values,
        'NONE Median': df_sig['none_median'].values,
    }
    
    # Create figure with subplots
    fig, axes = plt.subplots(1, 2, figsize=(14, max(8, len(features) * 0.3)))
    
    # Main heatmap: Cliff's Delta
    matrix_delta = df_sig['cliffs_delta'].values.reshape(-1, 1)
    sns.heatmap(
        matrix_delta,
        xticklabels=['Cliff\'s Delta'],
        yticklabels=features,
        cmap='RdBu_r',
        center=0,
        vmin=-1, vmax=1,
        annot=True,
        fmt='.3f',
        cbar_kws={'label': "Cliff's Delta"},
        ax=axes[0]
    )
    axes[0].set_title(f'ALL vs NONE (Top-{threshold})\nCliff\'s Delta Effect Size', fontweight='bold')
    axes[0].set_ylabel('Feature', fontweight='bold')
    
    # Secondary heatmap: Median values comparison
    matrix_medians = np.column_stack([
        df_sig['all_median'].values,
        df_sig['none_median'].values
    ])
    sns.heatmap(
        matrix_medians,
        xticklabels=['ALL Median', 'NONE Median'],
        yticklabels=features,
        cmap='YlOrRd',
        annot=True,
        fmt='.2f',
        cbar_kws={'label': 'Median Value'},
        ax=axes[1]
    )
    axes[1].set_title(f'Median Values Comparison', fontweight='bold')
    axes[1].set_ylabel('')
    
    plt.tight_layout()
    out_file = OUTPUT_DIR / f"all_vs_none_heatmap_top{threshold}.png"
    plt.savefig(out_file, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {out_file}")


def create_pairwise_heatmap(df: pd.DataFrame, threshold: int, top_n: int = 20):
    """Create heatmap for pairwise tool comparisons."""
    if df is None or len(df) == 0:
        return
    
    # Filter for practically significant features first, then significant, then just top by delta
    df_sig = df[df['practically_significant'] == True].copy()
    if len(df_sig) == 0:
        df_sig = df[df['significant'] == True].copy()
    
    # If still no significant features, use top features by absolute delta
    if len(df_sig) == 0:
        print(f"  No statistically significant features, using top features by effect size...")
        df_sig = df.copy()
        df_sig = df_sig.sort_values('abs_delta', ascending=False).head(top_n * 3)  # Get more candidates
    
    if len(df_sig) == 0:
        print(f"  No features available for pairwise_top{threshold}")
        return
    
    # Get top features per tool pair
    top_features_per_pair = {}
    for (tool_a, tool_b), group in df_sig.groupby(['tool_a', 'tool_b']):
        group_sorted = group.sort_values('abs_delta', ascending=False)
        top_features_per_pair[(tool_a, tool_b)] = group_sorted.head(top_n)['feature'].tolist()
    
    # Get all unique top features
    all_top_features = set()
    for features in top_features_per_pair.values():
        all_top_features.update(features)
    
    if not all_top_features:
        print(f"  No top features found for pairwise_top{threshold}")
        return
    
    # Create matrix: features x tool pairs
    pairs = sorted(top_features_per_pair.keys())
    features = sorted(all_top_features)
    
    matrix = np.zeros((len(features), len(pairs)))
    for j, (tool_a, tool_b) in enumerate(pairs):
        for i, feat in enumerate(features):
            subset = df_sig[
                (df_sig['tool_a'] == tool_a) &
                (df_sig['tool_b'] == tool_b) &
                (df_sig['feature'] == feat)
            ]
            if len(subset) > 0:
                matrix[i, j] = subset.iloc[0]['cliffs_delta']
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(max(10, len(pairs) * 2), max(10, len(features) * 0.3)))
    
    pair_labels = [f"{a}\nvs\n{b}" for a, b in pairs]
    
    sns.heatmap(
        matrix,
        xticklabels=pair_labels,
        yticklabels=features,
        cmap='RdBu_r',
        center=0,
        vmin=-1, vmax=1,
        annot=True,
        fmt='.3f',
        cbar_kws={'label': "Cliff's Delta"},
        ax=ax,
        linewidths=0.5
    )
    
    ax.set_title(f'Pairwise Tool Comparison (Top-{threshold})\nDiscriminative Features', 
                 fontweight='bold', fontsize=14)
    ax.set_xlabel('Tool Comparison', fontweight='bold')
    ax.set_ylabel('Feature', fontweight='bold')
    
    plt.tight_layout()
    out_file = OUTPUT_DIR / f"pairwise_heatmap_top{threshold}.png"
    plt.savefig(out_file, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {out_file}")


def create_tool_vs_rest_heatmap(df: pd.DataFrame, threshold: int, top_n: int = 20):
    """Create heatmap for tool vs rest comparisons."""
    if df is None or len(df) == 0:
        return
    
    # Filter for practically significant features first, then significant, then top by delta
    df_sig = df[df['practically_significant'] == True].copy()
    if len(df_sig) == 0:
        df_sig = df[df['significant'] == True].copy()
    
    # If still no significant features, use top features by absolute delta
    if len(df_sig) == 0:
        print(f"  No statistically significant features, using top features by effect size...")
        df_sig = df.copy()
        df_sig = df_sig.sort_values('abs_delta', ascending=False).head(top_n * 3)  # Get more candidates
    
    if len(df_sig) == 0:
        print(f"  No features available for tool_vs_rest_top{threshold}")
        return
    
    # Get top features per tool
    top_features_per_tool = {}
    for tool, group in df_sig.groupby('tool'):
        group_sorted = group.sort_values('abs_delta', ascending=False)
        top_features_per_tool[tool] = group_sorted.head(top_n)['feature'].tolist()
    
    # Get all unique top features
    all_top_features = set()
    for features in top_features_per_tool.values():
        all_top_features.update(features)
    
    if not all_top_features:
        print(f"  No top features found for tool_vs_rest_top{threshold}")
        return
    
    # Create matrix: features x tools
    tools = sorted(top_features_per_tool.keys())
    features = sorted(all_top_features)
    
    matrix = np.zeros((len(features), len(tools)))
    for j, tool in enumerate(tools):
        for i, feat in enumerate(features):
            subset = df_sig[
                (df_sig['tool'] == tool) &
                (df_sig['feature'] == feat)
            ]
            if len(subset) > 0:
                matrix[i, j] = subset.iloc[0]['cliffs_delta']
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(max(8, len(tools) * 1.5), max(10, len(features) * 0.3)))
    
    sns.heatmap(
        matrix,
        xticklabels=tools,
        yticklabels=features,
        cmap='RdBu_r',
        center=0,
        vmin=-1, vmax=1,
        annot=True,
        fmt='.3f',
        cbar_kws={'label': "Cliff's Delta"},
        ax=ax,
        linewidths=0.5
    )
    
    ax.set_title(f'Tool vs REST Comparison (Top-{threshold})\nUnique Strengths per Tool', 
                 fontweight='bold', fontsize=14)
    ax.set_xlabel('Tool', fontweight='bold')
    ax.set_ylabel('Feature', fontweight='bold')
    
    plt.tight_layout()
    out_file = OUTPUT_DIR / f"tool_vs_rest_heatmap_top{threshold}.png"
    plt.savefig(out_file, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {out_file}")


def main():
    print("=" * 80)
    print("CREATING HEATMAPS FOR TOOL COMPARISON RESULTS")
    print("=" * 80)
    print(f"Input directory: {INPUT_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")
    print("=" * 80)
    
    for threshold in THRESHOLDS:
        print(f"\n{'#'*80}")
        print(f"# THRESHOLD: Top-{threshold}")
        print(f"{'#'*80}")
        
        # 1. ALL vs NONE
        print(f"\n1. Creating ALL vs NONE heatmap...")
        all_vs_none_file = INPUT_DIR / f"all_vs_none_top{threshold}.csv"
        if all_vs_none_file.exists():
            df_all_none = pd.read_csv(all_vs_none_file)
            create_all_vs_none_heatmap(df_all_none, threshold, top_n=30)
        else:
            print(f"  File not found: {all_vs_none_file}")
        
        # 2. Pairwise comparisons
        print(f"\n2. Creating pairwise comparison heatmap...")
        pairwise_file = INPUT_DIR / f"pairwise_tool_comparison_top{threshold}.csv"
        if pairwise_file.exists():
            df_pairwise = pd.read_csv(pairwise_file)
            create_pairwise_heatmap(df_pairwise, threshold, top_n=20)
        else:
            print(f"  File not found: {pairwise_file}")
        
        # 3. Tool vs REST
        print(f"\n3. Creating tool vs REST heatmap...")
        tool_vs_rest_file = INPUT_DIR / f"tool_vs_rest_top{threshold}.csv"
        if tool_vs_rest_file.exists():
            df_tool_rest = pd.read_csv(tool_vs_rest_file)
            create_tool_vs_rest_heatmap(df_tool_rest, threshold, top_n=20)
        else:
            print(f"  File not found: {tool_vs_rest_file}")
    
    print("\n" + "=" * 80)
    print("HEATMAP GENERATION COMPLETE")
    print("=" * 80)
    print(f"\nGenerated heatmaps:")
    for threshold in THRESHOLDS:
        print(f"  Top-{threshold}:")
        print(f"    - all_vs_none_heatmap_top{threshold}.png")
        print(f"    - pairwise_heatmap_top{threshold}.png")
        print(f"    - tool_vs_rest_heatmap_top{threshold}.png")
    print("=" * 80)


if __name__ == "__main__":
    main()

