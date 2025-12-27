"""
Improved Visualizations for Cluster-Based Feature Analysis

Creates cleaner, more interpretable visualizations:
1. Heatmap of cluster × feature medians
2. Focused comparisons (unique successes vs. baseline)
3. Complementary failure comparisons
4. Key clusters only (remove clutter)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import pdist

# ======================================
# CONFIGURATION
# ======================================

DATA_DIR = Path(".")
OUT_DIR = DATA_DIR / "cluster_analysis"
MIN_CLUSTER_SIZE = 5

# ======================================
# VISUALIZATION FUNCTIONS
# ======================================

def create_cluster_feature_heatmap(df, cluster_col, feature_cols, out_dir):
    """
    Create a heatmap showing median feature values for each cluster.
    Rows = clusters, Columns = features, Values = median (standardized)
    """
    print("\n--- Creating Cluster Feature Heatmap ---")
    
    # Filter to clusters with sufficient size
    cluster_sizes = df[cluster_col].value_counts()
    valid_clusters = cluster_sizes[cluster_sizes >= MIN_CLUSTER_SIZE].index.tolist()
    
    if len(valid_clusters) < 2:
        print("Not enough valid clusters")
        return
    
    # Compute median for each cluster × feature
    cluster_medians = df[df[cluster_col].isin(valid_clusters)].groupby(cluster_col)[feature_cols].median()
    
    # Standardize features (z-score) for visualization
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    cluster_medians_scaled = pd.DataFrame(
        scaler.fit_transform(cluster_medians.T).T,
        index=cluster_medians.index,
        columns=cluster_medians.columns
    )
    
    # Select features with high variance across clusters
    feature_variance = cluster_medians_scaled.var(axis=0)
    top_features = feature_variance.nlargest(30).index.tolist()
    
    plot_data = cluster_medians_scaled[top_features]
    
    # Hierarchical clustering of features
    if plot_data.shape[1] > 1:
        distances = pdist(plot_data.T.fillna(0), metric='euclidean')
        linkage_matrix = linkage(distances, method='ward')
        dendro = dendrogram(linkage_matrix, no_plot=True)
        plot_data = plot_data.iloc[:, dendro['leaves']]
    
    # Sort clusters by type (all_fail, unique_*, only_*_failed, all_succeed)
    cluster_order = []
    for pattern in ['all_fail', 'unique_', 'only_', '2_of_', '3_of_', 'all_succeed']:
        matching = [c for c in plot_data.index if pattern in c]
        cluster_order.extend(sorted(matching))
    
    # Add any remaining
    remaining = [c for c in plot_data.index if c not in cluster_order]
    cluster_order.extend(sorted(remaining))
    
    plot_data = plot_data.loc[cluster_order]
    
    # Create plot
    fig_height = max(8, len(plot_data) * 0.5)
    fig_width = max(12, len(plot_data.columns) * 0.4)
    
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    
    sns.heatmap(
        plot_data,
        cmap='RdBu_r',
        center=0,
        vmin=-2,
        vmax=2,
        cbar_kws={'label': 'Standardized Median'},
        linewidths=0.5,
        linecolor='gray',
        ax=ax
    )
    
    ax.set_title('Cluster Feature Profiles (Standardized Medians)', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel('Feature', fontsize=11, fontweight='bold')
    ax.set_ylabel('Cluster', fontsize=11, fontweight='bold')
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=9)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(out_dir / "cluster_feature_heatmap.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_dir / 'cluster_feature_heatmap.png'}")


def create_focused_comparison_plots(df, cluster_col, feature_cols, out_dir):
    """
    Create focused boxplots comparing key clusters:
    1. All unique_* vs all_succeed
    2. All only_*_failed vs all_succeed
    3. all_fail vs all_succeed
    """
    print("\n--- Creating Focused Comparison Plots ---")
    
    # Get top varying features
    cluster_sizes = df[cluster_col].value_counts()
    valid_clusters = cluster_sizes[cluster_sizes >= MIN_CLUSTER_SIZE].index.tolist()
    
    if 'all_succeed' not in valid_clusters:
        print("all_succeed cluster not found")
        return
    
    # Compute feature variance across clusters
    cluster_medians = df[df[cluster_col].isin(valid_clusters)].groupby(cluster_col)[feature_cols].median()
    feature_variance = cluster_medians.var(axis=0).sort_values(ascending=False)
    top_features = feature_variance.head(15).index.tolist()
    
    # Comparison 1: Unique successes vs. baseline
    unique_clusters = [c for c in valid_clusters if 'unique_' in c]
    if unique_clusters:
        _plot_comparison(
            df, cluster_col, unique_clusters, 'all_succeed',
            top_features[:6], out_dir / "unique_vs_baseline.png",
            "Unique Tool Successes vs. All-Succeed Baseline"
        )
    
    # Comparison 2: Failures vs. baseline
    failure_clusters = [c for c in valid_clusters if 'only_' in c and '_failed' in c]
    if failure_clusters:
        _plot_comparison(
            df, cluster_col, failure_clusters, 'all_succeed',
            top_features[:6], out_dir / "failures_vs_baseline.png",
            "Tool-Specific Failures vs. All-Succeed Baseline"
        )
    
    # Comparison 3: all_fail vs all_succeed
    if 'all_fail' in valid_clusters:
        _plot_comparison(
            df, cluster_col, ['all_fail'], 'all_succeed',
            top_features[:6], out_dir / "all_fail_vs_all_succeed.png",
            "All Fail vs. All Succeed"
        )


def _plot_comparison(df, cluster_col, test_clusters, baseline_cluster, features, 
                     filename, title):
    """Helper to create comparison boxplot."""
    
    # Combine test clusters
    df_plot = df[df[cluster_col].isin(test_clusters + [baseline_cluster])].copy()
    
    # Simplify labels
    df_plot['comparison_group'] = df_plot[cluster_col].apply(
        lambda x: x if x == baseline_cluster else 'test_group'
    )
    
    # For unique successes, keep tool names
    if any('unique_' in c for c in test_clusters):
        df_plot['comparison_group'] = df_plot[cluster_col].apply(
            lambda x: x.replace('unique_', '') if 'unique_' in x else baseline_cluster
        )
    # For failures, keep tool names
    elif any('only_' in c and '_failed' in c for c in test_clusters):
        df_plot['comparison_group'] = df_plot[cluster_col].apply(
            lambda x: x.replace('only_', '').replace('_failed', '') if 'failed' in x else baseline_cluster
        )
    
    n_features = len(features)
    n_cols = 3
    n_rows = (n_features + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4*n_rows))
    axes = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes
    
    for idx, feat in enumerate(features):
        ax = axes[idx]
        
        sns.boxplot(
            data=df_plot,
            x='comparison_group',
            y=feat,
            ax=ax,
            palette='Set2'
        )
        
        ax.set_title(feat, fontsize=11, fontweight='bold')
        ax.set_xlabel('')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(axis='y', alpha=0.3)
    
    # Hide unused subplots
    for idx in range(len(features), len(axes)):
        axes[idx].set_visible(False)
    
    plt.suptitle(title, fontsize=14, fontweight='bold', y=1.0)
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {filename}")


def create_pairwise_failure_comparison(df, cluster_col, feature_cols, out_dir):
    """
    Create grid comparing only_A_failed vs only_B_failed for top features.
    Shows opposite tool weaknesses clearly.
    """
    print("\n--- Creating Pairwise Failure Comparison ---")
    
    failure_clusters = [c for c in df[cluster_col].unique() 
                       if 'only_' in c and '_failed' in c]
    
    cluster_sizes = df[cluster_col].value_counts()
    failure_clusters = [c for c in failure_clusters if cluster_sizes[c] >= MIN_CLUSTER_SIZE]
    
    if len(failure_clusters) < 2:
        print("Not enough failure clusters for comparison")
        return
    
    # Get top features that vary across failure clusters
    cluster_medians = df[df[cluster_col].isin(failure_clusters)].groupby(cluster_col)[feature_cols].median()
    feature_variance = cluster_medians.var(axis=0).sort_values(ascending=False)
    top_features = feature_variance.head(9).index.tolist()
    
    # Create grid
    n_features = len(top_features)
    n_cols = 3
    n_rows = (n_features + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4*n_rows))
    axes = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes
    
    df_plot = df[df[cluster_col].isin(failure_clusters)].copy()
    
    # Simplify labels (remove "only_" and "_failed")
    df_plot['tool'] = df_plot[cluster_col].str.replace('only_', '').str.replace('_failed', '')
    
    for idx, feat in enumerate(top_features):
        ax = axes[idx]
        
        sns.violinplot(
            data=df_plot,
            x='tool',
            y=feat,
            ax=ax,
            inner='box',
            palette='Set3'
        )
        
        ax.set_title(feat, fontsize=10, fontweight='bold')
        ax.set_xlabel('')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(axis='y', alpha=0.3)
    
    # Hide unused
    for idx in range(len(top_features), len(axes)):
        axes[idx].set_visible(False)
    
    plt.suptitle('Tool-Specific Failure Patterns: Feature Distributions', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(out_dir / "pairwise_failure_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_dir / 'pairwise_failure_comparison.png'}")


def create_effect_size_plot(results_df, out_dir, min_delta=0.3):
    """
    Create dot plot showing effect sizes for significant findings.
    X-axis = Cliff's delta, Y-axis = feature, Color = cluster
    """
    print("\n--- Creating Effect Size Plot ---")
    
    if results_df.empty:
        print("No results to plot")
        return
    
    # Filter for significant and practically significant
    sig_results = results_df[
        results_df['reject'] & 
        results_df['practically_significant'] &
        (results_df['abs_delta'] >= min_delta)
    ].copy()
    
    if sig_results.empty:
        print(f"No results with |δ| >= {min_delta}")
        return
    
    # Get top features by max effect size
    max_delta_per_feature = sig_results.groupby('feature')['abs_delta'].max()
    top_features = max_delta_per_feature.nlargest(20).index.tolist()
    
    plot_data = sig_results[sig_results['feature'].isin(top_features)]
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, max(8, len(top_features) * 0.4)))
    
    # Get unique clusters and assign colors
    clusters = plot_data['cluster'].unique()
    colors = sns.color_palette('Set2', len(clusters))
    cluster_colors = dict(zip(clusters, colors))
    
    # Plot points
    for cluster in clusters:
        cluster_data = plot_data[plot_data['cluster'] == cluster]
        ax.scatter(
            cluster_data['cliffs_delta'],
            cluster_data['feature'],
            label=cluster,
            s=100,
            alpha=0.7,
            color=cluster_colors[cluster],
            edgecolors='black',
            linewidth=0.5
        )
    
    # Add vertical line at 0
    ax.axvline(0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    
    # Add threshold lines
    ax.axvline(-min_delta, color='red', linestyle=':', linewidth=1, alpha=0.3)
    ax.axvline(min_delta, color='red', linestyle=':', linewidth=1, alpha=0.3)
    
    ax.set_xlabel("Cliff's Delta (vs. all_succeed)", fontsize=12, fontweight='bold')
    ax.set_ylabel('Feature', fontsize=12, fontweight='bold')
    ax.set_title('Effect Sizes for Cluster Differences', fontsize=14, fontweight='bold')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(out_dir / "effect_size_plot.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_dir / 'effect_size_plot.png'}")


def create_cluster_size_bar(df, cluster_col, out_dir):
    """Create clean bar chart of cluster sizes."""
    print("\n--- Creating Cluster Size Bar Chart ---")
    
    cluster_sizes = df[cluster_col].value_counts().sort_values(ascending=False)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    bars = ax.bar(range(len(cluster_sizes)), cluster_sizes.values, color='steelblue', alpha=0.7)
    
    # Highlight key clusters
    for i, cluster in enumerate(cluster_sizes.index):
        if 'all_succeed' in cluster:
            bars[i].set_color('green')
            bars[i].set_alpha(0.7)
        elif 'all_fail' in cluster:
            bars[i].set_color('red')
            bars[i].set_alpha(0.7)
        elif 'unique_' in cluster:
            bars[i].set_color('gold')
            bars[i].set_alpha(0.9)
    
    ax.set_xticks(range(len(cluster_sizes)))
    ax.set_xticklabels(cluster_sizes.index, rotation=45, ha='right', fontsize=10)
    ax.set_ylabel('Number of Bugs', fontsize=12, fontweight='bold')
    ax.set_title('Cluster Sizes: Bug Distribution by Tool Intersection Pattern', 
                 fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for i, (cluster, count) in enumerate(cluster_sizes.items()):
        ax.text(i, count + max(cluster_sizes)*0.01, str(count), 
               ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(out_dir / "cluster_sizes.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_dir / 'cluster_sizes.png'}")


# ======================================
# MAIN VISUALIZATION RUNNER
# ======================================

def create_improved_visualizations(df_clustered, cluster_col, feature_cols, out_dir):
    """Run all improved visualization functions."""
    
    print("\n" + "=" * 60)
    print("CREATING IMPROVED VISUALIZATIONS")
    print("=" * 60)
    
    # 1. Cluster sizes bar chart
    create_cluster_size_bar(df_clustered, cluster_col, out_dir)
    
    # 2. Heatmap of cluster × feature medians
    create_cluster_feature_heatmap(df_clustered, cluster_col, feature_cols, out_dir)
    
    # 3. Focused comparisons
    create_focused_comparison_plots(df_clustered, cluster_col, feature_cols, out_dir)
    
    # 4. Pairwise failure comparison
    create_pairwise_failure_comparison(df_clustered, cluster_col, feature_cols, out_dir)
    
    # 5. Effect size plot (if results exist)
    results_file = out_dir / "cluster_vs_baseline.csv"
    if results_file.exists():
        results_df = pd.read_csv(results_file)
        create_effect_size_plot(results_df, out_dir, min_delta=0.3)
    
    print("\n" + "=" * 60)
    print(f"ALL VISUALIZATIONS SAVED TO: {out_dir}")
    print("=" * 60)


if __name__ == "__main__":
    # This can be called standalone or imported
    print("Use this module by importing and calling create_improved_visualizations()")
    print("Or integrate into cluster_based_analysis.py")