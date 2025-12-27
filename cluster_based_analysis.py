"""
Cluster-Based Feature Analysis: Tool Intersection Patterns

This analysis creates clusters based on which tools found which bugs,
then compares feature characteristics across clusters to identify:
1. Tool-specific capabilities (unique successes)
2. Tool-specific weaknesses (unique failures)
3. Universal difficulty (all fail) vs. ease (all succeed)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mannwhitneyu
from pathlib import Path

# ======================================
# CONFIGURATION
# ======================================

DATA_DIR = Path(".")
OUT_DIR = DATA_DIR / "cluster_analysis"
OUT_DIR.mkdir(exist_ok=True, parents=True)

# Input files
IN_FILE_FEATURES = DATA_DIR / "full_feature_preproccessed/experimentA_full_dataset.csv"
IN_FILE_TOOL_COMPARISON = DATA_DIR / "tool_comparison_summary.csv"

# Settings
FOUND_THRESHOLD_MRR = 0.1  # MRR > 0.1 = "found"
MIN_CLUSTER_SIZE = 5  # Minimum bugs per cluster for statistical tests
ALPHA = 0.05

# ======================================
# HELPER FUNCTIONS
# ======================================

def cliffs_delta(x, y):
    """Compute Cliff's delta effect size."""
    nx, ny = len(x), len(y)
    if nx == 0 or ny == 0:
        return np.nan
    
    mat = np.subtract.outer(x, y)
    more = (mat > 0).sum()
    less = (mat < 0).sum()
    delta = (more - less) / (nx * ny)
    return delta


def create_cluster_labels(df, tools, use_rank_presence=True):
    """
    Create cluster labels based on which tools found each bug.
    
    Args:
        use_rank_presence: If True, "found" = rank exists (matches UpSet)
                          If False, "found" = MRR > 0.1 (stricter)
    
    Returns:
        df with added 'cluster' column and 'cluster_pattern' column
    """
    df = df.copy()
    
    # Create binary found indicators for each tool
    for tool in tools:
        if use_rank_presence:
            # Match UpSet: found = rank exists (any rank, even 200)
            rank_col = f'rank_{tool}'
            if rank_col in df.columns:
                df[f'found_{tool}'] = df[rank_col].notna().astype(int)
            else:
                # Fallback to MRR > 0 (found at all)
                mrr_col = f'mrr_{tool}'
                if mrr_col in df.columns:
                    df[f'found_{tool}'] = (df[mrr_col] > 0).astype(int)
                else:
                    raise ValueError(f"Neither rank nor MRR column found for {tool}")
        else:
            # Stricter: found = MRR > 0.1 (rank <= 10)
            mrr_col = f'mrr_{tool}'
            if mrr_col in df.columns:
                df[f'found_{tool}'] = (df[mrr_col] > 0.1).astype(int)
            else:
                # Fallback to rank <= 10
                rank_col = f'rank_{tool}'
                if rank_col in df.columns:
                    df[f'found_{tool}'] = ((df[rank_col].notna()) & (df[rank_col] <= 10)).astype(int)
                else:
                    raise ValueError(f"Neither MRR nor rank column found for {tool}")
    
    # Create pattern string (e.g., "1101" = FlexFL, Locus, BugLocator found; BoostNSift didn't)
    found_cols = [f'found_{t}' for t in tools]
    df['cluster_pattern'] = df[found_cols].astype(str).agg(''.join, axis=1)
    
    # Count how many tools found each bug
    df['n_tools_found'] = df[found_cols].sum(axis=1)
    
    # Create descriptive cluster labels
    def label_cluster(row):
        n_found = row['n_tools_found']
        pattern = row['cluster_pattern']
        
        if n_found == 0:
            return "all_fail"
        elif n_found == len(tools):
            return "all_succeed"
        elif n_found == 1:
            # Find which tool
            for i, tool in enumerate(tools):
                if pattern[i] == '1':
                    return f"unique_{tool}"
        elif n_found == len(tools) - 1:
            # Find which tool failed
            for i, tool in enumerate(tools):
                if pattern[i] == '0':
                    return f"only_{tool}_failed"
        else:
            return f"{n_found}_of_{len(tools)}_tools"
        
        return "other"
    
    df['cluster'] = df.apply(label_cluster, axis=1)
    
    return df


def analyze_cluster_features(df, cluster_col, feature_cols, reference_cluster="all_succeed"):
    """
    Compare each cluster to a reference cluster across all features.
    
    Returns:
        DataFrame with Mann-Whitney U test results
    """
    results = []
    
    reference_data = df[df[cluster_col] == reference_cluster]
    clusters = df[cluster_col].unique()
    
    for cluster in clusters:
        if cluster == reference_cluster:
            continue
        
        cluster_data = df[df[cluster_col] == cluster]
        
        if len(cluster_data) < MIN_CLUSTER_SIZE:
            continue
        
        for feat in feature_cols:
            x = cluster_data[feat].dropna()
            y = reference_data[feat].dropna()
            
            if len(x) < 3 or len(y) < 3:
                continue
            
            try:
                u_stat, p_val = mannwhitneyu(x, y, alternative='two-sided')
                delta = cliffs_delta(x.values, y.values)
                
                results.append({
                    'cluster': cluster,
                    'feature': feat,
                    'n_cluster': len(x),
                    'n_reference': len(y),
                    'median_cluster': x.median(),
                    'median_reference': y.median(),
                    'mean_cluster': x.mean(),
                    'mean_reference': y.mean(),
                    'u_stat': u_stat,
                    'p_value': p_val,
                    'cliffs_delta': delta
                })
            except:
                continue
    
    return pd.DataFrame(results)


def compare_complementary_failures(df, tools, feature_cols):
    """
    Compare bugs where only Tool A failed vs. only Tool B failed.
    This reveals tool-specific weaknesses.
    """
    results = []
    
    for tool_a in tools:
        for tool_b in tools:
            if tool_a >= tool_b:  # Avoid duplicates and self-comparison
                continue
            
            # Bugs where only tool_a failed
            cluster_a = f"only_{tool_a}_failed"
            cluster_b = f"only_{tool_b}_failed"
            
            data_a = df[df['cluster'] == cluster_a]
            data_b = df[df['cluster'] == cluster_b]
            
            if len(data_a) < MIN_CLUSTER_SIZE or len(data_b) < MIN_CLUSTER_SIZE:
                continue
            
            for feat in feature_cols:
                x = data_a[feat].dropna()
                y = data_b[feat].dropna()
                
                if len(x) < 3 or len(y) < 3:
                    continue
                
                try:
                    u_stat, p_val = mannwhitneyu(x, y, alternative='two-sided')
                    delta = cliffs_delta(x.values, y.values)
                    
                    results.append({
                        'comparison': f"{cluster_a}_vs_{cluster_b}",
                        'tool_a': tool_a,
                        'tool_b': tool_b,
                        'feature': feat,
                        'n_a': len(x),
                        'n_b': len(y),
                        'median_a': x.median(),
                        'median_b': y.median(),
                        'u_stat': u_stat,
                        'p_value': p_val,
                        'cliffs_delta': delta
                    })
                except:
                    continue
    
    return pd.DataFrame(results)


def visualize_cluster_distributions(df, cluster_col, top_n_features=10):
    """Create violin plots showing feature distributions across clusters."""
    
    # Get clusters with sufficient size
    cluster_sizes = df[cluster_col].value_counts()
    valid_clusters = cluster_sizes[cluster_sizes >= MIN_CLUSTER_SIZE].index.tolist()
    
    if len(valid_clusters) < 2:
        print("Not enough clusters with sufficient size for visualization")
        return
    
    # Select top varying features
    feature_variance = {}
    for col in df.select_dtypes(include=[np.number]).columns:
        if col.startswith('mrr_') or col.startswith('rank_') or col.startswith('found_'):
            continue
        var = df.groupby(cluster_col)[col].mean().var()
        feature_variance[col] = var
    
    top_features = sorted(feature_variance.items(), key=lambda x: x[1], reverse=True)[:top_n_features]
    top_features = [f[0] for f in top_features]
    
    # Create plots
    n_features = len(top_features)
    n_cols = 3
    n_rows = (n_features + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4*n_rows))
    axes = axes.flatten() if n_rows > 1 else [axes]
    
    for idx, feat in enumerate(top_features):
        ax = axes[idx]
        
        plot_data = df[df[cluster_col].isin(valid_clusters)]
        
        sns.violinplot(
            data=plot_data,
            x=cluster_col,
            y=feat,
            ax=ax,
            inner='box'
        )
        
        ax.set_title(feat, fontsize=10, fontweight='bold')
        ax.set_xlabel('')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(axis='y', alpha=0.3)
    
    # Hide unused subplots
    for idx in range(len(top_features), len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(OUT_DIR / "cluster_feature_distributions.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {OUT_DIR / 'cluster_feature_distributions.png'}")


def create_cluster_summary_table(df, cluster_col):
    """Create summary statistics for each cluster."""
    summary = []
    
    for cluster in df[cluster_col].unique():
        cluster_data = df[df[cluster_col] == cluster]
        
        summary.append({
            'cluster': cluster,
            'n_bugs': len(cluster_data),
            'pct_of_total': len(cluster_data) / len(df) * 100,
        })
    
    summary_df = pd.DataFrame(summary).sort_values('n_bugs', ascending=False)
    return summary_df


# ======================================
# MAIN ANALYSIS
# ======================================

def run_cluster_analysis():
    """Main function to run complete cluster-based analysis."""
    
    print("=" * 60)
    print("CLUSTER-BASED FEATURE ANALYSIS")
    print("=" * 60)
    
    # Load data
    print("\nLoading data...")
    df_features = pd.read_csv(IN_FILE_FEATURES)
    print(f"Loaded features: {df_features.shape}")
    
    # Get feature columns
    id_cols = [c for c in ["project", "bug_id", "id"] if c in df_features.columns]
    perf_cols = [c for c in df_features.columns if c.startswith("mrr_") or c.startswith("rank_")]
    numeric_cols = df_features.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [c for c in numeric_cols if c not in perf_cols + id_cols]
    feature_cols = [c for c in feature_cols if not (c.endswith("_is_missing") or "__is_missing" in c)]
    
    print(f"Feature columns: {len(feature_cols)}")
    
    # Get tools
    tools = sorted({c.split('_', 1)[1] for c in df_features.columns if c.startswith('mrr_')})
    print(f"Tools: {tools}")
    
    # Create clusters
    print("\nCreating clusters based on tool intersections...")
    print("Using definition: 'found' = tool produced any rank (matches UpSet)")
    df_clustered = create_cluster_labels(df_features, tools, use_rank_presence=True)
    
    # Summary
    cluster_summary = create_cluster_summary_table(df_clustered, 'cluster')
    cluster_summary.to_csv(OUT_DIR / "cluster_summary.csv", index=False)
    print("\nCluster Summary:")
    print(cluster_summary.to_string(index=False))
    print(f"\nSaved: {OUT_DIR / 'cluster_summary.csv'}")
    
    # Analysis 1: Each cluster vs. "all_succeed" baseline
    print("\n" + "=" * 60)
    print("ANALYSIS 1: Clusters vs. All-Succeed Baseline")
    print("=" * 60)
    
    cluster_vs_baseline = analyze_cluster_features(
        df_clustered, 
        'cluster', 
        feature_cols,
        reference_cluster="all_succeed"
    )
    
    if not cluster_vs_baseline.empty:
        # Apply multiple testing correction
        from statsmodels.stats.multitest import multipletests
        
        reject, p_adj, _, _ = multipletests(
            cluster_vs_baseline['p_value'], 
            alpha=ALPHA, 
            method='holm'
        )
        cluster_vs_baseline['pval_adj'] = p_adj
        cluster_vs_baseline['reject'] = reject
        cluster_vs_baseline['abs_delta'] = cluster_vs_baseline['cliffs_delta'].abs()
        cluster_vs_baseline['practically_significant'] = cluster_vs_baseline['abs_delta'] >= 0.2
        
        cluster_vs_baseline.to_csv(OUT_DIR / "cluster_vs_baseline.csv", index=False)
        print(f"Saved: {OUT_DIR / 'cluster_vs_baseline.csv'}")
        
        # Show significant findings
        sig_results = cluster_vs_baseline[
            cluster_vs_baseline['reject'] & 
            cluster_vs_baseline['practically_significant']
        ].sort_values('abs_delta', ascending=False)
        
        print(f"\nSignificant differences from baseline: {len(sig_results)}")
        if not sig_results.empty:
            print("\nTop 10 findings:")
            print(sig_results[['cluster', 'feature', 'cliffs_delta', 'pval_adj']].head(10).to_string(index=False))
    
    # Analysis 2: Complementary failures (only_A_failed vs. only_B_failed)
    print("\n" + "=" * 60)
    print("ANALYSIS 2: Tool-Specific Weaknesses")
    print("=" * 60)
    
    complementary_failures = compare_complementary_failures(df_clustered, tools, feature_cols)
    
    if not complementary_failures.empty:
        from statsmodels.stats.multitest import multipletests
        
        reject, p_adj, _, _ = multipletests(
            complementary_failures['p_value'],
            alpha=ALPHA,
            method='holm'
        )
        complementary_failures['pval_adj'] = p_adj
        complementary_failures['reject'] = reject
        complementary_failures['abs_delta'] = complementary_failures['cliffs_delta'].abs()
        complementary_failures['practically_significant'] = complementary_failures['abs_delta'] >= 0.2
        
        complementary_failures.to_csv(OUT_DIR / "complementary_failures.csv", index=False)
        print(f"Saved: {OUT_DIR / 'complementary_failures.csv'}")
        
        sig_comp = complementary_failures[
            complementary_failures['reject'] &
            complementary_failures['practically_significant']
        ].sort_values('abs_delta', ascending=False)
        
        print(f"\nSignificant complementary failure patterns: {len(sig_comp)}")
        if not sig_comp.empty:
            print("\nTop 10 findings:")
            print(sig_comp[['comparison', 'feature', 'cliffs_delta', 'pval_adj']].head(10).to_string(index=False))
    
    # Visualization
    print("\n" + "=" * 60)
    print("Creating visualizations...")
    print("=" * 60)
    
    # Import improved visualizations
    try:
        from improved_cluster_visualizations import create_improved_visualizations
        create_improved_visualizations(df_clustered, 'cluster', feature_cols, OUT_DIR)
    except ImportError:
        print("Improved visualizations module not found, using basic visualization...")
        visualize_cluster_distributions(df_clustered, 'cluster', top_n_features=12)
    
    print("\n" + "=" * 60)
    print(f"ANALYSIS COMPLETE. All outputs in: {OUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    run_cluster_analysis()