"""
Exploratory Feature Analysis for Bug Reports

This script performs exploratory data analysis on extracted bug features
to understand feature distributions, correlations, and patterns.

This can be done NOW, before fault localization tool results are available.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

# Configuration
FEATURES_FILE = "bug_features_v2.csv"
S2R_FILE = "s2r_correctness_results.csv"
OUTPUT_DIR = "analysis_outputs"

def load_and_merge_data():
    """Load feature data and merge with S2R results if available."""
    print("Loading data...")
    df_features = pd.read_csv(FEATURES_FILE)
    print(f"  Loaded {len(df_features)} bug reports with features")
    
    # Try to merge with S2R results
    try:
        df_s2r = pd.read_csv(S2R_FILE)
        df = df_features.merge(df_s2r[['id', 'overall_score', 'num_steps', 'num_hq_steps']], 
                               on='id', how='left')
        print(f"  Merged with S2R results: {df_s2r['id'].nunique()} reports")
    except FileNotFoundError:
        print("  S2R results not found, using features only")
        df = df_features
    
    return df


def analyze_feature_distributions(df):
    """Analyze distributions of key features."""
    print("\n" + "="*80)
    print("FEATURE DISTRIBUTION ANALYSIS")
    print("="*80)
    
    # Select numeric features (exclude id and embedding)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if 'id' in numeric_cols:
        numeric_cols.remove('id')
    
    # Key feature groups
    structural_features = ['has_stacktrace', 'has_steps', 'has_code', 'has_patch', 
                          'has_enumeration', 'has_os_info', 'has_version_info']
    quality_features = ['redundancy', 'ambiguity', 'completeness_score', 'specificity_score']
    readability_features = [c for c in numeric_cols if any(x in c.lower() 
                          for x in ['flesch', 'smog', 'gunning', 'coleman', 'ari'])]
    step_features = [c for c in numeric_cols if 'step' in c.lower()]
    causal_features = [c for c in numeric_cols if 'causal' in c.lower() or 'temporal' in c.lower()]
    
    print("\n1. STRUCTURAL FEATURES (Binary)")
    print("-" * 80)
    for feat in structural_features:
        if feat in df.columns:
            count = df[feat].sum()
            pct = df[feat].mean() * 100
            print(f"  {feat:30s}: {count:4d} ({pct:5.1f}%)")
    
    print("\n2. QUALITY METRICS (Continuous)")
    print("-" * 80)
    for feat in quality_features:
        if feat in df.columns:
            mean_val = df[feat].mean()
            median_val = df[feat].median()
            std_val = df[feat].std()
            print(f"  {feat:30s}: mean={mean_val:6.3f}, median={median_val:6.3f}, std={std_val:6.3f}")
    
    print("\n3. READABILITY SCORES")
    print("-" * 80)
    for feat in readability_features:
        if feat in df.columns:
            valid = df[feat].notna().sum()
            if valid > 0:
                mean_val = df[feat].mean()
                print(f"  {feat:30s}: mean={mean_val:6.2f} (valid: {valid}/{len(df)})")
    
    print("\n4. STEP-RELATED FEATURES")
    print("-" * 80)
    for feat in step_features:
        if feat in df.columns:
            mean_val = df[feat].mean()
            print(f"  {feat:30s}: mean={mean_val:6.2f}")
    
    return {
        'structural': structural_features,
        'quality': quality_features,
        'readability': readability_features,
        'step': step_features,
        'causal': causal_features
    }


def analyze_feature_correlations(df, feature_groups):
    """Analyze correlations between features."""
    print("\n" + "="*80)
    print("FEATURE CORRELATION ANALYSIS")
    print("="*80)
    
    # Select numeric features
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if 'id' in numeric_cols:
        numeric_cols.remove('id')
    
    # Remove embedding column if present
    if 'embedding' in numeric_cols:
        numeric_cols.remove('embedding')
    
    # Compute correlation matrix
    corr_matrix = df[numeric_cols].corr()
    
    # Find highly correlated pairs
    print("\n1. HIGHLY CORRELATED FEATURE PAIRS (|r| > 0.7)")
    print("-" * 80)
    high_corr_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            corr_val = corr_matrix.iloc[i, j]
            if abs(corr_val) > 0.7 and not np.isnan(corr_val):
                high_corr_pairs.append((
                    corr_matrix.columns[i],
                    corr_matrix.columns[j],
                    corr_val
                ))
    
    if high_corr_pairs:
        for feat1, feat2, corr in sorted(high_corr_pairs, key=lambda x: abs(x[2]), reverse=True)[:20]:
            print(f"  {feat1:30s} <-> {feat2:30s}: {corr:6.3f}")
    else:
        print("  No highly correlated pairs found (threshold: 0.7)")
    
    # Analyze correlations within feature groups
    print("\n2. WITHIN-GROUP CORRELATIONS")
    print("-" * 80)
    for group_name, features in feature_groups.items():
        if not features:
            continue
        available = [f for f in features if f in numeric_cols]
        if len(available) > 1:
            group_corr = df[available].corr()
            # Get mean absolute correlation (excluding diagonal)
            mask = np.triu(np.ones_like(group_corr, dtype=bool), k=1)
            mean_corr = group_corr.where(mask).abs().mean().mean()
            print(f"  {group_name:20s}: {len(available):2d} features, mean |r|={mean_corr:.3f}")
    
    return corr_matrix


def create_composite_features(df):
    """Create composite features that might be predictive."""
    print("\n" + "="*80)
    print("COMPOSITE FEATURE CREATION")
    print("="*80)
    
    df_new = df.copy()
    
    # 1. Information richness score
    if all(c in df.columns for c in ['n_tokens', 'n_sentences', 'has_stacktrace', 'has_code']):
        df_new['info_richness'] = (
            np.log1p(df['n_tokens']) * 0.3 +
            np.log1p(df['n_sentences']) * 0.2 +
            df['has_stacktrace'].astype(int) * 0.3 +
            df['has_code'].astype(int) * 0.2
        )
        print("  Created: info_richness (combines length, stacktrace, code)")
    
    # 2. Step quality composite
    if all(c in df.columns for c in ['completeness_score', 'specificity_score', 'num_steps']):
        df_new['step_quality_composite'] = (
            df['completeness_score'] * 0.4 +
            df['specificity_score'] * 0.4 +
            np.minimum(df['num_steps'] / 10.0, 1.0) * 0.2  # Normalize steps
        )
        print("  Created: step_quality_composite (combines completeness, specificity, step count)")
    
    # 3. Clarity score (inverse of ambiguity + redundancy)
    if all(c in df.columns for c in ['ambiguity', 'redundancy']):
        # Normalize to 0-1 range first
        amb_norm = (df['ambiguity'] - df['ambiguity'].min()) / (df['ambiguity'].max() - df['ambiguity'].min() + 1e-6)
        red_norm = (df['redundancy'] - df['redundancy'].min()) / (df['redundancy'].max() - df['redundancy'].min() + 1e-6)
        df_new['clarity_score'] = 1.0 - (amb_norm * 0.6 + red_norm * 0.4)
        print("  Created: clarity_score (inverse of ambiguity + redundancy)")
    
    # 4. Context richness
    if all(c in df.columns for c in ['num_os_mentions', 'num_browser_mentions', 'num_env_mentions', 'num_versions']):
        df_new['context_richness'] = (
            np.minimum(df['num_os_mentions'], 3) * 0.25 +
            np.minimum(df['num_browser_mentions'], 3) * 0.25 +
            np.minimum(df['num_env_mentions'], 3) * 0.25 +
            np.minimum(df['num_versions'], 3) * 0.25
        )
        print("  Created: context_richness (combines OS, browser, env, version mentions)")
    
    # 5. Structural completeness
    if all(c in df.columns for c in ['has_stacktrace', 'has_steps', 'has_code', 'has_enumeration']):
        df_new['structural_completeness'] = (
            df['has_stacktrace'].astype(int) * 0.3 +
            df['has_steps'].astype(int) * 0.4 +
            df['has_code'].astype(int) * 0.2 +
            df['has_enumeration'].astype(int) * 0.1
        )
        print("  Created: structural_completeness (combines structural flags)")
    
    return df_new


def cluster_bugs_by_features(df):
    """Cluster bugs based on feature similarity."""
    print("\n" + "="*80)
    print("BUG CLUSTERING ANALYSIS")
    print("="*80)
    
    # Select numeric features for clustering
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if 'id' in numeric_cols:
        numeric_cols.remove('id')
    if 'embedding' in numeric_cols:
        numeric_cols.remove('embedding')
    
    # Remove columns with too many missing values
    numeric_cols = [c for c in numeric_cols if df[c].notna().sum() > len(df) * 0.5]
    
    if len(numeric_cols) < 5:
        print("  Not enough features for clustering")
        return None
    
    # Prepare data
    X = df[numeric_cols].fillna(df[numeric_cols].median())
    
    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # PCA for visualization
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    print(f"\n  PCA explained variance: {pca.explained_variance_ratio_.sum():.2%}")
    print(f"  PC1: {pca.explained_variance_ratio_[0]:.2%}, PC2: {pca.explained_variance_ratio_[1]:.2%}")
    
    # K-means clustering
    n_clusters = 5
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)
    df['cluster'] = clusters
    
    print(f"\n  Clustered {len(df)} bugs into {n_clusters} groups:")
    for i in range(n_clusters):
        count = (clusters == i).sum()
        pct = count / len(df) * 100
        print(f"    Cluster {i}: {count:4d} bugs ({pct:5.1f}%)")
    
    # Analyze cluster characteristics
    print("\n  Cluster characteristics (mean values):")
    key_features = ['has_stacktrace', 'has_steps', 'has_code', 'num_steps', 
                   'redundancy', 'ambiguity', 'completeness_score']
    available_key = [f for f in key_features if f in numeric_cols]
    
    for i in range(n_clusters):
        cluster_data = df[df['cluster'] == i]
        print(f"\n    Cluster {i}:")
        for feat in available_key:
            if feat in cluster_data.columns:
                mean_val = cluster_data[feat].mean()
                print(f"      {feat:25s}: {mean_val:6.3f}")
    
    return df, X_pca, clusters


def prepare_for_tool_results(df):
    """Prepare data structure for when tool results arrive."""
    print("\n" + "="*80)
    print("PREPARING DATA STRUCTURE FOR TOOL RESULTS")
    print("="*80)
    
    # Create a template for tool results
    template = pd.DataFrame({
        'id': df['id'],
        'has_features': True
    })
    
    # Save template
    template_file = f"{OUTPUT_DIR}/tool_results_template.csv"
    template.to_csv(template_file, index=False)
    print(f"\n  Created template: {template_file}")
    print(f"  When tool results arrive, add columns like:")
    print(f"    - tool_name (e.g., 'Ochiai', 'Tarantula', 'DStar', etc.)")
    print(f"    - detected (boolean: did the tool find the bug?)")
    print(f"    - rank_at_fault (integer: rank of actual fault in suspiciousness list)")
    print(f"    - exam_score (integer: number of statements examined)")
    print(f"    - time_to_detect (float: time taken)")
    
    # Create feature summary for easy merging
    feature_summary = df[['id']].copy()
    key_features = [
        'has_stacktrace', 'has_steps', 'has_code', 'num_steps',
        'redundancy', 'ambiguity', 'completeness_score', 'specificity_score',
        'info_richness' if 'info_richness' in df.columns else None,
        'step_quality_composite' if 'step_quality_composite' in df.columns else None,
        'clarity_score' if 'clarity_score' in df.columns else None
    ]
    key_features = [f for f in key_features if f and f in df.columns]
    
    for feat in key_features:
        feature_summary[feat] = df[feat]
    
    summary_file = f"{OUTPUT_DIR}/key_features_summary.csv"
    feature_summary.to_csv(summary_file, index=False)
    print(f"\n  Created key features summary: {summary_file}")
    print(f"  Contains {len(key_features)} key features for easy merging with tool results")
    
    return template_file, summary_file


def main():
    """Main analysis pipeline."""
    import os
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load data
    df = load_and_merge_data()
    
    # 1. Feature distribution analysis
    feature_groups = analyze_feature_distributions(df)
    
    # 2. Correlation analysis
    corr_matrix = analyze_feature_correlations(df, feature_groups)
    
    # 3. Create composite features
    df = create_composite_features(df)
    
    # 4. Clustering
    try:
        df, X_pca, clusters = cluster_bugs_by_features(df)
    except Exception as e:
        print(f"\n  Clustering failed: {e}")
        df['cluster'] = -1
    
    # 5. Prepare for tool results
    template_file, summary_file = prepare_for_tool_results(df)
    
    # Save enhanced dataframe
    output_file = f"{OUTPUT_DIR}/features_with_composites.csv"
    df.to_csv(output_file, index=False)
    print(f"\n  Saved enhanced features to: {output_file}")
    
    # Summary
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print("\nWhat you can do NOW:")
    print("  1. Understand feature distributions and correlations")
    print("  2. Identify feature groups and patterns")
    print("  3. Use composite features for better analysis")
    print("  4. Explore bug clusters")
    print("\nWhat you need tool results for:")
    print("  1. Predict which features help/hurt fault localization")
    print("  2. Build predictive models")
    print("  3. Understand feature importance for tool success")
    print("\nNext steps:")
    print(f"  1. Review the analysis outputs in: {OUTPUT_DIR}/")
    print(f"  2. When tool results arrive, merge with: {summary_file}")
    print(f"  3. Use template: {template_file}")
    print("="*80)


if __name__ == "__main__":
    main()

