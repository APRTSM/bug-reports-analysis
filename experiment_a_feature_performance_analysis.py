"""
Experiment A: Feature-Performance Relationship Discovery

Objective: Establish the empirical foundation for understanding which syntactic and semantic 
bug characteristics predict fault localization success, and how these relationships differ 
between traditional IR-based tools and LLM-based agents.

Based on the experimental design from Section 4.2
"""

import pandas as pd
import numpy as np
from scipy.stats import spearmanr, pearsonr, wilcoxon
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.cluster.hierarchy import cophenet
from scipy.spatial.distance import pdist
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score, classification_report
from sklearn.metrics import silhouette_score, silhouette_samples
import matplotlib.pyplot as plt
import seaborn as sns
try:
    from statsmodels.stats.multitest import multipletests
    MULTIPLETESTS_AVAILABLE = True
except ImportError:
    MULTIPLETESTS_AVAILABLE = False
    print("Warning: statsmodels not available. Install with: pip install statsmodels")
    print("Continuing without Holm-Bonferroni correction...")

import warnings
import os
warnings.filterwarnings('ignore')

# Try to import SHAP, but continue without it if not available
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("Warning: SHAP not available. Install with: pip install shap")
    print("Continuing without SHAP values...")

# Create results directory
RESULTS_DIR = 'results_experimentA'
os.makedirs(RESULTS_DIR, exist_ok=True)
print(f"Results will be saved to: {RESULTS_DIR}/")

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (16, 12)

def calculate_mrr(rank):
    """Calculate Mean Reciprocal Rank (MRR) from rank"""
    # MRR = 1/rank if detected, 0 if not detected
    # Handle N/A ranks (not detected)
    if pd.isna(rank) or rank == 'N/A':
        return 0.0
    try:
        rank = float(rank)
        return 1.0 / rank if rank > 0 else 0.0
    except (ValueError, TypeError):
        return 0.0

def load_and_prepare_data():
    """Load and merge all data sources"""
    print("="*80)
    print("Loading and preparing data...")
    print("="*80)
    
    # Load feature data
    features_df = pd.read_csv('bug_feature_extraction/analysis_outputs/features_with_composites.csv')
    print(f"Features shape: {features_df.shape}")
    
    # Load tool comparison data
    tool_df = pd.read_csv('tool_comparison_summary.csv')
    print(f"Tool comparison shape: {tool_df.shape}")
    
    # Create bug_id
    tool_df['bug_id'] = tool_df['project'] + '-' + tool_df['bug_id'].astype(str)
    
    # Calculate MRR for each tool-bug combination
    tool_df['mrr'] = tool_df['rank'].apply(calculate_mrr)
    
    # Pivot to have one row per bug with columns for each tool
    performance_metrics = ['detected', 'rank', 'mrr', 'top@1', 'top@5', 'top@10', 'duration_seconds']
    
    tool_pivot_list = []
    for metric in performance_metrics:
        if metric in tool_df.columns:
            pivot = tool_df.pivot_table(
                index='bug_id',
                columns='tool',
                values=metric,
                aggfunc='first'
            )
            pivot.columns = [f'{tool}_{metric}' for tool in pivot.columns]
            tool_pivot_list.append(pivot)
    
    # Merge all pivots
    tool_pivot = tool_pivot_list[0]
    for p in tool_pivot_list[1:]:
        tool_pivot = tool_pivot.merge(p, left_index=True, right_index=True, how='outer')
    
    tool_pivot = tool_pivot.reset_index()
    
    # Merge with features
    merged_df = features_df.merge(
        tool_pivot,
        left_on='id',
        right_on='bug_id',
        how='inner'
    )
    
    print(f"Merged dataset shape: {merged_df.shape}")
    print(f"Available tools: {[col.split('_')[0] for col in tool_pivot.columns if '_detected' in col]}")
    
    return merged_df

def get_tool_categories():
    """Define tool categories based on available tools"""
    # Based on available tools: boostnsift, buglocator, locus
    # Adapting to what's available
    return {
        'IR-based': ['boostnsift', 'locus'],
        'Other': ['buglocator']  # buglocator might also be IR-based, but treating separately
    }

def univariate_feature_performance_analysis(merged_df, feature_cols):
    """
    Univariate Feature-Performance Analysis
    
    - Compute Spearman (ρ) and Pearson (r) correlations
    - Identify significant correlations (Holm-Bonferroni correction, α=0.05)
    - Create heatmaps and scatter plots
    """
    print("\n" + "="*80)
    print("1. UNIVARIATE FEATURE-PERFORMANCE ANALYSIS")
    print("="*80)
    
    # Get performance metrics for each tool
    tools = ['boostnsift', 'buglocator', 'locus']
    performance_metrics = ['mrr', 'top@1', 'top@5', 'top@10']
    
    results = {}
    
    for tool in tools:
        if tool not in [col.split('_')[0] for col in merged_df.columns]:
            continue
        
        print(f"\n{tool.upper()}:")
        print("-" * 60)
        
        tool_results = {}
        
        for metric in performance_metrics:
            metric_col = f'{tool}_{metric}'
            if metric_col not in merged_df.columns:
                continue
            
            y = merged_df[metric_col].fillna(0)
            
            correlations = []
            for feature in feature_cols:
                if feature not in merged_df.columns:
                    continue
                
                x = pd.to_numeric(merged_df[feature], errors='coerce').fillna(0)
                
                # Skip if constant
                if x.nunique() < 2:
                    continue
                
                # Pearson correlation
                try:
                    pearson_r, pearson_p = pearsonr(x, y)
                except:
                    pearson_r, pearson_p = np.nan, np.nan
                
                # Spearman correlation
                try:
                    spearman_rho, spearman_p = spearmanr(x, y)
                except:
                    spearman_rho, spearman_p = np.nan, np.nan
                
                correlations.append({
                    'feature': feature,
                    'pearson_r': pearson_r,
                    'pearson_p': pearson_p,
                    'spearman_rho': spearman_rho,
                    'spearman_p': spearman_p
                })
            
            corr_df = pd.DataFrame(correlations)
            
                # Holm-Bonferroni correction for multiple testing
            if len(corr_df) > 0:
                # Apply correction to both p-value columns
                for p_col in ['pearson_p', 'spearman_p']:
                    if p_col in corr_df.columns:
                        valid_p = corr_df[p_col].dropna()
                        if len(valid_p) > 0:
                            if MULTIPLETESTS_AVAILABLE:
                                try:
                                    _, corrected_p, _, _ = multipletests(
                                        valid_p, alpha=0.05, method='holm'
                                    )
                                    corr_df.loc[valid_p.index, f'{p_col}_corrected'] = corrected_p
                                    corr_df[f'{p_col}_significant'] = corr_df[f'{p_col}_corrected'] < 0.05
                                except:
                                    corr_df[f'{p_col}_significant'] = corr_df[p_col] < 0.05
                            else:
                                # Simple Bonferroni correction
                                n_tests = len(valid_p)
                                corr_df[f'{p_col}_significant'] = corr_df[p_col] < (0.05 / n_tests)
                
                # Sort by absolute correlation
                corr_df['abs_pearson'] = corr_df['pearson_r'].abs()
                corr_df['abs_spearman'] = corr_df['spearman_rho'].abs()
                corr_df = corr_df.sort_values('abs_spearman', ascending=False)
                
                tool_results[metric] = corr_df
                
                # Print top correlations
                print(f"\n  {metric.upper()} - Top 5 significant correlations:")
                significant = corr_df[corr_df.get('spearman_p_significant', corr_df['spearman_p'] < 0.05)].head(5)
                if len(significant) > 0:
                    print(significant[['feature', 'spearman_rho', 'spearman_p']].to_string(index=False))
                else:
                    print("  No significant correlations found")
        
        results[tool] = tool_results
    
    # Create correlation heatmaps
    create_correlation_heatmaps(merged_df, feature_cols, tools, performance_metrics, results)
    
    # Create scatter plots with LOWESS for top features
    create_scatter_plots(merged_df, feature_cols, tools, performance_metrics, results)
    
    return results

def create_correlation_heatmaps(merged_df, feature_cols, tools, performance_metrics, results):
    """Create heatmaps showing feature-tool correlation matrices"""
    print("\nCreating correlation heatmaps...")
    
    # For each metric, create a heatmap
    for metric in ['mrr']:  # Focus on MRR
        fig, axes = plt.subplots(1, len(tools), figsize=(6*len(tools), 8))
        if len(tools) == 1:
            axes = [axes]
        
        for idx, tool in enumerate(tools):
            if tool not in results:
                continue
            
            if metric not in results[tool]:
                continue
            
            corr_df = results[tool][metric]
            
            # Select top 20 features by absolute correlation
            top_features = corr_df.head(20)['feature'].tolist()
            
            # Create correlation matrix (just for visualization)
            # We'll show the correlation values as a bar chart instead
            ax = axes[idx]
            y_pos = np.arange(len(top_features))
            correlations = [corr_df[corr_df['feature'] == f]['spearman_rho'].iloc[0] 
                          for f in top_features]
            colors = ['green' if c > 0 else 'red' for c in correlations]
            
            ax.barh(y_pos, correlations, color=colors, alpha=0.7)
            ax.set_yticks(y_pos)
            ax.set_yticklabels(top_features, fontsize=8)
            ax.set_xlabel(f'Spearman ρ')
            ax.set_title(f'{tool} - {metric.upper()}')
            ax.axvline(x=0, color='black', linestyle='--', linewidth=0.5)
            ax.set_xlim([-1, 1])
        
        plt.tight_layout()
        plt.savefig(f'{RESULTS_DIR}/correlation_heatmap_{metric}.png', dpi=300, bbox_inches='tight')
        print(f"  Saved: {RESULTS_DIR}/correlation_heatmap_{metric}.png")

def create_scatter_plots(merged_df, feature_cols, tools, performance_metrics, results):
    """Create scatter plots with LOWESS curves for top-5 correlated features per tool"""
    print("\nCreating scatter plots with LOWESS curves...")
    
    try:
        from statsmodels.nonparametric.smoothers_lowess import lowess
        LOWESS_AVAILABLE = True
    except ImportError:
        LOWESS_AVAILABLE = False
        print("  Warning: LOWESS not available. Using linear fit instead.")
    
    for tool in tools:
        if tool not in results:
            continue
        
        for metric in ['mrr']:  # Focus on MRR
            corr_df = results[tool][metric]
            top_features = corr_df.head(5)['feature'].tolist()
            
            if len(top_features) == 0:
                continue
            
            fig, axes = plt.subplots(1, min(5, len(top_features)), figsize=(5*min(5, len(top_features)), 4))
            if len(top_features) == 1:
                axes = [axes]
            
            metric_col = f'{tool}_{metric}'
            y = merged_df[metric_col].fillna(0)
            
            for idx, feature in enumerate(top_features[:5]):
                if feature not in merged_df.columns:
                    continue
                
                ax = axes[idx]
                x = pd.to_numeric(merged_df[feature], errors='coerce').fillna(0)
                
                # Scatter plot
                ax.scatter(x, y, alpha=0.5, s=20)
                
                # LOWESS curve
                if LOWESS_AVAILABLE:
                    try:
                        sorted_indices = np.argsort(x)
                        x_sorted = x.iloc[sorted_indices].values
                        y_sorted = y.iloc[sorted_indices].values
                        lowess_result = lowess(y_sorted, x_sorted, frac=0.3)
                        ax.plot(lowess_result[:, 0], lowess_result[:, 1], 'r-', linewidth=2, label='LOWESS')
                    except:
                        # Fallback to linear fit
                        z = np.polyfit(x, y, 1)
                        p = np.poly1d(z)
                        ax.plot(sorted(x), p(sorted(x)), "r--", alpha=0.8, label='Linear fit')
                else:
                    # Linear fit
                    z = np.polyfit(x, y, 1)
                    p = np.poly1d(z)
                    ax.plot(sorted(x), p(sorted(x)), "r--", alpha=0.8, label='Linear fit')
                
                corr_val = corr_df[corr_df['feature'] == feature]['spearman_rho'].iloc[0]
                ax.set_xlabel(feature, fontsize=9)
                ax.set_ylabel(f'{metric.upper()}', fontsize=9)
                ax.set_title(f'ρ = {corr_val:.3f}', fontsize=10)
                ax.legend(fontsize=8)
            
            plt.tight_layout()
            plt.savefig(f'{RESULTS_DIR}/scatter_{tool}_{metric}.png', dpi=300, bbox_inches='tight')
            print(f"  Saved: {RESULTS_DIR}/scatter_{tool}_{metric}.png")

def tool_category_comparison(merged_df, feature_cols):
    """
    Tool Category Comparison (Traditional vs. LLM-based)
    
    - Stratify bugs into quartiles by feature value
    - Compute per-quartile performance: MRR_IR vs MRR_LLM
    - Statistical testing: Paired Wilcoxon signed-rank tests with effect sizes (Cliff's δ)
    """
    print("\n" + "="*80)
    print("2. TOOL CATEGORY COMPARISON")
    print("="*80)
    
    tool_categories = get_tool_categories()
    
    # Calculate average MRR per category
    for category, tools_in_category in tool_categories.items():
        mrr_cols = [f'{tool}_mrr' for tool in tools_in_category 
                   if f'{tool}_mrr' in merged_df.columns]
        if len(mrr_cols) > 0:
            merged_df[f'{category}_mrr'] = merged_df[mrr_cols].mean(axis=1)
    
    results = {}
    
    for feature in feature_cols[:20]:  # Analyze top 20 features to save time
        if feature not in merged_df.columns:
            continue
        
        x = pd.to_numeric(merged_df[feature], errors='coerce')
        valid_mask = ~x.isna()
        x_valid = x[valid_mask]
        
        if len(x_valid) < 4 or x_valid.nunique() < 2:  # Need at least 4 values and 2 unique values
            continue
        
        # Stratify into quartiles - handle cases with duplicates
        try:
            # First try with labels
            quartiles_valid, bins = pd.qcut(x_valid, q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'], 
                                      duplicates='drop', retbins=True)
            # Check if we got the expected number of bins
            n_bins = len(bins) - 1
            if n_bins < 4:
                # If we got fewer bins, recreate without labels and use numeric labels
                quartiles_valid = pd.qcut(x_valid, q=4, duplicates='drop')
                # Map to Q1-Q4 labels based on actual bins
                unique_bins = quartiles_valid.cat.categories
                label_map = {cat: f'Q{i+1}' for i, cat in enumerate(unique_bins)}
                quartiles_valid = quartiles_valid.map(label_map)
        except (ValueError, TypeError) as e:
            # If qcut fails, try with fewer quantiles or skip
            try:
                # Try with 3 quantiles instead
                quartiles_valid = pd.qcut(x_valid, q=3, labels=['Q1', 'Q2', 'Q3'], duplicates='drop')
            except:
                # Skip this feature if we can't create quantiles
                continue
        
        # Create full quartiles series aligned with merged_df index
        quartiles = pd.Series(index=merged_df.index, dtype='object')
        quartiles.loc[valid_mask] = quartiles_valid
        
        quartile_results = []
        
        for category, tools_in_category in tool_categories.items():
            mrr_col = f'{category}_mrr'
            if mrr_col not in merged_df.columns:
                continue
            
            # Get unique quartile labels that actually exist
            unique_quartiles = quartiles_valid.unique()
            for quartile in unique_quartiles:
                if pd.isna(quartile):
                    continue
                mask = quartiles == quartile
                if mask.sum() == 0:
                    continue
                
                mrr_values = merged_df.loc[mask, mrr_col].dropna()
                if len(mrr_values) > 0:
                    quartile_results.append({
                        'feature': feature,
                        'category': category,
                        'quartile': quartile,
                        'mean_mrr': mrr_values.mean(),
                        'median_mrr': mrr_values.median(),
                        'n': len(mrr_values)
                    })
        
        if len(quartile_results) >= 2:
            results[feature] = pd.DataFrame(quartile_results)
    
    # Statistical testing: Compare IR-based vs Other across quartiles
    if 'IR-based_mrr' in merged_df.columns and 'Other_mrr' in merged_df.columns:
        print("\nStatistical Testing (IR-based vs Other):")
        print("-" * 60)
        
        statistical_tests = []
        
        for feature in list(results.keys())[:10]:  # Test top 10
            x = pd.to_numeric(merged_df[feature], errors='coerce')
            valid_mask = ~x.isna()
            x_valid = x[valid_mask]
            
            if len(x_valid) < 4 or x_valid.nunique() < 2:
                continue
            
            try:
                quartiles_valid = pd.qcut(x_valid, q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'], duplicates='drop')
            except (ValueError, TypeError):
                try:
                    quartiles_valid = pd.qcut(x_valid, q=3, labels=['Q1', 'Q2', 'Q3'], duplicates='drop')
                except:
                    continue
            
            # Create full quartiles series aligned with merged_df index
            quartiles = pd.Series(index=merged_df.index, dtype='object')
            quartiles.loc[valid_mask] = quartiles_valid
            
            unique_quartiles = quartiles_valid.unique()
            for quartile in unique_quartiles:
                if pd.isna(quartile):
                    continue
                mask = quartiles == quartile
                ir_mrr = merged_df.loc[mask, 'IR-based_mrr'].dropna()
                other_mrr = merged_df.loc[mask, 'Other_mrr'].dropna()
                
                if len(ir_mrr) > 3 and len(other_mrr) > 3:
                    # Wilcoxon signed-rank test
                    try:
                        # Align by index
                        common_idx = ir_mrr.index.intersection(other_mrr.index)
                        if len(common_idx) > 3:
                            stat, p_value = wilcoxon(
                                ir_mrr.loc[common_idx],
                                other_mrr.loc[common_idx]
                            )
                            
                            # Cliff's delta (effect size)
                            cliff_delta = calculate_cliffs_delta(
                                ir_mrr.loc[common_idx],
                                other_mrr.loc[common_idx]
                            )
                            
                            statistical_tests.append({
                                'feature': feature,
                                'quartile': quartile,
                                'wilcoxon_stat': stat,
                                'wilcoxon_p': p_value,
                                'cliffs_delta': cliff_delta,
                                'n': len(common_idx)
                            })
                    except:
                        pass
        
        if statistical_tests:
            test_df = pd.DataFrame(statistical_tests)
            test_df = test_df.sort_values('cliffs_delta', key=abs, ascending=False)
            print(test_df.head(10).to_string(index=False))
            test_df.to_csv(f'{RESULTS_DIR}/tool_category_statistical_tests.csv', index=False)
            print(f"\n  Saved: {RESULTS_DIR}/tool_category_statistical_tests.csv")
    
    return results

def calculate_cliffs_delta(x, y):
    """Calculate Cliff's delta effect size"""
    n_x = len(x)
    n_y = len(y)
    
    # Count dominances
    dominance = 0
    for xi in x:
        for yj in y:
            if xi > yj:
                dominance += 1
            elif xi < yj:
                dominance -= 1
    
    delta = dominance / (n_x * n_y)
    return delta

def feature_importance_shap(merged_df, feature_cols):
    """
    Feature Importance for Tool-Specific Success Prediction
    
    - Train binary classifiers (Random Forest) per tool
    - Extract SHAP values to quantify feature contributions
    - Rank features by importance
    """
    print("\n" + "="*80)
    print("3. FEATURE IMPORTANCE WITH SHAP")
    print("="*80)
    
    tools = ['boostnsift', 'buglocator', 'locus']
    results = {}
    
    # Prepare feature matrix
    X = merged_df[feature_cols].select_dtypes(include=[np.number]).fillna(0)
    X = X.loc[:, ~X.columns.isin(['id', 'bug_id', 'cluster'])]
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
    
    for tool in tools:
        # Binary classification: Top-5 success
        top5_col = f'{tool}_top@5'
        if top5_col not in merged_df.columns:
            continue
        
        y = (merged_df[top5_col] == 1).astype(int)
        
        if y.sum() < 10 or (y == 0).sum() < 10:  # Need both classes
            print(f"\n{tool}: Insufficient class balance (skipping)")
            continue
        
        print(f"\n{tool.upper()}:")
        print("-" * 60)
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Train Random Forest
        rf = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            class_weight='balanced',
            max_depth=10
        )
        rf.fit(X_train, y_train)
        
        # Evaluate
        y_pred = rf.predict(X_test)
        auc = roc_auc_score(y_test, rf.predict_proba(X_test)[:, 1])
        f1 = f1_score(y_test, y_pred)
        
        print(f"  AUROC: {auc:.3f}")
        print(f"  F1-Score: {f1:.3f}")
        
        # Feature importance (Gini)
        importance_df = pd.DataFrame({
            'feature': X.columns,
            'importance': rf.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print(f"\n  Top 10 features (Gini importance):")
        print(importance_df.head(10)[['feature', 'importance']].to_string(index=False))
        
        # SHAP values
        if SHAP_AVAILABLE:
            try:
                print("\n  Computing SHAP values...")
                explainer = shap.TreeExplainer(rf)
                shap_values = explainer.shap_values(X_test[:100])  # Sample for speed
                
                # Average absolute SHAP value per feature
                if isinstance(shap_values, list):
                    shap_values = shap_values[1]  # Use positive class
                
                shap_importance = pd.DataFrame({
                    'feature': X.columns,
                    'shap_importance': np.abs(shap_values).mean(axis=0)
                }).sort_values('shap_importance', ascending=False)
                
                print(f"\n  Top 10 features (SHAP importance):")
                print(shap_importance.head(10)[['feature', 'shap_importance']].to_string(index=False))
                
                # Save SHAP plot
                try:
                    shap.summary_plot(shap_values, X_test[:100], show=False, max_display=20)
                    plt.savefig(f'{RESULTS_DIR}/shap_summary_{tool}.png', dpi=300, bbox_inches='tight')
                    plt.close()
                    print(f"  Saved: {RESULTS_DIR}/shap_summary_{tool}.png")
                except:
                    pass
                
                results[tool] = {
                    'gini_importance': importance_df,
                    'shap_importance': shap_importance,
                    'auc': auc,
                    'f1': f1
                }
            except Exception as e:
                print(f"  SHAP computation failed: {e}")
                results[tool] = {
                    'gini_importance': importance_df,
                    'auc': auc,
                    'f1': f1
                }
        else:
            results[tool] = {
                'gini_importance': importance_df,
                'auc': auc,
                'f1': f1
            }
        
        # Save importance
        importance_df.to_csv(f'{RESULTS_DIR}/feature_importance_{tool}_experiment_a.csv', index=False)
    
    return results

def bug_archetype_clustering(merged_df, feature_cols):
    """
    Bug Archetype Discovery via Clustering
    
    - Apply hierarchical clustering on feature space with Ward linkage
    - Determine optimal k via silhouette analysis
    - Characterize each cluster
    """
    print("\n" + "="*80)
    print("4. BUG ARCHETYPE DISCOVERY VIA CLUSTERING")
    print("="*80)
    
    # Prepare features
    X = merged_df[feature_cols].select_dtypes(include=[np.number]).fillna(0)
    X = X.loc[:, ~X.columns.isin(['id', 'bug_id', 'cluster'])]
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    print(f"Feature matrix shape: {X_scaled.shape}")
    
    # Hierarchical clustering with Ward linkage
    print("\nComputing hierarchical clustering (Ward linkage)...")
    linkage_matrix = linkage(X_scaled, method='ward')
    
    # Cophenetic correlation
    coph_dist = pdist(X_scaled)
    coph_corr, _ = cophenet(linkage_matrix, coph_dist)
    print(f"Cophenetic correlation: {coph_corr:.3f}")
    
    # Silhouette analysis to determine optimal k
    print("\nSilhouette analysis...")
    k_range = range(2, min(11, len(merged_df)//10))
    silhouette_scores = []
    
    for k in k_range:
        clusters = fcluster(linkage_matrix, k, criterion='maxclust')
        if len(np.unique(clusters)) > 1:
            score = silhouette_score(X_scaled, clusters)
            silhouette_scores.append({'k': k, 'silhouette_score': score})
            print(f"  k={k}: {score:.3f}")
    
    if silhouette_scores:
        silhouette_df = pd.DataFrame(silhouette_scores)
        optimal_k = silhouette_df.loc[silhouette_df['silhouette_score'].idxmax(), 'k']
        print(f"\nOptimal k: {optimal_k}")
        
        # Create clusters with optimal k
        clusters = fcluster(linkage_matrix, optimal_k, criterion='maxclust')
        merged_df['archetype_cluster'] = clusters
        
        # Characterize clusters
        print("\nCluster characterization:")
        print("-" * 60)
        
        cluster_characteristics = []
        
        for cluster_id in range(1, optimal_k + 1):
            mask = clusters == cluster_id
            cluster_data = merged_df[mask]
            
            # Feature statistics
            char = {
                'cluster': cluster_id,
                'size': mask.sum(),
            }
            
            # Average feature values
            for feature in ['summary_chars', 'description_chars', 'n_words', 
                          'has_stacktrace', 'has_code', 'clarity_score']:
                if feature in cluster_data.columns:
                    char[f'avg_{feature}'] = pd.to_numeric(
                        cluster_data[feature], errors='coerce'
                    ).mean()
            
            # Tool performance
            for tool in ['boostnsift', 'buglocator', 'locus']:
                mrr_col = f'{tool}_mrr'
                if mrr_col in cluster_data.columns:
                    char[f'{tool}_avg_mrr'] = cluster_data[mrr_col].mean()
                    char[f'{tool}_top5_rate'] = (cluster_data.get(f'{tool}_top@5', 0) == 1).mean()
            
            cluster_characteristics.append(char)
        
        char_df = pd.DataFrame(cluster_characteristics)
        print(char_df.to_string(index=False))
        char_df.to_csv(f'{RESULTS_DIR}/bug_archetype_characteristics.csv', index=False)
        print(f"\n  Saved: {RESULTS_DIR}/bug_archetype_characteristics.csv")
        
        # Create dendrogram
        plt.figure(figsize=(20, 10))
        dendrogram(linkage_matrix, truncate_mode='level', p=5, show_leaf_counts=True)
        plt.title('Hierarchical Clustering Dendrogram (Ward Linkage)')
        plt.xlabel('Bug Index')
        plt.ylabel('Distance')
        plt.savefig(f'{RESULTS_DIR}/dendrogram.png', dpi=300, bbox_inches='tight')
        print(f"  Saved: {RESULTS_DIR}/dendrogram.png")
        
        return clusters, char_df
    
    return None, None

def main():
    """Main analysis pipeline"""
    print("="*80)
    print("EXPERIMENT A: FEATURE-PERFORMANCE RELATIONSHIP DISCOVERY")
    print("="*80)
    
    # Load data
    merged_df = load_and_prepare_data()
    
    # Get feature columns (exclude non-feature columns)
    exclude_cols = ['id', 'bug_id', 'embedding', 'cluster', 'archetype_cluster']
    feature_cols = [col for col in merged_df.columns 
                    if col not in exclude_cols 
                    and not col.startswith(('boostnsift_', 'buglocator_', 'locus_'))]
    
    # Remove embedding column if it exists (too high dimensional)
    feature_cols = [col for col in feature_cols if col != 'embedding']
    
    print(f"\nAnalyzing {len(feature_cols)} features")
    
    # 1. Univariate analysis
    correlation_results = univariate_feature_performance_analysis(merged_df, feature_cols)
    
    # Save correlation results
    for tool, tool_results in correlation_results.items():
        for metric, corr_df in tool_results.items():
            corr_df.to_csv(f'{RESULTS_DIR}/correlations_{tool}_{metric}.csv', index=False)
    
    # 2. Tool category comparison
    category_results = tool_category_comparison(merged_df, feature_cols)
    
    # 3. Feature importance with SHAP
    importance_results = feature_importance_shap(merged_df, feature_cols)
    
    # 4. Bug archetype clustering
    clusters, archetype_df = bug_archetype_clustering(merged_df, feature_cols)
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print(f"\nAll results saved to: {RESULTS_DIR}/")
    print("\nOutput files created:")
    print(f"  - {RESULTS_DIR}/correlation_heatmap_*.png")
    print(f"  - {RESULTS_DIR}/scatter_*.png")
    print(f"  - {RESULTS_DIR}/correlations_*.csv")
    print(f"  - {RESULTS_DIR}/tool_category_statistical_tests.csv")
    print(f"  - {RESULTS_DIR}/feature_importance_*_experiment_a.csv")
    if SHAP_AVAILABLE:
        print(f"  - {RESULTS_DIR}/shap_summary_*.png")
    print(f"  - {RESULTS_DIR}/bug_archetype_characteristics.csv")
    print(f"  - {RESULTS_DIR}/dendrogram.png")

if __name__ == "__main__":
    main()

