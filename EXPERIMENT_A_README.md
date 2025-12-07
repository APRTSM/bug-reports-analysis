# Experiment A: Feature-Performance Relationship Discovery

This script implements the experimental design from Section 4.2, analyzing which bug features predict tool detection success.

## Installation

Install additional dependencies:

```bash
pip install statsmodels shap
```

Optional but recommended:
- `statsmodels` - For Holm-Bonferroni multiple testing correction
- `shap` - For SHAP values (more interpretable than Gini importance)

The script will work without these, but with reduced functionality.

## Usage

```bash
python experiment_a_feature_performance_analysis.py
```

All results will be saved to the `results_experimentA/` folder, which will be created automatically if it doesn't exist.

## What It Does

### 1. Univariate Feature-Performance Analysis

- Computes **Spearman (ρ)** and **Pearson (r)** correlations between each feature and tool performance metrics
- Applies **Holm-Bonferroni correction** for multiple testing (α=0.05)
- Creates:
  - Correlation heatmaps
  - Scatter plots with LOWESS curves for top-5 correlated features per tool

**Output:**
- `results_experimentA/correlation_heatmap_mrr.png` - Heatmap of correlations
- `results_experimentA/scatter_{tool}_mrr.png` - Scatter plots for each tool
- `results_experimentA/correlations_{tool}_{metric}.csv` - Detailed correlation results

### 2. Tool Category Comparison

- Groups tools into categories (IR-based vs Other)
- Stratifies bugs into **quartiles** by feature value
- Computes per-quartile performance (MRR)
- Statistical testing:
  - **Wilcoxon signed-rank tests** comparing categories
  - **Cliff's delta** effect sizes

**Output:**
- `results_experimentA/tool_category_statistical_tests.csv` - Statistical test results

### 3. Feature Importance with SHAP

- Trains **Random Forest classifiers** per tool (binary: Top-5 success)
- Extracts **SHAP values** to quantify feature contributions
- Ranks features by:
  - Average importance across tools
  - Tool-specific importance

**Output:**
- `results_experimentA/feature_importance_{tool}_experiment_a.csv` - Feature rankings
- `results_experimentA/shap_summary_{tool}.png` - SHAP summary plots (if SHAP available)

### 4. Bug Archetype Discovery

- **Hierarchical clustering** with Ward linkage
- **Silhouette analysis** to determine optimal k
- Characterizes each cluster:
  - Feature distributions
  - Dominant tool per cluster
  - Performance variance

**Output:**
- `results_experimentA/bug_archetype_characteristics.csv` - Cluster characteristics
- `results_experimentA/dendrogram.png` - Clustering dendrogram

## Key Differences from Previous Analysis

1. **Statistical Rigor:**
   - Multiple testing correction (Holm-Bonferroni)
   - Effect sizes (Cliff's delta)
   - Non-parametric tests (Wilcoxon)

2. **SHAP Values:**
   - More interpretable than Gini importance
   - Shows feature contributions per prediction
   - Better for understanding model behavior

3. **Quartile Analysis:**
   - Stratifies bugs by feature value
   - Reveals non-linear relationships
   - Shows how performance changes across feature ranges

4. **Hierarchical Clustering:**
   - Ward linkage (better for continuous features)
   - Silhouette analysis for optimal k
   - More interpretable than K-means

## Interpreting Results

### Correlation Analysis

- **Significant correlations** (after correction) indicate predictive features
- **Spearman ρ** captures monotonic relationships (non-linear)
- **Pearson r** captures linear relationships

### Tool Category Comparison

- **Cliff's delta** interpretation:
  - |δ| < 0.147: negligible
  - |δ| < 0.33: small
  - |δ| < 0.474: medium
  - |δ| ≥ 0.474: large

- **Quartile analysis** shows which features cause performance divergence

### SHAP Values

- **Positive SHAP** = feature increases probability of detection
- **Negative SHAP** = feature decreases probability
- **Magnitude** = strength of effect

### Bug Archetypes

- Each cluster represents a distinct bug type
- Compare feature profiles across clusters
- Identify which tools work best for each archetype

## Example Output

```
boostnsift - MRR - Top 5 significant correlations:
         feature  spearman_rho  spearman_p
  summary_chars          0.342       0.001
description_chars          0.298       0.002
       n_words          0.275       0.003
    smog_index          0.231       0.005
  has_stacktrace          0.189       0.012
```

## Troubleshooting

### Missing Dependencies

If `statsmodels` is not available:
- Script will use simple Bonferroni correction instead
- Results will be slightly less conservative

If `shap` is not available:
- Script will use Gini importance only
- Feature importance still computed, just less interpretable

### Memory Issues

For large datasets:
- Reduce number of features analyzed
- Sample data for SHAP computation (already done in code)
- Use fewer clusters in silhouette analysis

## References

- Holm-Bonferroni correction: https://en.wikipedia.org/wiki/Holm%E2%80%93Bonferroni_method
- Cliff's delta: https://en.wikipedia.org/wiki/Effect_size#Cliff's_delta
- SHAP values: https://shap.readthedocs.io/
- Hierarchical clustering: https://scikit-learn.org/stable/modules/clustering.html#hierarchical-clustering

