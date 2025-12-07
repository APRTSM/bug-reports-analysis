# Bug Reports Analysis

This repository contains tools for extracting comprehensive text features from bug reports and analyzing which tools can detect which bugs based on those features.

## Overview

The project includes:
1. **Feature Extraction**: Extract ~70 features from bug reports (text statistics, readability, coherence, semantic similarity)
2. **Tool Detection Clustering Analysis**: Understand which bug features predict tool detection success

---

## Part 1: Feature Extraction

The `BugReportFeatureExtractor` class extracts the following feature categories:

1. **TextDescriptives Features** (~48 features): Descriptive statistics, readability scores, POS proportions, and dependency distances
2. **Coherence Momentum** (1 feature): Semantic consistency across sentences
3. **Outlier Score** (1 feature): How atypical the bug report is compared to others
4. **Semantic Similarity** (3 features): BERTScore similarity to demonstration examples (max, mean, min)

**Total: ~48-70 features per bug report** (depending on available metrics)

### Installation

```bash
# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download SpaCy language model
python -m spacy download en_core_web_sm
```

### Usage

```bash
# Process Defects4J dataset (default)
python bug_report_feature_extractor.py

# Process GHRB dataset
python bug_report_feature_extractor.py --dataset GHRB

# Process both datasets
python bug_report_feature_extractor.py --dataset both
```

### Output

Generates `outputs/all_bug_report_features.csv` with all extracted features.

---

## Part 2: Tool Detection Clustering Analysis

This analysis helps you understand:
- **Which bug features predict tool detection success**
- **What types of bugs each tool handles best**
- **How to improve bug reports for better tool detection**

### Quick Start

```bash
# Install required packages
pip install pandas numpy scikit-learn matplotlib seaborn scipy

# Run basic analysis
python tool_detection_clustering_analysis.py

# Run advanced analysis (correlations, category analysis, etc.)
python advanced_clustering_analysis.py
```

### What the Analysis Does

1. **Data Integration**: Merges bug features, categorizations, quality ratings, and tool detection results
2. **Clustering**: Groups bugs into clusters based on feature similarity using K-means
3. **Feature Importance**: Uses Random Forest to identify which features predict tool detection
4. **Visualization**: Creates 2D PCA visualizations showing clusters and detection patterns

### Output Files

**Basic Analysis:**
- `tool_detection_clustering.png` - Visual overview
- `cluster_analysis_results.csv` - Cluster statistics
- `feature_importance_*.csv` - Feature rankings per tool

**Advanced Analysis:**
- `advanced_clustering_analysis.png` - Comprehensive visualization
- `correlations_*.csv` - Feature correlations with detection
- `category_detection_*.csv` - Detection rates by bug category
- `feature_category_importance_*.csv` - Category-level insights

### Interpreting Results

#### High Detection Clusters
- Bugs with features that tools can easily match
- Well-documented, specific bugs
- Good candidates for automated detection

#### Low Detection Clusters
- Bugs that tools struggle with
- May have vague descriptions, missing context, or ambiguous issues

#### Feature Importance
- Top features indicate what tools rely on
- Can guide bug report improvement
- Helps understand tool limitations

### Key Questions Answered

1. **Which bug features correlate with tool detection?**
   - See feature importance rankings in `feature_importance_*.csv`

2. **Are there distinct groups of bugs that tools handle differently?**
   - See cluster analysis results in `cluster_analysis_results.csv`

3. **Which tools work best for which types of bugs?**
   - See detection rates by cluster and tool

4. **What makes a bug hard to detect?**
   - Compare features of low-detection vs high-detection clusters

### Example Workflow

```bash
# 1. Run basic analysis
python tool_detection_clustering_analysis.py

# 2. Review cluster_analysis_results.csv
# Find: Cluster 2 has 90% detection rate, Cluster 4 has 20%

# 3. Check feature_importance_any_tool.csv
# Find: Top features are clarity, specificity, has_stacktrace

# 4. Run advanced analysis
python advanced_clustering_analysis.py

# 5. Review correlations_any_tool.csv
# Find: clarity correlates 0.45 with detection (p < 0.001)

# 6. Action: Update bug report template to emphasize clarity
```

### Actionable Insights

**For Bug Report Writers:**
- Increase clarity and specificity (if these correlate with detection)
- Include stack traces (if `has_stacktrace` is important)
- Add code examples (if `has_code` helps)
- Improve readability (if readability metrics matter)

**For Tool Selection:**
- Check tool-specific feature importance files
- Match tools to bug characteristics
- Use `category_detection_*.csv` to see which tool works best for your bug category

**For Tool Developers:**
- Identify gaps: Clusters with low detection = opportunities
- Feature engineering: Focus on features that correlate with detection

---

## Methodology

### Data Pipeline

```
bug_features_v2.csv          ┐
gemini_bug_categorization.csv├─→ Merge on bug_id ─→ Feature Matrix
gemini_bug_ratings.csv       │
tool_comparison_summary.csv  ┘
```

### Feature Categories

**Text Features:**
- Length metrics: `summary_chars`, `description_chars`, `n_tokens`, `n_words`, `n_sentences`
- Readability: `flesch_reading_ease`, `smog_index`, `gunning_fog`
- Structure: `has_stacktrace`, `has_code`, `has_patch`, `num_steps`

**Linguistic Features:**
- Causal markers: `num_causal_markers`, `causal_density`
- Temporal markers: `num_temporal_markers`, `temporal_density`
- Verb analysis: `behavior_verb_count`, `behavior_verb_density`
- Modal verbs: `num_modal_verbs`, `modal_density`

**Quality Metrics:**
- `actionability`, `clarity`, `specificity`
- `technical_depth`, `repair_difficulty`
- `completeness_score`, `specificity_score`

### Clustering Approach

- **K-Means Clustering**: Groups bugs into k clusters based on feature similarity
- **Standardization**: Features are standardized before clustering
- **Analysis**: Detection rates per cluster, average feature values, category distribution

### Feature Importance Analysis

- **Random Forest Classifier**: Predicts tool detection from features
- **Feature Importance**: Ranks features by how much they predict detection
- **Handles**: Non-linear relationships and many features

### Statistical Methods

- **Pearson Correlation**: Measures linear relationships between features and detection
- **Hypothesis Testing**: Tests statistical significance (α = 0.05)
- **Cross-Validation**: K-fold cross-validation for robust estimates

---

## Common Questions

**Q: Which tool should I use?**
A: Check `category_detection_*.csv` to see which tool works best for your bug category.

**Q: Why aren't my bugs being detected?**
A: Check `cluster_analysis_results.csv` - if your bugs are in low-detection clusters, improve the features that matter (see feature importance).

**Q: How many clusters should I use?**
A: Start with 5. You can adjust in the code: `perform_clustering(X, n_clusters=10)`

**Q: Can I use this for prediction?**
A: Yes! The Random Forest models in the code can predict detection probability.

---

## Troubleshooting

### Missing Data
- The script handles missing values by filling with median
- Check for columns with too many missing values

### Imbalanced Classes
- Some tools may detect very few bugs
- Consider class weighting in models

### TextDescriptives Errors
1. Update TextDescriptives:
   ```bash
   pip install --upgrade textdescriptives
   ```
2. Run the diagnostic script:
   ```bash
   python textdescriptives_diagnostic.py
   ```

### Model Loading Issues
The script may take a few minutes to initialize on first run as it downloads models. Ensure you have sufficient disk space and a stable internet connection.

---

## Repository Structure

```
bug-reports-analysis/
├── bug_report_feature_extractor.py      # Feature extraction script
├── tool_detection_clustering_analysis.py # Basic clustering analysis
├── advanced_clustering_analysis.py       # Advanced analysis
├── textdescriptives_diagnostic.py       # Diagnostic tool
├── requirements.txt                      # Python dependencies
├── bug_reports/                          # Bug report datasets
│   ├── Defects4J/                       # 835 JSON bug reports
│   └── GHRB/                            # 97 JSON bug reports
└── outputs/                              # Output directory
```

---

## Performance

- **Feature Extraction**: ~1-5 seconds per bug report
- **Full Dataset Processing**: ~60-120 minutes for all 932 bug reports
- **Memory Usage**: ~2-4 GB (due to model loading)
- **Clustering Analysis**: ~1-5 minutes depending on dataset size

---

## License

© APRTSM Lab — Bilkent University.
Distributed under the APRTSM Lab Research License (non-commercial use only).

---

## Acknowledgments

- [Defects4J](https://github.com/rjust/defects4j) dataset
- [GHRB](https://github.com/soarsmu/GHRB) dataset
- [TextDescriptives](https://github.com/hlasse/TextDescriptives) library
- [BERTScore](https://github.com/Tiiiger/bert_score) for semantic similarity
- [Cleanlab](https://github.com/cleanlab/cleanlab) for outlier detection
- scikit-learn for clustering and machine learning

---

## References

- Scikit-learn documentation: https://scikit-learn.org/
- Clustering algorithms: https://scikit-learn.org/stable/modules/clustering.html
- Feature importance: https://scikit-learn.org/stable/modules/permutation_importance.html
