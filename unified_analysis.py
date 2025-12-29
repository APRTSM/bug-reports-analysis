"""
Unified Analysis Script for Bug Report Feature Analysis - ENHANCED FOR OUTLIER DETECTION

This script combines five analysis types with focus on tool-specific outlier effects:
1. Correlation Analysis: Spearman correlations with standardized metrics
2. Success/Failure Analysis: Improved label definitions and robustness checks
3. Outlier Analysis: NEW - Unique success, extreme advantages, and outlier visualization
4. Clustered Heatmaps: Feature-clustered heatmaps from correlation results
5. Venn/UpSet Diagrams: Tool intersection analysis

ENHANCEMENTS FOR OUTLIER DETECTION:
- Added unique success analysis (when only one tool succeeds)
- Added extreme advantage detection (>2 SD outliers)
- Added continuous advantage analysis preserving magnitude
- Added outlier visualization for top features
- Added missingness reporting for transparency
- Added feature subgroup analysis (low/med/high bins)
"""

import itertools
import re
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.stats import spearmanr, mannwhitneyu, pointbiserialr
from statsmodels.stats.multitest import multipletests
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore', category=RuntimeWarning)

# ======================================
# CONFIGURATION
# ======================================
DATA_DIR = Path(".")

# Input files
# Choose which dataset to use for analysis:
# - IN_FILE_PREPROCESSED: Has standardized features + derived features (text stats, one-hot encodings, etc.)
# - IN_FILE_FULL: Has raw (unscaled) features only (fewer columns, but no standardization)
# Note: For rank-based analyses (Spearman, Mann-Whitney U, Cliff's Delta), both work identically
#       since these methods are scale-invariant. Use IN_FILE_FULL if you want raw values.
USE_RAW_DATASET = True  # Set to True to use experimentA_full_dataset.csv instead of preprocessed
IN_FILE_PREPROCESSED = DATA_DIR / "full_feature_preproccessed/experimentA_preprocessed_rich.csv"
IN_FILE_FULL = DATA_DIR / "full_feature_preproccessed/experimentA_full_dataset.csv"  # For raw feature statistics
IN_FILE_TOOL_COMPARISON = DATA_DIR / "tool_comparison_summary.csv"

# Select the input file based on configuration
IN_FILE = IN_FILE_FULL if USE_RAW_DATASET else IN_FILE_PREPROCESSED

# Output directories
OUT_DIR_CORR = DATA_DIR / "experimentA_gap_corr_results_unprocessed"
OUT_DIR_SUCCESS = DATA_DIR / "experimentA_success_failure_unprocessed"
OUT_DIR_OUTLIER = DATA_DIR / "experimentA_outlier_analysis_unprocessed"  # NEW
OUT_DIR_CLUSTERED = DATA_DIR / "clustered_heatmaps_gap_unprocessed"
OUT_DIR_VENN = DATA_DIR / "tool_intersections"
OUT_DIR_DIAGNOSTICS = DATA_DIR / "analysis_diagnostics"

# Analysis flags
RUN_CORRELATION_ANALYSIS = True
RUN_SUCCESS_FAILURE_ANALYSIS = True
RUN_OUTLIER_ANALYSIS = True  # NEW: Unique success and extreme advantage analysis
RUN_CLUSTERED_HEATMAPS = True
RUN_VENN_DIAGRAMS = True
RUN_INTERSECTION_FEATURE_ANALYSIS = True  # NEW: Feature analysis based on tool intersections
RUN_UNIQUE_TOOL_SUCCESS_ANALYSIS = True  # NEW: Analysis of bugs uniquely found by each tool
RUN_DIAGNOSTICS = True

# Shared settings
ALPHA = 0.05
INCLUDE_MISSINGNESS_FEATURES = True
PRACTICAL_SIG_CORR = 0.2
PRACTICAL_SIG_DELTA = 0.2

# Correlation Analysis settings
BASE_METRIC_PREFIXES = ["mrr"]
TOP_N_FEATURES_HEATMAP = 15
STANDARDIZE_METRICS = True
STRATIFY_BY_PROJECT = False
BOOTSTRAP_CI = True
N_BOOTSTRAP = 100

# Success/Failure Analysis settings
BASE_PREFIX = "mrr"
LABEL_MODE = "stratified"
RANK_THRESHOLD_HIGH = 5
RANK_THRESHOLD_LOW = 6
ADV_EPS = 1e-12
TOP_K_PLOTS = 6
MIN_GROUP_N = 8
SUCCESS_HEATMAP_TOP_N = 30          # NEW: Number of features in success/failure heatmap
SUCCESS_HEATMAP_MIN_TOOLS = 2       # NEW: Min tools a feature must affect to include

# NEW: Outlier Analysis settings
UNIQUE_SUCCESS_THRESHOLD = 0.1  # MRR threshold for "success"
EXTREME_ADV_Z_THRESHOLD = 2.0   # Z-score threshold for extreme advantages
TOP_N_OUTLIER_FEATURES = 10     # Number of features to visualize
N_FEATURE_BINS = 3              # Low/medium/high bins for subgroup analysis

# Clustered Heatmaps settings
MAX_ROWS_PER_CLUSTER = 25
TOP_BOTTOM_SPLIT = True
SELECTION_MODE = "variance"
IN_CORR_FILE = OUT_DIR_CORR / "gap_corr_spearman.csv"

# Venn Diagrams settings
FOUND_DEF = "rank"

# ======================================
# SHARED HELPER FUNCTIONS
# ======================================

def safe_name(s: str) -> str:
    """Convert string to safe filename."""
    # Convert to string if not already
    if not isinstance(s, str):
        s = str(s)
    s = re.sub(r"[^a-zA-Z0-9_]+", "_", s)
    return s.strip("_").lower()

def get_tools(df, prefix):
    """Extract tool names from columns with given prefix."""
    cols = [c for c in df.columns if c.startswith(prefix + "_")]
    # Filter out missingness indicators (columns ending with __is_missing or _is_missing)
    cols = [c for c in cols if not (c.endswith("__is_missing") or c.endswith("_is_missing"))]
    tools = sorted({c.split("_", 1)[1] for c in cols})
    return tools

def tools_for_prefix(df, prefix: str):
    """Get tools and columns for a given prefix."""
    cols = [c for c in df.columns if c.startswith(prefix + "_")]
    # Filter out missingness indicators (columns ending with __is_missing or _is_missing)
    cols = [c for c in cols if not (c.endswith("__is_missing") or c.endswith("_is_missing"))]
    tools = sorted({c.split("_", 1)[1] for c in cols})
    return tools, cols

def apply_holm(df_corr, alpha=0.05, pval_col="pval"):
    """Apply Holm-Bonferroni correction for multiple testing."""
    df_corr = df_corr.copy()
    mask = df_corr[pval_col].notna()
    pvals = df_corr.loc[mask, pval_col].values
    if len(pvals) == 0:
        df_corr["pval_adj"] = np.nan
        df_corr["reject"] = False
        return df_corr

    reject, p_adj, _, _ = multipletests(pvals, alpha=alpha, method="holm")
    df_corr.loc[mask, "pval_adj"] = p_adj
    df_corr.loc[mask, "reject"] = reject
    df_corr["reject"] = df_corr["reject"].fillna(False).astype(bool)
    return df_corr

def load_data_and_features(in_file):
    """Load data and extract feature columns."""
    df = pd.read_csv(in_file)
    print(f"Loaded: {df.shape} from {in_file}")

    id_cols = [c for c in ["project", "bug_id", "id"] if c in df.columns]
    
    # FIXED: Exclude rank columns along with other performance metrics
    perf_cols = [c for c in df.columns if c.startswith("mrr_") or c.startswith("top@") or c.startswith("rank_")]
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [c for c in numeric_cols if c not in perf_cols + id_cols]

    if not INCLUDE_MISSINGNESS_FEATURES:
        feature_cols = [c for c in feature_cols if not (c.endswith("_is_missing") or "__is_missing" in c)]
    
    # Exclude redundant columns that are identical to other features
    redundant_features = ['fine_grained_description_length']  # Identical to description_length
    feature_cols = [c for c in feature_cols if c not in redundant_features]

    # Sanity check: verify no performance metrics in features
    suspicious = [f for f in feature_cols if any(x in f.lower() for x in ['rank_', 'mrr_', 'top@'])]
    if suspicious:
        print(f"[WARNING] Performance metrics found in feature list: {suspicious}")
        print("These will be removed from features.")
        feature_cols = [f for f in feature_cols if f not in suspicious]
    
    print(f"Feature columns: {len(feature_cols)}")
    print(f"Performance columns (excluded): {len(perf_cols)}")
    print(f"Sample features: {feature_cols[:10] if len(feature_cols) >= 10 else feature_cols}")
    
    return df, feature_cols, id_cols, perf_cols

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

# ======================================
# NEW: ENHANCED DIAGNOSTIC FUNCTIONS
# ======================================

def check_metric_distributions(df, perf_cols, out_dir):
    """Create diagnostic plots for metric distributions."""
    print("\n--- Diagnostic: Metric Distributions ---")
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for idx, col in enumerate(perf_cols[:6]):
        if idx >= len(axes):
            break
        ax = axes[idx]
        data = df[col].dropna()
        ax.hist(data, bins=30, alpha=0.7, edgecolor='black')
        ax.set_title(f"{col}\nMean={data.mean():.3f}, Std={data.std():.3f}")
        ax.set_xlabel("Value")
        ax.set_ylabel("Frequency")
        ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(out_dir / "metric_distributions.png", dpi=300)
    plt.close()
    print(f"Saved: {out_dir / 'metric_distributions.png'}")

def check_feature_correlations(df, feature_cols, out_dir, max_features=50):
    """Check for highly correlated features (multicollinearity)."""
    print("\n--- Diagnostic: Feature Correlations ---")
    
    if len(feature_cols) > max_features:
        feature_cols = np.random.choice(feature_cols, max_features, replace=False).tolist()
    
    corr_matrix = df[feature_cols].corr().abs()
    
    high_corr_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            if corr_matrix.iloc[i, j] > 0.8:
                high_corr_pairs.append({
                    'feature1': corr_matrix.columns[i],
                    'feature2': corr_matrix.columns[j],
                    'correlation': corr_matrix.iloc[i, j]
                })
    
    if high_corr_pairs:
        high_corr_df = pd.DataFrame(high_corr_pairs)
        high_corr_df.to_csv(out_dir / "high_feature_correlations.csv", index=False)
        print(f"Found {len(high_corr_pairs)} highly correlated feature pairs (|r| > 0.8)")
        print(f"Saved: {out_dir / 'high_feature_correlations.csv'}")
    else:
        print("No highly correlated feature pairs found.")

def check_feature_missingness(df, feature_cols, out_dir):
    """NEW: Report missingness patterns for transparency."""
    print("\n--- Diagnostic: Feature Missingness ---")
    
    miss_stats = []
    for feat in feature_cols:
        miss_pct = df[feat].isna().mean()
        if miss_pct > 0:
            miss_stats.append({
                'feature': feat,
                'missing_pct': miss_pct,
                'n_valid': df[feat].notna().sum(),
                'n_missing': df[feat].isna().sum()
            })
    
    if miss_stats:
        miss_df = pd.DataFrame(miss_stats).sort_values('missing_pct', ascending=False)
        miss_df.to_csv(out_dir / "feature_missingness.csv", index=False)
        
        high_miss = miss_df[miss_df['missing_pct'] > 0.2]
        very_high_miss = miss_df[miss_df['missing_pct'] > 0.5]
        
        print(f"Total features with any missingness: {len(miss_df)}")
        print(f"Features with >20% missing: {len(high_miss)}")
        print(f"Features with >50% missing: {len(very_high_miss)}")
        
        if len(very_high_miss) > 0:
            print("\nFeatures with >50% missingness:")
            print(very_high_miss[['feature', 'missing_pct']].to_string(index=False))
        
        print(f"Saved: {out_dir / 'feature_missingness.csv'}")
    else:
        print("No missing values found in any feature.")

def check_project_effects(df, perf_cols, out_dir):
    """Check if project has strong effects on performance."""
    print("\n--- Diagnostic: Project Effects ---")
    
    if 'project' not in df.columns:
        print("No 'project' column found. Skipping project effects check.")
        return
    
    project_stats = []
    for col in perf_cols[:6]:
        for project in df['project'].unique():
            proj_data = df[df['project'] == project][col].dropna()
            if len(proj_data) > 0:
                project_stats.append({
                    'metric': col,
                    'project': project,
                    'mean': proj_data.mean(),
                    'std': proj_data.std(),
                    'n': len(proj_data)
                })
    
    if project_stats:
        stats_df = pd.DataFrame(project_stats)
        stats_df.to_csv(out_dir / "project_effects.csv", index=False)
        print(f"Saved: {out_dir / 'project_effects.csv'}")
        
        for col in perf_cols[:6]:
            subset = stats_df[stats_df['metric'] == col]
            if len(subset) > 1:
                between_var = subset['mean'].var()
                within_var = (subset['std']**2 * subset['n']).sum() / subset['n'].sum()
                ratio = between_var / (between_var + within_var) if (between_var + within_var) > 0 else 0
                print(f"{col}: Between-project variance ratio = {ratio:.3f}")

def _compute_text_stats_feature(df_raw, feat_name):
    """
    Compute a single text stats feature from raw text column.
    Handles txt_* features that are derived from text columns.
    """
    # Extract the source column name from feature name
    # e.g., "txt_fine_grained_reasoning_char_len" -> "fine_grained_reasoning"
    if not feat_name.startswith("txt_"):
        return None
    
    parts = feat_name.replace("txt_", "").split("_")
    # Find the source text column by matching prefix
    # Try common text column names
    text_cols = ["title", "description", "reasoning", "reasoning_ratings", 
                 "fine_grained_reasoning", "fine_grained_title", 
                 "likely_impacted_code_concepts"]
    
    source_col = None
    for col in text_cols:
        if feat_name.startswith(f"txt_{col}_"):
            source_col = col
            break
    
    if source_col is None or source_col not in df_raw.columns:
        return None
    
    # Extract the stat type from feature name
    stat_type = feat_name.replace(f"txt_{source_col}_", "")
    
    # Compute the feature using the same logic as text_stats
    s = df_raw[source_col].fillna("").astype(str)
    _word_re = re.compile(r"[A-Za-z0-9_]+")
    
    if stat_type == "char_len":
        return s.str.len()
    elif stat_type == "word_count":
        return s.apply(lambda t: len(_word_re.findall(t)))
    elif stat_type == "line_count":
        return s.apply(lambda t: t.count("\n") + (1 if t else 0))
    elif stat_type == "avg_word_len":
        char_len = s.str.len()
        word_counts = s.apply(lambda t: len(_word_re.findall(t)))
        return char_len / word_counts.replace(0, np.nan)
    elif stat_type == "avg_words_per_line":
        word_counts = s.apply(lambda t: len(_word_re.findall(t)))
        line_counts = s.apply(lambda t: t.count("\n") + (1 if t else 0))
        return word_counts / line_counts.replace(0, np.nan)
    elif stat_type == "avg_sentence_len":
        def avg_sent_len(t: str) -> float:
            sentences = re.split(r'[.!?]+', t)
            sentences = [sent.strip() for sent in sentences if sent.strip()]
            if not sentences:
                return np.nan
            return len(_word_re.findall(t)) / len(sentences)
        return s.apply(avg_sent_len)
    elif stat_type == "hedge_count":
        lower = s.str.lower()
        hedge_markers = ["maybe", "might", "likely", "possibly", "unclear", "unsure",
                        "seems", "appears", "probably", "could", "cannot", "can't",
                        "unknown", "approximately", "guess"]
        return sum(lower.str.count(re.escape(m)) for m in hedge_markers)
    elif stat_type == "hedge_density":
        lower = s.str.lower()
        hedge_markers = ["maybe", "might", "likely", "possibly", "unclear", "unsure",
                        "seems", "appears", "probably", "could", "cannot", "can't",
                        "unknown", "approximately", "guess"]
        hedge_count = sum(lower.str.count(re.escape(m)) for m in hedge_markers)
        word_counts = s.apply(lambda t: len(_word_re.findall(t)))
        return hedge_count / word_counts.replace(0, np.nan)
    elif stat_type == "has_code_like":
        lower = s.str.lower()
        return lower.str.contains(
            r"\b(stack trace|exception|nullpointer|assert|traceback|line \d+)\b", 
            regex=True
        ).astype(int)
    elif stat_type == "question_density":
        word_counts = s.apply(lambda t: len(_word_re.findall(t)))
        return s.str.count(r"\?") / word_counts.replace(0, np.nan)
    elif stat_type == "exclaim_density":
        word_counts = s.apply(lambda t: len(_word_re.findall(t)))
        return s.str.count(r"!") / word_counts.replace(0, np.nan)
    elif stat_type == "digit_density":
        char_len = s.str.len()
        return s.str.count(r"\d") / char_len.replace(0, np.nan)
    elif stat_type == "uniq_word_ratio":
        def uniq_ratio(t: str) -> float:
            toks = _word_re.findall(t.lower())
            if not toks:
                return np.nan
            return len(set(toks)) / len(toks)
        return s.apply(uniq_ratio)
    elif stat_type == "is_missing":
        return df_raw[source_col].isna().astype(int)
    else:
        return None

def generate_feature_summary_table(df, feature_cols, out_dir, use_raw_values=True):
    """
    Generate a summary table with min, max, mean, and median for each feature.
    
    Args:
        df: DataFrame with features (may be preprocessed/scaled)
        feature_cols: List of feature column names
        out_dir: Output directory for saving the table
        use_raw_values: If True, load raw values from full dataset instead of scaled values
    """
    print("\n" + "=" * 60)
    print("GENERATING FEATURE SUMMARY TABLE")
    print("=" * 60)
    
    # If use_raw_values is True, load the full dataset for raw feature values
    if use_raw_values and IN_FILE_FULL.exists():
        print(f"Loading raw feature values from: {IN_FILE_FULL}")
        df_raw = pd.read_csv(IN_FILE_FULL)
        print(f"  Loaded {df_raw.shape} rows")
        # Use raw dataset for statistics, but keep same feature column names
        df_for_stats = df_raw.copy()
        
        # Compute missing txt_* features from raw text columns
        missing_txt_features = [f for f in feature_cols if f.startswith("txt_") and f not in df_for_stats.columns]
        if missing_txt_features:
            print(f"  Computing {len(missing_txt_features)} text-derived features from raw text columns...")
            computed_count = 0
            for feat in missing_txt_features:
                computed = _compute_text_stats_feature(df_raw, feat)
                if computed is not None:
                    df_for_stats[feat] = computed
                    computed_count += 1
                    if computed_count <= 5:  # Print first 5
                        print(f"    ✓ Computed {feat} (min={computed.min():.2f}, max={computed.max():.2f})")
            print(f"  Successfully computed {computed_count}/{len(missing_txt_features)} text features")
    else:
        df_for_stats = df
    
    summary_records = []
    
    for feat in feature_cols:
        # Check if feature exists in the dataset we're using for statistics
        if feat not in df_for_stats.columns:
            # Try to find it in the original df (for missingness info)
            if feat not in df.columns:
                continue
            # Feature exists in df but not in df_for_stats - use df for stats
            # This means it's a derived feature that couldn't be computed from raw data
            # Use the preprocessed version but warn
            print(f"  [WARN] Using preprocessed values for {feat} (may be standardized)")
            values = df[feat].dropna()
        else:
            values = df_for_stats[feat].dropna()
        
        # For missingness, use the original df
        n_missing = df[feat].isna().sum() if feat in df.columns else len(df_for_stats) - len(values)
        
        if len(values) == 0:
            summary_records.append({
                'feature': feat,
                'min': np.nan,
                'max': np.nan,
                'mean': np.nan,
                'median': np.nan,
                'std': np.nan,
                'scale': "N/A (all missing)",
                'n_valid': 0,
                'n_missing': len(df),
                'missing_pct': 100.0
            })
            continue
        
        min_val = float(values.min())
        max_val = float(values.max())
        
        # Determine scale description
        if min_val == 0.0 and max_val == 1.0 and len(values.unique()) <= 2:
            scale = "0-1 (binary)"
        elif min_val == 0.0 and max_val == 0.0:
            scale = f"constant ({min_val:.3g})"
        elif min_val >= 0.0 and max_val <= 1.0 and len(values.unique()) <= 2:
            scale = "0-1 (binary)"
        elif isinstance(min_val, (int, np.integer)) and isinstance(max_val, (int, np.integer)) and min_val >= 0:
            # Integer count/scale
            if max_val <= 10:
                scale = f"0-{int(max_val)} (integer)"
            else:
                scale = f"{int(min_val)}-{int(max_val)} (integer)"
        else:
            # Continuous scale - show range
            if abs(min_val) < 0.001 and abs(max_val) < 0.001:
                scale = f"{min_val:.3e}-{max_val:.3e}"
            else:
                # Format based on magnitude
                if max_val < 1.0:
                    scale = f"{min_val:.3f}-{max_val:.3f}"
                elif max_val < 1000:
                    scale = f"{min_val:.1f}-{max_val:.1f}"
                else:
                    scale = f"{min_val:.0f}-{max_val:.0f}"
        
        summary_records.append({
            'feature': feat,
            'min': min_val,
            'max': max_val,
            'mean': float(values.mean()),
            'median': float(values.median()),
            'std': float(values.std()),
            'scale': scale,
            'n_valid': int(len(values)),
            'n_missing': int(n_missing),
            'missing_pct': float(n_missing / len(df) * 100)
        })
    
    summary_df = pd.DataFrame(summary_records)
    summary_df = summary_df.sort_values('feature')
    
    # Save to CSV
    output_file = out_dir / "feature_summary_statistics.csv"
    summary_df.to_csv(output_file, index=False)
    print(f"Saved: {output_file}")
    print(f"  Total features: {len(summary_df)}")
    print(f"  Features with no missing data: {len(summary_df[summary_df['n_missing'] == 0])}")
    print(f"  Features with >50% missing: {len(summary_df[summary_df['missing_pct'] > 50])}")
    
    # Print sample
    print("\nSample of summary table (first 10 features):")
    print(summary_df.head(10).to_string(index=False))
    
    return summary_df

def check_feature_types(df, feature_cols, perf_cols, out_dir):
    """NEW: Verify that features don't include performance metrics."""
    print("\n--- Diagnostic: Feature Type Validation ---")
    
    # Check for performance metrics that slipped into features
    perf_in_features = [f for f in feature_cols if f in perf_cols]
    rank_in_features = [f for f in feature_cols if 'rank_' in f.lower()]
    mrr_in_features = [f for f in feature_cols if 'mrr_' in f.lower()]
    top_in_features = [f for f in feature_cols if 'top@' in f.lower()]
    
    issues = []
    
    if perf_in_features:
        issues.append(f"Performance metrics in features: {perf_in_features}")
    if rank_in_features:
        issues.append(f"Rank columns in features: {rank_in_features}")
    if mrr_in_features:
        issues.append(f"MRR columns in features: {mrr_in_features}")
    if top_in_features:
        issues.append(f"Top@ columns in features: {top_in_features}")
    
    if issues:
        print("[ERROR] Feature contamination detected:")
        for issue in issues:
            print(f"  - {issue}")
        
        report = {
            'issue_type': ['Performance in features', 'Rank in features', 'MRR in features', 'Top@ in features'],
            'count': [len(perf_in_features), len(rank_in_features), len(mrr_in_features), len(top_in_features)],
            'columns': [str(perf_in_features), str(rank_in_features), str(mrr_in_features), str(top_in_features)]
        }
        report_df = pd.DataFrame(report)
        report_df.to_csv(out_dir / "feature_contamination_report.csv", index=False)
        print(f"Saved: {out_dir / 'feature_contamination_report.csv'}")
        print("\n[ACTION REQUIRED] Fix feature extraction to exclude these columns.")
    else:
        print("✓ No performance metrics found in feature list")
        print(f"✓ {len(feature_cols)} valid features")
        print(f"✓ {len(perf_cols)} performance columns properly excluded")
    
    # Save feature manifest
    feature_manifest = pd.DataFrame({
        'feature': feature_cols,
        'type': ['feature'] * len(feature_cols)
    })
    feature_manifest.to_csv(out_dir / "feature_manifest.csv", index=False)
    print(f"Saved: {out_dir / 'feature_manifest.csv'}")

def run_diagnostic_checks(df, feature_cols, perf_cols):
    """Run all diagnostic checks."""
    print("\n" + "=" * 60)
    print("RUNNING DIAGNOSTIC CHECKS")
    print("=" * 60)
    
    OUT_DIR_DIAGNOSTICS.mkdir(exist_ok=True, parents=True)
    
    # Generate feature summary table first
    generate_feature_summary_table(df, feature_cols, OUT_DIR_DIAGNOSTICS)
    
    check_feature_types(df, feature_cols, perf_cols, OUT_DIR_DIAGNOSTICS)  # NEW: Run first!
    check_metric_distributions(df, perf_cols, OUT_DIR_DIAGNOSTICS)
    check_feature_correlations(df, feature_cols, OUT_DIR_DIAGNOSTICS)
    check_feature_missingness(df, feature_cols, OUT_DIR_DIAGNOSTICS)
    check_project_effects(df, perf_cols, OUT_DIR_DIAGNOSTICS)
    
    print(f"\nDiagnostics saved to: {OUT_DIR_DIAGNOSTICS}")

# ======================================
# 1. CORRELATION ANALYSIS (ENHANCED)
# ======================================

def shorten_target(t):
    """Shorten target names for display."""
    t = str(t)
    t = t.replace("boostnsift", "BNS")
    t = t.replace("buglocator", "BL")
    t = t.replace("locus", "LOC")
    t = t.replace("FlexFL", "FFL")
    t = t.replace("flexfl", "FFL")

    if t.startswith("adv_mrr_"):
        core = t.replace("adv_mrr_", "")
        if core.endswith("_is_missing"):
            core = core.replace("_is_missing", "")
            return f"Adv({core.upper()})*"
        return f"Adv({core.upper()})"

    if t.startswith("gap_mrr_"):
        core = t.replace("gap_mrr_", "")
        core = core.replace("_minus_", "−")
        return f"Δ({core.upper()})"
    
    # NEW: Handle unique success and extreme advantage
    if t.startswith("unique_success_"):
        tool = t.replace("unique_success_", "")
        return f"UniqSucc({tool.upper()})"
    
    if t.startswith("extreme_adv_"):
        tool = t.replace("extreme_adv_", "")
        return f"ExtrAdv({tool.upper()})"

    return t

def compute_spearman_table(df, feature_cols, target_cols):
    """Compute Spearman correlations with missingness tracking."""
    records = []
    for target in target_cols:
        y = df[target].to_numpy(dtype=float)
        for feat in feature_cols:
            x = df[feat].to_numpy(dtype=float)

            mask = ~np.isnan(x) & ~np.isnan(y)
            n_valid = mask.sum()
            missingness_pct = 1 - (n_valid / len(df))  # NEW
            
            if n_valid < 3:
                records.append({
                    "target": target, 
                    "feature": feat, 
                    "corr": np.nan, 
                    "pval": np.nan, 
                    "n": int(n_valid),
                    "missingness_pct": missingness_pct  # NEW
                })
                continue

            x_clean = x[mask]
            y_clean = y[mask]

            if np.nanstd(x_clean) == 0 or np.nanstd(y_clean) == 0:
                records.append({
                    "target": target, 
                    "feature": feat, 
                    "corr": np.nan, 
                    "pval": np.nan, 
                    "n": int(n_valid),
                    "missingness_pct": missingness_pct  # NEW
                })
                continue

            try:
                rho, p = spearmanr(x_clean, y_clean)
            except Exception:
                rho, p = np.nan, np.nan

            records.append({
                "target": target, 
                "feature": feat, 
                "corr": rho, 
                "pval": p, 
                "n": int(n_valid),
                "missingness_pct": missingness_pct  # NEW
            })

    return pd.DataFrame(records)

def compute_raw_mrr_correlations(df, feature_cols, tools, prefix="mrr", out_dir=None):
    """
    Compute Spearman correlations between features and raw MRR values.
    
    This captures both "did tool find bug?" (MRR=0 vs MRR>0) and 
    "how well ranked?" (MRR=0.1 vs MRR=1.0) in a single metric.
    """
    print("\n--- Computing Raw MRR Correlations ---")
    
    records = []
    for tool in tools:
        mrr_col = f"{prefix}_{tool}"
        if mrr_col not in df.columns:
            continue
            
        for feat in feature_cols:
            x = df[feat].to_numpy(dtype=float)
            y = df[mrr_col].to_numpy(dtype=float)
            
            mask = ~np.isnan(x) & ~np.isnan(y)
            n_valid = mask.sum()
            
            if n_valid < 10:
                continue
            
            x_clean = x[mask]
            y_clean = y[mask]
            
            if np.std(x_clean) == 0 or np.std(y_clean) == 0:
                continue
            
            try:
                rho, p = spearmanr(x_clean, y_clean)
            except:
                rho, p = np.nan, np.nan
            
            records.append({
                'tool': tool,
                'feature': feat,
                'corr': rho,
                'pval': p,
                'n': int(n_valid),
                'missingness_pct': 1 - (n_valid / len(df))
            })
    
    result_df = pd.DataFrame(records)
    
    if not result_df.empty:
        # Apply multiple testing correction
        from statsmodels.stats.multitest import multipletests
        mask = result_df['pval'].notna()
        if mask.sum() > 0:
            reject, p_adj, _, _ = multipletests(
                result_df.loc[mask, 'pval'], 
                alpha=0.05, 
                method='holm'
            )
            result_df.loc[mask, 'pval_adj'] = p_adj
            result_df.loc[mask, 'reject'] = reject
        
        result_df['practically_significant'] = result_df['corr'].abs() >= 0.2
        
        if out_dir:
            result_df.to_csv(out_dir / "raw_mrr_correlations.csv", index=False)
            print(f"Saved: {out_dir / 'raw_mrr_correlations.csv'}")
        
        # Summary statistics
        sig_results = result_df[result_df['reject'] & result_df['practically_significant']]
        print(f"\nSignificant correlations: {len(sig_results)}")
        print(f"Average |correlation| (significant): {sig_results['corr'].abs().mean():.3f}")
        print(f"Max |correlation|: {result_df['corr'].abs().max():.3f}")
    
    return result_df


def compute_two_part_analysis(df, feature_cols, tools, prefix="mrr", rank_prefix="rank", out_dir=None):
    """
    Two-part analysis separating "found vs not found" from "rank quality among found".
    
    Part 1: Point-biserial correlation with binary "found" indicator
    Part 2: Spearman correlation with ranks (only among found bugs)
    """
    print("\n--- Computing Two-Part Found/Rank Analysis ---")
    
    part1_records = []  # Found vs not found
    part2_records = []  # Rank quality among found
    
    for tool in tools:
        mrr_col = f"{prefix}_{tool}"
        rank_col = f"{rank_prefix}_{tool}"
        
        if mrr_col not in df.columns:
            continue
        
        # Create binary "found" indicator
        # MRR > 0 means bug was found (even if ranked poorly)
        found = (df[mrr_col] > 0).astype(int)
        n_found = found.sum()
        n_not_found = (found == 0).sum()
        
        print(f"\n{tool}: {n_found} found, {n_not_found} not found ({n_found/(n_found+n_not_found)*100:.1f}% found rate)")
        
        # Part 1: Point-biserial correlation (binary found vs continuous feature)
        for feat in feature_cols:
            x = df[feat].to_numpy(dtype=float)
            
            mask = ~np.isnan(x)
            if mask.sum() < 10:
                continue
            
            x_clean = x[mask]
            found_clean = found[mask]
            
            if np.std(x_clean) == 0 or len(np.unique(found_clean)) < 2:
                continue
            
            try:
                r, p = pointbiserialr(found_clean, x_clean)
            except:
                r, p = np.nan, np.nan
            
            part1_records.append({
                'tool': tool,
                'feature': feat,
                'corr': r,
                'pval': p,
                'n': int(mask.sum()),
                'n_found': int(found_clean.sum()),
                'n_not_found': int((found_clean == 0).sum()),
                'analysis': 'found_vs_notfound'
            })
        
        # Part 2: Spearman correlation with ranks (only among found bugs)
        if rank_col in df.columns:
            found_mask = df[rank_col].notna()
            
            for feat in feature_cols:
                # Only bugs where both feature and rank are available
                mask = found_mask & df[feat].notna()
                
                if mask.sum() < 10:
                    continue
                
                x_clean = df.loc[mask, feat].values
                rank_clean = df.loc[mask, rank_col].values
                
                if np.std(x_clean) == 0 or np.std(rank_clean) == 0:
                    continue
                
                try:
                    rho, p = spearmanr(x_clean, rank_clean)
                except:
                    rho, p = np.nan, np.nan
                
                part2_records.append({
                    'tool': tool,
                    'feature': feat,
                    'corr': rho,
                    'pval': p,
                    'n': int(mask.sum()),
                    'analysis': 'rank_quality_among_found'
                })
    
    # Create DataFrames
    part1_df = pd.DataFrame(part1_records)
    part2_df = pd.DataFrame(part2_records)
    
    # Apply multiple testing correction to each part separately
    for df_part, name in [(part1_df, 'part1'), (part2_df, 'part2')]:
        if df_part.empty:
            continue
        
        mask = df_part['pval'].notna()
        if mask.sum() > 0:
            from statsmodels.stats.multitest import multipletests
            reject, p_adj, _, _ = multipletests(
                df_part.loc[mask, 'pval'],
                alpha=0.05,
                method='holm'
            )
            df_part.loc[mask, 'pval_adj'] = p_adj
            df_part.loc[mask, 'reject'] = reject
        
        df_part['practically_significant'] = df_part['corr'].abs() >= 0.2
    
    # Save results
    if out_dir:
        if not part1_df.empty:
            part1_df.to_csv(out_dir / "two_part_found_correlations.csv", index=False)
            print(f"\nSaved: {out_dir / 'two_part_found_correlations.csv'}")
            
            sig1 = part1_df[part1_df['reject'] & part1_df['practically_significant']]
            print(f"Part 1 - Significant 'found' correlations: {len(sig1)}")
            if len(sig1) > 0:
                print(f"  Average |r|: {sig1['corr'].abs().mean():.3f}")
                print(f"  Top feature: {sig1.nlargest(1, 'corr')['feature'].values[0]}")
        
        if not part2_df.empty:
            part2_df.to_csv(out_dir / "two_part_rank_correlations.csv", index=False)
            print(f"Saved: {out_dir / 'two_part_rank_correlations.csv'}")
            
            sig2 = part2_df[part2_df['reject'] & part2_df['practically_significant']]
            print(f"Part 2 - Significant rank correlations (among found): {len(sig2)}")
            if len(sig2) > 0:
                print(f"  Average |ρ|: {sig2['corr'].abs().mean():.3f}")
    
    return part1_df, part2_df

def compute_pointbiserial_table(df, feature_cols, binary_target_cols):
    """NEW: Compute point-biserial correlations for binary targets (unique success, extreme adv)."""
    records = []
    for target in binary_target_cols:
        y = df[target].to_numpy(dtype=float)
        
        if y.sum() == 0 or y.sum() == len(y):  # All 0s or all 1s
            continue
            
        for feat in feature_cols:
            x = df[feat].to_numpy(dtype=float)

            mask = ~np.isnan(x) & ~np.isnan(y)
            n_valid = mask.sum()
            missingness_pct = 1 - (n_valid / len(df))
            
            if n_valid < 10:
                continue

            x_clean = x[mask]
            y_clean = y[mask]

            if np.nanstd(x_clean) == 0 or len(np.unique(y_clean)) < 2:
                continue

            try:
                r, p = pointbiserialr(y_clean, x_clean)
            except Exception:
                r, p = np.nan, np.nan

            records.append({
                "target": target, 
                "feature": feat, 
                "corr": r, 
                "pval": p, 
                "n": int(n_valid),
                "missingness_pct": missingness_pct,
                "n_positive": int(y_clean.sum()),
                "n_negative": int((y_clean == 0).sum())
            })

    return pd.DataFrame(records)

def bootstrap_correlation_ci(df, feature_cols, target_cols, n_boot=100):
    """Compute bootstrap confidence intervals for correlations."""
    print(f"\nComputing bootstrap CIs (n={n_boot})...")
    
    ci_records = []
    for target in target_cols:
        for feat in feature_cols:
            boot_rhos = []
            
            mask = df[[feat, target]].notna().all(axis=1)
            if mask.sum() < 10:
                continue
            
            data = df.loc[mask, [feat, target]]
            
            for _ in range(n_boot):
                sample = data.sample(frac=1.0, replace=True)
                try:
                    rho, _ = spearmanr(sample[feat], sample[target])
                    if not np.isnan(rho):
                        boot_rhos.append(rho)
                except:
                    pass
            
            if len(boot_rhos) >= n_boot * 0.8:
                ci_low, ci_high = np.percentile(boot_rhos, [2.5, 97.5])
                ci_records.append({
                    'target': target,
                    'feature': feat,
                    'ci_low': ci_low,
                    'ci_high': ci_high,
                    'ci_width': ci_high - ci_low,
                    'ci_excludes_zero': (ci_low > 0 or ci_high < 0)
                })
    
    return pd.DataFrame(ci_records)

def select_top_bottom_features(df_corr, feature_col="feature", value_col="corr", 
                                 top_n=15, mode="max_abs"):
    """Select features for visualization."""
    # Ensure target column exists before dropna
    df_corr = df_corr.copy()
    if "tool" in df_corr.columns and "target" not in df_corr.columns:
        df_corr["target"] = df_corr["tool"]
    
    d = df_corr.dropna(subset=[value_col]).copy()
    
    if d.empty:
        return []
    
    if mode == "max_abs":
        scores = (
            d.groupby(feature_col)[value_col]
            .apply(lambda s: s.abs().max())
            .sort_values(ascending=False)
        )
        top_feats = scores.head(top_n).index.tolist()
        bottom_feats = scores.tail(top_n).index.tolist()
        
    elif mode == "variance":
        # Handle both 'target' and 'tool' column names
        target_col = "target" if "target" in d.columns else "tool"
        if target_col not in d.columns:
            raise ValueError(f"Neither 'target' nor 'tool' column found in data. Available columns: {d.columns.tolist()}")
        if d.empty:
            return []
        pivot = d.pivot_table(index=feature_col, columns=target_col, values=value_col)
        if pivot.empty:
            return []
        variance = pivot.std(axis=1, skipna=True)
        scores = variance.sort_values(ascending=False)
        top_feats = scores.head(top_n * 2).index.tolist()
        bottom_feats = []
        
    elif mode == "representative":
        all_corrs = d.groupby(feature_col)[value_col].apply(lambda s: s.abs().max())
        
        bins = np.linspace(0, 1, 6)
        features_by_bin = []
        for i in range(len(bins) - 1):
            in_bin = all_corrs[(all_corrs >= bins[i]) & (all_corrs < bins[i+1])]
            if len(in_bin) > 0:
                n_sample = min(len(in_bin), top_n // (len(bins) - 1))
                features_by_bin.extend(in_bin.nlargest(n_sample).index.tolist())
        
        top_feats = features_by_bin[:top_n * 2]
        bottom_feats = []
    
    else:
        raise ValueError(f"Unknown mode: {mode}")
    
    return top_feats + bottom_feats

def make_heatmap(df_corr, title, filename, out_dir, top_n=15, mode="variance"):
    """Create heatmap from correlation results."""
    if df_corr.empty:
        print(f"[WARN] No data for heatmap: {filename}")
        return
    
    # Handle both 'target' and 'tool' column names
    df_corr = df_corr.copy()
    if "tool" in df_corr.columns and "target" not in df_corr.columns:
        df_corr["target"] = df_corr["tool"]
    
    # Ensure required columns exist
    if "target" not in df_corr.columns and "tool" not in df_corr.columns:
        print(f"[WARN] No 'target' or 'tool' column found for heatmap: {filename}")
        return
    if "feature" not in df_corr.columns:
        print(f"[WARN] No 'feature' column found for heatmap: {filename}")
        return
    if "corr" not in df_corr.columns:
        print(f"[WARN] No 'corr' column found for heatmap: {filename}")
        return
    
    feats = select_top_bottom_features(df_corr, top_n=top_n, mode=mode)
    if not feats:
        print(f"[WARN] No features selected for heatmap: {filename}")
        return
    
    d = df_corr[df_corr["feature"].isin(feats)].copy()
    if d.empty:
        print(f"[WARN] No data for heatmap after feature selection: {filename}")
        return

    # Ensure target column exists
    if "target" not in d.columns and "tool" in d.columns:
        d["target"] = d["tool"]
    
    pivot = d.pivot(index="feature", columns="target", values="corr")
    
    from scipy.cluster.hierarchy import linkage, dendrogram
    from scipy.spatial.distance import pdist
    
    if pivot.shape[0] > 1:
        distances = pdist(pivot.fillna(0), metric='euclidean')
        linkage_matrix = linkage(distances, method='ward')
        dendro = dendrogram(linkage_matrix, no_plot=True)
        pivot = pivot.iloc[dendro['leaves']]
    
    plt.figure(figsize=(max(10, pivot.shape[1] * 0.9), max(10, pivot.shape[0] * 0.3)))
    sns.heatmap(pivot, cmap="RdBu_r", center=0, annot=False, vmin=-0.6, vmax=0.6,
                cbar_kws={'label': 'Spearman ρ'})
    plt.title(title, fontsize=12, fontweight='bold')
    plt.xlabel("Gap/Advantage Metric", fontsize=10)
    plt.ylabel("Feature", fontsize=10)
    plt.xticks(rotation=45, ha='right', fontsize=9)
    plt.yticks(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_dir / filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved heatmap: {out_dir / filename}")

def run_correlation_analysis_for_threshold(df, feature_cols, tools, threshold: int = None, suffix: str = ""):
    """
    Run correlation analysis for a specific rank threshold.
    
    Args:
        df: Full DataFrame
        feature_cols: List of feature column names
        tools: List of tool names
        threshold: Rank threshold (1, 5, 10, or None for all data)
        suffix: Suffix for output files (e.g., "_top1", "_top5", "_top10")
    """
    print(f"\n--- Correlation Analysis for {'All Data' if threshold is None else f'Top@{threshold}'} ---")
    
    # Filter data by threshold if specified
    if threshold is not None:
        # Filter to only bugs where at least one tool found it within threshold
        rank_cols = [f"rank_{t}" for t in tools if f"rank_{t}" in df.columns]
        if not rank_cols:
            print(f"[WARN] No rank columns found for threshold filtering. Skipping threshold {threshold}.")
            return
        
        # Create mask: bug is included if ANY tool found it within threshold
        threshold_mask = pd.Series(False, index=df.index)
        for col in rank_cols:
            threshold_mask |= (df[col].notna()) & (df[col] <= threshold)
        
        df_filtered = df[threshold_mask].copy()
        print(f"  Filtered to {len(df_filtered)} bugs (from {len(df)} total)")
        
        if len(df_filtered) < 10:
            print(f"[WARN] Too few bugs for threshold {threshold}. Skipping.")
            return
    else:
        df_filtered = df.copy()
    
    # Compute MRR for the threshold if needed
    if threshold is not None:
        for tool in tools:
            rank_col = f"rank_{tool}"
            mrr_col = f"{BASE_PREFIX}_{tool}"
            if rank_col in df_filtered.columns:
                # Compute MRR@threshold: 1/rank if rank <= threshold, else 0
                df_filtered[f"mrr{threshold}_{tool}"] = np.where(
                    (df_filtered[rank_col].notna()) & (df_filtered[rank_col] <= threshold),
                    1.0 / df_filtered[rank_col],
                    0.0
                )
                # Also create rank column for this threshold
                df_filtered[f"rank{threshold}_{tool}"] = np.where(
                    (df_filtered[rank_col].notna()) & (df_filtered[rank_col] <= threshold),
                    df_filtered[rank_col],
                    np.nan
                )
    
    # Use threshold-specific columns if available, otherwise use original
    if threshold is not None:
        prefix_to_use = f"mrr{threshold}"
        rank_prefix_to_use = f"rank{threshold}"
    else:
        prefix_to_use = BASE_PREFIX
        rank_prefix_to_use = "rank"
    
    # Standardize metrics if requested
    if STANDARDIZE_METRICS:
        print(f"\nStandardizing metrics for prefix '{prefix_to_use}'...")
        scaler = StandardScaler()
        tools_list, cols = tools_for_prefix(df_filtered, prefix_to_use)
        if len(cols) > 0:
            df_filtered[cols] = scaler.fit_transform(df_filtered[cols].fillna(0))
            print(f"  Standardized {len(cols)} columns")

    # Compute gaps and advantages
    gap_targets = {}
    adv_targets = {}
    
    tools_list, cols = tools_for_prefix(df_filtered, prefix_to_use)
    if len(tools_list) < 2:
        print(f"[WARN] Not enough tools found for prefix {prefix_to_use}. Skipping.")
        return

    for a, b in itertools.combinations(tools_list, 2):
        col_a = f"{prefix_to_use}_{a}"
        col_b = f"{prefix_to_use}_{b}"
        if col_a in df_filtered.columns and col_b in df_filtered.columns:
            name_ab = f"gap_{prefix_to_use}_{a}_minus_{b}"
            name_ba = f"gap_{prefix_to_use}_{b}_minus_{a}"
            gap_targets[name_ab] = df_filtered[col_a] - df_filtered[col_b]
            gap_targets[name_ba] = df_filtered[col_b] - df_filtered[col_a]

    mat = np.column_stack([df_filtered[f"{prefix_to_use}_{t}"].to_numpy(dtype=float) for t in tools_list])
    for j, t in enumerate(tools_list):
        others = np.delete(mat, j, axis=1)
        if others.shape[1] > 0:
            mean_other = np.nanmean(others, axis=1)
            adv_targets[f"adv_{prefix_to_use}_{t}"] = mat[:, j] - mean_other

    gap_df = pd.DataFrame(gap_targets)
    adv_df = pd.DataFrame(adv_targets)

    df_gap = pd.concat([df_filtered, gap_df, adv_df], axis=1)

    # Compute correlations
    print("\nComputing Spearman correlations...")
    gap_corr = compute_spearman_table(df_gap, feature_cols, gap_df.columns.tolist())
    gap_corr = apply_holm(gap_corr, alpha=ALPHA)
    gap_corr["target"] = gap_corr["target"].apply(shorten_target)
    gap_corr["practically_significant"] = gap_corr["corr"].abs() >= PRACTICAL_SIG_CORR
    
    gap_corr.to_csv(OUT_DIR_CORR / f"gap_corr_spearman{suffix}.csv", index=False)
    print(f"Saved: {OUT_DIR_CORR / f'gap_corr_spearman{suffix}.csv'}")

    adv_corr = compute_spearman_table(df_gap, feature_cols, adv_df.columns.tolist())
    adv_corr = apply_holm(adv_corr, alpha=ALPHA)
    adv_corr["target"] = adv_corr["target"].apply(shorten_target)
    adv_corr["practically_significant"] = adv_corr["corr"].abs() >= PRACTICAL_SIG_CORR
    adv_corr.to_csv(OUT_DIR_CORR / f"adv_corr_spearman{suffix}.csv", index=False)
    print(f"Saved: {OUT_DIR_CORR / f'adv_corr_spearman{suffix}.csv'}")

    # Heatmaps
    make_heatmap(gap_corr, f"Feature × Gap Correlations{suffix.replace('_', ' ').title()}", 
                 f"gap_corr_heatmap{suffix}.png", OUT_DIR_CORR, TOP_N_FEATURES_HEATMAP, SELECTION_MODE)
    make_heatmap(adv_corr, f"Feature × Advantage Correlations{suffix.replace('_', ' ').title()}", 
                 f"adv_corr_heatmap{suffix}.png", OUT_DIR_CORR, TOP_N_FEATURES_HEATMAP, SELECTION_MODE)

    # Raw MRR correlations
    print(f"\n--- Raw MRR Correlations for {'All Data' if threshold is None else f'Top@{threshold}'} ---")
    raw_mrr_corr = compute_raw_mrr_correlations(
        df_filtered, 
        feature_cols, 
        tools_list, 
        prefix=prefix_to_use,
        out_dir=None  # Don't save here, we'll save with suffix
    )
    
    if not raw_mrr_corr.empty:
        raw_mrr_corr.to_csv(OUT_DIR_CORR / f"raw_mrr_correlations{suffix}.csv", index=False)
        print(f"Saved: {OUT_DIR_CORR / f'raw_mrr_correlations{suffix}.csv'}")
        make_heatmap(raw_mrr_corr, f"Raw MRR Correlations{suffix.replace('_', ' ').title()}", 
                     f"raw_mrr_corr_heatmap{suffix}.png", OUT_DIR_CORR, TOP_N_FEATURES_HEATMAP, SELECTION_MODE)
    
    # Two-part analysis
    print(f"\n--- Two-Part Found/Rank Analysis for {'All Data' if threshold is None else f'Top@{threshold}'} ---")
    found_corr, rank_corr = compute_two_part_analysis(
        df_filtered,
        feature_cols,
        tools_list,
        prefix=prefix_to_use,
        rank_prefix=rank_prefix_to_use,
        out_dir=None  # Don't save here, we'll save with suffix
    )
    
    if not found_corr.empty:
        found_corr.to_csv(OUT_DIR_CORR / f"two_part_found_correlations{suffix}.csv", index=False)
        print(f"Saved: {OUT_DIR_CORR / f'two_part_found_correlations{suffix}.csv'}")
        make_heatmap(found_corr, f"Found vs Not Found Correlations{suffix.replace('_', ' ').title()}", 
                     f"found_rank_corr_heatmap{suffix}.png", OUT_DIR_CORR, TOP_N_FEATURES_HEATMAP, SELECTION_MODE)
    
    if not rank_corr.empty:
        rank_corr.to_csv(OUT_DIR_CORR / f"two_part_rank_correlations{suffix}.csv", index=False)
        print(f"Saved: {OUT_DIR_CORR / f'two_part_rank_correlations{suffix}.csv'}")
        make_heatmap(rank_corr, f"Rank Quality Correlations{suffix.replace('_', ' ').title()}", 
                     f"rank_corr_heatmap{suffix}.png", OUT_DIR_CORR, TOP_N_FEATURES_HEATMAP, SELECTION_MODE)

def run_correlation_analysis():
    """Run correlation analysis between features and tool performance gaps/advantages."""
    print("\n" + "=" * 60)
    print("CORRELATION ANALYSIS (CORRECTED)")
    print("=" * 60)

    OUT_DIR_CORR.mkdir(exist_ok=True, parents=True)

    df, feature_cols, id_cols, perf_cols = load_data_and_features(IN_FILE)
    tools = get_tools(df, BASE_PREFIX)
    
    # Run for each threshold: top@1, top@5, top@10, and all data
    thresholds = [1, 5, 10, None]
    threshold_labels = ["Top@1", "Top@5", "Top@10", "All Data"]
    
    for threshold, label in zip(thresholds, threshold_labels):
        suffix = f"_top{threshold}" if threshold is not None else ""
        try:
            run_correlation_analysis_for_threshold(df, feature_cols, tools, threshold=threshold, suffix=suffix)
        except Exception as e:
            print(f"[ERROR] Failed to run correlation analysis for {label}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"\nAll correlation analysis outputs saved to: {OUT_DIR_CORR}")

# ======================================
# 2. SUCCESS/FAILURE ANALYSIS (ENHANCED WITH HEATMAP)
# ======================================

def create_success_failure_heatmap(combined_stats, out_dir, top_n=30, min_tools=2, suffix=""):
    """
    Create heatmap showing features × tools for success/failure effects.
    
    Args:
        combined_stats: DataFrame with columns [tool, feature, cliffs_delta, reject, practically_significant]
        out_dir: Output directory
        top_n: Number of top features to show
        min_tools: Minimum number of tools a feature must affect to be included
    """
    print("\n--- Creating Success/Failure Heatmap ---")
    
    if combined_stats.empty:
        print("[WARN] No data for success/failure heatmap")
        return
    
    # Filter for significant features only
    sig_stats = combined_stats[
        combined_stats["reject"] & 
        combined_stats["practically_significant"]
    ].copy()
    
    if sig_stats.empty:
        print("[WARN] No significant features for heatmap")
        return
    
    # Count how many tools each feature affects
    feature_counts = sig_stats.groupby('feature')['tool'].count()
    
    # Filter features that affect at least min_tools
    multi_tool_features = feature_counts[feature_counts >= min_tools].index.tolist()
    
    if not multi_tool_features:
        print(f"[WARN] No features affect {min_tools}+ tools. Using all significant features.")
        multi_tool_features = sig_stats['feature'].unique()
    
    # Select top features by maximum absolute effect size across tools
    feature_max_delta = sig_stats.groupby('feature')['cliffs_delta'].apply(lambda x: x.abs().max())
    top_features = feature_max_delta.nlargest(top_n).index.tolist()
    
    # Also include multi-tool features even if not in top N
    all_selected = list(set(top_features) | set(multi_tool_features[:top_n]))
    
    # Create pivot: features × tools, values = Cliff's delta
    # Include ALL features (not just significant) for selected features to show full picture
    pivot_data = combined_stats[combined_stats['feature'].isin(all_selected)].copy()
    pivot = pivot_data.pivot_table(
        index='feature',
        columns='tool',
        values='cliffs_delta',
        aggfunc='first'
    )
    
    if pivot.empty or pivot.shape[0] < 2:
        print("[WARN] Insufficient data for heatmap after filtering")
        return
    
    # Sort features by hierarchical clustering
    from scipy.cluster.hierarchy import linkage, dendrogram
    from scipy.spatial.distance import pdist
    
    distances = pdist(pivot.fillna(0), metric='euclidean')
    linkage_matrix = linkage(distances, method='ward')
    dendro = dendrogram(linkage_matrix, no_plot=True)
    pivot_sorted = pivot.iloc[dendro['leaves']]
    
    # Create significance mask (for hatching)
    sig_mask = pivot_data.pivot_table(
        index='feature',
        columns='tool',
        values='reject',
        aggfunc='first'
    ).fillna(False)
    sig_mask = sig_mask.reindex_like(pivot_sorted).fillna(False)
    
    # Create plot
    fig_height = max(10, pivot_sorted.shape[0] * 0.35)
    fig_width = max(8, pivot_sorted.shape[1] * 1.5)
    
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    
    # Create heatmap
    sns.heatmap(
        pivot_sorted,
        cmap="RdBu_r",
        center=0,
        annot=False,
        vmin=-0.8,
        vmax=0.8,
        cbar_kws={'label': "Cliff's δ"},
        linewidths=0.5,
        linecolor='gray',
        ax=ax
    )
    
    # Add hatching for non-significant cells
    for i, feature in enumerate(pivot_sorted.index):
        for j, tool in enumerate(pivot_sorted.columns):
            if not sig_mask.loc[feature, tool]:
                ax.add_patch(plt.Rectangle((j, i), 1, 1, fill=False, 
                                          edgecolor='white', lw=2, linestyle='--'))
    
    ax.set_title(
        f"Success/Failure Feature Effects Across Tools\n"
        f"(Cliff's δ: Red = helps success, Blue = helps failure)\n"
        f"Dashed boxes = not significant (p≥0.05 or |δ|<0.2)",
        fontsize=13,
        fontweight='bold',
        pad=20
    )
    ax.set_xlabel("Tool", fontsize=11, fontweight='bold')
    ax.set_ylabel("Feature", fontsize=11, fontweight='bold')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=10)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=9)
    
    plt.tight_layout()
    
    # Save full heatmap
    heatmap_file = out_dir / f"success_failure_heatmap_all_tools{suffix}.png"
    plt.savefig(heatmap_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {heatmap_file}")
    
    # Also create version with only significant cells (for clarity)
    pivot_sig_only = pivot_sorted.copy()
    for feature in pivot_sig_only.index:
        for tool in pivot_sig_only.columns:
            if not sig_mask.loc[feature, tool]:
                pivot_sig_only.loc[feature, tool] = np.nan
    
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    sns.heatmap(
        pivot_sig_only,
        cmap="RdBu_r",
        center=0,
        annot=False,
        vmin=-0.8,
        vmax=0.8,
        cbar_kws={'label': "Cliff's δ"},
        linewidths=0.5,
        linecolor='gray',
        ax=ax,
        mask=pivot_sig_only.isna()
    )
    
    title_suffix = suffix.replace('_', ' ').title() if suffix else ""
    ax.set_title(
        f"Success/Failure Feature Effects - Significant Only{title_suffix}\n"
        f"(Cliff's δ: Red = helps success, Blue = helps failure)",
        fontsize=13,
        fontweight='bold',
        pad=20
    )
    ax.set_xlabel("Tool", fontsize=11, fontweight='bold')
    ax.set_ylabel("Feature", fontsize=11, fontweight='bold')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=10)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=9)
    
    plt.tight_layout()
    
    heatmap_sig_file = out_dir / f"success_failure_heatmap_significant_only{suffix}.png"
    plt.savefig(heatmap_sig_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {heatmap_sig_file}")
    
    # Save summary statistics
    summary_stats = []
    for feature in all_selected:
        feat_data = sig_stats[sig_stats['feature'] == feature]
        if not feat_data.empty:
            summary_stats.append({
                'feature': feature,
                'n_tools_affected': len(feat_data),
                'max_abs_delta': feat_data['cliffs_delta'].abs().max(),
                'mean_delta': feat_data['cliffs_delta'].mean(),
                'tools_positive': ', '.join(feat_data[feat_data['cliffs_delta'] > 0]['tool'].tolist()),
                'tools_negative': ', '.join(feat_data[feat_data['cliffs_delta'] < 0]['tool'].tolist())
            })
    
    if summary_stats:
        summary_df = pd.DataFrame(summary_stats).sort_values('max_abs_delta', ascending=False)
        summary_df.to_csv(out_dir / "success_failure_feature_summary.csv", index=False)
        print(f"Saved: {out_dir / 'success_failure_feature_summary.csv'}")
        
        print(f"\nHeatmap includes {len(all_selected)} features:")
        print(f"  - Features affecting {min_tools}+ tools: {len(multi_tool_features)}")
        print(f"  - Top features by effect size: {len(top_features)}")

# ======================================

def build_labels(df, tools, prefix, mode="stratified", adv_eps=1e-12, rank_high=5, rank_low=6, rank_prefix="rank"):
    """
    Build binary labels for success/failure analysis.
    
    Args:
        rank_prefix: Prefix for rank columns (e.g., "rank" or "rank1" for threshold-specific)
    """
    labels = {}
    
    for t in tools:
        if mode == "success":
            col = f"{prefix}_{t}"
            if col not in df.columns:
                raise KeyError(f"Column {col} not found.")
            vals = df[col].to_numpy(dtype=float)
            labels[t] = (vals > 0).astype(int)
            
        elif mode == "winner":
            tools_list, cols = tools_for_prefix(df, prefix)
            mat = np.column_stack([df[c].to_numpy(dtype=float) for c in cols])
            if mat.shape[1] < 2:
                raise ValueError("Need at least 2 tools for 'winner' mode.")
            
            tool_idx = tools_list.index(t)
            col_t = mat[:, tool_idx]
            others = np.delete(mat, tool_idx, axis=1)
            max_other = np.nanmax(others, axis=1)
            labels[t] = (col_t > max_other + adv_eps).astype(int)
            
        elif mode == "advantage":
            tools_list, cols = tools_for_prefix(df, prefix)
            mat = np.column_stack([df[c].to_numpy(dtype=float) for c in cols])
            tool_idx = tools_list.index(t)
            col_t = mat[:, tool_idx]
            others = np.delete(mat, tool_idx, axis=1)
            
            if others.shape[1] > 0:
                mean_other = np.nanmean(others, axis=1)
                labels[t] = (col_t > mean_other + adv_eps).astype(int)
            else:
                labels[t] = np.zeros(len(df), dtype=int)
                
        elif mode == "stratified":
            rank_col = f"{rank_prefix}_{t}"
            mrr_col = f"{prefix}_{t}"
            
            # Try to get rank column directly (with specified prefix)
            if rank_col in df.columns:
                ranks = df[rank_col].to_numpy(dtype=float)
            elif f"rank_{t}" in df.columns:
                # Fallback to standard rank column
                ranks = df[f"rank_{t}"].to_numpy(dtype=float)
            elif mrr_col in df.columns:
                # Calculate rank from MRR: rank = 1/mrr (when mrr > 0)
                mrr_vals = df[mrr_col].to_numpy(dtype=float)
                ranks = np.full(len(df), np.nan)
                mask = mrr_vals > 0
                ranks[mask] = 1.0 / mrr_vals[mask]
                # If MRR is 0, rank is undefined (NaN)
            else:
                raise KeyError(f"Neither {rank_col}, rank_{t}, nor {mrr_col} found for stratified mode.")
            
            label_arr = np.full(len(df), np.nan)
            label_arr[ranks <= rank_high] = 1
            label_arr[ranks >= rank_low] = 0
            labels[t] = label_arr
            
        elif mode == "relative_quartile":
            tools_list, cols = tools_for_prefix(df, prefix)
            mat = np.column_stack([df[c].to_numpy(dtype=float) for c in cols])
            tool_idx = tools_list.index(t)
            
            others = np.delete(mat, tool_idx, axis=1)
            if others.shape[1] > 0:
                mean_other = np.nanmean(others, axis=1)
                relative = mat[:, tool_idx] - mean_other
                
                q25, q75 = np.nanpercentile(relative[~np.isnan(relative)], [25, 75])
                label_arr = np.full(len(df), np.nan)
                label_arr[relative >= q75] = 1
                label_arr[relative <= q25] = 0
                labels[t] = label_arr
            else:
                labels[t] = np.full(len(df), np.nan)
        else:
            raise ValueError("Unknown LABEL_MODE: " + mode)

    return labels

def run_success_failure_analysis_for_threshold(df, feature_cols, tools, threshold: int = None, suffix: str = ""):
    """
    Run success/failure analysis for a specific rank threshold.
    
    Args:
        df: Full DataFrame
        feature_cols: List of feature column names
        tools: List of tool names
        threshold: Rank threshold (1, 5, 10, or None for all data)
        suffix: Suffix for output files (e.g., "_top1", "_top5", "_top10")
    """
    print(f"\n--- Success/Failure Analysis for {'All Data' if threshold is None else f'Top@{threshold}'} ---")
    
    # Filter data by threshold if specified
    if threshold is not None:
        # Filter to only bugs where at least one tool found it within threshold
        rank_cols = [f"rank_{t}" for t in tools if f"rank_{t}" in df.columns]
        if not rank_cols:
            print(f"[WARN] No rank columns found for threshold filtering. Skipping threshold {threshold}.")
            return
        
        # Create mask: bug is included if ANY tool found it within threshold
        threshold_mask = pd.Series(False, index=df.index)
        for col in rank_cols:
            threshold_mask |= (df[col].notna()) & (df[col] <= threshold)
        
        df_filtered = df[threshold_mask].copy()
        print(f"  Filtered to {len(df_filtered)} bugs (from {len(df)} total)")
        
        if len(df_filtered) < 10:
            print(f"[WARN] Too few bugs for threshold {threshold}. Skipping.")
            return
        
        # For threshold analysis, adjust rank thresholds to be relative to the threshold
        # Success = rank <= threshold (i.e., rank <= 1 for top@1, rank <= 5 for top@5, etc.)
        # Failure = rank > threshold (but still found, so rank is not NaN)
        rank_high = threshold  # Success: rank <= threshold
        rank_low = threshold + 1  # Failure: rank > threshold (i.e., rank >= threshold + 1)
    else:
        df_filtered = df.copy()
        rank_high = RANK_THRESHOLD_HIGH
        rank_low = RANK_THRESHOLD_LOW
    
    # Use threshold-specific MRR columns if available
    if threshold is not None:
        prefix_to_use = f"mrr{threshold}"
        # Create MRR@threshold columns if they don't exist
        for tool in tools:
            rank_col = f"rank_{tool}"
            if rank_col in df_filtered.columns:
                df_filtered[f"mrr{threshold}_{tool}"] = np.where(
                    (df_filtered[rank_col].notna()) & (df_filtered[rank_col] <= threshold),
                    1.0 / df_filtered[rank_col],
                    0.0
                )
                # Also create rank column for this threshold
                df_filtered[f"rank{threshold}_{tool}"] = np.where(
                    (df_filtered[rank_col].notna()) & (df_filtered[rank_col] <= threshold),
                    df_filtered[rank_col],
                    np.nan
                )
    else:
        prefix_to_use = BASE_PREFIX

    # Determine rank prefix to use
    if threshold is not None:
        rank_prefix_to_use = f"rank{threshold}"
    else:
        rank_prefix_to_use = "rank"
    
    labels = build_labels(df_filtered, tools, prefix_to_use, LABEL_MODE, 
                          adv_eps=ADV_EPS, 
                          rank_high=rank_high,
                          rank_low=rank_low,
                          rank_prefix=rank_prefix_to_use)

    all_tool_stats = []

    for tool in tools:
        label = labels[tool]
        label_name = f"{LABEL_MODE}_{prefix_to_use}_{tool}"
        
        valid_mask = ~np.isnan(label)
        df_valid = df[valid_mask].copy()
        label_valid = label[valid_mask]
        
        df_valid[label_name] = label_valid.astype(int)

        n_pos = int(label_valid.sum())
        n_neg = int((label_valid == 0).sum())
        n_excluded = int((~valid_mask).sum())
        
        print(f"\n=== {tool} | label={label_name} ===")
        print(f"Pos: {n_pos}, Neg: {n_neg}, Excluded: {n_excluded}")

        # Special handling for tools with insufficient negative cases
        # Use other tools' negative cases as comparison group if needed
        use_other_tools_negative = False
        if n_neg < MIN_GROUP_N and n_pos >= MIN_GROUP_N:
            if n_neg == 0:
                print(f"Note: {tool} has no negative cases (all detections rank <= {rank_high})")
                print(f"  This indicates excellent performance.")
            else:
                print(f"Note: {tool} has only {n_neg} negative cases (need {MIN_GROUP_N} for valid comparison)")
            
            # Use other tools' negative cases as comparison group
            other_tools = [t for t in tools if t != tool]
            other_negative_mask = np.zeros(len(df_filtered), dtype=bool)
            for other_tool in other_tools:
                other_label = labels.get(other_tool)
                if other_label is not None:
                    other_negative_mask |= (~np.isnan(other_label)) & (other_label == 0)
            
            n_other_neg = int(other_negative_mask.sum())
            
            # If other tools' negatives are insufficient, try using ALL tools' combined negatives
            if n_other_neg < MIN_GROUP_N:
                # Get combined negatives from ALL tools (including current tool)
                all_negative_mask = np.zeros(len(df_filtered), dtype=bool)
                for all_tool in tools:
                    all_label = labels.get(all_tool)
                    if all_label is not None:
                        all_negative_mask |= (~np.isnan(all_label)) & (all_label == 0)
                
                n_all_neg = int(all_negative_mask.sum())
                if n_all_neg >= MIN_GROUP_N:
                    print(f"  Other tools' negatives insufficient ({n_other_neg} < {MIN_GROUP_N})")
                    print(f"  Using ALL tools' combined negative cases ({n_all_neg} cases) as comparison group")
                    use_other_tools_negative = True
                    # Create combined dataset: tool positives + all tools' negatives
                    combined_mask = valid_mask | all_negative_mask
                    df_valid = df_filtered[combined_mask].copy()
                    
                    # Create combined labels: tool positives = 1, all tools' negatives = 0
                    label_valid = np.full(len(df_valid), np.nan)
                    # Set tool positives (from original valid_mask)
                    tool_pos_in_combined = valid_mask[combined_mask]
                    label_valid[tool_pos_in_combined] = 1
                    # Set all tools' negatives
                    all_neg_in_combined = all_negative_mask[combined_mask]
                    label_valid[all_neg_in_combined] = 0
                    
                    df_valid[label_name] = label_valid.astype(int)
                    n_neg = int((label_valid == 0).sum())
                    print(f"  Updated: Pos: {n_pos}, Neg: {n_neg} (using all tools' combined failures)")
                else:
                    print(f"  Skipping {tool}: insufficient comparison cases (other tools: {n_other_neg}, all tools: {n_all_neg}, min={MIN_GROUP_N})")
                    continue
            else:
                print(f"  Using other tools' negative cases ({n_other_neg} cases) as comparison group")
                use_other_tools_negative = True
                # Create combined dataset: tool positives + other tools' negatives
                combined_mask = valid_mask | other_negative_mask
                df_valid = df_filtered[combined_mask].copy()
                
                # Create combined labels: tool positives = 1, other tools' negatives = 0
                label_valid = np.full(len(df_valid), np.nan)
                # Set tool positives (from original valid_mask)
                tool_pos_in_combined = valid_mask[combined_mask]
                label_valid[tool_pos_in_combined] = 1
                # Set other tools' negatives
                other_neg_in_combined = other_negative_mask[combined_mask]
                label_valid[other_neg_in_combined] = 0
                
                df_valid[label_name] = label_valid.astype(int)
                n_neg = int((label_valid == 0).sum())
                print(f"  Updated: Pos: {n_pos}, Neg: {n_neg} (using other tools' failures)")
        elif n_pos < MIN_GROUP_N or n_neg < MIN_GROUP_N:
            print(f"Skipping {tool}: insufficient group sizes (min={MIN_GROUP_N})")
            continue

        records = []
        for feat in feature_cols:
            x = df_valid.loc[df_valid[label_name] == 1, feat].dropna()
            y = df_valid.loc[df_valid[label_name] == 0, feat].dropna()

            if len(x) < MIN_GROUP_N or len(y) < MIN_GROUP_N:
                records.append({
                    "tool": tool,
                    "label_mode": LABEL_MODE,
                    "base_prefix": BASE_PREFIX,
                    "feature": feat,
                    "n_pos": len(x),
                    "n_neg": len(y),
                    "median_pos": x.median() if len(x) else np.nan,
                    "median_neg": y.median() if len(y) else np.nan,
                    "u_stat": np.nan,
                    "p_value": np.nan,
                    "cliffs_delta": np.nan
                })
                continue

            u_stat, p_val = mannwhitneyu(x, y, alternative="two-sided", method="asymptotic")
            delta = cliffs_delta(x.values, y.values)

            records.append({
                "tool": tool,
                "label_mode": LABEL_MODE,
                "base_prefix": prefix_to_use,
                "feature": feat,
                "n_pos": len(x),
                "n_neg": len(y),
                "median_pos": x.median(),
                "median_neg": y.median(),
                "u_stat": u_stat,
                "p_value": p_val,
                "cliffs_delta": delta
            })

        stats_df = pd.DataFrame(records)
        stats_df = apply_holm(stats_df, alpha=ALPHA, pval_col="p_value")
        stats_df["abs_delta"] = stats_df["cliffs_delta"].abs()
        stats_df["practically_significant"] = stats_df["abs_delta"] >= PRACTICAL_SIG_DELTA
        
        sig_df = stats_df[stats_df["reject"] & stats_df["practically_significant"]].copy()
        sig_df = sig_df.sort_values("abs_delta", ascending=False)

        out_file = OUT_DIR_SUCCESS / f"{label_name}_stats{suffix}.csv"
        stats_df.to_csv(out_file, index=False)
        print(f"Saved: {out_file}")

        if not sig_df.empty:
            sig_out = OUT_DIR_SUCCESS / f"{label_name}_significant{suffix}.csv"
            sig_df.to_csv(sig_out, index=False)
            print(f"Saved: {sig_out}")
            print(f"  Significant features: {len(sig_df)}")
        else:
            print("  No significant features found.")

        all_tool_stats.append(stats_df)

        if not sig_df.empty:
            top_feats = sig_df.head(TOP_K_PLOTS)["feature"].tolist()
            
            for feat_name in top_feats:
                row = sig_df[sig_df["feature"] == feat_name].iloc[0]
                
                x = df_valid.loc[df_valid[label_name] == 1, feat_name].dropna()
                y = df_valid.loc[df_valid[label_name] == 0, feat_name].dropna()
                
                fig, ax = plt.subplots(figsize=(8, 5))
                positions = [1, 2]
                ax.boxplot([x, y], positions=positions, widths=0.5, patch_artist=True,
                          boxprops=dict(facecolor='lightblue', alpha=0.7),
                          medianprops=dict(color='red', linewidth=2))
                
                ax.set_xticks(positions)
                ax.set_xticklabels([f"Success (n={len(x)})", f"Failure (n={len(y)})"])
                ax.set_ylabel(feat_name, fontsize=10)
                title_suffix = f" (Top@{threshold})" if threshold is not None else ""
                ax.set_title(f"{tool} | {feat_name}{title_suffix}\nδ={row['cliffs_delta']:.3f}, p_adj={row['pval_adj']:.4f}", 
                           fontsize=11, fontweight='bold')
                ax.grid(axis='y', alpha=0.3)
                
                plt.tight_layout()
                plot_file = OUT_DIR_SUCCESS / f"{label_name}_{safe_name(feat_name)}_boxplot{suffix}.png"
                plt.savefig(plot_file, dpi=300)
                plt.close()

    if all_tool_stats:
        combined = pd.concat(all_tool_stats, ignore_index=True)
        combined.to_csv(OUT_DIR_SUCCESS / f"all_tools_combined_stats{suffix}.csv", index=False)
        print(f"\nSaved combined stats: {OUT_DIR_SUCCESS / f'all_tools_combined_stats{suffix}.csv'}")
        
        # Create success/failure heatmap
        create_success_failure_heatmap(
            combined, 
            OUT_DIR_SUCCESS, 
            top_n=SUCCESS_HEATMAP_TOP_N,
            min_tools=SUCCESS_HEATMAP_MIN_TOOLS,
            suffix=suffix
        )

def run_success_failure_analysis():
    """Run success/failure analysis using Mann-Whitney U tests for top@1, top@5, top@10."""
    print("\n" + "=" * 60)
    print("SUCCESS/FAILURE ANALYSIS (CORRECTED)")
    print("=" * 60)

    OUT_DIR_SUCCESS.mkdir(exist_ok=True, parents=True)

    df, feature_cols, id_cols, perf_cols = load_data_and_features(IN_FILE)

    tools = get_tools(df, BASE_PREFIX)
    print(f"Tools: {tools}")

    if len(tools) < 2:
        raise RuntimeError(f"Need at least 2 tools for BASE_PREFIX={BASE_PREFIX}")

    # Run for each threshold: top@1, top@5, top@10, and all data
    thresholds = [1, 5, 10, None]
    threshold_labels = ["Top@1", "Top@5", "Top@10", "All Data"]
    
    for threshold, label in zip(thresholds, threshold_labels):
        suffix = f"_top{threshold}" if threshold is not None else ""
        try:
            run_success_failure_analysis_for_threshold(df, feature_cols, tools, threshold=threshold, suffix=suffix)
        except Exception as e:
            print(f"[ERROR] Failed to run success/failure analysis for {label}: {e}")
            import traceback
            traceback.print_exc()
            continue

    print(f"\nAll success/failure analysis outputs saved to: {OUT_DIR_SUCCESS}")

# ======================================
# 3. NEW: OUTLIER ANALYSIS
# ======================================

def compute_unique_success_targets(df, tools, prefix="mrr", threshold=0.1):
    """
    Find bugs where exactly one tool succeeds (MRR > threshold) 
    while all others fail (MRR ≤ threshold).
    """
    print(f"\n--- Computing Unique Success Targets (threshold={threshold}) ---")
    
    unique_targets = {}
    mat = np.column_stack([df[f"{prefix}_{t}"].to_numpy(dtype=float) for t in tools])
    
    for j, t in enumerate(tools):
        tool_col = mat[:, j]
        others = np.delete(mat, j, axis=1)
        
        tool_succeeds = tool_col > threshold
        all_others_fail = np.all(others <= threshold, axis=1)
        
        unique_success = tool_succeeds & all_others_fail
        unique_targets[f"unique_success_{t}"] = unique_success.astype(int)
        
        n_unique = unique_success.sum()
        pct_unique = (n_unique / len(df)) * 100
        print(f"  {t}: {n_unique} unique successes ({pct_unique:.1f}%)")
    
    return unique_targets

def compute_extreme_advantage_targets(df, tools, prefix="mrr", z_threshold=2.0):
    """
    Flag bugs where one tool has extreme advantage over others (z-score > threshold).
    """
    print(f"\n--- Computing Extreme Advantage Targets (z > {z_threshold}) ---")
    
    extreme_targets = {}
    mat = np.column_stack([df[f"{prefix}_{t}"].to_numpy(dtype=float) for t in tools])
    
    for j, t in enumerate(tools):
        tool_col = mat[:, j]
        others = np.delete(mat, j, axis=1)
        
        if others.shape[1] > 0:
            mean_other = np.nanmean(others, axis=1)
            advantage = tool_col - mean_other
            
            z_adv = (advantage - np.nanmean(advantage)) / np.nanstd(advantage)
            
            extreme_adv = z_adv > z_threshold
            extreme_targets[f"extreme_adv_{t}"] = extreme_adv.astype(int)
            
            n_extreme = extreme_adv.sum()
            pct_extreme = (n_extreme / len(df)) * 100
            print(f"  {t}: {n_extreme} extreme advantages ({pct_extreme:.1f}%)")
        else:
            extreme_targets[f"extreme_adv_{t}"] = np.zeros(len(df), dtype=int)
    
    return extreme_targets

def analyze_feature_subgroups(df, feature, advantage_col, n_bins=3, out_dir=None):
    """
    Split bugs into feature bins (low/med/high) and compute
    Cliff's delta between bins for tool advantage.
    """
    try:
        df_work = df[[feature, advantage_col]].dropna().copy()
        
        if len(df_work) < 20:
            return None
        
        df_work['feat_bin'] = pd.qcut(df_work[feature], q=n_bins, labels=['low', 'med', 'high'], duplicates='drop')
        
        low_group = df_work[df_work['feat_bin'] == 'low'][advantage_col].values
        high_group = df_work[df_work['feat_bin'] == 'high'][advantage_col].values
        
        if len(low_group) > 5 and len(high_group) > 5:
            delta = cliffs_delta(high_group, low_group)
            
            result = {
                'feature': feature,
                'advantage_metric': advantage_col,
                'delta': delta,
                'n_low': len(low_group),
                'n_high': len(high_group),
                'median_low': np.median(low_group),
                'median_high': np.median(high_group),
                'mean_low': np.mean(low_group),
                'mean_high': np.mean(high_group)
            }
            
            # Create visualization if output directory provided
            if out_dir is not None:
                fig, ax = plt.subplots(figsize=(8, 5))
                ax.boxplot([low_group, high_group], labels=['Low', 'High'], patch_artist=True,
                          boxprops=dict(facecolor='lightblue', alpha=0.7))
                ax.set_xlabel(f'{feature} (binned)')
                ax.set_ylabel(advantage_col)
                ax.set_title(f"Subgroup Analysis: {feature}\nCliff's δ = {delta:.3f}", fontweight='bold')
                ax.grid(axis='y', alpha=0.3)
                plt.tight_layout()
                
                safe_feat = safe_name(feature)
                safe_adv = safe_name(advantage_col)
                plot_file = out_dir / f"subgroup_{safe_feat}_{safe_adv}.png"
                plt.savefig(plot_file, dpi=300)
                plt.close()
            
            return result
    except Exception as e:
        print(f"  [WARN] Subgroup analysis failed for {feature}: {e}")
        return None

def plot_outlier_scatterplot(df, feature, advantage_col, tool_name, out_dir, top_n=20):
    """Plot feature vs advantage, highlighting extreme cases."""
    df_plot = df[[feature, advantage_col]].dropna().copy()
    
    if len(df_plot) < 10:
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot all bugs
    ax.scatter(df_plot[feature], df_plot[advantage_col], alpha=0.3, s=30, c='gray')
    
    # Highlight top N advantages
    df_sorted = df_plot.nlargest(top_n, advantage_col)
    ax.scatter(df_sorted[feature], df_sorted[advantage_col], 
               color='red', s=100, alpha=0.7, label=f'Top {top_n} advantages', edgecolors='black')
    
    # Fit trend line
    from scipy.stats import linregress
    slope, intercept, r_value, p_value, std_err = linregress(df_plot[feature], df_plot[advantage_col])
    line_x = np.array([df_plot[feature].min(), df_plot[feature].max()])
    line_y = slope * line_x + intercept
    ax.plot(line_x, line_y, 'b--', alpha=0.5, label=f'Trend (r={r_value:.3f})')
    
    ax.set_xlabel(feature, fontsize=11)
    ax.set_ylabel(f'{tool_name} Advantage', fontsize=11)
    ax.set_title(f'Outlier Analysis: {feature} vs {tool_name} Advantage', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    safe_feat = safe_name(feature)
    safe_tool = safe_name(tool_name)
    plot_file = out_dir / f"outlier_scatter_{safe_tool}_{safe_feat}.png"
    plt.savefig(plot_file, dpi=300)
    plt.close()
    
    print(f"  Saved outlier scatter: {plot_file.name}")

def run_outlier_analysis():
    """NEW: Run comprehensive outlier analysis for tool-specific effects."""
    print("\n" + "=" * 60)
    print("OUTLIER ANALYSIS - TOOL-SPECIFIC EFFECTS")
    print("=" * 60)
    
    OUT_DIR_OUTLIER.mkdir(exist_ok=True, parents=True)
    
    df, feature_cols, id_cols, perf_cols = load_data_and_features(IN_FILE)
    
    tools = get_tools(df, BASE_PREFIX)
    print(f"Tools: {tools}")
    
    if len(tools) < 2:
        print("[WARN] Need at least 2 tools for outlier analysis. Skipping.")
        return
    
    # 1. Compute unique success targets
    unique_targets = compute_unique_success_targets(df, tools, BASE_PREFIX, UNIQUE_SUCCESS_THRESHOLD)
    unique_df = pd.DataFrame(unique_targets)
    
    # 2. Compute extreme advantage targets
    extreme_targets = compute_extreme_advantage_targets(df, tools, BASE_PREFIX, EXTREME_ADV_Z_THRESHOLD)
    extreme_df = pd.DataFrame(extreme_targets)
    
    # Combine with original data
    df_outlier = pd.concat([df, unique_df, extreme_df], axis=1)
    
    # 3. Point-biserial correlations for unique success
    print("\n--- Analyzing Unique Success Correlations ---")
    unique_corr = compute_pointbiserial_table(df_outlier, feature_cols, unique_df.columns.tolist())
    
    if not unique_corr.empty:
        unique_corr = apply_holm(unique_corr, alpha=ALPHA)
        unique_corr["target"] = unique_corr["target"].apply(shorten_target)
        unique_corr["practically_significant"] = unique_corr["corr"].abs() >= PRACTICAL_SIG_CORR
        unique_corr.to_csv(OUT_DIR_OUTLIER / "unique_success_correlations.csv", index=False)
        print(f"Saved: {OUT_DIR_OUTLIER / 'unique_success_correlations.csv'}")
        
        sig_unique = unique_corr[unique_corr["reject"] & unique_corr["practically_significant"]]
        if not sig_unique.empty:
            print(f"  Significant unique success correlations: {len(sig_unique)}")
            sig_unique.to_csv(OUT_DIR_OUTLIER / "unique_success_significant.csv", index=False)
    else:
        print("  No unique success patterns found.")
    
    # 4. Point-biserial correlations for extreme advantages
    print("\n--- Analyzing Extreme Advantage Correlations ---")
    extreme_corr = compute_pointbiserial_table(df_outlier, feature_cols, extreme_df.columns.tolist())
    
    if not extreme_corr.empty:
        extreme_corr = apply_holm(extreme_corr, alpha=ALPHA)
        extreme_corr["target"] = extreme_corr["target"].apply(shorten_target)
        extreme_corr["practically_significant"] = extreme_corr["corr"].abs() >= PRACTICAL_SIG_CORR
        extreme_corr.to_csv(OUT_DIR_OUTLIER / "extreme_advantage_correlations.csv", index=False)
        print(f"Saved: {OUT_DIR_OUTLIER / 'extreme_advantage_correlations.csv'}")
        
        sig_extreme = extreme_corr[extreme_corr["reject"] & extreme_corr["practically_significant"]]
        if not sig_extreme.empty:
            print(f"  Significant extreme advantage correlations: {len(sig_extreme)}")
            sig_extreme.to_csv(OUT_DIR_OUTLIER / "extreme_advantage_significant.csv", index=False)
    else:
        print("  No extreme advantage patterns found.")
    
    # 5. Feature subgroup analysis (continuous advantages)
    print("\n--- Running Feature Subgroup Analysis ---")
    subgroup_dir = OUT_DIR_OUTLIER / "subgroup_plots"
    subgroup_dir.mkdir(exist_ok=True, parents=True)
    
    subgroup_results = []
    
    # For each tool, analyze top correlated features
    for tool in tools:
        adv_col = f"adv_{BASE_PREFIX}_{tool}"
        if adv_col not in df_outlier.columns:
            continue
        
        # Get top features correlated with this tool's advantage
        tool_corrs = []
        for feat in feature_cols:
            mask = df_outlier[[feat, adv_col]].notna().all(axis=1)
            if mask.sum() > 20:
                try:
                    rho, p = spearmanr(df_outlier.loc[mask, feat], df_outlier.loc[mask, adv_col])
                    if not np.isnan(rho):
                        tool_corrs.append((feat, abs(rho)))
                except:
                    pass
        
        tool_corrs.sort(key=lambda x: x[1], reverse=True)
        top_features = [f for f, _ in tool_corrs[:TOP_N_OUTLIER_FEATURES]]
        
        print(f"\n  {tool}: Analyzing top {len(top_features)} features")
        
        for feat in top_features:
            result = analyze_feature_subgroups(df_outlier, feat, adv_col, N_FEATURE_BINS, subgroup_dir)
            if result:
                result['tool'] = tool
                subgroup_results.append(result)
    
    if subgroup_results:
        subgroup_df = pd.DataFrame(subgroup_results)
        subgroup_df = subgroup_df.sort_values('delta', key=abs, ascending=False)
        subgroup_df.to_csv(OUT_DIR_OUTLIER / "subgroup_analysis_results.csv", index=False)
        print(f"\nSaved: {OUT_DIR_OUTLIER / 'subgroup_analysis_results.csv'}")
        print(f"  Total subgroup comparisons: {len(subgroup_df)}")
    
    # 6. Outlier scatter plots for top features
    print("\n--- Creating Outlier Scatter Plots ---")
    scatter_dir = OUT_DIR_OUTLIER / "outlier_scatters"
    scatter_dir.mkdir(exist_ok=True, parents=True)
    
    for tool in tools:
        adv_col = f"adv_{BASE_PREFIX}_{tool}"
        if adv_col not in df_outlier.columns:
            continue
        
        # Get top 5 features for this tool
        tool_corrs = []
        for feat in feature_cols:
            mask = df_outlier[[feat, adv_col]].notna().all(axis=1)
            if mask.sum() > 20:
                try:
                    rho, p = spearmanr(df_outlier.loc[mask, feat], df_outlier.loc[mask, adv_col])
                    if not np.isnan(rho) and abs(rho) >= PRACTICAL_SIG_CORR:
                        tool_corrs.append((feat, abs(rho)))
                except:
                    pass
        
        tool_corrs.sort(key=lambda x: x[1], reverse=True)
        top_5_features = [f for f, _ in tool_corrs[:5]]
        
        for feat in top_5_features:
            plot_outlier_scatterplot(df_outlier, feat, adv_col, tool, scatter_dir)
    
    print(f"\nDone. Outlier analysis outputs in: {OUT_DIR_OUTLIER}")

# ======================================
# 4. CLUSTERED HEATMAPS (unchanged)
# ======================================

def select_rows_for_cluster(dsub, max_rows):
    """Select representative rows from a cluster."""
    if len(dsub) <= max_rows:
        return dsub
    
    if SELECTION_MODE == "variance":
        pivot = dsub.pivot_table(index="feature", columns="target", values="corr")
        variance = pivot.std(axis=1, skipna=True).sort_values(ascending=False)
        top_features = variance.head(max_rows).index
        return dsub[dsub["feature"].isin(top_features)]
        
    elif SELECTION_MODE == "max_abs":
        scores = dsub.groupby("feature")["corr"].apply(lambda s: s.abs().max())
        top_features = scores.nlargest(max_rows).index
        return dsub[dsub["feature"].isin(top_features)]
        
    elif SELECTION_MODE == "representative":
        all_corrs = dsub.groupby("feature")["corr"].apply(lambda s: s.abs().max())
        bins = np.linspace(0, 1, 6)
        selected = []
        per_bin = max(1, max_rows // (len(bins) - 1))
        
        for i in range(len(bins) - 1):
            in_bin = all_corrs[(all_corrs >= bins[i]) & (all_corrs < bins[i+1])]
            if len(in_bin) > 0:
                n_sample = min(len(in_bin), per_bin)
                selected.extend(in_bin.nlargest(n_sample).index.tolist())
        
        return dsub[dsub["feature"].isin(selected[:max_rows])]
    
    return dsub.head(max_rows)

def plot_cluster_heatmap(dsub, cluster_name, out_dir):
    """Plot heatmap for a feature cluster."""
    pivot = dsub.pivot(index="feature", columns="target", values="corr")
    
    if pivot.empty or pivot.shape[0] < 2:
        return
    
    from scipy.cluster.hierarchy import linkage, dendrogram
    from scipy.spatial.distance import pdist
    
    distances = pdist(pivot.fillna(0), metric='euclidean')
    linkage_matrix = linkage(distances, method='ward')
    dendro = dendrogram(linkage_matrix, no_plot=True)
    pivot_sorted = pivot.iloc[dendro['leaves']]
    
    fig_height = max(8, pivot_sorted.shape[0] * 0.25)
    fig_width = max(10, pivot_sorted.shape[1] * 0.6)
    
    plt.figure(figsize=(fig_width, fig_height))
    sns.heatmap(pivot_sorted, cmap="RdBu_r", center=0, annot=False,
                vmin=-0.6, vmax=0.6, cbar_kws={'label': 'Spearman ρ'})
    
    plt.title(f"Cluster: {cluster_name}", fontsize=12, fontweight='bold')
    plt.xlabel("Target", fontsize=10)
    plt.ylabel("Feature", fontsize=10)
    plt.xticks(rotation=45, ha='right', fontsize=9)
    plt.yticks(fontsize=8)
    plt.tight_layout()
    
    filename = f"cluster_{safe_name(cluster_name)}_heatmap.png"
    plt.savefig(out_dir / filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_dir / filename}")

def run_clustered_heatmaps():
    """Generate feature-clustered heatmaps."""
    print("\n" + "=" * 60)
    print("CLUSTERED HEATMAPS")
    print("=" * 60)
    
    OUT_DIR_CLUSTERED.mkdir(exist_ok=True, parents=True)
    
    if not IN_CORR_FILE.exists():
        print(f"[WARN] Input file not found: {IN_CORR_FILE}")
        print("Run correlation analysis first.")
        return
    
    df = pd.read_csv(IN_CORR_FILE)
    print(f"Loaded: {df.shape}")
    
    if "corr" not in df.columns or "feature" not in df.columns or "target" not in df.columns:
        raise ValueError("Expected columns: corr, feature, target")
    
    df = df.dropna(subset=["corr"])
    
    if df.empty:
        print("[WARN] No valid correlations after dropping NaNs.")
        return
    
    pivot_for_clustering = df.pivot_table(index="feature", columns="target", values="corr")
    pivot_filled = pivot_for_clustering.fillna(0)
    
    from scipy.cluster.hierarchy import linkage, fcluster
    from scipy.spatial.distance import pdist
    
    distances = pdist(pivot_filled, metric='euclidean')
    linkage_matrix = linkage(distances, method='ward')
    
    n_clusters = min(5, pivot_filled.shape[0])
    cluster_labels = fcluster(linkage_matrix, n_clusters, criterion='maxclust')
    
    pivot_filled["cluster"] = cluster_labels
    df = df.merge(pivot_filled[["cluster"]], left_on="feature", right_index=True, how="left")
    
    print("\nCluster counts:")
    print(df["cluster"].value_counts())
    
    for cluster_name in sorted(df["cluster"].unique()):
        dsub = df[df["cluster"] == cluster_name].copy()
        if dsub.empty:
            continue
        
        dsub = select_rows_for_cluster(dsub, MAX_ROWS_PER_CLUSTER)
        
        if dsub["feature"].nunique() < 2:
            continue
        
        plot_cluster_heatmap(dsub, cluster_name, OUT_DIR_CLUSTERED)
    
    print(f"\nDone. Clustered heatmaps in: {OUT_DIR_CLUSTERED}")

# ======================================
# 5. VENN/UpSet DIAGRAMS (unchanged)
# ======================================

def pivot_success_long(perf_long: pd.DataFrame, threshold: int = None) -> pd.DataFrame:
    """
    Convert long-format performance data to wide boolean format.
    
    Args:
        perf_long: Long format DataFrame with tool performance data
        threshold: Rank threshold (1, 5, or 10). If None, uses FOUND_DEF setting.
    
    Returns:
        Wide format DataFrame with boolean columns indicating if bug was found at threshold.
    """
    df = perf_long.copy()

    if "project" not in df.columns or "bug_id" not in df.columns:
        raise ValueError("Expected columns project and bug_id in IN_FILE.")
    if "tool" not in df.columns:
        raise ValueError("Expected a 'tool' column (long format) in IN_FILE.")

    # Determine "found" based on threshold or FOUND_DEF
    if threshold is not None:
        # Use rank threshold (top@1, top@5, top@10)
        if "rank" not in df.columns:
            raise ValueError(f"threshold={threshold} specified but 'rank' column not found.")
        df["found"] = (df["rank"].notna()) & (df["rank"] <= threshold)
    elif FOUND_DEF == "rank":
        if "rank" not in df.columns:
            raise ValueError("FOUND_DEF='rank' but 'rank' column not found.")
        df["found"] = df["rank"].notna()
    elif FOUND_DEF == "mrr":
        if "mrr" in df.columns:
            df["found"] = df["mrr"].fillna(0.0) > 0
        elif "rank" in df.columns:
            df["found"] = df["rank"].notna()
        else:
            raise ValueError("FOUND_DEF='mrr' but neither 'mrr' nor 'rank' columns exist.")
    else:
        raise ValueError("FOUND_DEF must be 'rank' or 'mrr'.")

    wide = df.pivot_table(
        index=["project", "bug_id"],
        columns="tool",
        values="found",
        aggfunc="max",
        fill_value=False
    )

    wide.columns = [f"found_{c}" for c in wide.columns]
    wide = wide.reset_index()
    return wide

def compute_intersections(wide: pd.DataFrame, tool_cols: list) -> pd.DataFrame:
    """Compute intersection patterns and counts."""
    M = wide[tool_cols].astype(int)
    pattern = M.astype(str).agg("".join, axis=1)
    out = pattern.value_counts().rename_axis("pattern").reset_index(name="count")

    tool_names = [c.replace("found_", "") for c in tool_cols]
    def label_from_pattern(p):
        yes = [tool_names[i] for i, ch in enumerate(p) if ch == "1"]
        return " & ".join(yes) if yes else "None"

    out["label"] = out["pattern"].apply(label_from_pattern)
    return out

def save_basic_summary(wide: pd.DataFrame, tool_cols: list, suffix: str = "") -> pd.DataFrame:
    """Save summary statistics for tool intersections."""
    tool_names = [c.replace("found_", "") for c in tool_cols]
    M = wide[tool_cols].astype(bool).to_numpy()

    counts = {}
    for j, t in enumerate(tool_names):
        counts[f"found_{t}"] = int(M[:, j].sum())

    for j, t in enumerate(tool_names):
        others = np.delete(M, j, axis=1)
        unique = M[:, j] & (~others.any(axis=1) if others.shape[1] else True)
        counts[f"unique_{t}"] = int(unique.sum())

    counts["found_all_tools"] = int(M.all(axis=1).sum())
    counts["found_none"] = int((~M.any(axis=1)).sum())

    summary = pd.DataFrame([counts])
    summary.to_csv(OUT_DIR_VENN / f"intersection_summary{suffix}.csv", index=False)
    return summary

def plot_venn(wide: pd.DataFrame, tool_cols: list, threshold: int = None, suffix: str = ""):
    """Plot Venn diagram (2 or 3 tools only)."""
    tool_names = [c.replace("found_", "") for c in tool_cols]
    sets = [set(wide.loc[wide[c], ["project", "bug_id"]].apply(tuple, axis=1)) for c in tool_cols]

    title_suffix = f" (Top@{threshold})" if threshold else ""
    filename_suffix = f"_top{threshold}" if threshold else ""

    if len(tool_cols) == 2:
        from matplotlib_venn import venn2
        plt.figure(figsize=(6, 5))
        venn2(subsets=sets, set_labels=tool_names)
        plt.title(f"Bug intersections (found by tool){title_suffix}")
        plt.tight_layout()
        plt.savefig(OUT_DIR_VENN / f"venn2_tools{suffix}{filename_suffix}.png", dpi=300)
        plt.close()
    elif len(tool_cols) == 3:
        from matplotlib_venn import venn3
        plt.figure(figsize=(7, 6))
        venn3(subsets=sets, set_labels=tool_names)
        plt.title(f"Bug intersections (found by tool){title_suffix}")
        plt.tight_layout()
        plt.savefig(OUT_DIR_VENN / f"venn3_tools{suffix}{filename_suffix}.png", dpi=300)
        plt.close()
    else:
        raise ValueError("Venn plotting supports only 2 or 3 tools.")

def plot_upset(wide: pd.DataFrame, tool_cols: list, threshold: int = None, suffix: str = ""):
    """Plot UpSet diagram (for 4+ tools)."""
    try:
        from upsetplot import UpSet, from_indicators
    except ImportError:
        raise ImportError(
            "upsetplot is not installed. Install with: pip install upsetplot\n"
            "Or reduce to 3 tools for a Venn diagram."
        )

    tool_names = [c.replace("found_", "") for c in tool_cols]
    data = wide[tool_cols].copy()
    data.columns = tool_names

    title_suffix = f" (Top@{threshold})" if threshold else ""
    filename_suffix = f"_top{threshold}" if threshold else ""

    upset_data = from_indicators(tool_names, data=data)
    plt.figure(figsize=(10, 6))
    UpSet(upset_data, show_counts=True, sort_by="cardinality").plot()
    plt.suptitle(f"Bug intersections (found by tool){title_suffix}")
    plt.tight_layout()
    plt.savefig(OUT_DIR_VENN / f"upset_tools{suffix}{filename_suffix}.png", dpi=300)
    plt.close()

def run_venn_diagrams():
    """Run Venn/UpSet diagram analysis for top@1, top@5, and top@10."""
    print("\n" + "=" * 60)
    print("VENN/UPSET DIAGRAMS")
    print("=" * 60)

    OUT_DIR_VENN.mkdir(exist_ok=True, parents=True)

    if not IN_FILE_TOOL_COMPARISON.exists():
        print(f"[WARN] Tool comparison file not found: {IN_FILE_TOOL_COMPARISON}")
        print("Skipping Venn diagrams.")
        return

    perf = pd.read_csv(IN_FILE_TOOL_COMPARISON)
    print(f"Loaded: {perf.shape} from {IN_FILE_TOOL_COMPARISON}")

    # Check if rank column exists
    if "rank" not in perf.columns:
        print("[WARN] 'rank' column not found. Cannot generate top@1/5/10 diagrams.")
        print("Falling back to default found/not found analysis.")
        thresholds = [None]  # Use default FOUND_DEF
    else:
        thresholds = [1, 5, 10]  # Generate for top@1, top@5, top@10

    for threshold in thresholds:
        threshold_label = f"Top@{threshold}" if threshold else "All Found"
        print(f"\n--- Generating diagrams for {threshold_label} ---")
        
        wide = pivot_success_long(perf, threshold=threshold)
        tool_cols = [c for c in wide.columns if c.startswith("found_")]
        if len(tool_cols) < 2:
            print(f"[WARN] Need at least 2 tools for {threshold_label}. Skipping.")
            continue

        print(f"Tools: {[c.replace('found_', '') for c in tool_cols]}")

        # Compute intersections
        intersections = compute_intersections(wide, tool_cols)
        suffix = f"_top{threshold}" if threshold else ""
        intersections.to_csv(OUT_DIR_VENN / f"intersection_patterns{suffix}.csv", index=False)
        print(f"Saved: {OUT_DIR_VENN / f'intersection_patterns{suffix}.csv'}")

        # Save summary
        summary = save_basic_summary(wide, tool_cols, suffix=suffix)
        print(f"Saved: {OUT_DIR_VENN / f'intersection_summary{suffix}.csv'}")

        # Generate diagrams
        if len(tool_cols) <= 3:
            plot_venn(wide, tool_cols, threshold=threshold, suffix=suffix)
        else:
            plot_upset(wide, tool_cols, threshold=threshold, suffix=suffix)

    print(f"\nDone. Outputs in: {OUT_DIR_VENN}")

# ======================================
# 6. INTERSECTION-BASED FEATURE ANALYSIS
# ======================================

def analyze_intersection_features(df, feature_cols, tools, threshold=None, suffix=""):
    """
    Analyze features of bugs based on which tools can find them.
    
    Compares features between:
    - Bugs that NO tools can find (e.g., "None")
    - Bugs that ALL tools can find (e.g., "All")
    - Other intersection patterns
    
    Args:
        df: DataFrame with features and performance data
        feature_cols: List of feature column names
        tools: List of tool names
        threshold: Rank threshold (1, 5, or 10). If None, uses FOUND_DEF.
        suffix: Suffix for output files
    """
    print(f"\n{'='*80}")
    print(f"INTERSECTION-BASED FEATURE ANALYSIS{suffix}")
    print(f"{'='*80}")
    
    # Use the same approach as run_venn_diagrams: load from tool_comparison_summary.csv
    # This ensures consistency with the Venn/UpSet diagrams
    if not IN_FILE_TOOL_COMPARISON.exists():
        print(f"[WARN] Tool comparison file not found: {IN_FILE_TOOL_COMPARISON}")
        print("Skipping intersection feature analysis.")
        return
    
    perf = pd.read_csv(IN_FILE_TOOL_COMPARISON)
    print(f"Loaded performance data: {perf.shape} from {IN_FILE_TOOL_COMPARISON}")
    
    # Create wide format with found flags (same as Venn diagrams)
    wide = pivot_success_long(perf, threshold=threshold)
    
    # Merge with features from main dataset
    # First, ensure we have a consistent ID column
    if "project" in df.columns and "bug_id" in df.columns:
        merge_cols = ["project", "bug_id"]
    elif "id" in df.columns:
        # Try to split id into project and bug_id
        if "project" not in df.columns or "bug_id" not in df.columns:
            # Create project and bug_id from id if needed
            if "id" in df.columns and "project" not in df.columns:
                df["project"] = df["id"].str.split("-").str[0]
                df["bug_id"] = df["id"].str.split("-").str[1]
        merge_cols = ["project", "bug_id"]
    else:
        print("[WARN] Cannot merge features: missing project/bug_id columns.")
        return
    
    # Merge with features
    wide = wide.merge(df[merge_cols + feature_cols], 
                      on=merge_cols, 
                      how="left")
    
    # Get tool columns
    tool_cols = [c for c in wide.columns if c.startswith("found_")]
    
    # Create intersection pattern
    M = wide[tool_cols].astype(int)
    pattern = M.astype(str).agg("".join, axis=1)
    wide["intersection_pattern"] = pattern
    
    # Create readable labels
    tool_names = [c.replace("found_", "") for c in tool_cols]
    def label_from_pattern(p):
        yes = [tool_names[i] for i, ch in enumerate(p) if ch == "1"]
        return " & ".join(yes) if yes else "None"
    
    wide["intersection_label"] = wide["intersection_pattern"].apply(label_from_pattern)
    
    # Count tools that found each bug
    wide["num_tools_found"] = M.sum(axis=1)
    wide["all_tools"] = (wide["num_tools_found"] == len(tool_cols))
    wide["no_tools"] = (wide["num_tools_found"] == 0)
    
    # Analyze features for different groups
    print(f"\nIntersection pattern counts:")
    pattern_counts = wide["intersection_label"].value_counts()
    for label, count in pattern_counts.items():
        print(f"  {label}: {count}")
    
    # Focus on "None" vs "All" comparison
    none_bugs = wide[wide["no_tools"]].copy()
    all_bugs = wide[wide["all_tools"]].copy()
    
    print(f"\nComparing features:")
    print(f"  Bugs that NO tools found: {len(none_bugs)}")
    print(f"  Bugs that ALL tools found: {len(all_bugs)}")
    
    if len(none_bugs) < 5 or len(all_bugs) < 5:
        print(f"[WARN] Insufficient samples for comparison (min=5). Skipping statistical tests.")
        return
    
    # Statistical comparison for each feature
    results = []
    for feat in feature_cols:
        if feat not in wide.columns:
            continue
        
        x = none_bugs[feat].dropna()
        y = all_bugs[feat].dropna()
        
        if len(x) < 5 or len(y) < 5:
            continue
        
        # Mann-Whitney U test
        try:
            from scipy.stats import mannwhitneyu
            u_stat, p_val = mannwhitneyu(x, y, alternative="two-sided", method="asymptotic")
        except:
            continue
        
        # Cliff's Delta
        delta = cliffs_delta(x.values, y.values)
        
        # Effect size interpretation
        if abs(delta) < 0.147:
            effect_size = "negligible"
        elif abs(delta) < 0.33:
            effect_size = "small"
        elif abs(delta) < 0.474:
            effect_size = "medium"
        else:
            effect_size = "large"
        
        results.append({
            "feature": feat,
            "none_count": len(x),
            "all_count": len(y),
            "none_median": x.median(),
            "all_median": y.median(),
            "none_mean": x.mean(),
            "all_mean": y.mean(),
            "none_std": x.std(),
            "all_std": y.std(),
            "u_statistic": u_stat,
            "p_value": p_val,
            "cliffs_delta": delta,
            "effect_size": effect_size,
            "median_diff": y.median() - x.median(),
            "mean_diff": y.mean() - x.mean()
        })
    
    results_df = pd.DataFrame(results)
    
    if len(results_df) == 0:
        print(f"[WARN] No valid feature comparisons. Skipping.")
        return
    
    # Apply Holm-Bonferroni correction
    results_df = apply_holm(results_df, alpha=ALPHA, pval_col="p_value")
    
    # Add "significant" column for compatibility (based on "reject" from apply_holm)
    if "reject" in results_df.columns:
        results_df["significant"] = results_df["reject"]
    else:
        results_df["significant"] = results_df["p_value"] < ALPHA
    
    # Sort by absolute Cliff's Delta
    results_df["abs_delta"] = results_df["cliffs_delta"].abs()
    results_df = results_df.sort_values("abs_delta", ascending=False)
    
    # Save results
    out_file = OUT_DIR_VENN / f"intersection_feature_comparison{suffix}.csv"
    results_df.to_csv(out_file, index=False)
    print(f"\nSaved: {out_file}")
    
    # Print top features
    print(f"\nTop 10 features with largest differences (None vs All):")
    print("=" * 80)
    top_features = results_df.head(10)
    for _, row in top_features.iterrows():
        # Check if significant column exists (from apply_holm)
        sig = "***" if "significant" in row and row["significant"] else ""
        print(f"{row['feature']:<50} | δ={row['cliffs_delta']:>6.3f} ({row['effect_size']:<8}) | "
              f"p={row['p_value']:.4f} {sig}")
        print(f"  None: median={row['none_median']:.2f}, mean={row['none_mean']:.2f}")
        print(f"  All:  median={row['all_median']:.2f}, mean={row['all_mean']:.2f}")
    
    # Create visualization: heatmap of top features
    top_n = min(20, len(results_df))
    top_results = results_df.head(top_n)
    
    # Prepare data for heatmap
    heatmap_data = []
    for _, row in top_results.iterrows():
        heatmap_data.append({
            "feature": row["feature"],
            "Cliff's Delta": row["cliffs_delta"],
            "p_value": row["p_value"],
            "significant": row["significant"]
        })
    
    heatmap_df = pd.DataFrame(heatmap_data)
    heatmap_df = heatmap_df.set_index("feature")
    
    # Create heatmap
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    plt.figure(figsize=(8, max(6, top_n * 0.3)))
    sns.heatmap(heatmap_df[["Cliff's Delta"]], 
                annot=False, 
                cmap="RdBu_r", 
                center=0,
                vmin=-1, vmax=1,
                cbar_kws={"label": "Cliff's Delta (None vs All)"})
    plt.title(f"Feature Differences: Bugs Found by None vs All Tools{suffix}")
    plt.ylabel("")
    plt.tight_layout()
    
    out_file = OUT_DIR_VENN / f"intersection_feature_heatmap{suffix}.png"
    plt.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_file}")
    
    # Create boxplots for top features
    top_5_features = results_df.head(5)["feature"].tolist()
    for feat in top_5_features:
        if feat not in wide.columns:
            continue
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        data_to_plot = []
        labels = []
        for group_name, group_data in [("None", none_bugs), ("All", all_bugs)]:
            values = group_data[feat].dropna()
            if len(values) > 0:
                data_to_plot.append(values)
                labels.append(f"{group_name}\n(n={len(values)})")
        
        if len(data_to_plot) == 2:
            bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True)
            bp["boxes"][0].set_facecolor("lightcoral")
            bp["boxes"][1].set_facecolor("lightblue")
            
            ax.set_ylabel(feat)
            ax.set_title(f"Feature Distribution: Bugs Found by None vs All Tools{suffix}")
            plt.tight_layout()
            
            out_file = OUT_DIR_VENN / f"intersection_boxplot_{safe_name(feat)}{suffix}.png"
            plt.savefig(out_file, dpi=300, bbox_inches="tight")
            plt.close()
            print(f"Saved: {out_file}")
    
    print(f"\nDone. Outputs in: {OUT_DIR_VENN}")

def run_intersection_feature_analysis():
    """Run intersection-based feature analysis for different thresholds."""
    df, feature_cols, id_cols, perf_cols = load_data_and_features(IN_FILE)
    
    # Get tools from performance data
    tools = get_tools(df, "mrr")
    if not tools:
        print("[WARN] No tools found. Skipping intersection feature analysis.")
        return
    
    # Run for different thresholds
    thresholds = [1, 5, 10, None]
    for threshold in thresholds:
        suffix = f"_top{threshold}" if threshold else ""
        print(f"\n{'='*80}")
        print(f"Running intersection feature analysis for threshold: {threshold if threshold else 'All'}")
        print(f"{'='*80}")
        analyze_intersection_features(df, feature_cols, tools, threshold=threshold, suffix=suffix)

# ======================================
# 7. UNIQUE TOOL SUCCESS ANALYSIS
# ======================================

def analyze_unique_tool_successes(df, feature_cols, tools, threshold=None, suffix=""):
    """
    Analyze features of bugs that are uniquely found by each tool.
    
    For each tool, identifies bugs that:
    - Are found by that tool within threshold
    - Are NOT found by any other tool within threshold
    
    Then compares features of these unique bugs across tools and vs bugs found by all tools.
    
    Args:
        df: DataFrame with features and performance data
        feature_cols: List of feature column names
        tools: List of tool names
        threshold: Rank threshold (1, 5, or 10). If None, uses FOUND_DEF.
        suffix: Suffix for output files
    """
    print(f"\n{'='*80}")
    print(f"UNIQUE TOOL SUCCESS ANALYSIS{suffix}")
    print(f"{'='*80}")
    
    # Use the same approach as intersection analysis: load from tool_comparison_summary.csv
    if not IN_FILE_TOOL_COMPARISON.exists():
        print(f"[WARN] Tool comparison file not found: {IN_FILE_TOOL_COMPARISON}")
        print("Skipping unique tool success analysis.")
        return
    
    perf = pd.read_csv(IN_FILE_TOOL_COMPARISON)
    print(f"Loaded performance data: {perf.shape} from {IN_FILE_TOOL_COMPARISON}")
    
    # Create wide format with found flags
    wide = pivot_success_long(perf, threshold=threshold)
    
    # Merge with features from main dataset
    if "project" in df.columns and "bug_id" in df.columns:
        merge_cols = ["project", "bug_id"]
    elif "id" in df.columns:
        if "project" not in df.columns or "bug_id" not in df.columns:
            if "id" in df.columns and "project" not in df.columns:
                df["project"] = df["id"].str.split("-").str[0]
                df["bug_id"] = df["id"].str.split("-").str[1]
        merge_cols = ["project", "bug_id"]
    else:
        print("[WARN] Cannot merge features: missing project/bug_id columns.")
        return
    
    # Merge with features
    wide = wide.merge(df[merge_cols + feature_cols], 
                      on=merge_cols, 
                      how="left")
    
    # Get tool columns
    tool_cols = [c for c in wide.columns if c.startswith("found_")]
    tool_names = [c.replace("found_", "") for c in tool_cols]
    
    # For each tool, identify bugs that are uniquely found by that tool
    print(f"\nIdentifying unique bugs for each tool:")
    print("=" * 60)
    
    unique_bugs_by_tool = {}
    for tool_name in tool_names:
        tool_col = f"found_{tool_name}"
        if tool_col not in wide.columns:
            continue
        
        # Bugs found by this tool
        found_by_tool = wide[wide[tool_col] == True].copy()
        
        # Bugs found by other tools
        other_tool_cols = [c for c in tool_cols if c != tool_col]
        found_by_others = wide[other_tool_cols].any(axis=1)
        
        # Unique bugs: found by this tool but NOT by any other tool
        unique_mask = (wide[tool_col] == True) & (~found_by_others)
        unique_bugs = wide[unique_mask].copy()
        
        unique_bugs_by_tool[tool_name] = unique_bugs
        
        print(f"\n{tool_name}:")
        print(f"  Total found by {tool_name}: {len(found_by_tool)}")
        print(f"  Unique to {tool_name} (not found by others): {len(unique_bugs)}")
    
    # Compare features of unique bugs across tools
    print(f"\n\nComparing features of unique bugs across tools:")
    print("=" * 60)
    
    # Statistical comparison: each tool's unique bugs vs all other unique bugs combined
    all_results = []
    
    for tool_name, unique_bugs in unique_bugs_by_tool.items():
        if len(unique_bugs) < 5:
            print(f"\n{tool_name}: Skipping (only {len(unique_bugs)} unique bugs, need at least 5)")
            continue
        
        # Combine unique bugs from all other tools for comparison
        other_bugs_list = [
            bugs for other_tool, bugs in unique_bugs_by_tool.items() 
            if other_tool != tool_name and len(bugs) >= 5
        ]
        
        if len(other_bugs_list) == 0:
            print(f"\n{tool_name}: Skipping (no other tools with unique bugs for comparison)")
            continue
        
        other_unique_bugs = pd.concat(other_bugs_list, ignore_index=True)
        
        if len(other_unique_bugs) < 5:
            print(f"\n{tool_name}: Skipping (insufficient comparison group: {len(other_unique_bugs)} bugs)")
            continue
        
        print(f"\n{tool_name} (n={len(unique_bugs)}) vs Other tools' unique bugs (n={len(other_unique_bugs)}):")
        
        for feat in feature_cols:
            if feat not in wide.columns:
                continue
            
            x = unique_bugs[feat].dropna()
            y = other_unique_bugs[feat].dropna()
            
            if len(x) < 5 or len(y) < 5:
                continue
            
            # Mann-Whitney U test
            try:
                from scipy.stats import mannwhitneyu
                u_stat, p_val = mannwhitneyu(x, y, alternative="two-sided", method="asymptotic")
            except:
                continue
            
            # Cliff's Delta
            delta = cliffs_delta(x.values, y.values)
            
            # Effect size interpretation
            if abs(delta) < 0.147:
                effect_size = "negligible"
            elif abs(delta) < 0.33:
                effect_size = "small"
            elif abs(delta) < 0.474:
                effect_size = "medium"
            else:
                effect_size = "large"
            
            all_results.append({
                "tool": tool_name,
                "feature": feat,
                "unique_count": len(x),
                "other_count": len(y),
                "unique_median": x.median(),
                "other_median": y.median(),
                "unique_mean": x.mean(),
                "other_mean": y.mean(),
                "unique_std": x.std(),
                "other_std": y.std(),
                "u_statistic": u_stat,
                "p_value": p_val,
                "cliffs_delta": delta,
                "effect_size": effect_size,
                "median_diff": x.median() - y.median(),
                "mean_diff": x.mean() - y.mean()
            })
    
    if len(all_results) == 0:
        print(f"[WARN] No valid comparisons. Skipping.")
        return
    
    results_df = pd.DataFrame(all_results)
    
    # Apply Holm-Bonferroni correction per tool
    results_df_list = []
    for tool_name in results_df["tool"].unique():
        tool_results = results_df[results_df["tool"] == tool_name].copy()
        tool_results = apply_holm(tool_results, alpha=ALPHA, pval_col="p_value")
        if "reject" in tool_results.columns:
            tool_results["significant"] = tool_results["reject"]
        else:
            tool_results["significant"] = tool_results["p_value"] < ALPHA
        results_df_list.append(tool_results)
    
    results_df = pd.concat(results_df_list, ignore_index=True)
    
    # Sort by absolute Cliff's Delta
    results_df["abs_delta"] = results_df["cliffs_delta"].abs()
    results_df = results_df.sort_values(["tool", "abs_delta"], ascending=[True, False])
    
    # Save results
    out_file = OUT_DIR_VENN / f"unique_tool_success_comparison{suffix}.csv"
    results_df.to_csv(out_file, index=False)
    print(f"\nSaved: {out_file}")
    
    # Create visualization: heatmap showing top features for each tool
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Get top 10 features per tool
    top_n_per_tool = 10
    heatmap_data = []
    
    for tool_name in sorted(results_df["tool"].unique()):
        tool_results = results_df[results_df["tool"] == tool_name].head(top_n_per_tool)
        for _, row in tool_results.iterrows():
            heatmap_data.append({
                "tool": tool_name,
                "feature": row["feature"],
                "Cliff's Delta": row["cliffs_delta"]
            })
    
    if heatmap_data:
        heatmap_df = pd.DataFrame(heatmap_data)
        pivot_heatmap = heatmap_df.pivot(index="feature", columns="tool", values="Cliff's Delta")
        
        # Sort by average absolute delta across tools
        pivot_heatmap["avg_abs_delta"] = pivot_heatmap.abs().mean(axis=1)
        pivot_heatmap = pivot_heatmap.sort_values("avg_abs_delta", ascending=False)
        pivot_heatmap = pivot_heatmap.drop("avg_abs_delta", axis=1)
        
        # Create heatmap
        plt.figure(figsize=(max(8, len(pivot_heatmap.columns) * 1.5), max(8, len(pivot_heatmap) * 0.3)))
        sns.heatmap(pivot_heatmap, 
                    annot=False, 
                    cmap="RdBu_r", 
                    center=0,
                    vmin=-1, vmax=1,
                    cbar_kws={"label": "Cliff's Delta (Tool's Unique vs Others' Unique)"},
                    yticklabels=True)
        plt.title(f"Feature Differences: Unique Tool Successes{suffix}\n(Each tool's unique bugs vs other tools' unique bugs)")
        plt.ylabel("")
        plt.xlabel("Tool")
        plt.tight_layout()
        
        out_file = OUT_DIR_VENN / f"unique_tool_success_heatmap{suffix}.png"
        plt.savefig(out_file, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved: {out_file}")
    
    # Create boxplots for top features per tool
    top_features_per_tool = 3
    for tool_name in sorted(results_df["tool"].unique()):
        tool_results = results_df[results_df["tool"] == tool_name].head(top_features_per_tool)
        unique_bugs = unique_bugs_by_tool.get(tool_name)
        
        if unique_bugs is None or len(unique_bugs) < 5:
            continue
        
        # Get other tools' unique bugs for comparison
        other_bugs_list = [
            bugs for other_tool, bugs in unique_bugs_by_tool.items() 
            if other_tool != tool_name and len(bugs) >= 5
        ]
        
        if len(other_bugs_list) == 0:
            continue
        
        other_unique_bugs = pd.concat(other_bugs_list, ignore_index=True)
        
        if len(other_unique_bugs) < 5:
            continue
        
        for _, row in tool_results.iterrows():
            feat = row["feature"]
            if feat not in wide.columns:
                continue
            
            fig, ax = plt.subplots(figsize=(8, 6))
            
            data_to_plot = []
            labels = []
            
            tool_values = unique_bugs[feat].dropna()
            other_values = other_unique_bugs[feat].dropna()
            
            if len(tool_values) > 0 and len(other_values) > 0:
                data_to_plot.append(tool_values)
                labels.append(f"{tool_name}\n(n={len(tool_values)})")
                data_to_plot.append(other_values)
                labels.append(f"Other tools\n(n={len(other_values)})")
            
            if len(data_to_plot) == 2:
                bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True)
                bp["boxes"][0].set_facecolor("lightgreen")
                bp["boxes"][1].set_facecolor("lightgray")
                
                ax.set_ylabel(feat)
                ax.set_title(f"Feature Distribution: {tool_name}'s Unique Bugs vs Others' Unique Bugs{suffix}")
                plt.tight_layout()
                
                out_file = OUT_DIR_VENN / f"unique_tool_boxplot_{safe_name(tool_name)}_{safe_name(feat)}{suffix}.png"
                plt.savefig(out_file, dpi=300, bbox_inches="tight")
                plt.close()
                print(f"Saved: {out_file}")
    
    # Also compare each tool's unique bugs vs bugs found by ALL tools
    print(f"\n\nComparing each tool's unique bugs vs bugs found by ALL tools:")
    print("=" * 60)
    
    # Get bugs found by all tools
    all_tools_mask = wide[tool_cols].all(axis=1)
    all_tools_bugs = wide[all_tools_mask].copy()
    
    if len(all_tools_bugs) >= 5:
        all_vs_unique_results = []
        
        for tool_name, unique_bugs in unique_bugs_by_tool.items():
            if len(unique_bugs) < 5:
                continue
            
            print(f"\n{tool_name} unique (n={len(unique_bugs)}) vs All tools (n={len(all_tools_bugs)}):")
            
            for feat in feature_cols:
                if feat not in wide.columns:
                    continue
                
                x = unique_bugs[feat].dropna()
                y = all_tools_bugs[feat].dropna()
                
                if len(x) < 5 or len(y) < 5:
                    continue
                
                try:
                    from scipy.stats import mannwhitneyu
                    u_stat, p_val = mannwhitneyu(x, y, alternative="two-sided", method="asymptotic")
                except:
                    continue
                
                delta = cliffs_delta(x.values, y.values)
                
                if abs(delta) < 0.147:
                    effect_size = "negligible"
                elif abs(delta) < 0.33:
                    effect_size = "small"
                elif abs(delta) < 0.474:
                    effect_size = "medium"
                else:
                    effect_size = "large"
                
                all_vs_unique_results.append({
                    "tool": tool_name,
                    "feature": feat,
                    "unique_count": len(x),
                    "all_count": len(y),
                    "unique_median": x.median(),
                    "all_median": y.median(),
                    "unique_mean": x.mean(),
                    "all_mean": y.mean(),
                    "u_statistic": u_stat,
                    "p_value": p_val,
                    "cliffs_delta": delta,
                    "effect_size": effect_size
                })
        
        if len(all_vs_unique_results) > 0:
            all_vs_unique_df = pd.DataFrame(all_vs_unique_results)
            
            # Apply Holm-Bonferroni correction per tool
            all_vs_unique_list = []
            for tool_name in all_vs_unique_df["tool"].unique():
                tool_results = all_vs_unique_df[all_vs_unique_df["tool"] == tool_name].copy()
                tool_results = apply_holm(tool_results, alpha=ALPHA, pval_col="p_value")
                if "reject" in tool_results.columns:
                    tool_results["significant"] = tool_results["reject"]
                else:
                    tool_results["significant"] = tool_results["p_value"] < ALPHA
                all_vs_unique_list.append(tool_results)
            
            all_vs_unique_df = pd.concat(all_vs_unique_list, ignore_index=True)
            all_vs_unique_df["abs_delta"] = all_vs_unique_df["cliffs_delta"].abs()
            all_vs_unique_df = all_vs_unique_df.sort_values(["tool", "abs_delta"], ascending=[True, False])
            
            out_file = OUT_DIR_VENN / f"unique_vs_all_tools_comparison{suffix}.csv"
            all_vs_unique_df.to_csv(out_file, index=False)
            print(f"\nSaved: {out_file}")
            
            # Create heatmap: unique vs all
            top_n_per_tool = 10
            heatmap_data = []
            
            for tool_name in sorted(all_vs_unique_df["tool"].unique()):
                tool_results = all_vs_unique_df[all_vs_unique_df["tool"] == tool_name].head(top_n_per_tool)
                for _, row in tool_results.iterrows():
                    heatmap_data.append({
                        "tool": tool_name,
                        "feature": row["feature"],
                        "Cliff's Delta": row["cliffs_delta"]
                    })
            
            if heatmap_data:
                heatmap_df = pd.DataFrame(heatmap_data)
                pivot_heatmap = heatmap_df.pivot(index="feature", columns="tool", values="Cliff's Delta")
                
                # Sort by average absolute delta
                pivot_heatmap["avg_abs_delta"] = pivot_heatmap.abs().mean(axis=1)
                pivot_heatmap = pivot_heatmap.sort_values("avg_abs_delta", ascending=False)
                pivot_heatmap = pivot_heatmap.drop("avg_abs_delta", axis=1)
                
                plt.figure(figsize=(max(8, len(pivot_heatmap.columns) * 1.5), max(8, len(pivot_heatmap) * 0.3)))
                sns.heatmap(pivot_heatmap, 
                            annot=False, 
                            cmap="RdBu_r", 
                            center=0,
                            vmin=-1, vmax=1,
                            cbar_kws={"label": "Cliff's Delta (Tool's Unique vs All Tools)"},
                            yticklabels=True)
                plt.title(f"Feature Differences: Each Tool's Unique Bugs vs Bugs Found by All Tools{suffix}")
                plt.ylabel("")
                plt.xlabel("Tool")
                plt.tight_layout()
                
                out_file = OUT_DIR_VENN / f"unique_vs_all_tools_heatmap{suffix}.png"
                plt.savefig(out_file, dpi=300, bbox_inches="tight")
                plt.close()
                print(f"Saved: {out_file}")
    
    print(f"\nDone. Outputs in: {OUT_DIR_VENN}")

def run_unique_tool_success_analysis():
    """Run unique tool success analysis for different thresholds."""
    df, feature_cols, id_cols, perf_cols = load_data_and_features(IN_FILE)
    
    # Get tools from performance data
    tools = get_tools(df, "mrr")
    if not tools:
        print("[WARN] No tools found. Skipping unique tool success analysis.")
        return
    
    # Run for different thresholds
    thresholds = [1, 5, 10, None]
    for threshold in thresholds:
        suffix = f"_top{threshold}" if threshold else ""
        print(f"\n{'='*80}")
        print(f"Running unique tool success analysis for threshold: {threshold if threshold else 'All'}")
        print(f"{'='*80}")
        analyze_unique_tool_successes(df, feature_cols, tools, threshold=threshold, suffix=suffix)

# ======================================
# MAIN EXECUTION
# ======================================

if __name__ == "__main__":
    print("=" * 60)
    print("UNIFIED ANALYSIS SCRIPT - ENHANCED FOR OUTLIER DETECTION")
    print("=" * 60)
    print(f"Correlation Analysis: {RUN_CORRELATION_ANALYSIS}")
    print(f"  - Standardize metrics: {STANDARDIZE_METRICS}")
    print(f"  - Bootstrap CI: {BOOTSTRAP_CI}")
    print(f"  - Stratify by project: {STRATIFY_BY_PROJECT}")
    print(f"  - Practical significance: |ρ| ≥ {PRACTICAL_SIG_CORR}")
    print(f"Success/Failure Analysis: {RUN_SUCCESS_FAILURE_ANALYSIS}")
    print(f"  - Label mode: {LABEL_MODE}")
    print(f"  - Practical significance: |δ| ≥ {PRACTICAL_SIG_DELTA}")
    print(f"Outlier Analysis: {RUN_OUTLIER_ANALYSIS}  [NEW]")
    print(f"  - Unique success threshold: {UNIQUE_SUCCESS_THRESHOLD}")
    print(f"  - Extreme advantage z-threshold: {EXTREME_ADV_Z_THRESHOLD}")
    print(f"  - Top features visualized: {TOP_N_OUTLIER_FEATURES}")
    print(f"Clustered Heatmaps: {RUN_CLUSTERED_HEATMAPS}")
    print(f"Venn Diagrams: {RUN_VENN_DIAGRAMS}")
    print(f"Intersection Feature Analysis: {RUN_INTERSECTION_FEATURE_ANALYSIS}")
    print(f"Unique Tool Success Analysis: {RUN_UNIQUE_TOOL_SUCCESS_ANALYSIS}")
    print(f"Diagnostics: {RUN_DIAGNOSTICS}")
    print("=" * 60)

    if RUN_DIAGNOSTICS:
        df, feature_cols, id_cols, perf_cols = load_data_and_features(IN_FILE)
        run_diagnostic_checks(df, feature_cols, perf_cols)

    if RUN_CORRELATION_ANALYSIS:
        run_correlation_analysis()

    if RUN_SUCCESS_FAILURE_ANALYSIS:
        run_success_failure_analysis()
    
    if RUN_OUTLIER_ANALYSIS:
        run_outlier_analysis()

    if RUN_CLUSTERED_HEATMAPS:
        run_clustered_heatmaps()

    if RUN_VENN_DIAGRAMS:
        run_venn_diagrams()
    
    if RUN_INTERSECTION_FEATURE_ANALYSIS:
        run_intersection_feature_analysis()
    
    if RUN_UNIQUE_TOOL_SUCCESS_ANALYSIS:
        run_unique_tool_success_analysis()

    print("\n" + "=" * 60)
    print("ALL ANALYSES COMPLETE")
    print("=" * 60)