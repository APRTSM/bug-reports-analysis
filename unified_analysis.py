"""
Enhanced Focused Tool Comparison Feature Analysis

This script analyzes features that distinguish between different tool performance patterns:
1. Bugs ALL tools find vs bugs NONE can find
2. Bugs only BugLocator finds vs bugs only FlexFL finds (pairwise comparisons)
3. Bugs uniquely found by each tool vs bugs found by all others
4. Tool complementarity patterns with UpSet diagrams

ENHANCEMENTS:
- Support for enhanced semantic features (embeddings, clustering)
- Support for LLM interaction features
- UpSet diagrams for Top-1, Top-5, Top-10
- Better feature categorization and filtering
"""

import re
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests
from itertools import combinations

warnings.filterwarnings('ignore', category=RuntimeWarning)

# Try to import upsetplot for UpSet diagrams
try:
    from upsetplot import UpSet, from_memberships
    UPSET_AVAILABLE = True
except ImportError:
    print("Warning: upsetplot not installed. UpSet diagrams will be skipped.")
    print("Install with: pip install upsetplot")
    UPSET_AVAILABLE = False

# ======================================
# CONFIGURATION
# ======================================
DATA_DIR = Path(".")

# Input files - UPDATED to use enhanced preprocessed data
IN_FILE = DATA_DIR / "full_feature_preproccessed_fixed/experimentA_preprocessed_rich.csv"
IN_FILE_TOOL_COMPARISON = DATA_DIR / "tool_comparison_summary.csv"

# Output directory
OUT_DIR = DATA_DIR / "tool_comparison_results_fixed"
OUT_DIR.mkdir(exist_ok=True)

# Analysis settings
ALPHA = 0.05
PRACTICAL_SIG_DELTA = 0.2  # Minimum Cliff's delta to consider meaningful
MIN_GROUP_SIZE = 8  # Minimum bugs per group for reliable comparison

# Tool names (will be auto-detected but can be specified)
EXPECTED_TOOLS = ["buglocator", "flexfl", "locus", "boostnsift"]


# Rank thresholds to analyze
THRESHOLDS = [1, 5, 10]  # Top-1, Top-5, Top-10

# NEW: Feature categorization for targeted analysis
FEATURE_CATEGORIES = {
    'syntactic': [
        'n_tokens', 'n_words', 'n_sentences', 'flesch_reading_ease', 'smog_index',
        'has_stacktrace', 'has_code', 'has_patch', 'has_enumeration',
        'num_causal_markers', 'num_temporal_markers', 'completeness_score'
    ],
    'semantic_diversity': [
        'semantic_entropy', 'semantic_spread_pc1', 'semantic_spread_pc2',
        'semantic_coherence', 'redundancy', 'ambiguity'
    ],
    'embedding': [
        'embedding_norm', 'embedding_mean', 'embedding_std', 'embedding_sparsity',
        'embedding_cluster', 'embedding_cluster_distance'
    ],
    'exception': [
        'primary_exception_type', 'num_exception_types', 'stacktrace_depth',
        'exc_cat_null_pointer', 'exc_cat_type_error', 'exc_cat_io_error',
        'exception_user_frame_ratio'
    ],
    'llm_quality': [
        'actionability', 'clarity', 'specificity', 'technical_depth',
        'quality_composite', 'technical_completeness'
    ],
    'llm_reasoning': [
        'causal_reasoning_quality', 'reasoning_composite', 'hidden_s2r_present',
        'causal_reasoning_consistency'
    ],
    'ambiguity_types': [
        'has_reproduction_ambiguity', 'has_input_ambiguity', 'has_error_ambiguity',
        'ambiguity_type_count', 'ambiguity_category_coverage'
    ],
    'concepts': [
        'concept_network_concept_diversity', 'concept_network_has_cross_layer_concepts',
        'concept_network_concept_breadth'
    ]
}

# ======================================
# HELPER FUNCTIONS
# ======================================

def safe_name(s: str) -> str:
    """Convert string to safe filename."""
    if not isinstance(s, str):
        s = str(s)
    s = re.sub(r"[^a-zA-Z0-9_]+", "_", s)
    return s.strip("_").lower()

def get_tools(df, prefix="mrr"):
    """Extract tool names from columns with given prefix."""
    cols = [c for c in df.columns if c.startswith(prefix + "_")]
    cols = [c for c in cols if not (c.endswith("__is_missing") or c.endswith("_is_missing"))]
    tools = sorted({c.split("_", 1)[1] for c in cols})
    return tools

def cliffs_delta(x, y):
    """Compute Cliff's delta effect size."""
    x, y = np.asarray(x), np.asarray(y)
    x, y = x[~np.isnan(x)], y[~np.isnan(y)]
    if len(x) == 0 or len(y) == 0:
        return np.nan
    
    n_x, n_y = len(x), len(y)
    dominance = sum(1 for xi in x for yi in y if xi > yi)
    subordinance = sum(1 for xi in x for yi in y if xi < yi)
    
    delta = (dominance - subordinance) / (n_x * n_y)
    return delta

def apply_holm(df_results, alpha=0.05, pval_col="p_value"):
    """Apply Holm-Bonferroni correction for multiple testing."""
    df_results = df_results.copy()
    mask = df_results[pval_col].notna()
    pvals = df_results.loc[mask, pval_col].values
    
    if len(pvals) == 0:
        df_results["pval_adj"] = np.nan
        df_results["reject"] = False
        df_results["significant"] = False
        return df_results
    
    reject, p_adj, _, _ = multipletests(pvals, alpha=alpha, method="holm")
    df_results.loc[mask, "pval_adj"] = p_adj
    df_results.loc[mask, "reject"] = reject
    df_results.loc[mask, "significant"] = reject
    df_results["reject"] = df_results["reject"].fillna(False).astype(bool)
    df_results["significant"] = df_results["significant"].fillna(False).astype(bool)
    
    return df_results

def effect_size_label(delta):
    """Get effect size label for Cliff's delta."""
    abs_delta = abs(delta)
    if abs_delta < 0.147:
        return "negligible"
    elif abs_delta < 0.33:
        return "small"
    elif abs_delta < 0.474:
        return "medium"
    else:
        return "large"

def categorize_features(feature_cols, available_cols):
    """
    Categorize features into groups and detect which enhanced features are present.
    Returns dict of {category: [features]}.
    """
    categorized = {}
    uncategorized = []
    
    for feature in feature_cols:
        if feature not in available_cols:
            continue
            
        found_category = False
        for category, patterns in FEATURE_CATEGORIES.items():
            # Check if feature matches any pattern in category
            if any(pattern in feature for pattern in patterns):
                if category not in categorized:
                    categorized[category] = []
                categorized[category].append(feature)
                found_category = True
                break
        
        if not found_category:
            uncategorized.append(feature)
    
    if uncategorized:
        categorized['other'] = uncategorized
    
    return categorized

def load_data():
    """Load feature data and tool comparison data."""
    # Load feature data
    df_features = pd.read_csv(IN_FILE)
    print(f"Loaded feature data: {df_features.shape}")
    
    # Extract feature columns (numeric, excluding performance metrics)
    id_cols = [c for c in ["project", "bug_id", "id"] if c in df_features.columns]
    perf_cols = [c for c in df_features.columns if c.startswith("mrr_") or c.startswith("top@") or c.startswith("rank_")]
    numeric_cols = df_features.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [c for c in numeric_cols if c not in perf_cols + id_cols]
    
    # Exclude missingness features
    feature_cols = [c for c in feature_cols if not (c.endswith("_is_missing") or "__is_missing" in c)]
    
    # Exclude redundant features (identical to other features)
    redundant = ['fine_grained_description_length', 'description_length_fine_grained']
    feature_cols = [c for c in feature_cols if c not in redundant]
    
    # Exclude text-derived word_count and char_len features
    text_stats_to_exclude = [c for c in feature_cols if c.startswith('txt_') and 
                             (c.endswith('_word_count') or c.endswith('_char_len'))]
    feature_cols = [c for c in feature_cols if c not in text_stats_to_exclude]
    if text_stats_to_exclude:
        print(f"  Excluded {len(text_stats_to_exclude)} text-derived word_count/char_len features from analysis")
        print(f"    Examples: {text_stats_to_exclude[:5]}")
    
    # Deduplicate text-derived features: keep one of each pair (prefer _density over _count)
    txt_features = [c for c in feature_cols if c.startswith('txt_')]
    if txt_features:
        # First, handle special case: avg_sentence_len vs avg_words_per_line
        # Group by source (everything before _avg_)
        sentence_line_pairs = {}
        for feat in txt_features:
            if '_avg_sentence_len' in feat:
                source = feat.replace('_avg_sentence_len', '')
                if source not in sentence_line_pairs:
                    sentence_line_pairs[source] = {'sentence_len': None, 'words_per_line': None}
                sentence_line_pairs[source]['sentence_len'] = feat
            elif '_avg_words_per_line' in feat:
                source = feat.replace('_avg_words_per_line', '')
                if source not in sentence_line_pairs:
                    sentence_line_pairs[source] = {'sentence_len': None, 'words_per_line': None}
                sentence_line_pairs[source]['words_per_line'] = feat
        
        # For pairs where both exist, keep avg_sentence_len (drop avg_words_per_line)
        features_to_drop_special = []
        for source, pair in sentence_line_pairs.items():
            if pair['sentence_len'] and pair['words_per_line']:
                features_to_drop_special.append(pair['words_per_line'])
        
        # Remove the special case features from txt_features for the general deduplication
        txt_features_filtered = [f for f in txt_features if f not in features_to_drop_special]
        
        # Group features by base name (everything except the last suffix)
        feature_groups = {}
        for feat in txt_features_filtered:
            parts = feat.split('_')
            if len(parts) >= 3:
                base = '_'.join(parts[:-1])  # Everything except last part
                suffix = parts[-1]
                if base not in feature_groups:
                    feature_groups[base] = []
                feature_groups[base].append((feat, suffix))
        
        # For groups with multiple features, keep one based on priority
        features_to_keep = set()
        features_to_drop = []
        
        # Priority order: prefer density over count, then alphabetical
        priority_order = ['density', 'count', 'ratio', 'avg', 'median', 'min', 'max']
        
        for base, features in feature_groups.items():
            if len(features) > 1:
                # Sort by priority
                def get_priority(item):
                    suffix = item[1]
                    if suffix in priority_order:
                        return priority_order.index(suffix)
                    return len(priority_order)  # Lower priority for others
                
                features_sorted = sorted(features, key=get_priority)
                # Keep the first (highest priority)
                keep_feat = features_sorted[0][0]
                features_to_keep.add(keep_feat)
                # Drop the rest
                for feat, _ in features_sorted[1:]:
                    features_to_drop.append(feat)
            else:
                # Single feature, keep it
                features_to_keep.add(features[0][0])
        
        # Combine all features to drop
        all_features_to_drop = features_to_drop + features_to_drop_special
        
        # Update feature_cols: keep non-txt features and selected txt features
        feature_cols = [c for c in feature_cols if not c.startswith('txt_')] + list(features_to_keep)
        
        if all_features_to_drop:
            print(f"  Deduplicated text features: kept {len(features_to_keep)} txt_ features, dropped {len(all_features_to_drop)} duplicates")
            if features_to_drop_special:
                print(f"    Dropped avg_words_per_line features (kept avg_sentence_len): {features_to_drop_special}")
            if features_to_drop:
                print(f"    Examples of other dropped: {features_to_drop[:5]}")
    
    # Exclude confidence features from analysis (but keep in summary statistics)
    confidence_features = [c for c in feature_cols if 'confidence' in c.lower()]
    feature_cols = [c for c in feature_cols if c not in confidence_features]
    if confidence_features:
        print(f"  Excluded {len(confidence_features)} confidence features from analysis: {confidence_features}")
    
    # Categorize features
    categorized_features = categorize_features(feature_cols, df_features.columns)
    
    print(f"Feature columns: {len(feature_cols)}")
    print(f"\nFeature breakdown by category:")
    for category, features in categorized_features.items():
        print(f"  {category}: {len(features)} features")
    
    # Check for enhanced features
    enhanced_indicators = ['semantic_', 'embedding_', 'exc_cat_', 'quality_composite', 
                          'concept_network_', 'ambiguity_type']
    enhanced_features = [f for f in feature_cols if any(ind in f for ind in enhanced_indicators)]
    if enhanced_features:
        print(f"\n✓ Enhanced features detected: {len(enhanced_features)} features")
        print(f"  Examples: {enhanced_features[:5]}")
    else:
        print(f"\n⚠ No enhanced features detected. Consider using enhanced preprocessing.")
    
    # Load tool comparison data
    df_tools_raw = pd.read_csv(IN_FILE_TOOL_COMPARISON)
    print(f"\nLoaded tool comparison: {df_tools_raw.shape}")
    
    # Pivot from long to wide format if needed
    if "tool" in df_tools_raw.columns:
        # Data is in long format, pivot to wide
        print("  Pivoting tool comparison data from long to wide format...")
        
        # Get unique tools (normalize to lowercase for consistency)
        tools_raw = df_tools_raw["tool"].unique()
        tools = sorted([t.lower() for t in tools_raw])
        print(f"  Tools found: {tools}")
        
        # Pivot rank, mrr, and top@ columns
        pivot_cols = ["rank", "mrr", "top@1", "top@5", "top@10"]
        available_pivot_cols = [c for c in pivot_cols if c in df_tools_raw.columns]
        
        if not available_pivot_cols:
            print("  [WARN] No pivotable columns found. Trying to use 'detected' column...")
            # Fallback: use detected column
            df_tools = df_tools_raw.pivot_table(
                index=["project", "bug_id"],
                columns="tool",
                values="detected",
                aggfunc="first"
            )
            df_tools.columns = [f"detected_{tool}" for tool in df_tools.columns]
            df_tools = df_tools.reset_index()
        else:
            # Pivot available columns
            df_tools = df_tools_raw.pivot_table(
                index=["project", "bug_id"],
                columns="tool",
                values=available_pivot_cols,
                aggfunc="first"
            )
            # Flatten column names: (metric, tool) -> metric_tool
            # Handle @ symbol in column names by replacing with underscore
            # Normalize tool names to lowercase
            new_cols = []
            for metric, tool_orig in df_tools.columns:
                # Replace @ with underscore for valid column names
                metric_clean = metric.replace("@", "_")
                # Normalize tool name to lowercase
                tool_normalized = tool_orig.lower()
                new_cols.append(f"{metric_clean}_{tool_normalized}")
            df_tools.columns = new_cols
            df_tools = df_tools.reset_index()
        
        print(f"  Pivoted shape: {df_tools.shape}")
    else:
        # Data is already in wide format
        df_tools = df_tools_raw
        # Get tool names from columns (try multiple prefixes)
        tools = get_tools(df_tools, "rank")
        if not tools:
            tools = get_tools(df_tools, "mrr")
        if not tools:
            tools = get_tools(df_features, "mrr")
        if not tools:
            tools = [t.lower() for t in EXPECTED_TOOLS]
    
    # Normalize tool names to lowercase for consistency
    tools = [t.lower() for t in tools]
    print(f"Tools detected: {tools}")
    
    return df_features, df_tools, feature_cols, categorized_features, tools, id_cols

def create_success_flags(df_tools, tools, threshold):
    """Create binary success flags for each tool at given threshold."""
    result = df_tools[["project", "bug_id"]].copy()
    
    for tool in tools:
        rank_col = f"rank_{tool}"
        # Handle @ symbol in column names (pivoted columns use underscore)
        top_col = f"top_{threshold}_{tool}" if threshold in [1, 5, 10] else None
        mrr_col = f"mrr_{tool}"
        detected_col = f"detected_{tool}"
        
        # Try multiple methods to determine if bug was found
        found = None
        
        # Method 1: Use rank column if available
        if rank_col in df_tools.columns:
            found = df_tools[rank_col] <= threshold
        # Method 2: Use top@N column if available
        elif top_col and top_col in df_tools.columns:
            found = df_tools[top_col] == 1
        # Method 3: Calculate rank from MRR (rank = 1/mrr if mrr > 0)
        elif mrr_col in df_tools.columns:
            mrr = df_tools[mrr_col].fillna(0)
            calculated_rank = np.where(mrr > 0, 1.0 / mrr, np.inf)
            found = calculated_rank <= threshold
        # Method 4: Use detected column if available
        elif detected_col in df_tools.columns:
            found = df_tools[detected_col].fillna("No").str.contains("Yes", case=False, na=False)
        else:
            print(f"[WARN] Missing column: {rank_col} (and no fallback columns found)")
            result[f"found_{tool}"] = False
            continue
        
        result[f"found_{tool}"] = found
    
    return result


# ======================================
# NEW: UPSET DIAGRAM FUNCTION
# ======================================

def create_upset_diagram(df_tools, tools, threshold, suffix=""):
    """
    Create UpSet diagram showing tool intersections.
    """
    if not UPSET_AVAILABLE:
        print(f"Skipping UpSet diagram (upsetplot not installed)")
        return
    
    print(f"\nCreating UpSet diagram for Top-{threshold}...")
    
    # Create success flags
    flags = create_success_flags(df_tools, tools, threshold)
    
    # Create membership list for each bug
    memberships = []
    for _, row in flags.iterrows():
        bug_tools = tuple([tool for tool in tools if row[f"found_{tool}"]])
        if bug_tools:  # Only include bugs found by at least one tool
            memberships.append(bug_tools)
    
    if not memberships:
        print(f"  No bugs found by any tool at threshold {threshold}")
        return
    
    # Create UpSet plot
    try:
        upset_data = from_memberships(memberships)
        
        fig = plt.figure(figsize=(12, 6))
        upset = UpSet(upset_data, 
                     subset_size='count',
                     show_counts=True,
                     sort_by='cardinality',
                     sort_categories_by='cardinality')
        upset.plot(fig=fig)
        
        plt.suptitle(f'Tool Intersection Analysis (Top-{threshold})', 
                    fontsize=14, fontweight='bold', y=0.98)
        
        out_file = OUT_DIR / f"upset_diagram_top{threshold}{suffix}.png"
        plt.savefig(out_file, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {out_file}")
        
        # Print summary statistics
        print(f"\n  UpSet Summary (Top-{threshold}):")
        print(f"    Total bugs found by at least one tool: {len(memberships)}")
        
        # Count bugs found by all tools
        all_tools_bugs = sum(1 for m in memberships if len(m) == len(tools))
        print(f"    Bugs found by all {len(tools)} tools: {all_tools_bugs}")
        
        # Count bugs found by only one tool (unique)
        unique_bugs = sum(1 for m in memberships if len(m) == 1)
        print(f"    Bugs found by only one tool: {unique_bugs}")
        
        # Per-tool statistics
        for tool in tools:
            tool_count = sum(1 for m in memberships if tool in m)
            unique_count = sum(1 for m in memberships if m == (tool,))
            print(f"    {tool}: {tool_count} total, {unique_count} unique")
        
    except Exception as e:
        print(f"  Error creating UpSet diagram: {e}")


# ======================================
# ANALYSIS 1: ALL vs NONE
# ======================================

def analyze_all_vs_none(df_features, df_tools, feature_cols, tools, threshold, categorized_features=None):
    """
    Compare features of bugs that:
    - ALL tools find (within threshold)
    - NONE of the tools find (beyond threshold)
    """
    print(f"\n{'='*80}")
    print(f"ANALYSIS 1: ALL TOOLS FIND vs NONE FIND (Top-{threshold})")
    print(f"{'='*80}")
    
    # Create success flags
    flags = create_success_flags(df_tools, tools, threshold)
    
    # Identify bug groups
    found_cols = [f"found_{tool}" for tool in tools]
    flags["num_found"] = flags[found_cols].sum(axis=1)
    
    all_bugs = flags[flags["num_found"] == len(tools)]
    none_bugs = flags[flags["num_found"] == 0]
    
    print(f"Bugs found by ALL tools: {len(all_bugs)}")
    print(f"Bugs found by NONE: {len(none_bugs)}")
    
    if len(all_bugs) < MIN_GROUP_SIZE or len(none_bugs) < MIN_GROUP_SIZE:
        print(f"[SKIP] Insufficient bugs for reliable comparison (min={MIN_GROUP_SIZE})")
        return None
    
    # Merge with features
    merged = pd.merge(flags, df_features, on=["project", "bug_id"], how="inner")
    
    all_data = merged[merged["num_found"] == len(tools)]
    none_data = merged[merged["num_found"] == 0]
    
    # Compare features
    results = []
    for feat in feature_cols:
        if feat not in merged.columns:
            continue
        
        x_all = all_data[feat].dropna()
        x_none = none_data[feat].dropna()
        
        if len(x_all) < MIN_GROUP_SIZE or len(x_none) < MIN_GROUP_SIZE:
            continue
        
        # Mann-Whitney U test
        try:
            u_stat, p_val = mannwhitneyu(x_all, x_none, alternative="two-sided")
        except:
            continue
        
        # Cliff's delta
        delta = cliffs_delta(x_all.values, x_none.values)
        
        results.append({
            "feature": feat,
            "all_n": len(x_all),
            "none_n": len(x_none),
            "all_median": x_all.median(),
            "none_median": x_none.median(),
            "all_mean": x_all.mean(),
            "none_mean": x_none.mean(),
            "median_diff": x_all.median() - x_none.median(),
            "u_statistic": u_stat,
            "p_value": p_val,
            "cliffs_delta": delta,
            "effect_size": effect_size_label(delta)
        })
    
    if not results:
        print("[WARN] No valid comparisons")
        return None
    
    results_df = pd.DataFrame(results)
    results_df = apply_holm(results_df, alpha=ALPHA)
    results_df["abs_delta"] = results_df["cliffs_delta"].abs()
    
    # Filter for practical significance
    results_df["practically_significant"] = (
        results_df["significant"] & 
        (results_df["abs_delta"] >= PRACTICAL_SIG_DELTA)
    )
    
    results_df = results_df.sort_values("abs_delta", ascending=False)
    
    # Save results
    out_file = OUT_DIR / f"all_vs_none_top{threshold}.csv"
    results_df.to_csv(out_file, index=False)
    print(f"Saved: {out_file}")
    
    # Print top features by category if available
    if categorized_features:
        print(f"\nTop discriminative features by category:")
        for category, cat_features in categorized_features.items():
            cat_results = results_df[results_df['feature'].isin(cat_features)]
            if len(cat_results) > 0:
                top_feat = cat_results.iloc[0]
                if top_feat['practically_significant']:
                    print(f"  {category}: {top_feat['feature']} (δ={top_feat['cliffs_delta']:.3f}, p={top_feat['pval_adj']:.4f})")
    
    # Print overall top features
    print(f"\nTop 10 discriminative features (ALL vs NONE):")
    print("=" * 120)
    for _, row in results_df.head(10).iterrows():
        sig = "***" if row["practically_significant"] else ("*" if row["significant"] else "")
        print(f"{row['feature']:<40} | δ={row['cliffs_delta']:>6.3f} ({row['effect_size']:<10}) | p={row['pval_adj']:.4f} {sig}")
        print(f"  ALL:  median={row['all_median']:>8.2f}, mean={row['all_mean']:>8.2f} (n={row['all_n']})")
        print(f"  NONE: median={row['none_median']:>8.2f}, mean={row['none_mean']:>8.2f} (n={row['none_n']})")
    
    return results_df


# ======================================
# ANALYSIS 2: PAIRWISE TOOL COMPARISON
# ======================================

def analyze_tool_pairwise(df_features, df_tools, feature_cols, tools, threshold):
    """
    Compare features of bugs uniquely found by each tool pair.
    """
    print(f"\n{'='*80}")
    print(f"ANALYSIS 2: PAIRWISE TOOL COMPARISONS (Top-{threshold})")
    print(f"{'='*80}")
    
    # Create success flags
    flags = create_success_flags(df_tools, tools, threshold)
    
    # For each tool, identify bugs it finds that others don't
    unique_bugs = {}
    for tool in tools:
        found_col = f"found_{tool}"
        other_cols = [f"found_{t}" for t in tools if t != tool]
        
        # Bugs this tool finds AND no other tool finds
        mask_found = flags[found_col]
        mask_others_not = ~flags[other_cols].any(axis=1)
        unique_bugs[tool] = flags[mask_found & mask_others_not]
        
        print(f"{tool}: {len(unique_bugs[tool])} unique bugs")
    
    # Merge with features
    merged = pd.merge(flags, df_features, on=["project", "bug_id"], how="inner")
    
    # Pairwise comparisons
    all_pairwise_results = []
    
    for tool_a, tool_b in combinations(tools, 2):
        bugs_a = unique_bugs[tool_a]
        bugs_b = unique_bugs[tool_b]
        
        if len(bugs_a) < MIN_GROUP_SIZE or len(bugs_b) < MIN_GROUP_SIZE:
            print(f"\n[SKIP] {tool_a} vs {tool_b}: insufficient bugs")
            continue
        
        print(f"\n{tool_a} (n={len(bugs_a)}) vs {tool_b} (n={len(bugs_b)}):")
        
        # Get feature data for each group
        data_a = merged[merged[["project", "bug_id"]].apply(tuple, axis=1).isin(
            bugs_a[["project", "bug_id"]].apply(tuple, axis=1)
        )]
        data_b = merged[merged[["project", "bug_id"]].apply(tuple, axis=1).isin(
            bugs_b[["project", "bug_id"]].apply(tuple, axis=1)
        )]
        
        results = []
        for feat in feature_cols:
            if feat not in merged.columns:
                continue
            
            x_a = data_a[feat].dropna()
            x_b = data_b[feat].dropna()
            
            if len(x_a) < MIN_GROUP_SIZE or len(x_b) < MIN_GROUP_SIZE:
                continue
            
            try:
                u_stat, p_val = mannwhitneyu(x_a, x_b, alternative="two-sided")
            except:
                continue
            
            delta = cliffs_delta(x_a.values, x_b.values)
            
            results.append({
                "tool_a": tool_a,
                "tool_b": tool_b,
                "feature": feat,
                "n_a": len(x_a),
                "n_b": len(x_b),
                "median_a": x_a.median(),
                "median_b": x_b.median(),
                "mean_a": x_a.mean(),
                "mean_b": x_b.mean(),
                "median_diff": x_a.median() - x_b.median(),
                "u_statistic": u_stat,
                "p_value": p_val,
                "cliffs_delta": delta,
                "effect_size": effect_size_label(delta)
            })
        
        if not results:
            continue
        
        # Apply correction per pair
        pair_df = pd.DataFrame(results)
        pair_df = apply_holm(pair_df, alpha=ALPHA)
        pair_df["abs_delta"] = pair_df["cliffs_delta"].abs()
        pair_df["practically_significant"] = (
            pair_df["significant"] & 
            (pair_df["abs_delta"] >= PRACTICAL_SIG_DELTA)
        )
        
        all_pairwise_results.append(pair_df)
        
        # Print top 5 for this pair
        pair_df = pair_df.sort_values("abs_delta", ascending=False)
        print(f"\nTop 5 discriminative features ({tool_a} vs {tool_b}):")
        for _, row in pair_df.head(5).iterrows():
            sig = "***" if row["practically_significant"] else ""
            print(f"  {row['feature']:<35} | δ={row['cliffs_delta']:>6.3f} | p={row['pval_adj']:.4f} {sig}")
    
    if not all_pairwise_results:
        print("\n[WARN] No valid pairwise comparisons")
        return None
    
    # Combine all pairwise results
    combined_df = pd.concat(all_pairwise_results, ignore_index=True)
    combined_df = combined_df.sort_values(["tool_a", "tool_b", "abs_delta"], ascending=[True, True, False])
    
    # Save
    out_file = OUT_DIR / f"pairwise_tool_comparison_top{threshold}.csv"
    combined_df.to_csv(out_file, index=False)
    print(f"\nSaved: {out_file}")
    
    return combined_df


# ======================================
# ANALYSIS 3: TOOL vs REST
# ======================================

def analyze_tool_vs_rest(df_features, df_tools, feature_cols, tools, threshold):
    """
    For each tool, compare bugs it uniquely finds vs bugs found by all OTHER tools.
    """
    print(f"\n{'='*80}")
    print(f"ANALYSIS 3: EACH TOOL vs REST (Top-{threshold})")
    print(f"{'='*80}")
    
    flags = create_success_flags(df_tools, tools, threshold)
    merged = pd.merge(flags, df_features, on=["project", "bug_id"], how="inner")
    
    all_tool_results = []
    
    for tool in tools:
        found_col = f"found_{tool}"
        other_cols = [f"found_{t}" for t in tools if t != tool]
        
        # Bugs this tool finds that others don't
        mask_unique = flags[found_col] & ~flags[other_cols].any(axis=1)
        unique_bugs = flags[mask_unique]
        
        # Bugs all other tools find (regardless of this tool)
        mask_others = flags[other_cols].all(axis=1)
        others_bugs = flags[mask_others]
        
        print(f"\n{tool}:")
        print(f"  Unique bugs: {len(unique_bugs)}")
        print(f"  Bugs found by all others: {len(others_bugs)}")
        
        if len(unique_bugs) < MIN_GROUP_SIZE or len(others_bugs) < MIN_GROUP_SIZE:
            print(f"  [SKIP] Insufficient bugs")
            continue
        
        # Get feature data
        data_unique = merged[merged[["project", "bug_id"]].apply(tuple, axis=1).isin(
            unique_bugs[["project", "bug_id"]].apply(tuple, axis=1)
        )]
        data_others = merged[merged[["project", "bug_id"]].apply(tuple, axis=1).isin(
            others_bugs[["project", "bug_id"]].apply(tuple, axis=1)
        )]
        
        results = []
        for feat in feature_cols:
            if feat not in merged.columns:
                continue
            
            x_unique = data_unique[feat].dropna()
            x_others = data_others[feat].dropna()
            
            if len(x_unique) < MIN_GROUP_SIZE or len(x_others) < MIN_GROUP_SIZE:
                continue
            
            try:
                u_stat, p_val = mannwhitneyu(x_unique, x_others, alternative="two-sided")
            except:
                continue
            
            delta = cliffs_delta(x_unique.values, x_others.values)
            
            results.append({
                "tool": tool,
                "feature": feat,
                "unique_n": len(x_unique),
                "others_n": len(x_others),
                "unique_median": x_unique.median(),
                "others_median": x_others.median(),
                "unique_mean": x_unique.mean(),
                "others_mean": x_others.mean(),
                "median_diff": x_unique.median() - x_others.median(),
                "u_statistic": u_stat,
                "p_value": p_val,
                "cliffs_delta": delta,
                "effect_size": effect_size_label(delta)
            })
        
        if not results:
            continue
        
        tool_df = pd.DataFrame(results)
        tool_df = apply_holm(tool_df, alpha=ALPHA)
        tool_df["abs_delta"] = tool_df["cliffs_delta"].abs()
        tool_df["practically_significant"] = (
            tool_df["significant"] & 
            (tool_df["abs_delta"] >= PRACTICAL_SIG_DELTA)
        )
        tool_df = tool_df.sort_values("abs_delta", ascending=False)
        
        all_tool_results.append(tool_df)
        
        # Print top 5 for this tool
        print(f"\nTop 5 features distinguishing {tool}'s unique strengths:")
        for _, row in tool_df.head(5).iterrows():
            sig = "***" if row["practically_significant"] else ""
            direction = "higher" if row["cliffs_delta"] > 0 else "lower"
            print(f"  {row['feature']:<35} | δ={row['cliffs_delta']:>6.3f} ({direction}) | p={row['pval_adj']:.4f} {sig}")
    
    if not all_tool_results:
        print("\n[WARN] No valid tool-vs-rest comparisons")
        return None
    
    combined_df = pd.concat(all_tool_results, ignore_index=True)
    combined_df = combined_df.sort_values(["tool", "abs_delta"], ascending=[True, False])
    
    out_file = OUT_DIR / f"tool_vs_rest_top{threshold}.csv"
    combined_df.to_csv(out_file, index=False)
    print(f"\nSaved: {out_file}")
    
    return combined_df


# ======================================
# VISUALIZATION
# ======================================

def create_summary_heatmap(pairwise_df, threshold):
    """Create heatmap showing which features distinguish tool pairs."""
    if pairwise_df is None or len(pairwise_df) == 0:
        return
    
    # Get top features per pair
    top_features_per_pair = {}
    for (tool_a, tool_b), group in pairwise_df.groupby(["tool_a", "tool_b"]):
        group_sig = group[group["practically_significant"]]
        if len(group_sig) > 0:
            top_features_per_pair[(tool_a, tool_b)] = group_sig.head(10)["feature"].tolist()
    
    # Get all features that appear in any top list
    all_top_features = set()
    for features in top_features_per_pair.values():
        all_top_features.update(features)
    
    if not all_top_features:
        return
    
    # Create matrix: features x tool pairs
    pairs = sorted(top_features_per_pair.keys())
    features = sorted(all_top_features)
    
    matrix = np.zeros((len(features), len(pairs)))
    for j, pair in enumerate(pairs):
        for i, feat in enumerate(features):
            if feat in top_features_per_pair.get(pair, []):
                # Find the delta for this feature-pair combination
                subset = pairwise_df[
                    (pairwise_df["tool_a"] == pair[0]) & 
                    (pairwise_df["tool_b"] == pair[1]) & 
                    (pairwise_df["feature"] == feat)
                ]
                if len(subset) > 0:
                    matrix[i, j] = subset.iloc[0]["cliffs_delta"]
    
    # Plot
    fig, ax = plt.subplots(figsize=(max(8, len(pairs) * 1.5), max(8, len(features) * 0.3)))
    
    pair_labels = [f"{a}\nvs\n{b}" for a, b in pairs]
    
    sns.heatmap(matrix, 
                xticklabels=pair_labels,
                yticklabels=features,
                cmap="RdBu_r",
                center=0,
                vmin=-1, vmax=1,
                annot=False,
                cbar_kws={"label": "Cliff's Delta"},
                ax=ax)
    
    ax.set_title(f"Tool-Pair Discriminative Features (Top-{threshold})")
    ax.set_xlabel("Tool Comparison")
    ax.set_ylabel("Feature")
    
    plt.tight_layout()
    out_file = OUT_DIR / f"pairwise_heatmap_top{threshold}.png"
    plt.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved visualization: {out_file}")


# ======================================
# MAIN EXECUTION
# ======================================

def main():
    print("=" * 80)
    print("ENHANCED FOCUSED TOOL COMPARISON FEATURE ANALYSIS")
    print("=" * 80)
    print(f"Output directory: {OUT_DIR}")
    print(f"Thresholds: {THRESHOLDS}")
    print(f"Minimum group size: {MIN_GROUP_SIZE}")
    print(f"Practical significance threshold: |δ| ≥ {PRACTICAL_SIG_DELTA}")
    print(f"UpSet diagrams: {'Enabled' if UPSET_AVAILABLE else 'Disabled (install upsetplot)'}")
    print("=" * 80)
    
    # Load data
    df_features, df_tools, feature_cols, categorized_features, tools, id_cols = load_data()
    
    # Run analyses for each threshold
    for threshold in THRESHOLDS:
        print(f"\n{'#'*80}")
        print(f"# THRESHOLD: Top-{threshold}")
        print(f"{'#'*80}")
        
        # NEW: Create UpSet diagram first
        create_upset_diagram(df_tools, tools, threshold)
        
        # Analysis 1: ALL vs NONE
        all_vs_none_df = analyze_all_vs_none(df_features, df_tools, feature_cols, tools, threshold, categorized_features)
        
        # Analysis 2: Pairwise tool comparisons
        pairwise_df = analyze_tool_pairwise(df_features, df_tools, feature_cols, tools, threshold)
        
        # Analysis 3: Tool vs rest
        tool_vs_rest_df = analyze_tool_vs_rest(df_features, df_tools, feature_cols, tools, threshold)
        
        # Create visualizations
        if pairwise_df is not None:
            create_summary_heatmap(pairwise_df, threshold)
    
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print(f"Results saved to: {OUT_DIR}")
    print("=" * 80)
    print("\nGenerated files:")
    print("  - all_vs_none_top{1,5,10}.csv")
    print("  - pairwise_tool_comparison_top{1,5,10}.csv")
    print("  - tool_vs_rest_top{1,5,10}.csv")
    print("  - pairwise_heatmap_top{1,5,10}.png")
    if UPSET_AVAILABLE:
        print("  - upset_diagram_top{1,5,10}.png")
    print("=" * 80)

if __name__ == "__main__":
    main()