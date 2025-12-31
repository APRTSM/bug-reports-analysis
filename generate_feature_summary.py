#!/usr/bin/env python3
"""
Generate a summary table of all features with min, max, mean, and median.

Usage:
    python generate_feature_summary.py
"""

import pandas as pd
import numpy as np
import re
from pathlib import Path

# Configuration
IN_FILE_PREPROCESSED = Path("full_feature_preproccessed/experimentA_preprocessed_rich.csv")
IN_FILE_FULL = Path("full_feature_preproccessed/experimentA_full_dataset.csv")  # For raw feature values
OUT_FILE = Path("feature_summary_statistics.csv")
USE_RAW_VALUES = True  # Set to True to use raw values from full dataset, False to use scaled values

def load_data_and_features(in_file):
    """Load data and extract feature columns."""
    df = pd.read_csv(in_file)
    print(f"Loaded: {df.shape} from {in_file}")

    id_cols = [c for c in ["project", "bug_id", "id"] if c in df.columns]
    
    # Exclude performance metrics
    perf_cols = [c for c in df.columns if c.startswith("mrr_") or c.startswith("top@") or c.startswith("rank_")]
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [c for c in numeric_cols if c not in perf_cols + id_cols]

    # Exclude missingness indicators if desired (set to False to include them)
    INCLUDE_MISSINGNESS_FEATURES = True
    if not INCLUDE_MISSINGNESS_FEATURES:
        feature_cols = [c for c in feature_cols if not (c.endswith("_is_missing") or "__is_missing" in c)]

    # Sanity check: verify no performance metrics in features
    suspicious = [f for f in feature_cols if any(x in f.lower() for x in ['rank_', 'mrr_', 'top@'])]
    if suspicious:
        print(f"[WARNING] Performance metrics found in feature list: {suspicious}")
        print("These will be removed from features.")
        feature_cols = [f for f in feature_cols if f not in suspicious]
    
    print(f"Feature columns: {len(feature_cols)}")
    print(f"Performance columns (excluded): {len(perf_cols)}")
    
    # Exclude redundant columns that are identical to other features
    # Note: confidence features are kept in summary statistics (only excluded from analysis)
    redundant_features = ['fine_grained_description_length', 'description_length_fine_grained']  # Identical to description_length
    feature_cols = [c for c in feature_cols if c not in redundant_features]
    
    return df, feature_cols

def _compute_text_stats_feature(df_raw, feat_name):
    """
    Compute a single text stats feature from raw text column.
    Handles txt_* features that are derived from text columns.
    """
    # Extract the source column name from feature name
    # e.g., "txt_fine_grained_reasoning_char_len" -> "fine_grained_reasoning"
    if not feat_name.startswith("txt_"):
        return None
    
    # Find the source text column by matching prefix
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

def generate_feature_summary_table(df, feature_cols, out_file, use_raw_values=True):
    """
    Generate a summary table with min, max, mean, and median for each feature.
    
    Args:
        df: DataFrame with features (may be preprocessed/scaled)
        feature_cols: List of feature column names
        out_file: Output file path
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
        if use_raw_values:
            print(f"Warning: {IN_FILE_FULL} not found. Using preprocessed values (may be scaled).")
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
                'n_missing': int(n_missing),
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
    summary_df.to_csv(out_file, index=False)
    print(f"\nSaved: {out_file}")
    print(f"  Total features: {len(summary_df)}")
    print(f"  Features with no missing data: {len(summary_df[summary_df['n_missing'] == 0])}")
    print(f"  Features with >50% missing: {len(summary_df[summary_df['missing_pct'] > 50])}")
    
    # Print sample
    print("\nSample of summary table (first 10 features):")
    print(summary_df.head(10).to_string(index=False))
    
    return summary_df

if __name__ == "__main__":
    if not IN_FILE_PREPROCESSED.exists():
        print(f"Error: Input file not found: {IN_FILE_PREPROCESSED}")
        print("Please make sure the preprocessed dataset exists.")
        exit(1)
    
    df, feature_cols = load_data_and_features(IN_FILE_PREPROCESSED)
    summary_df = generate_feature_summary_table(df, feature_cols, OUT_FILE, use_raw_values=USE_RAW_VALUES)
    
    print(f"\n✓ Feature summary table generated successfully!")
    print(f"  Output: {OUT_FILE}")

