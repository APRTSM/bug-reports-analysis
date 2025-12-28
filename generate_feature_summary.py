#!/usr/bin/env python3
"""
Generate a summary table of all features with min, max, mean, and median.

Usage:
    python generate_feature_summary.py
"""

import pandas as pd
import numpy as np
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
    
    return df, feature_cols

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
        df_for_stats = df_raw
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

