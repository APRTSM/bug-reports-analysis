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
IN_FILE = Path("full_feature_preproccessed/experimentA_preprocessed_rich.csv")
OUT_FILE = Path("feature_summary_statistics.csv")

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

def generate_feature_summary_table(df, feature_cols, out_file):
    """
    Generate a summary table with min, max, mean, and median for each feature.
    """
    print("\n" + "=" * 60)
    print("GENERATING FEATURE SUMMARY TABLE")
    print("=" * 60)
    
    summary_records = []
    
    for feat in feature_cols:
        if feat not in df.columns:
            continue
        
        values = df[feat].dropna()
        
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
                'n_missing': int(df[feat].isna().sum()),
                'missing_pct': 100.0
            })
            continue
        
        min_val = float(values.min())
        max_val = float(values.max())
        
        # Determine scale description
        if min_val == 0.0 and max_val == 1.0:
            scale = "0-1"
        elif min_val == 0.0 and max_val == 0.0:
            scale = "constant (0)"
        elif min_val >= 0.0 and max_val <= 1.0:
            scale = f"{min_val:.3f}-{max_val:.3f}"
        elif min_val.is_integer() and max_val.is_integer() and min_val >= 0:
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
                scale = f"{min_val:.3f}-{max_val:.3f}"
        
        summary_records.append({
            'feature': feat,
            'min': min_val,
            'max': max_val,
            'mean': float(values.mean()),
            'median': float(values.median()),
            'std': float(values.std()),
            'scale': scale,
            'n_valid': int(len(values)),
            'n_missing': int(df[feat].isna().sum()),
            'missing_pct': float(df[feat].isna().sum() / len(df) * 100)
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
    if not IN_FILE.exists():
        print(f"Error: Input file not found: {IN_FILE}")
        print("Please make sure the preprocessed dataset exists.")
        exit(1)
    
    df, feature_cols = load_data_and_features(IN_FILE)
    summary_df = generate_feature_summary_table(df, feature_cols, OUT_FILE)
    
    print(f"\n✓ Feature summary table generated successfully!")
    print(f"  Output: {OUT_FILE}")

