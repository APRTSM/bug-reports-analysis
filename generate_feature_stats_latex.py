"""
Generate LaTeX table with feature statistics (Mean, STD, Min, Max) from original (non-standardized) values.
Uses experimentA_full_dataset.csv which contains original values before standardization.
Pure Python implementation (no numpy)
"""

import csv
import math
from pathlib import Path
from collections import defaultdict

# Configuration
# Use full dataset which has original values before standardization
INPUT_FILE = Path("full_feature_preproccessed_fixed/experimentA_full_dataset.csv")
OUTPUT_FILE = Path("results/feature_statistics.tex")

def escape_latex(text):
    """Escape special LaTeX characters."""
    if not text:
        return ""
    text = str(text)
    # Escape special characters in order
    text = text.replace("\\", "\\textbackslash{}")
    text = text.replace("&", "\\&")
    text = text.replace("%", "\\%")
    text = text.replace("$", "\\$")
    text = text.replace("#", "\\#")
    text = text.replace("^", "\\textasciicircum{}")
    text = text.replace("_", "\\_")
    text = text.replace("{", "\\{")
    text = text.replace("}", "\\}")
    text = text.replace("~", "\\textasciitilde{}")
    return text

def format_feature_name(feature):
    """Format feature name for LaTeX, making it more readable."""
    # Replace underscores with spaces and capitalize
    feature = str(feature)
    words = feature.replace("_", " ").split()
    words = [w.capitalize() for w in words]
    return " ".join(words)

def try_float(value):
    """Try to convert value to float, return None if not possible."""
    try:
        return float(value)
    except (ValueError, TypeError):
        return None

def compute_stats(values):
    """Compute mean, std, min, max from a list of values."""
    if not values:
        return None
    
    n = len(values)
    mean = sum(values) / n
    variance = sum((x - mean) ** 2 for x in values) / n
    std = math.sqrt(variance)
    min_val = min(values)
    max_val = max(values)
    
    return {
        "mean": mean,
        "std": std,
        "min": min_val,
        "max": max_val
    }

def main():
    print("=" * 80)
    print("GENERATING FEATURE STATISTICS LaTeX TABLE")
    print("=" * 80)
    print(f"Input file: {INPUT_FILE}")
    print(f"Output file: {OUTPUT_FILE}")
    print("=" * 80)
    
    # First pass: read header and identify feature columns
    print("\nReading header...")
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        header = next(reader)
    
    print(f"  Found {len(header)} columns")
    
    # Identify feature columns (exclude metadata and performance metrics)
    exclude_patterns = [
        "project", "bug_id", "id",  # Metadata
        "mrr_", "rank_", "top@",  # Performance metrics
    ]
    
    feature_cols = []
    feature_indices = []
    for i, col in enumerate(header):
        if any(col.startswith(pattern) for pattern in exclude_patterns):
            continue
        feature_cols.append(col)
        feature_indices.append(i)
    
    print(f"  Identified {len(feature_cols)} feature columns to process")
    
    # Second pass: collect values for each feature
    print("\nProcessing data (this may take a moment)...")
    feature_values = defaultdict(list)
    total_rows = 0
    
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        next(reader)  # Skip header
        
        for row in reader:
            total_rows += 1
            for col, idx in zip(feature_cols, feature_indices):
                if idx < len(row):
                    value = try_float(row[idx])
                    if value is not None:
                        feature_values[col].append(value)
            
            if total_rows % 100 == 0:
                print(f"  Processed {total_rows} rows...", end='\r')
    
    print(f"\n  Processed {total_rows} rows")
    
    # Compute statistics for each feature
    print("\nComputing statistics...")
    stats_data = []
    
    for col in feature_cols:
        values = feature_values[col]
        
        if len(values) == 0:
            continue
        
        stats = compute_stats(values)
        if stats:
            stats_data.append({
                "feature": col,
                **stats
            })
    
    # Sort by feature name
    stats_data.sort(key=lambda x: x["feature"])
    
    print(f"  Computed statistics for {len(stats_data)} features")
    
    # Generate LaTeX table
    print("\nGenerating LaTeX table...")
    
    latex = []
    latex.append("% Feature Statistics Table")
    latex.append("% Generated from: experimentA_full_dataset.csv (original, non-standardized values)")
    latex.append("")
    latex.append("\\begin{table}[h]")
    latex.append("\\centering")
    latex.append("\\caption{Feature Statistics (Mean, Standard Deviation, Min, Max)}")
    latex.append("\\label{tab:feature_statistics}")
    latex.append("\\begin{tabular}{lrrrr}")
    latex.append("\\toprule")
    latex.append("\\textbf{Feature} & \\textbf{Mean} & \\textbf{STD} & \\textbf{Min} & \\textbf{Max} \\\\")
    latex.append("\\midrule")
    
    # Add rows
    for stat in stats_data:
        feature = format_feature_name(stat['feature'])
        mean = stat['mean']
        std = stat['std']
        min_val = stat['min']
        max_val = stat['max']
        
        feature_escaped = escape_latex(feature)
        
        # Format numbers with appropriate precision
        latex.append(f"{feature_escaped} & ${mean:.4f}$ & ${std:.4f}$ & ${min_val:.4f}$ & ${max_val:.4f}$ \\\\")
    
    latex.append("\\bottomrule")
    latex.append("\\end{tabular}")
    latex.append("\\end{table}")
    latex.append("")
    
    # Save to file
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        f.write("\n".join(latex))
    
    print(f"\nSaved LaTeX table to: {OUTPUT_FILE}")
    print(f"  Rows: {len(stats_data)}")
    print("\n" + "=" * 80)
    print("LaTeX TABLE GENERATION COMPLETE")
    print("=" * 80)
    print("\nNote: This LaTeX file requires the following packages:")
    print("  \\usepackage{booktabs}  % For \\toprule, \\midrule, \\bottomrule")
    print("=" * 80)

if __name__ == "__main__":
    main()
