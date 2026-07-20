"""
Generate LaTeX tables for top 12 impactful features per tool in tool vs rest analysis.
"""

import pandas as pd
from pathlib import Path
import re

# Configuration
INPUT_DIR = Path("tool_comparison_results_fixed")
OUTPUT_DIR = Path("results")
THRESHOLDS = [1, 5, 10]
TOP_N_FEATURES = 12

def escape_latex(text):
    """Escape special LaTeX characters."""
    if pd.isna(text):
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
    # Replace underscores with spaces for readability
    feature = str(feature).replace("_", " ")
    # Capitalize first letter of each word
    words = feature.split()
    words = [w.capitalize() for w in words]
    return " ".join(words)

def generate_single_threshold_table(df_tool, tool_name, threshold):
    """Generate LaTeX table for a single tool at a specific threshold."""
    
    # Select top N features
    df_top = df_tool.head(TOP_N_FEATURES).copy()
    
    if len(df_top) == 0:
        return None
    
    # Create safe label
    safe_label = re.sub(r'[^a-zA-Z0-9_]', '_', str(tool_name).lower())
    
    # Start LaTeX table
    latex = []
    latex.append(f"% Top {TOP_N_FEATURES} impactful features for {tool_name} (Top-{threshold})")
    latex.append(f"\\begin{{table}}[h]")
    latex.append(f"\\centering")
    latex.append(f"\\caption{{Top {TOP_N_FEATURES} Impactful Features for {escape_latex(tool_name)} (Top-{threshold})}}")
    latex.append(f"\\label{{tab:tool_features_{safe_label}_top{threshold}}}")
    latex.append(f"\\begin{{tabular}}{{lrrrr}}")
    latex.append(f"\\toprule")
    latex.append(f"Feature & $\\delta$ & Effect Size & $p$-value \\\\")
    latex.append(f"\\midrule")
    
    # Add rows
    for idx, row in df_top.iterrows():
        feature = format_feature_name(row['feature'])
        delta = float(row['cliffs_delta'])
        effect_size = str(row['effect_size'])
        pval = float(row['pval_adj'])
        
        # Format p-value
        if pval < 0.001:
            pval_str = "$<0.001$"
        else:
            pval_str = f"${pval:.3f}$"
        
        # Add significance marker
        sig_marker = ""
        if row.get('practically_significant', False):
            sig_marker = "$^{***}$"
        elif row.get('significant', False):
            sig_marker = "$^{*}$"
        
        feature_escaped = escape_latex(feature)
        effect_size_escaped = escape_latex(effect_size)
        
        latex.append(f"{feature_escaped}{sig_marker} & ${delta:+.3f}$ & {effect_size_escaped} & {pval_str} \\\\")
    
    latex.append(f"\\bottomrule")
    latex.append(f"\\end{{tabular}}")
    latex.append(f"\\end{{table}}")
    latex.append("")
    
    return "\n".join(latex)

def generate_combined_latex_table(df_all_thresholds, tool_name):
    """Generate LaTeX table for a single tool combining all thresholds."""
    
    if len(df_all_thresholds) == 0:
        return None
    
    # Create safe label
    safe_label = re.sub(r'[^a-zA-Z0-9_]', '_', str(tool_name).lower())
    
    # Start LaTeX table
    latex = []
    latex.append(f"% Top {TOP_N_FEATURES} impactful features for {tool_name} (combined Top-1, Top-5, Top-10)")
    latex.append(f"\\begin{{table}}[h]")
    latex.append(f"\\centering")
    latex.append(f"\\caption{{Top {TOP_N_FEATURES} Impactful Features for {escape_latex(tool_name)} (Top-1, Top-5, Top-10)}}")
    latex.append(f"\\label{{tab:tool_features_{safe_label}_combined}}")
    latex.append(f"\\begin{{tabular}}{{lcrrr}}")
    latex.append(f"\\toprule")
    latex.append(f"Feature & Threshold & $\\delta$ & Effect Size & $p$-value \\\\")
    latex.append(f"\\midrule")
    
    # Add rows
    for idx, row in df_all_thresholds.iterrows():
        feature = format_feature_name(row['feature'])
        threshold = int(row['threshold'])
        delta = float(row['cliffs_delta'])
        effect_size = str(row['effect_size'])
        pval = float(row['pval_adj'])
        
        # Format p-value
        if pval < 0.001:
            pval_str = "$<0.001$"
        else:
            pval_str = f"${pval:.3f}$"
        
        # Add significance marker
        sig_marker = ""
        if row.get('practically_significant', False):
            sig_marker = "$^{***}$"
        elif row.get('significant', False):
            sig_marker = "$^{*}$"
        
        feature_escaped = escape_latex(feature)
        effect_size_escaped = escape_latex(effect_size)
        
        latex.append(f"{feature_escaped}{sig_marker} & Top-{threshold} & ${delta:+.3f}$ & {effect_size_escaped} & {pval_str} \\\\")
    
    latex.append(f"\\bottomrule")
    latex.append(f"\\end{{tabular}}")
    latex.append(f"\\end{{table}}")
    latex.append("")
    
    return "\n".join(latex)

def main():
    """Generate LaTeX tables for all tools - both individual threshold files and combined file."""
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("GENERATING LATEX TABLES FOR TOOL VS REST FEATURES")
    print("=" * 80)
    
    # Load all threshold data
    all_data = []
    threshold_data = {}
    
    for threshold in THRESHOLDS:
        input_file = INPUT_DIR / f"tool_vs_rest_top{threshold}.csv"
        
        if not input_file.exists():
            print(f"  Warning: File not found: {input_file}")
            continue
        
        print(f"\nReading: {input_file}")
        try:
            df = pd.read_csv(input_file)
            df['threshold'] = threshold  # Add threshold column
            all_data.append(df)
            threshold_data[threshold] = df
        except Exception as e:
            print(f"  Error reading file: {e}")
            continue
    
    if not all_data:
        print("  No data files found!")
        return
    
    # Combine all data
    combined_df = pd.concat(all_data, ignore_index=True)
    
    # Get unique tools
    tools = sorted(combined_df['tool'].unique())
    print(f"\nFound {len(tools)} tools: {', '.join(tools)}")
    
    # Generate individual threshold files
    print("\n" + "=" * 80)
    print("GENERATING INDIVIDUAL THRESHOLD FILES")
    print("=" * 80)
    
    for threshold in THRESHOLDS:
        if threshold not in threshold_data:
            continue
        
        print(f"\n{'#'*80}")
        print(f"# THRESHOLD: Top-{threshold}")
        print(f"{'#'*80}")
        
        all_tables = []
        all_tables.append(f"% Top {TOP_N_FEATURES} Impactful Features per Tool (Top-{threshold})")
        all_tables.append(f"% Generated from: tool_comparison_results_fixed/tool_vs_rest_top{threshold}.csv")
        all_tables.append("")
        
        df_threshold = threshold_data[threshold]
        
        for tool in tools:
            print(f"\n  Processing {tool}...")
            df_tool = df_threshold[df_threshold['tool'] == tool].copy()
            
            # Sort by absolute delta (descending)
            df_tool = df_tool.sort_values('abs_delta', ascending=False)
            
            # Generate LaTeX table
            try:
                table_latex = generate_single_threshold_table(df_tool, tool, threshold)
                
                if table_latex:
                    all_tables.append(table_latex)
                    print(f"    Generated table with {min(TOP_N_FEATURES, len(df_tool))} features")
                else:
                    print(f"    No features found for {tool}")
            except Exception as e:
                print(f"    Error generating table for {tool}: {e}")
                continue
        
        # Save individual threshold file
        output_file = OUTPUT_DIR / f"tool_features_top{threshold}.tex"
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write("\n".join(all_tables))
            print(f"\n  Saved: {output_file}")
        except Exception as e:
            print(f"  Error saving file: {e}")
    
    # Generate combined file
    print("\n" + "=" * 80)
    print("GENERATING COMBINED FILE")
    print("=" * 80)
    
    all_tables_combined = []
    all_tables_combined.append(f"% Top {TOP_N_FEATURES} Impactful Features per Tool (Combined Top-1, Top-5, Top-10)")
    all_tables_combined.append(f"% Generated from: tool_comparison_results_fixed/tool_vs_rest_top*.csv")
    all_tables_combined.append("")
    
    for tool in tools:
        print(f"\n  Processing {tool}...")
        df_tool = combined_df[combined_df['tool'] == tool].copy()
        
        # Sort by absolute delta (descending), then by threshold
        df_tool = df_tool.sort_values(['abs_delta', 'threshold'], ascending=[False, True])
        
        # Get top N features across all thresholds
        df_top = df_tool.head(TOP_N_FEATURES).copy()
        
        # Generate LaTeX table
        try:
            table_latex = generate_combined_latex_table(df_top, tool)
            
            if table_latex:
                all_tables_combined.append(table_latex)
                print(f"    Generated table with {len(df_top)} features")
            else:
                print(f"    No features found for {tool}")
        except Exception as e:
            print(f"    Error generating table for {tool}: {e}")
            continue
    
    # Save combined file
    output_file = OUTPUT_DIR / "tool_features_combined.tex"
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("\n".join(all_tables_combined))
        print(f"\n  Saved: {output_file}")
    except Exception as e:
        print(f"  Error saving file: {e}")
    
    print("\n" + "=" * 80)
    print("LATEX TABLE GENERATION COMPLETE")
    print("=" * 80)
    print(f"\nGenerated files:")
    for threshold in THRESHOLDS:
        print(f"  - results/tool_features_top{threshold}.tex")
    print(f"  - results/tool_features_combined.tex")
    print("=" * 80)
    print("\nNote: These LaTeX files require the following packages:")
    print("  \\usepackage{booktabs}  % For \\toprule, \\midrule, \\bottomrule")
    print("=" * 80)

if __name__ == "__main__":
    main()
