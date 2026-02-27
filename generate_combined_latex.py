#!/usr/bin/env python3
"""
Generate combined LaTeX tables for top 12 impactful features per tool.
Combines Top-1, Top-5, and Top-10 into single tables.
"""

import csv
from pathlib import Path
import re

INPUT_DIR = Path("tool_comparison_results_fixed")
OUTPUT_DIR = Path("results")
THRESHOLDS = [1, 5, 10]
TOP_N_FEATURES = 12

def escape_latex(text):
    """Escape special LaTeX characters."""
    if not text:
        return ""
    text = str(text)
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
    """Format feature name for LaTeX."""
    feature = str(feature).replace("_", " ")
    words = feature.split()
    words = [w.capitalize() for w in words]
    return " ".join(words)

def load_tool_data():
    """Load data from all threshold files."""
    all_data = {}
    
    for threshold in THRESHOLDS:
        input_file = INPUT_DIR / f"tool_vs_rest_top{threshold}.csv"
        if not input_file.exists():
            continue
        
        with open(input_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                tool = row['tool']
                if tool not in all_data:
                    all_data[tool] = []
                
                # Add threshold to row
                row['threshold'] = threshold
                all_data[tool].append(row)
    
    return all_data

def generate_combined_table(tool_name, features_data):
    """Generate LaTeX table for a tool combining all thresholds."""
    safe_label = re.sub(r'[^a-zA-Z0-9_]', '_', str(tool_name).lower())
    
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
    
    for feat_data in features_data:
        feature = format_feature_name(feat_data['feature'])
        threshold = int(feat_data['threshold'])
        delta = float(feat_data['cliffs_delta'])
        effect_size = str(feat_data['effect_size'])
        pval = float(feat_data['pval_adj'])
        
        pval_str = "$<0.001$" if pval < 0.001 else f"${pval:.3f}$"
        
        sig_marker = ""
        if feat_data.get('practically_significant', 'False') == 'True':
            sig_marker = "$^{***}$"
        elif feat_data.get('significant', 'False') == 'True':
            sig_marker = "$^{*}$"
        
        feature_escaped = escape_latex(feature)
        effect_size_escaped = escape_latex(effect_size)
        
        latex.append(f"{feature_escaped}{sig_marker} & Top-{threshold} & ${delta:+.3f}$ & {effect_size_escaped} & {pval_str} \\\\")
    
    latex.append(f"\\bottomrule")
    latex.append(f"\\end{{tabular}}")
    latex.append(f"\\end{{table}}")
    latex.append("")
    
    return "\n".join(latex)

def generate_single_threshold_table(tool_name, features_data, threshold):
    """Generate LaTeX table for a tool at a specific threshold."""
    safe_label = re.sub(r'[^a-zA-Z0-9_]', '_', str(tool_name).lower())
    
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
    
    for feat_data in features_data:
        feature = format_feature_name(feat_data['feature'])
        delta = float(feat_data['cliffs_delta'])
        effect_size = str(feat_data['effect_size'])
        pval = float(feat_data['pval_adj'])
        
        pval_str = "$<0.001$" if pval < 0.001 else f"${pval:.3f}$"
        
        sig_marker = ""
        if feat_data.get('practically_significant', 'False') == 'True':
            sig_marker = "$^{***}$"
        elif feat_data.get('significant', 'False') == 'True':
            sig_marker = "$^{*}$"
        
        feature_escaped = escape_latex(feature)
        effect_size_escaped = escape_latex(effect_size)
        
        latex.append(f"{feature_escaped}{sig_marker} & ${delta:+.3f}$ & {effect_size_escaped} & {pval_str} \\\\")
    
    latex.append(f"\\bottomrule")
    latex.append(f"\\end{{tabular}}")
    latex.append(f"\\end{{table}}")
    latex.append("")
    
    return "\n".join(latex)

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("GENERATING LATEX TABLES FOR TOOL VS REST FEATURES")
    print("=" * 80)
    
    print("\nLoading data...")
    all_data = load_tool_data()
    
    tools = sorted(all_data.keys())
    print(f"Found {len(tools)} tools: {', '.join(tools)}")
    
    # Generate individual threshold files
    print("\n" + "=" * 80)
    print("GENERATING INDIVIDUAL THRESHOLD FILES")
    print("=" * 80)
    
    for threshold in THRESHOLDS:
        print(f"\n{'#'*80}")
        print(f"# THRESHOLD: Top-{threshold}")
        print(f"{'#'*80}")
        
        all_tables = []
        all_tables.append(f"% Top {TOP_N_FEATURES} Impactful Features per Tool (Top-{threshold})")
        all_tables.append("")
        
        for tool in tools:
            print(f"Processing {tool}...")
            # Filter features for this threshold
            tool_features = [f for f in all_data[tool] if int(f['threshold']) == threshold]
            
            # Sort by absolute delta (descending)
            tool_features.sort(key=lambda x: -float(x['abs_delta']))
            
            # Get top N
            top_features = tool_features[:TOP_N_FEATURES]
            
            if top_features:
                table = generate_single_threshold_table(tool, top_features, threshold)
                all_tables.append(table)
                print(f"  Generated table with {len(top_features)} features")
        
        output_file = OUTPUT_DIR / f"tool_features_top{threshold}.tex"
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("\n".join(all_tables))
        print(f"\nSaved: {output_file}")
    
    # Generate combined file
    print("\n" + "=" * 80)
    print("GENERATING COMBINED FILE")
    print("=" * 80)
    
    all_tables_combined = []
    all_tables_combined.append(f"% Top {TOP_N_FEATURES} Impactful Features per Tool (Combined Top-1, Top-5, Top-10)")
    all_tables_combined.append("")
    
    for tool in tools:
        print(f"Processing {tool}...")
        tool_features = all_data[tool]
        
        # Sort by absolute delta (descending), then by threshold (ascending)
        tool_features.sort(key=lambda x: (-float(x['abs_delta']), int(x['threshold'])))
        
        # Get top N
        top_features = tool_features[:TOP_N_FEATURES]
        
        table = generate_combined_table(tool, top_features)
        all_tables_combined.append(table)
        print(f"  Generated table with {len(top_features)} features")
    
    output_file = OUTPUT_DIR / "tool_features_combined.tex"
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("\n".join(all_tables_combined))
    
    print(f"\nSaved: {output_file}")
    
    print("\n" + "=" * 80)
    print("LATEX TABLE GENERATION COMPLETE")
    print("=" * 80)
    print(f"\nGenerated files:")
    for threshold in THRESHOLDS:
        print(f"  - results/tool_features_top{threshold}.tex")
    print(f"  - results/tool_features_combined.tex")
    print("=" * 80)

if __name__ == "__main__":
    main()
