# Bug Reports Analysis

This repository is a replication package for the paper *Beyond Keywords: Understanding Bug Report Features Impact on Fault Localization*.


## Repository Structure

```
bug-reports-analysis/
├── run_full_pipeline.py                 # End-to-end pipeline runner
├── bug_feature_extraction/             # Feature extraction scripts
│   ├── extract_bug_features.py
│   ├── gemini_bug_categorization_overall.py
│   ├── gemini_bug_ratings.py
│   └── fine_grained_gemini_catg.py
├── final_feature_set.csv               #Post pruning feature set which analysis was performed on
├── removed_features_log.txt            #All features removed during pruning    
├── merge_features_and_performance.py   #Merges output from scripts above with tool performance results, and performs redundancy analysis
├── unified_analysis.py                 #Outputs upset diagrams, and all vs none and tool vs rest comparisons (RQ3)
├── predictor.py                        #Performs prediction tasks from RQ4
├── requirements.txt
├── tool_comparison_summary.csv         #Tool performance for each bug
├── prompts_and_regex.md                #All prompts and regex used for feature extraction
└── defects4j_xml/                      # Defects4J XML inputs (used by extraction)
```

Run the pipeline once to generate all results (feature integration, statistical comparisons, complementarity visualizations, and prediction outputs).

## Installation

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

## Run the pipeline

```bash
python run_full_pipeline.py
```

Preview commands without executing:

```bash
python run_full_pipeline.py --dry-run
```

## Unified analysis outputs (`tool_comparison_results_fixed/`)

The unified analysis stage (implemented in `unified_analysis.py`) generates outputs for **Top-1**, **Top-5**, and **Top-10**.

- **Tool complementarity (UpSet diagrams)**:
  - `tool_comparison_results_fixed/upset_diagram_top1.png`
  - `tool_comparison_results_fixed/upset_diagram_top5.png`
  - `tool_comparison_results_fixed/upset_diagram_top10.png`

- **All tools vs none (feature differences)**:
  - `tool_comparison_results_fixed/all_vs_none_top1.csv`
  - `tool_comparison_results_fixed/all_vs_none_top5.csv`
  - `tool_comparison_results_fixed/all_vs_none_top10.csv`

- **Bug difficulty spectrum (by number of tools that succeed)**:
  - `tool_comparison_results_fixed/bug_difficulty_spectrum_top1.csv`
  - `tool_comparison_results_fixed/bug_difficulty_spectrum_top5.csv`
  - `tool_comparison_results_fixed/bug_difficulty_spectrum_top10.csv`

- **Each tool vs the rest (unique signals)**:
  - `tool_comparison_results_fixed/tool_vs_rest_top1.csv`
  - `tool_comparison_results_fixed/tool_vs_rest_top5.csv`
  - `tool_comparison_results_fixed/tool_vs_rest_top10.csv`







## Predictor outputs (`delta_score_outputs/`)

The predictor stage writes:
- `delta_score_outputs/cv_fold_results.csv`
- `delta_score_outputs/cv_auc_summary.csv`
- `delta_score_outputs/score_vs_success_top5.csv`
- `delta_score_outputs/full_dataset_deltas.csv`
- `delta_score_outputs/full_dataset_scaling.csv`
- `delta_score_outputs/cascade_routing_topk_sweep.csv`
- `delta_score_outputs/plots/`

## Notes

# 5. Review correlations_any_tool.csv
## Find (Example): clarity correlates 0.45 with detection (p < 0.001)

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

## License

© APRTSM Lab — Bilkent University. Distributed under the APRTSM Lab Research License (non-commercial use only).
