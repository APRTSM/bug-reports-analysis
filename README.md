# Bug Reports Analysis

This repository is a replication package for the paper *Beyond Keywords: Understanding Bug Report Features Impact on Fault Localization*.


## Repository Structure

```
bug-reports-analysis/
├── run_full_pipeline.py                          # End-to-end pipeline runner
├── bug_feature_extraction/                       # Feature extraction scripts (run in this order)
│   ├── extract_bug_features.py
│   ├── gemini_bug_categorization_overall.py
│   ├── fine_grained_gemini_catg.py
│   └── gemini_bug_ratings.py
├── tool_feature_analysis/
│   ├── merge_features_and_performance.py         # Merges extracted features with tool performance,
│   │                                              # performs redundancy pruning
│   └── tool_comparison_summary_main.csv          # Per-bug, per-tool FL results (rank/MRR/MAP/top@k)
├── full_feature_preproccessed_fixed/
│   └── final_feature_set_bug_reports_analysis.csv  # Post-pruning feature set analysis is performed on
├── tool_comparison_results_fixed/
│   ├── unified_analysis.py                       # Upset diagrams, all-vs-none and tool-vs-rest (RQ3)
│   └── plot_tool_vs_rest.py                      # Lollipop + heatmap figures from tool-vs-rest CSVs
│                                                  # (manual step, not part of run_full_pipeline.py)
├── delta_score_outputs/
│   ├── predictor.py                              # Delta-weighted scorer + cascade routing (RQ4)
│   └── scale_confound_analysis.py                # Standalone: tests report-feature associations for
│                                                  # confounding with project scale (not auto-run)
├── energy_consumption/                           # Measured per-bug energy/CO2 logs (IR, BRaIn, FlexFL),
│                                                  # used by predictor.py's cascade energy estimate
├── results/                                       # Formatted LaTeX tables used in the paper
├── removed_features_log.txt                      # All features removed during pruning
├── prompts_and_regex.md                           # All prompts and regex used for feature extraction
├── requirements.txt
└── defects4j_xml/                                 # Defects4J XML inputs (used by extraction)
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

This runs, in order: the `bug_feature_extraction/` scripts, `tool_feature_analysis/merge_features_and_performance.py`, `tool_comparison_results_fixed/unified_analysis.py`, then `delta_score_outputs/predictor.py`.

Two scripts are **not** part of this automatic run and are invoked manually when needed (see below): `tool_comparison_results_fixed/plot_tool_vs_rest.py` and `delta_score_outputs/scale_confound_analysis.py`.

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

Run manually to generate figures from the tool-vs-rest CSVs above:

```bash
python tool_comparison_results_fixed/plot_tool_vs_rest.py
```

Writes lollipop small-multiples and heatmaps (PDF + PNG) to `tool_comparison_results_fixed/figures/`.

## Predictor outputs (`delta_score_outputs/`)

The predictor stage (`predictor.py`) writes:
- `delta_score_outputs/cv_fold_results.csv`
- `delta_score_outputs/cv_auc_summary.csv`
- `delta_score_outputs/score_vs_success_top5.csv`
- `delta_score_outputs/full_dataset_deltas.csv`
- `delta_score_outputs/full_dataset_scaling.csv`
- `delta_score_outputs/cascade_routing_topk_sweep.csv` — three-tier (IR → BRaIn → FlexFL) cascade
  routing sweep: hit rate/MRR/MAP for the scorer-based and matched-size random-routing baselines at
  each tier-size split, plus a measured energy/CO2 estimate per split (`energy_kwh_mean`,
  `co2_kg_mean`, `energy_saved_pct_vs_full_cascade`, and the random-routing equivalents) when
  `energy_consumption/` is present. If those energy log files are missing, this estimate is skipped
  and the rest of the pipeline runs unaffected.
- `delta_score_outputs/plots/` — delta stability, ROC, score distribution, and cascade routing plots

### Scale-confound check (optional, manual)

```bash
python delta_score_outputs/scale_confound_analysis.py
```

Tests whether the report-feature associations behind the all-vs-none and tool-vs-rest analyses
survive controlling for project scale (number of Java files), via paired raw-vs-adjusted logistic
regression plus a tertile-stratified Cliff's delta cross-check. Writes
`delta_score_outputs/scale_confound_analysis.csv`.

## License

© APRTSM Lab — Bilkent University. Distributed under the APRTSM Lab Research License (non-commercial use only).
