# Bug Reports Analysis - Feature Extraction

This repository contains tools for extracting comprehensive text features from bug reports in the Defects4J and GHRB datasets. The feature extractor analyzes bug reports and generates ~70 features including descriptive statistics, readability metrics, coherence scores, outlier detection, and semantic similarity.

## Overview

The `BugReportFeatureExtractor` class extracts the following feature categories:

1. **TextDescriptives Features** (~48 features): Descriptive statistics, readability scores, POS proportions, and dependency distances
2. **Coherence Momentum** (1 feature): Semantic consistency across sentences
3. **Outlier Score** (1 feature): How atypical the bug report is compared to others
4. **Semantic Similarity** (3 features): BERTScore similarity to demonstration examples (max, mean, min)

**Total: ~48-70 features per bug report** (depending on available metrics)

## Repository Structure

```
bug-reports-analysis/
├── bug_report_feature_extractor.py  # Main feature extraction script
├── textdescriptives_diagnostic.py   # Diagnostic tool for TextDescriptives
├── requirements.txt                  # Python dependencies
├── bug_reports/                      # Bug report datasets
│   ├── Defects4J/                   # 835 JSON bug reports
│   └── GHRB/                        # 97 JSON bug reports
└── outputs/                          # Output directory for CSV files
```

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup

1. **Clone the repository** (if applicable) or navigate to the project directory:
   ```bash
   cd bug-reports-analysis
   ```

2. **Create a virtual environment** (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Download SpaCy language model**:
   ```bash
   python -m spacy download en_core_web_sm
   ```

## Usage

### Basic Usage

Process all bug reports from a specific dataset:

```bash
# Process Defects4J dataset (default)
python bug_report_feature_extractor.py

# Process GHRB dataset
python bug_report_feature_extractor.py --dataset GHRB

# Process both datasets
python bug_report_feature_extractor.py --dataset both
```

### Command-Line Options

- `--dataset`: Choose which dataset to process (`Defects4J`, `GHRB`, or `both`). Default: `Defects4J`
- `--output-dir`: Specify output directory for CSV files. Default: `outputs`
- `--bug-reports-dir`: Specify directory containing bug report subdirectories. Default: `bug_reports`

### Examples

```bash
# Process both datasets and save to custom output directory
python bug_report_feature_extractor.py --dataset both --output-dir results

# Process only GHRB with custom bug reports directory
python bug_report_feature_extractor.py --dataset GHRB --bug-reports-dir /path/to/bug_reports
```

## Output

The script generates a single CSV file containing all extracted features:

- **Output file**: `outputs/all_bug_report_features.csv`
- **Format**: CSV with columns:
  - `bug_id`: Unique identifier (filename or title)
  - `dataset`: Source dataset (Defects4J or GHRB)
  - Feature columns: All extracted features (prefixed with `td_` for TextDescriptives, `bertscore_` for similarity scores, etc.)

### Output Structure

```
bug_id,dataset,td_n_tokens,td_n_sentences,...,coherence_momentum,outlier_score,bertscore_max,bertscore_mean,bertscore_min
Chart-1.json,Defects4J,45,3,...,0.75,0.23,0.85,0.82,0.78
Lang-1.json,Defects4J,52,4,...,0.72,0.19,0.88,0.84,0.80
...
```

## Features Extracted

### TextDescriptives Features

Extracted using the [TextDescriptives](https://github.com/hlasse/TextDescriptives) library:

- **Descriptive Statistics**: Token count, sentence count, character count, average word/sentence length
- **Readability Scores**: Flesch Reading Ease, Dale-Chall, etc.
- **POS Proportions**: Part-of-speech tag distributions
- **Dependency Distance**: Syntactic dependency metrics

### Coherence Momentum

Measures semantic consistency across sentences using sentence embeddings and cosine similarity between consecutive sentences.

### Outlier Score

Uses Cleanlab's OutOfDistribution detector to identify how atypical a bug report is compared to the dataset. Falls back to average distance metric if Cleanlab is unavailable.

### Semantic Similarity (BERTScore)

Calculates semantic similarity to demonstration examples using BERTScore, providing max, mean, and min similarity scores.

## Dependencies

- **numpy** (>=1.21.0): Numerical operations
- **pandas** (>=1.3.0): Data manipulation
- **textdescriptives** (>=2.8.0): Text feature extraction
- **bert-score** (>=0.3.13): Semantic similarity
- **cleanlab** (>=2.6.0): Outlier detection
- **sentence-transformers** (>=2.2.0): Sentence embeddings
- **spacy** (>=3.0.0): NLP processing
- **torch** (>=1.9.0): Deep learning backend

## Troubleshooting

### TextDescriptives Errors

If you encounter errors about missing TextDescriptives metrics:

1. Update TextDescriptives:
   ```bash
   pip install --upgrade textdescriptives
   ```

2. Run the diagnostic script:
   ```bash
   python textdescriptives_diagnostic.py
   ```

### Model Loading Issues

The script may take a few minutes to initialize on first run as it downloads models:
- SentenceTransformer model (`all-MiniLM-L6-v2`)
- BERTScore models (RoBERTa-large)

If model loading hangs, ensure you have:
- Sufficient disk space
- Stable internet connection (for first-time downloads)
- Proper permissions to write to cache directories

### Common Warnings

- **RobertaModel warnings**: These are informational messages from BERTScore and can be safely ignored
- **Missing metrics**: Some TextDescriptives metrics may not be available in all versions - the script will fall back to available features

## Bug Report Format

Expected JSON format for bug reports:

```json
{
    "title": "Bug report title",
    "description": "Detailed bug description..."
}
```

The script will also use the filename as `bug_id` if available.

## Performance

- Processing time: ~1-5 seconds per bug report (depending on text length and available hardware)
- Full dataset processing: ~60-120 minutes for all 932 bug reports
- Memory usage: ~2-4 GB (due to model loading)

## License

© APRTSM Lab — Bilkent University.
Distributed under the APRTSM Lab Research License (non-commercial use only).


## Acknowledgments

- [Defects4J](https://github.com/rjust/defects4j) dataset
- [GHRB](https://github.com/soarsmu/GHRB) dataset
- [TextDescriptives](https://github.com/hlasse/TextDescriptives) library
- [BERTScore](https://github.com/Tiiiger/bert_score) for semantic similarity
- [Cleanlab](https://github.com/cleanlab/cleanlab) for outlier detection

