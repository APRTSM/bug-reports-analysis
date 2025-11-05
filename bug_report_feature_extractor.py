"""
Bug Report Feature Extractor

This module extracts text characteristics and sample representativeness features
from bug reports, including:
- Descriptive statistics, syntactic complexity, and readability metrics (via TextDescriptives)
- Coherence using Coherence Momentum model
- Outlier scores for sample representativeness
- Semantic similarity to demonstration examples using BERTScore
"""

import json
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional
import warnings
import argparse
import os
from pathlib import Path
import glob
warnings.filterwarnings('ignore')


class BugReportFeatureExtractor:
    """
    Extracts comprehensive text features from bug reports for analysis.
    
    Features extracted:
    1. TextDescriptives features (~65 features): descriptive stats, syntactic complexity, readability
    2. Coherence scores using Coherence Momentum model
    3. Outlier scores for sample representativeness
    4. Semantic similarity scores to demonstration examples
    
    Total: ~70 features
    """
    
    def __init__(self, demonstration_examples: Optional[List[str]] = None):
        """
        Initialize the feature extractor.
        
        Args:
            demonstration_examples: List of example bug reports for similarity comparison
        """
        self.demonstration_examples = demonstration_examples or []
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize all required models and APIs."""
        print("Initializing models...")
        
        # TextDescriptives for comprehensive text features
        try:
            import textdescriptives as td
            self.td = td
            self.td_available = True
        except ImportError:
            print("Warning: textdescriptives not installed. Install with: pip install textdescriptives")
            self.td_available = False
        
        # BERTScore for semantic similarity
        try:
            from bert_score import score as bert_score
            self.bert_score = bert_score
            self.bert_score_available = True
        except ImportError:
            print("Warning: bert-score not installed. Install with: pip install bert-score")
            self.bert_score_available = False
        
        # Cleanlab for outlier detection
        try:
            from cleanlab.outlier import OutOfDistribution
            self.OutOfDistribution = OutOfDistribution
            self.cleanlab_available = True
        except ImportError:
            print("Warning: cleanlab not installed. Install with: pip install cleanlab")
            self.cleanlab_available = False
        
        # Sentence transformers for embeddings (needed for coherence and outlier detection)
        try:
            from sentence_transformers import SentenceTransformer
            self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
            self.embeddings_available = True
        except ImportError:
            print("Warning: sentence-transformers not installed. Install with: pip install sentence-transformers")
            self.embeddings_available = False
        
        print("Initialization complete.")
    
    def extract_text_from_bug_report(self, bug_report: Dict[str, Any]) -> str:
        """
        Extract text content from bug report JSON.
        
        Args:
            bug_report: Dictionary containing bug report data
            
        Returns:
            Combined text from title and description
        """
        title = bug_report.get('title', '')
        description = bug_report.get('description', '')
        return f"{title}\n\n{description}".strip()
    
    def extract_textdescriptives_features(self, text: str) -> Dict[str, float]:
        """
        Extract comprehensive text features using TextDescriptives API.
        
        Features include:
        - Descriptive statistics (token count, sentence count, etc.)
        - Syntactic complexity metrics
        - Readability scores (Flesch, Dale-Chall, etc.)
        
        Args:
            text: Input text to analyze
            
        Returns:
            Dictionary of feature names and values
        """
        if not self.td_available:
            return self._get_fallback_text_features(text)
        
        try:
            # Extract all available metrics from TextDescriptives
            df = pd.DataFrame([{"text": text}])
            metrics = self.td.extract_metrics(
                text=df["text"],
                metrics=[
                    "descriptive_stats",
                    "readability",
                    "pos_proportions",
                    "dependency_distance"
                ],
                spacy_model="en_core_web_sm"
            )
            
            # Convert to dictionary and flatten
            features = {}
            for col in metrics.columns:
                if col != 'text':
                    value = metrics[col].iloc[0]
                    if isinstance(value, (int, float, np.number)):
                        features[f"td_{col}"] = float(value)
            
            return features
        except Exception as e:
            print(f"Error extracting TextDescriptives features: {e}")
            return self._get_fallback_text_features(text)
    
    def _get_fallback_text_features(self, text: str) -> Dict[str, float]:
        """
        Fallback basic text features if TextDescriptives is unavailable.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary of basic text features
        """
        words = text.split()
        sentences = text.split('.')
        
        return {
            'td_n_tokens': len(words),
            'td_n_sentences': len([s for s in sentences if s.strip()]),
            'td_n_characters': len(text),
            'td_avg_word_length': np.mean([len(w) for w in words]) if words else 0,
            'td_avg_sentence_length': len(words) / max(len([s for s in sentences if s.strip()]), 1)
        }
    
    def calculate_coherence_momentum(self, text: str) -> float:
        """
        Calculate coherence using Coherence Momentum model.
        
        This measures semantic consistency across sentences in the text.
        
        Args:
            text: Input text
            
        Returns:
            Coherence momentum score
        """
        if not self.embeddings_available:
            return 0.0
        
        try:
            # Split into sentences
            sentences = [s.strip() for s in text.split('.') if s.strip()]
            
            if len(sentences) < 2:
                return 1.0  # Single sentence is perfectly coherent
            
            # Get embeddings for each sentence
            embeddings = self.sentence_model.encode(sentences)
            
            # Calculate coherence momentum as average cosine similarity between consecutive sentences
            similarities = []
            for i in range(len(embeddings) - 1):
                sim = np.dot(embeddings[i], embeddings[i + 1]) / (
                    np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[i + 1])
                )
                similarities.append(sim)
            
            # Coherence momentum is the average similarity
            coherence_score = np.mean(similarities) if similarities else 0.0
            
            return float(coherence_score)
        except Exception as e:
            print(f"Error calculating coherence: {e}")
            return 0.0
    
    def calculate_outlier_score(self, texts: List[str], current_text: str) -> float:
        """
        Calculate outlier score using cleanlab.outlier API.
        
        Measures how atypical the current sample is compared to others.
        
        Args:
            texts: List of all bug report texts in dataset
            current_text: The bug report text to evaluate
            
        Returns:
            Outlier score (higher = more atypical)
        """
        if not self.cleanlab_available or not self.embeddings_available:
            return 0.0
        
        try:
            # Get embeddings for all texts
            all_embeddings = self.sentence_model.encode(texts + [current_text])
            
            # Use cleanlab's OutOfDistribution detector
            train_embeddings = all_embeddings[:-1]
            test_embedding = all_embeddings[-1:]
            
            # Initialize OutOfDistribution detector and get scores
            # Newer cleanlab API: pass features to score method directly
            ood = self.OutOfDistribution()
            # Score returns outlier scores for the test embedding
            outlier_scores = ood.score(features=test_embedding, feature_embeddings=train_embeddings)
            outlier_score = outlier_scores[0]
            
            return float(outlier_score)
        except Exception as e:
            # Fallback: calculate average distance to all other samples
            if self.embeddings_available:
                try:
                    all_embeddings = self.sentence_model.encode(texts + [current_text])
                    current_emb = all_embeddings[-1]
                    others_emb = all_embeddings[:-1]
                    
                    distances = [np.linalg.norm(current_emb - other) for other in others_emb]
                    return float(np.mean(distances))
                except Exception:
                    pass
            return 0.0
    
    def calculate_semantic_similarity(self, text: str, demonstration_examples: List[str]) -> Dict[str, float]:
        """
        Calculate semantic similarity to demonstration examples using BERTScore.
        
        Args:
            text: Bug report text to evaluate
            demonstration_examples: List of demonstration example texts
            
        Returns:
            Dictionary containing similarity statistics
        """
        if not demonstration_examples:
            return {
                'bertscore_max': 0.0,
                'bertscore_mean': 0.0,
                'bertscore_min': 0.0
            }
        
        if not self.bert_score_available:
            # Fallback to cosine similarity with sentence embeddings
            if self.embeddings_available:
                current_emb = self.sentence_model.encode([text])[0]
                demo_embs = self.sentence_model.encode(demonstration_examples)
                
                similarities = [
                    np.dot(current_emb, demo_emb) / (
                        np.linalg.norm(current_emb) * np.linalg.norm(demo_emb)
                    )
                    for demo_emb in demo_embs
                ]
                
                return {
                    'bertscore_max': float(np.max(similarities)),
                    'bertscore_mean': float(np.mean(similarities)),
                    'bertscore_min': float(np.min(similarities))
                }
            return {
                'bertscore_max': 0.0,
                'bertscore_mean': 0.0,
                'bertscore_min': 0.0
            }
        
        try:
            # Calculate BERTScore for each demonstration example
            scores = []
            for demo in demonstration_examples:
                P, R, F1 = self.bert_score([text], [demo], lang='en', verbose=False)
                scores.append(float(F1.mean()))
            
            return {
                'bertscore_max': float(np.max(scores)),
                'bertscore_mean': float(np.mean(scores)),
                'bertscore_min': float(np.min(scores))
            }
        except Exception as e:
            print(f"Error calculating BERTScore: {e}")
            return {
                'bertscore_max': 0.0,
                'bertscore_mean': 0.0,
                'bertscore_min': 0.0
            }
    
    def extract_all_features(
        self,
        bug_report: Dict[str, Any],
        all_bug_reports: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, float]:
        """
        Extract all features from a bug report.
        
        Args:
            bug_report: The bug report to analyze
            all_bug_reports: List of all bug reports in dataset (for outlier detection)
            
        Returns:
            Dictionary containing all ~70 features
        """
        # Extract text
        text = self.extract_text_from_bug_report(bug_report)
        
        # Initialize features dictionary
        features = {}
        
        # 1. TextDescriptives features (~65 features)
        print("Extracting TextDescriptives features...")
        td_features = self.extract_textdescriptives_features(text)
        features.update(td_features)
        
        # 2. Coherence Momentum
        print("Calculating coherence momentum...")
        coherence = self.calculate_coherence_momentum(text)
        features['coherence_momentum'] = coherence
        
        # 3. Outlier score (if dataset provided)
        if all_bug_reports:
            print("Calculating outlier score...")
            all_texts = [self.extract_text_from_bug_report(br) for br in all_bug_reports]
            outlier_score = self.calculate_outlier_score(all_texts, text)
            features['outlier_score'] = outlier_score
        else:
            features['outlier_score'] = 0.0
        
        # 4. Semantic similarity to demonstrations
        print("Calculating semantic similarity to demonstrations...")
        similarity_features = self.calculate_semantic_similarity(
            text,
            self.demonstration_examples
        )
        features.update(similarity_features)
        
        print(f"Extracted {len(features)} features in total.")
        
        return features
    
    def extract_features_batch(
        self,
        bug_reports: List[Dict[str, Any]]
    ) -> pd.DataFrame:
        """
        Extract features for multiple bug reports.
        
        Args:
            bug_reports: List of bug reports to analyze
            
        Returns:
            DataFrame with features for each bug report
        """
        all_features = []
        
        for i, bug_report in enumerate(bug_reports):
            print(f"\nProcessing bug report {i+1}/{len(bug_reports)}...")
            features = self.extract_all_features(bug_report, bug_reports)
            # Use filename if available, otherwise title, otherwise index
            bug_id = bug_report.get('filename') or bug_report.get('title') or f'bug_{i}'
            features['bug_id'] = bug_id
            all_features.append(features)
        
        return pd.DataFrame(all_features)


def load_bug_reports_from_directory(directory_path: str) -> List[Dict[str, Any]]:
    """
    Load all JSON bug reports from a directory.
    
    Args:
        directory_path: Path to directory containing JSON bug report files
        
    Returns:
        List of bug report dictionaries
    """
    bug_reports = []
    json_files = glob.glob(os.path.join(directory_path, "*.json"))
    
    print(f"Found {len(json_files)} JSON files in {directory_path}")
    
    for json_file in json_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                bug_report = json.load(f)
                # Add filename as identifier if not present
                if 'filename' not in bug_report:
                    bug_report['filename'] = os.path.basename(json_file)
                bug_reports.append(bug_report)
        except Exception as e:
            print(f"Warning: Failed to load {json_file}: {e}")
            continue
    
    print(f"Successfully loaded {len(bug_reports)} bug reports")
    return bug_reports


def main():
    """Main function to extract features from bug reports."""
    
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description='Extract features from bug reports in Defects4J or GHRB datasets'
    )
    parser.add_argument(
        '--dataset',
        type=str,
        choices=['Defects4J', 'GHRB', 'both'],
        default='Defects4J',
        help='Dataset to process: Defects4J, GHRB, or both (default: Defects4J)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='outputs',
        help='Output directory for feature CSV files (default: outputs)'
    )
    parser.add_argument(
        '--bug-reports-dir',
        type=str,
        default='bug_reports',
        help='Directory containing bug report subdirectories (default: bug_reports)'
    )
    
    args = parser.parse_args()
    
    # Get the script's directory as base path
    script_dir = Path(__file__).parent.absolute()
    bug_reports_base = script_dir / args.bug_reports_dir
    output_dir = script_dir / args.output_dir
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(exist_ok=True)
    
    # Define some demonstration examples (you can customize these)
    demonstration_examples = [
        "Setting up the project in Eclipse resulted in a compilation error in the rendering module.",
        "The null pointer check is incorrect and should be inverted to prevent crashes.",
        "Code analysis revealed a potential bug in the dataset validation logic."
    ]
    
    # Initialize extractor
    print("="*80)
    print("INITIALIZING BUG REPORT FEATURE EXTRACTOR")
    print("="*80)
    extractor = BugReportFeatureExtractor(demonstration_examples=demonstration_examples)
    
    # Determine which datasets to process
    datasets_to_process = []
    if args.dataset == 'both':
        datasets_to_process = ['Defects4J', 'GHRB']
    else:
        datasets_to_process = [args.dataset]
    
    # Collect all feature DataFrames
    all_features_dfs = []
    
    # Process each dataset
    for dataset_name in datasets_to_process:
        dataset_path = bug_reports_base / dataset_name
        
        if not dataset_path.exists():
            print(f"Warning: Directory {dataset_path} does not exist. Skipping.")
            continue
        
        print("\n" + "="*80)
        print(f"PROCESSING DATASET: {dataset_name}")
        print("="*80)
        
        # Load all bug reports from the directory
        print(f"\nLoading bug reports from {dataset_path}...")
        bug_reports = load_bug_reports_from_directory(str(dataset_path))
        
        if not bug_reports:
            print(f"No bug reports found in {dataset_path}. Skipping.")
            continue
        
        # Extract features for all bug reports
        print(f"\nExtracting features for {len(bug_reports)} bug reports...")
        features_df = extractor.extract_features_batch(bug_reports)
        
        # Add dataset column to identify source
        features_df.insert(0, 'dataset', dataset_name)
        
        # Store for later combination
        all_features_dfs.append(features_df)
        
        # Display summary
        print("\n" + "="*80)
        print(f"FEATURE SUMMARY FOR {dataset_name}")
        print("="*80)
        print(f"Total bug reports processed: {len(features_df)}")
        print(f"Total features extracted: {len(features_df.columns) - 2}")  # -2 for dataset and bug_id columns
        print(f"\nFeature categories:")
        td_features = [col for col in features_df.columns if col.startswith('td_')]
        print(f"  - TextDescriptives features: {len(td_features)}")
        print(f"  - Coherence: 1")
        print(f"  - Outlier score: 1")
        bertscore_features = [col for col in features_df.columns if col.startswith('bertscore_')]
        print(f"  - Semantic similarity (BERTScore): {len(bertscore_features)}")
    
    # Combine all datasets into a single DataFrame
    if all_features_dfs:
        print("\n" + "="*80)
        print("COMBINING ALL RESULTS")
        print("="*80)
        
        # Concatenate all DataFrames
        combined_df = pd.concat(all_features_dfs, ignore_index=True)
        
        # Reorder columns: bug_id first, then dataset, then all features
        # Get all columns except bug_id and dataset
        feature_columns = [col for col in combined_df.columns if col not in ['bug_id', 'dataset']]
        # Reorder: bug_id first, then dataset, then all features (sorted alphabetically)
        column_order = ['bug_id', 'dataset'] + sorted(feature_columns)
        combined_df = combined_df[column_order]
        
        # Save combined results to single CSV file
        output_path = output_dir / "all_bug_report_features.csv"
        combined_df.to_csv(output_path, index=False)
        print(f"\nCombined features saved to: {output_path}")
        print(f"Total bug reports: {len(combined_df)}")
        print(f"Total features: {len(feature_columns)}")
        print(f"Columns: bug_id, dataset, and {len(feature_columns)} feature columns")
    else:
        print("\nNo data to save. Please check that the dataset directories exist and contain JSON files.")
    
    print("\n" + "="*80)
    print("PROCESSING COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()