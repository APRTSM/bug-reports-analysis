"""
TextDescriptives Feature Diagnostic Script

This script helps you understand:
1. Which TextDescriptives metrics are available
2. How many features each metric category produces
3. What the actual feature count is for your installation

Run this to diagnose why you're getting 48 features instead of 70.
"""

import warnings
warnings.filterwarnings('ignore')

def check_textdescriptives_features():
    """Check what features are available in TextDescriptives."""
    
    print("="*80)
    print("TEXTDESCRIPTIVES FEATURE DIAGNOSTIC")
    print("="*80)
    
    # Try to import textdescriptives
    try:
        import textdescriptives as td
        print("✓ TextDescriptives is installed\n")
    except ImportError:
        print("✗ TextDescriptives is NOT installed")
        print("Install with: pip install textdescriptives")
        return
    
    # Sample text
    sample_text = """
    This is a sample bug report. The application crashes when trying to process 
    null values in the input validation module. The error occurs consistently 
    during the data preprocessing phase.
    """
    
    # List of all possible metrics
    all_possible_metrics = [
        "descriptive_stats",
        "readability",
        "pos_proportions",
        "dependency_distance",
        "coherence",
        "information_theory",
        "quality",
        "morphology"
    ]
    
    print("Testing each metric category:\n")
    print("-"*80)
    
    total_features = 0
    successful_metrics = []
    
    for metric in all_possible_metrics:
        try:
            import pandas as pd
            df = pd.DataFrame([{"text": sample_text}])
            result = td.extract_metrics(
                text=df["text"],
                metrics=[metric],
                spacy_model="en_core_web_sm"
            )
            
            # Count features (excluding 'text' column)
            feature_count = len([col for col in result.columns if col != 'text'])
            total_features += feature_count
            successful_metrics.append(metric)
            
            print(f"✓ {metric:25s} - {feature_count:3d} features")
            
            # Show first few feature names
            features = [col for col in result.columns if col != 'text']
            if features:
                print(f"  Sample features: {', '.join(features[:5])}")
                if len(features) > 5:
                    print(f"  ... and {len(features) - 5} more")
            print()
            
        except Exception as e:
            print(f"✗ {metric:25s} - FAILED: {str(e)[:60]}")
            print()
    
    print("-"*80)
    print(f"\nSUMMARY:")
    print(f"  Successful metrics: {len(successful_metrics)}/{len(all_possible_metrics)}")
    print(f"  Total features from TextDescriptives: {total_features}")
    print(f"  Working metrics: {', '.join(successful_metrics)}")
    
    # Calculate expected total
    print(f"\n  Expected total (with all additional features):")
    print(f"  - TextDescriptives: {total_features}")
    print(f"  - Coherence momentum (custom): 1")
    print(f"  - Outlier score: 1")
    print(f"  - Semantic similarity (BERTScore): 6")
    print(f"  - TOTAL: {total_features + 8}")
    
    # Compare to paper's 70
    print(f"\n  Paper's target: 70 features")
    print(f"  Your current: {total_features + 8} features")
    difference = 70 - (total_features + 8)
    if difference > 0:
        print(f"  Missing: {difference} features")
    else:
        print(f"  Surplus: {abs(difference)} features")
    
    print("\n" + "="*80)
    print("RECOMMENDATIONS:")
    print("="*80)
    
    if total_features < 60:
        print("\n1. Your TextDescriptives might be missing some metric categories.")
        print("   Try updating: pip install --upgrade textdescriptives")
        print("\n2. Check if spacy model is installed:")
        print("   python -m spacy download en_core_web_sm")
    
    if 'coherence' not in successful_metrics:
        print("\n3. Built-in coherence metric is not available.")
        print("   You'll rely on the custom coherence momentum implementation.")
    
    if 'information_theory' not in successful_metrics:
        print("\n4. Information theory metrics (entropy, perplexity) are missing.")
        print("   These may require additional dependencies.")
    
    print("\n" + "="*80)


def check_all_dependencies():
    """Check if all required dependencies are installed."""
    
    print("\n" + "="*80)
    print("DEPENDENCY CHECK")
    print("="*80 + "\n")
    
    dependencies = {
        'textdescriptives': 'TextDescriptives API',
        'bert_score': 'BERTScore for similarity',
        'cleanlab': 'Cleanlab for outlier detection',
        'sentence_transformers': 'Sentence embeddings',
        'spacy': 'SpaCy NLP',
        'pandas': 'Data manipulation',
        'numpy': 'Numerical operations'
    }
    
    for package, description in dependencies.items():
        try:
            __import__(package)
            print(f"✓ {description:40s} - INSTALLED")
        except ImportError:
            print(f"✗ {description:40s} - NOT INSTALLED")
            print(f"  Install with: pip install {package}")
    
    # Check spacy model
    print("\nChecking SpaCy model:")
    try:
        import spacy
        nlp = spacy.load("en_core_web_sm")
        print("✓ SpaCy model 'en_core_web_sm' - INSTALLED")
    except:
        print("✗ SpaCy model 'en_core_web_sm' - NOT INSTALLED")
        print("  Install with: python -m spacy download en_core_web_sm")
    
    print("\n" + "="*80)


if __name__ == "__main__":
    check_all_dependencies()
    print("\n")
    check_textdescriptives_features()
    
    print("\n" + "="*80)
    print("DIAGNOSTIC COMPLETE")
    print("="*80)
    print("\nNext steps:")
    print("1. Install any missing dependencies")
    print("2. Update textdescriptives if needed")
    print("3. Run the improved feature extractor script")
    print("="*80)