"""
Enhanced Merge and Preprocess Script with LLM Feature Engineering

This script:
1. Loads multiple data sources (features, ratings, categories, performance metrics, bee_results)
2. Merges them into a unified dataset
3. Performs rich preprocessing including:
   - Text feature extraction
   - LLM interaction features (NEW)
   - Ambiguity type encoding (NEW)
   - Exception category features (NEW - from enhanced extraction)
   - Concept network features (NEW)
   - Category encoding
   - Concept multi-hot encoding
   - Feature standardization
4. Outputs:
   - experimentA_full_dataset.csv: Full dataset with all original columns
   - experimentA_preprocessed_rich.csv: Preprocessed dataset ready for modeling
   - feature_scaler.pkl: StandardScaler for feature scaling
"""

import re
import json
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Set
from sklearn.preprocessing import StandardScaler

DATA_DIR = Path(".")

# Configuration
ENABLE_STANDARDIZATION = True  # Set to False to skip standardization

# Input files - UPDATED to use enhanced features
FEATURES_FILE = DATA_DIR / "bug_features_cleaned.csv" 
RATINGS_FILE  = DATA_DIR / "gemini_ratings/gemini_bug_ratings.csv"
CATEG_FILE    = DATA_DIR / "gemini_ratings/gemini_bug_categorization.csv"
FINE_GRAINED_CATEG_FILE = DATA_DIR / "gemini_ratings/fine_grained_gemini_categorization.csv"
PERF_FILE     = DATA_DIR / "tool_comparison_summary.csv"
BEE_RESULTS_FILE = DATA_DIR / "bee_results.jsonl"

# Output files
OUT_FULL = DATA_DIR / "full_feature_preproccessed_fixed/experimentA_full_dataset.csv"
OUT_PREP = DATA_DIR / "full_feature_preproccessed_fixed/experimentA_preprocessed_rich.csv"
SCALER_FILE = DATA_DIR / "feature_scaler.pkl"

# -----------------------------
# Helpers
# -----------------------------
def split_id(df: pd.DataFrame) -> pd.DataFrame:
    """Parse project & bug_id from 'id' like 'Chart-1'."""
    if "id" not in df.columns:
        raise ValueError("Expected an 'id' column.")
    proj_bug = df["id"].astype(str).str.split("-", n=1, expand=True)
    df = df.copy()
    df["project"] = proj_bug[0]
    df["bug_id"]  = pd.to_numeric(proj_bug[1], errors="coerce").astype("Int64")
    return df

def safe_str(x) -> str:
    if pd.isna(x):
        return ""
    return str(x)

_word_re = re.compile(r"[A-Za-z0-9_]+")

def text_stats(series: pd.Series, prefix: str) -> pd.DataFrame:
    """Extract lightweight numeric features from free text with normalization."""
    s = series.fillna("").astype(str)

    char_len = s.str.len()
    word_counts = s.apply(lambda t: len(_word_re.findall(t)))
    line_counts = s.apply(lambda t: t.count("\n") + (1 if t else 0))

    avg_word_len = char_len / word_counts.replace(0, np.nan)
    avg_words_per_line = word_counts / line_counts.replace(0, np.nan)

    lower = s.str.lower()
    hedge_markers = [
        "maybe", "might", "likely", "possibly", "unclear", "unsure",
        "seems", "appears", "probably", "could", "cannot", "can't",
        "unknown", "approximately", "guess"
    ]
    hedge_count = sum(lower.str.count(re.escape(m)) for m in hedge_markers)
    hedge_density = hedge_count / word_counts.replace(0, np.nan)

    has_code_like = lower.str.contains(
        r"\b(stack trace|exception|nullpointer|assert|traceback|line \d+)\b", 
        regex=True
    ).astype(int)
    question_density = s.str.count(r"\?") / word_counts.replace(0, np.nan)
    exclaim_density = s.str.count(r"!") / word_counts.replace(0, np.nan)
    digit_density = s.str.count(r"\d") / char_len.replace(0, np.nan)

    def uniq_ratio(t: str) -> float:
        toks = _word_re.findall(t.lower())
        if not toks:
            return np.nan
        return len(set(toks)) / len(toks)

    uniq_word_ratio = s.apply(uniq_ratio)

    def avg_sent_len(t: str) -> float:
        sentences = re.split(r'[.!?]+', t)
        sentences = [sent.strip() for sent in sentences if sent.strip()]
        if not sentences:
            return np.nan
        words_per_sent = [len(_word_re.findall(sent)) for sent in sentences]
        return np.mean(words_per_sent) if words_per_sent else np.nan

    avg_sentence_len = s.apply(avg_sent_len)

    return pd.DataFrame({
        f"{prefix}_char_len": char_len,
        f"{prefix}_word_count": word_counts,
        f"{prefix}_line_count": line_counts,
        f"{prefix}_avg_word_len": avg_word_len,
        f"{prefix}_avg_words_per_line": avg_words_per_line,
        f"{prefix}_avg_sentence_len": avg_sentence_len,
        f"{prefix}_hedge_count": hedge_count,
        f"{prefix}_hedge_density": hedge_density,
        f"{prefix}_has_code_like": has_code_like,
        f"{prefix}_question_density": question_density,
        f"{prefix}_exclaim_density": exclaim_density,
        f"{prefix}_digit_density": digit_density,
        f"{prefix}_uniq_word_ratio": uniq_word_ratio,
        f"{prefix}_is_missing": (s.str.strip() == "").astype(int),
    })

def parse_concepts_to_list(val: str) -> List[str]:
    """Parse a 'concepts' field that may be comma/semicolon/JSON list."""
    raw = safe_str(val).strip()
    if not raw:
        return []

    # try json
    if (raw.startswith("[") and raw.endswith("]")) or (raw.startswith("{") and raw.endswith("}")):
        try:
            obj = json.loads(raw)
            if isinstance(obj, list):
                items = [safe_str(x) for x in obj]
            elif isinstance(obj, dict):
                items = [safe_str(k) for k in obj.keys()]
            else:
                items = [raw]
            items = [re.sub(r"\s+", " ", it.strip().lower()) for it in items if it.strip()]
            return items
        except Exception:
            pass

    parts = re.split(r"[;,|\n]+", raw)
    parts = [re.sub(r"\s+", " ", p.strip().lower()) for p in parts]
    parts = [p for p in parts if p]
    return parts

def multi_hot_from_concepts(df: pd.DataFrame, col: str, prefix: str, min_freq: int = 30) -> pd.DataFrame:
    """Create multi-hot concept flags, keeping only concepts that appear at least min_freq times."""
    concepts_lists = df[col].apply(parse_concepts_to_list) if col in df.columns else pd.Series([[]] * len(df))

    vocab_counts: Dict[str, int] = {}
    for lst in concepts_lists:
        for c in set(lst):
            vocab_counts[c] = vocab_counts.get(c, 0) + 1

    keep_concepts = {c for c, cnt in vocab_counts.items() if cnt >= min_freq}
    
    print(f"  Concept encoding: {len(keep_concepts)} concepts kept (min_freq={min_freq})")
    if len(keep_concepts) > 0:
        print(f"    Top concepts: {sorted(keep_concepts)[:10]}")

    out = pd.DataFrame(index=df.index)
    out[f"{prefix}_num_concepts"] = concepts_lists.apply(len)
    out[f"{prefix}_is_missing"] = concepts_lists.apply(lambda x: int(len(x) == 0))

    for c in keep_concepts:
        clean_name = re.sub(r'[^a-z0-9_]+', '_', c)[:60]
        out[f"{prefix}__{clean_name}"] = concepts_lists.apply(lambda lst: int(c in set(lst)))

    return out

def one_hot_topk_categories(df: pd.DataFrame, col: str, prefix: str, min_freq: int = 20) -> pd.DataFrame:
    """One-hot categories but keep rare categories grouped as 'other'."""
    if col not in df.columns:
        return pd.DataFrame(index=df.index)

    s = df[col].fillna("").astype(str).str.strip()
    is_missing = (s == "").astype(int)

    freq = s.value_counts()
    keep = set(freq[freq >= min_freq].index.tolist())
    
    print(f"Category encoding for '{col}': {len(keep)} categories kept (min_freq={min_freq})")
    print(f"  Categories: {sorted(keep)}")
    
    s_norm = s.apply(lambda x: x if x in keep and x != "" else ("__other__" if x != "" else "__missing__"))

    oh = pd.get_dummies(s_norm, prefix=prefix)
    oh[f"{prefix}_is_missing"] = is_missing
    return oh

def add_missingness_indicators(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    """Add _is_missing indicator columns for selected features."""
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[f"{c}__is_missing"] = out[c].isna().astype(int)
    return out

def load_bee_results(jsonl_path: Path) -> pd.DataFrame:
    """Load bee_results.jsonl and extract the 9 stats features."""
    bee_records = []
    
    if not jsonl_path.exists():
        print(f"  Warning: {jsonl_path} not found, skipping bee_results merge")
        return pd.DataFrame()
    
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                bug_id = record.get('bug_id')
                stats = record.get('stats', {})
                
                if bug_id and stats:
                    bee_records.append({
                        'id': bug_id,
                        'has_OB': stats.get('has_OB'),
                        'has_EB': stats.get('has_EB'),
                        'has_S2R': stats.get('has_S2R'),
                        'missing_OB': stats.get('missing_OB'),
                        'missing_EB': stats.get('missing_EB'),
                        'missing_S2R': stats.get('missing_S2R'),
                        'num_OB': stats.get('num_OB'),
                        'num_EB': stats.get('num_EB'),
                        'num_S2R': stats.get('num_S2R'),
                    })
            except json.JSONDecodeError as e:
                print(f"  Warning: Failed to parse line in bee_results: {e}")
                continue
    
    if bee_records:
        bee_df = pd.DataFrame(bee_records)
        print(f"  Loaded {len(bee_df)} records from bee_results.jsonl")
        return bee_df
    else:
        return pd.DataFrame()


# ============================================================================
# NEW: LLM FEATURE ENGINEERING
# ============================================================================

def create_llm_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create interaction terms between LLM ratings.
    These capture relationships between different quality dimensions.
    """
    print("\nCreating LLM interaction features...")
    
    out = pd.DataFrame(index=df.index)
    
    # Check which LLM rating columns are available
    required_cols = ['actionability', 'clarity', 'specificity', 'technical_depth',
                     'expected_observed_alignment', 'causal_reasoning_quality']
    
    available = [c for c in required_cols if c in df.columns]
    
    if len(available) < 3:
        print(f"  Warning: Not enough LLM rating columns found. Skipping interaction features.")
        return out
    
    print(f"  Found {len(available)} LLM rating columns")
    
    # Clarity-Specificity gap (clear but not specific = vague)
    if 'clarity' in df.columns and 'specificity' in df.columns:
        out['clarity_specificity_gap'] = df['clarity'] - df['specificity']
        out['clarity_specificity_ratio'] = df['clarity'] / (df['specificity'] + 0.1)  # Avoid div by 0
    
    # Actionability-Technical depth interaction
    if 'actionability' in df.columns and 'technical_depth' in df.columns:
        out['actionability_depth_product'] = df['actionability'] * df['technical_depth']
        out['actionability_depth_ratio'] = df['actionability'] / (df['technical_depth'] + 0.1)
    
    # Quality composite score (average of actionability, clarity, specificity)
    quality_cols = [c for c in ['actionability', 'clarity', 'specificity'] if c in df.columns]
    if len(quality_cols) >= 2:
        out['quality_composite'] = df[quality_cols].mean(axis=1)
        out['quality_std'] = df[quality_cols].std(axis=1)
    
    # Technical completeness (depth + alignment)
    if 'technical_depth' in df.columns and 'expected_observed_alignment' in df.columns:
        out['technical_completeness'] = (df['technical_depth'] + df['expected_observed_alignment']) / 2
    
    # Reasoning quality composite
    if 'causal_reasoning_quality' in df.columns:
        reasoning_score = df['causal_reasoning_quality'].copy()
        if 'hidden_s2r_present' in df.columns:
            # Bonus points for hidden S2R
            reasoning_score = reasoning_score + df['hidden_s2r_present'].astype(float) * 2
        out['reasoning_composite'] = reasoning_score / 7.0  # Normalize to 0-1
    
    # Contradiction penalty
    if 'contradiction_present' in df.columns and 'clarity' in df.columns:
        # Contradiction reduces effective clarity
        out['clarity_minus_contradiction'] = df['clarity'] - df['contradiction_present'].astype(float) * 2
    
    print(f"  Created {len(out.columns)} interaction features")
    return out


def encode_ambiguity_types(df: pd.DataFrame, col: str = 'ambiguity_types') -> pd.DataFrame:
    """
    Convert ambiguity types list into structured binary features.
    """
    print("\nEncoding ambiguity types...")
    
    out = pd.DataFrame(index=df.index)
    
    if col not in df.columns:
        print(f"  Warning: Column '{col}' not found. Skipping ambiguity encoding.")
        return out
    
    # Define ambiguity categories
    ambiguity_categories = {
        'reproduction_ambiguity': ['missing steps', 'unclear reproduction', 'ambiguous steps', 'unclear steps'],
        'input_ambiguity': ['vague inputs', 'missing inputs', 'unclear parameters', 'unclear input'],
        'error_ambiguity': ['unclear error messages', 'missing error details', 'vague error'],
        'context_ambiguity': ['missing context', 'missing environment info', 'unclear setup', 'missing environment'],
        'behavior_ambiguity': ['unclear expected behavior', 'vague description', 'unclear behavior']
    }
    
    # Parse ambiguity_types column (could be JSON list or comma-separated)
    def parse_ambiguity_list(val):
        if pd.isna(val):
            return []
        
        val_str = str(val).strip()
        if not val_str or val_str == '[]':
            return []
        
        # Try JSON
        if val_str.startswith('['):
            try:
                return json.loads(val_str)
            except:
                pass
        
        # Try comma-separated
        return [item.strip().lower() for item in val_str.split(',') if item.strip()]
    
    ambiguity_lists = df[col].apply(parse_ambiguity_list)
    
    # Create binary features for each category
    for category, keywords in ambiguity_categories.items():
        out[f'has_{category}'] = ambiguity_lists.apply(
            lambda lst: int(any(kw in item.lower() for item in lst for kw in keywords))
        )
    
    # Count total ambiguity types
    out['ambiguity_type_count'] = ambiguity_lists.apply(len)
    
    # Ambiguity coverage (how many categories are hit)
    out['ambiguity_category_coverage'] = out[[f'has_{cat}' for cat in ambiguity_categories.keys()]].sum(axis=1)
    
    print(f"  Created {len(out.columns)} ambiguity features")
    print(f"  Ambiguity category distribution:")
    for category in ambiguity_categories.keys():
        count = out[f'has_{category}'].sum()
        pct = (count / len(df)) * 100
        print(f"    {category}: {count} ({pct:.1f}%)")
    
    return out


def create_concept_network_features(df: pd.DataFrame, col: str = 'likely_impacted_code_concepts') -> pd.DataFrame:
    """
    Analyze concept co-occurrence and cross-layer patterns.
    """
    print("\nCreating concept network features...")
    
    out = pd.DataFrame(index=df.index)
    
    if col not in df.columns:
        print(f"  Warning: Column '{col}' not found. Skipping concept network features.")
        return out
    
    # Define code layers
    layers = {
        'ui': ['ui', 'gui', 'rendering', 'display', 'view', 'frontend', 'button', 'screen'],
        'business_logic': ['validation', 'calculation', 'processing', 'algorithm', 'logic', 'rule'],
        'data': ['database', 'query', 'storage', 'persistence', 'sql', 'data', 'cache'],
        'network': ['http', 'api', 'request', 'connection', 'rest', 'socket', 'network'],
        'infrastructure': ['configuration', 'deployment', 'logging', 'security', 'authentication']
    }
    
    # Parse concepts
    concepts_lists = df[col].apply(parse_concepts_to_list)
    
    def analyze_concepts(concept_list):
        if not concept_list:
            return {
                'concept_diversity': 0,
                'has_cross_layer_concepts': 0,
                'concept_breadth': 0,
                'ui_concepts': 0,
                'logic_concepts': 0,
                'data_concepts': 0,
                'network_concepts': 0,
                'infra_concepts': 0
            }
        
        # Identify which layers are touched
        concept_layers = set()
        layer_counts = {layer: 0 for layer in layers.keys()}
        
        for concept in concept_list:
            concept_lower = concept.lower()
            for layer, keywords in layers.items():
                if any(kw in concept_lower for kw in keywords):
                    concept_layers.add(layer)
                    layer_counts[layer] += 1
        
        return {
            'concept_diversity': len(concept_layers),
            'has_cross_layer_concepts': int(len(concept_layers) > 1),
            'concept_breadth': len(set(concept_list)),
            'ui_concepts': layer_counts['ui'],
            'logic_concepts': layer_counts['business_logic'],
            'data_concepts': layer_counts['data'],
            'network_concepts': layer_counts['network'],
            'infra_concepts': layer_counts['infrastructure']
        }
    
    # Apply analysis
    concept_features = concepts_lists.apply(analyze_concepts).apply(pd.Series)
    
    for col_name in concept_features.columns:
        out[f'concept_network_{col_name}'] = concept_features[col_name]
    
    print(f"  Created {len(out.columns)} concept network features")
    print(f"  Cross-layer concepts: {out['concept_network_has_cross_layer_concepts'].sum()} "
          f"({out['concept_network_has_cross_layer_concepts'].mean()*100:.1f}%)")
    
    return out


def create_semantic_consistency_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Check consistency between syntactic features and LLM assessments.
    This helps identify when LLM ratings match/mismatch actual content.
    """
    print("\nCreating semantic consistency features...")
    
    out = pd.DataFrame(index=df.index)
    
    # Technical depth consistency
    if 'technical_depth' in df.columns:
        # Check if technical depth matches presence of technical indicators
        technical_indicators = 0
        
        if 'has_code' in df.columns:
            technical_indicators += df['has_code'].astype(float)
        if 'has_stacktrace' in df.columns:
            technical_indicators += df['has_stacktrace'].astype(float)
        if 'has_version_info' in df.columns:
            technical_indicators += df['has_version_info'].astype(float)
        
        if technical_indicators is not 0:  # At least one indicator present
            # Normalize both to 0-1 scale
            indicators_norm = technical_indicators / 3.0
            depth_norm = df['technical_depth'] / 5.0
            
            # Consistency: 1 - absolute difference
            out['technical_depth_consistency'] = 1.0 - np.abs(indicators_norm - depth_norm)
    
    # Causal reasoning consistency
    if 'causal_reasoning_quality' in df.columns and 'num_causal_markers' in df.columns:
        # Does causal rating match actual causal markers?
        causal_markers_norm = np.minimum(df['num_causal_markers'] / 5.0, 1.0)
        causal_rating_norm = df['causal_reasoning_quality'] / 5.0
        
        out['causal_reasoning_consistency'] = 1.0 - np.abs(causal_markers_norm - causal_rating_norm)
    
    # Overall consistency composite
    consistency_cols = [c for c in out.columns if 'consistency' in c]
    if len(consistency_cols) > 0:
        out['consistency_composite'] = out[consistency_cols].mean(axis=1)
    
    if len(out.columns) > 0:
        print(f"  Created {len(out.columns)} consistency features")
    else:
        print(f"  No consistency features created (missing required columns)")
    
    return out


# -----------------------------
# MAIN EXECUTION
# -----------------------------
print("=" * 60)
print("ENHANCED MERGE AND PREPROCESSING")
print("=" * 60)
print(f"Using enhanced features: {FEATURES_FILE}")
print("=" * 60)

# -----------------------------
# 1. Load
# -----------------------------
print("\nLoading data...")
features_df = pd.read_csv(FEATURES_FILE)
ratings_df  = pd.read_csv(RATINGS_FILE, sep=';')
categ_df    = pd.read_csv(CATEG_FILE)
fine_grained_categ_df = pd.read_csv(FINE_GRAINED_CATEG_FILE) if FINE_GRAINED_CATEG_FILE.exists() else pd.DataFrame()
perf_df     = pd.read_csv(PERF_FILE)

print(f"  Features: {features_df.shape}")
print(f"  Ratings:  {ratings_df.shape}")
print(f"  Categories: {categ_df.shape}")
print(f"  Fine-grained categories: {fine_grained_categ_df.shape if not fine_grained_categ_df.empty else 'Not found'}")
print(f"  Performance: {perf_df.shape}")

# Check for new enhanced features
enhanced_cols = [c for c in features_df.columns if any(x in c for x in [
    'semantic_', 'embedding_', 'exc_cat_', 'has_exc_', 'exception_user'
])]
if enhanced_cols:
    print(f"\n  ✓ Enhanced features detected: {len(enhanced_cols)} new columns")
    print(f"    Sample: {enhanced_cols[:5]}")
else:
    print(f"\n  ⚠ Warning: No enhanced features detected. Using basic features.")

# -----------------------------
# 2. Parse IDs
# -----------------------------
print("\nParsing bug IDs...")
features_df = split_id(features_df)
ratings_df  = split_id(ratings_df)
categ_df    = split_id(categ_df)
if not fine_grained_categ_df.empty:
    fine_grained_categ_df = split_id(fine_grained_categ_df)

# -----------------------------
# 3. Compute per-tool MRR from rank
# -----------------------------
print("\nComputing MRR from ranks...")
perf_df = perf_df.copy()

# FIX: Some tools (e.g. BRaIn) store bug_id as a composite "Project-N" string
# instead of just the numeric ID. This MUST run BEFORE pd.to_numeric() below,
# otherwise "Chart-1" becomes NaN and all BRaIn rows vanish silently.
_composite_mask = perf_df["bug_id"].astype(str).str.contains("-", na=False)
if _composite_mask.any():
    _split = perf_df.loc[_composite_mask, "bug_id"].str.split("-", n=1, expand=True)
    perf_df.loc[_composite_mask, "project"] = _split[0]
    perf_df.loc[_composite_mask, "bug_id"]  = _split[1]
    print(f"  Fixed {_composite_mask.sum()} rows with composite bug_id (e.g. 'Chart-1' → project='Chart', bug_id='1')")

# Ensure consistent data types for merge keys
if "bug_id" in perf_df.columns:
    perf_df["bug_id"] = pd.to_numeric(perf_df["bug_id"], errors="coerce").astype("Int64")
if "project" in perf_df.columns:
    perf_df["project"] = perf_df["project"].astype(str)

perf_df["mrr"] = np.where(perf_df["rank"].notna(), 1.0 / perf_df["rank"], 0.0)

perf_subset = perf_df[["project", "bug_id", "tool", "rank", "mrr", "top@1", "top@5"]].copy()

# -----------------------------
# 4. Pivot performance wide
# -----------------------------
print("Pivoting performance metrics...")
perf_wide = perf_subset.pivot_table(
    index=["project", "bug_id"],
    columns="tool",
    values=["rank", "mrr", "top@1", "top@5"],
    aggfunc="first"
)
perf_wide.columns = [f"{metric}_{tool}" for metric, tool in perf_wide.columns]
perf_wide = perf_wide.reset_index()

# Ensure consistent data types in perf_wide after pivot
if "bug_id" in perf_wide.columns:
    perf_wide["bug_id"] = pd.to_numeric(perf_wide["bug_id"], errors="coerce").astype("Int64")
if "project" in perf_wide.columns:
    perf_wide["project"] = perf_wide["project"].astype(str)

# Ensure consistent data types in features_df (should already be done by split_id, but double-check)
if "bug_id" in features_df.columns:
    features_df["bug_id"] = pd.to_numeric(features_df["bug_id"], errors="coerce").astype("Int64")
if "project" in features_df.columns:
    features_df["project"] = features_df["project"].astype(str)

# -----------------------------
# 5. Merge all sources
# -----------------------------
print("\nMerging all data sources...")
merged = features_df.merge(perf_wide, on=["project", "bug_id"], how="left")

drop_from_ratings = ["description_length"]
ratings_for_merge = ratings_df.drop(columns=drop_from_ratings, errors="ignore")

# Ensure consistent data types in ratings_for_merge
if "bug_id" in ratings_for_merge.columns:
    ratings_for_merge["bug_id"] = pd.to_numeric(ratings_for_merge["bug_id"], errors="coerce").astype("Int64")
if "project" in ratings_for_merge.columns:
    ratings_for_merge["project"] = ratings_for_merge["project"].astype(str)

merged = merged.merge(
    ratings_for_merge,
    on=["project", "bug_id"],
    how="left",
    suffixes=("", "_ratings")
)

categ_for_merge = categ_df.copy()
# Ensure consistent data types in categ_for_merge
if "bug_id" in categ_for_merge.columns:
    categ_for_merge["bug_id"] = pd.to_numeric(categ_for_merge["bug_id"], errors="coerce").astype("Int64")
if "project" in categ_for_merge.columns:
    categ_for_merge["project"] = categ_for_merge["project"].astype(str)

merged = merged.merge(
    categ_for_merge,
    on=["project", "bug_id"],
    how="left",
    suffixes=("", "_categ")
)

if not fine_grained_categ_df.empty:
    fine_grained_for_merge = fine_grained_categ_df.copy()
    # Ensure consistent data types in fine_grained_for_merge
    if "bug_id" in fine_grained_for_merge.columns:
        fine_grained_for_merge["bug_id"] = pd.to_numeric(fine_grained_for_merge["bug_id"], errors="coerce").astype("Int64")
    if "project" in fine_grained_for_merge.columns:
        fine_grained_for_merge["project"] = fine_grained_for_merge["project"].astype(str)
    
    merged = merged.merge(
        fine_grained_for_merge,
        on=["project", "bug_id"],
        how="left",
        suffixes=("", "_fine_grained")
    )
    # Encode fine-grained categories immediately before duplicate cleanup drops category_fine_grained
    if "category_fine_grained" in merged.columns:
        print("\n6.6.1 Encoding fine-grained categories...")
        fine_grained_oh = one_hot_topk_categories(merged, "category_fine_grained", prefix="fg_cat", min_freq=10)
        merged = pd.concat([merged, fine_grained_oh], axis=1)
        merged.drop(columns=["category_fine_grained"], inplace=True)

# Load and merge bee_results
bee_df = load_bee_results(BEE_RESULTS_FILE)
if not bee_df.empty:
    # Ensure consistent data types in bee_df (it uses 'id' column, not project/bug_id)
    # The merge is on 'id', so we don't need to normalize project/bug_id for bee_df
    merged = merged.merge(bee_df, on="id", how="left", suffixes=("", "_bee"))

print(f"\nMerged shape: {merged.shape}")

# Handle duplicate columns from merging
print("\nHandling duplicate columns...")
cols_to_drop = []
for col in merged.columns:
    if col.endswith("_ratings") or col.endswith("_categ") or col.endswith("_fine_grained") or col.endswith("_bee"):
        base_col = col.rsplit("_", 1)[0]
        if base_col in merged.columns and base_col not in ["project", "bug_id", "id"]:
            cols_to_drop.append(col)

if cols_to_drop:
    print(f"  Dropping {len(cols_to_drop)} redundant columns")
    merged = merged.drop(columns=cols_to_drop)

# Save full dataset
print(f"\nFull merged shape: {merged.shape}")
OUT_FULL.parent.mkdir(parents=True, exist_ok=True)
merged.to_csv(OUT_FULL, index=False)
print(f"Saved full dataset: {OUT_FULL}")

# -----------------------------
# 6. Rich preprocessing with LLM feature engineering
# -----------------------------
print("\n" + "=" * 60)
print("Starting rich preprocessing with LLM feature engineering...")
print("=" * 60)
df = merged.copy()

# 6.1 Process performance metrics
print("\n6.1 Processing performance metrics...")
perf_cols = [c for c in df.columns if c.startswith("mrr_") or c.startswith("top@")]
df = add_missingness_indicators(df, perf_cols)
df[perf_cols] = df[perf_cols].fillna(0.0)
print(f"  Performance columns: {len(perf_cols)}")

# 6.2 NEW: Create LLM interaction features
llm_interaction_feats = create_llm_interaction_features(df)
if not llm_interaction_feats.empty:
    df = pd.concat([df, llm_interaction_feats], axis=1)

# 6.3 NEW: Encode ambiguity types
ambiguity_feats = encode_ambiguity_types(df, col='ambiguity_types')
if not ambiguity_feats.empty:
    df = pd.concat([df, ambiguity_feats], axis=1)

# 6.4 NEW: Create concept network features
concept_network_feats = create_concept_network_features(df, col='likely_impacted_code_concepts')
if not concept_network_feats.empty:
    df = pd.concat([df, concept_network_feats], axis=1)

# 6.5 NEW: Create semantic consistency features
consistency_feats = create_semantic_consistency_features(df)
if not consistency_feats.empty:
    df = pd.concat([df, consistency_feats], axis=1)

# 6.6 Encode categories
print("\n6.6 Encoding categories...")
for cat_col in ["category", "category_categ", "bug_category", "bug_category_categ"]:
    if cat_col in df.columns:
        cat_oh = one_hot_topk_categories(df, cat_col, prefix="cat", min_freq=20)
        df = pd.concat([df, cat_oh], axis=1)
        df.drop(columns=[cat_col], inplace=True)
        break



# 6.7 Extract text statistics
print("\n6.7 Extracting text statistics...")
text_candidates = [
    "title", "description", "reasoning", "reasoning_ratings",
    "fine_grained_reasoning", "fine_grained_title",
    "likely_impacted_code_concepts", "root_cause_guess"
]
for col in text_candidates:
    if col in df.columns:
        print(f"  Processing text column: {col}")
        stats = text_stats(df[col], prefix=f"txt_{col}")
        df = pd.concat([df, stats], axis=1)

# 6.8 Convert concepts into multi-hot flags
print("\n6.8 Creating concept multi-hot encoding...")
if "likely_impacted_code_concepts" in df.columns:
    concept_feats = multi_hot_from_concepts(
        df,
        col="likely_impacted_code_concepts",
        prefix="concept",
        min_freq=30
    )
    df = pd.concat([df, concept_feats], axis=1)

# 6.9 Drop raw text columns from preprocessed dataset
print("\n6.9 Dropping raw text columns from preprocessed dataset...")
drop_raw_text_cols = [
    "title", "description", "reasoning", "reasoning_ratings", 
    "fine_grained_reasoning", "fine_grained_title",
    "likely_impacted_code_concepts", "ambiguity_types",
    "root_cause_guess"
]
df.drop(columns=drop_raw_text_cols, inplace=True, errors="ignore")

# 6.10 Reorder columns
print("\n6.10 Reordering columns...")
key_cols = [c for c in ["project", "bug_id", "id"] if c in df.columns]
other_cols = [c for c in df.columns if c not in key_cols]
df = df[key_cols + other_cols]

# 6.10b NEW: Feature redundancy elimination
print("\n6.10b Removing redundant features...")

import re

def collapse_statistical_feature_families(cols):
    """
    Collapse expansion families such as:
    feature__orig
    feature__orig_mean
    feature__orig_std
    feature__orig_min
    feature__orig_median
    feature__orig_max

    Keep only ONE representative.
    """

    base_map = {}

    for col in cols:

        base = re.sub(
            r"__orig(_(mean|std|min|median|max))?$",
            "",
            col
        )

        if base not in base_map:
            base_map[base] = []

        base_map[base].append(col)

    selected = []
    removed = []

    for base, family in base_map.items():

        if base in family:
            selected.append(base)
            removed.extend([f for f in family if f != base])

        else:
            median = [f for f in family if "median" in f]

            if median:
                selected.append(median[0])
                removed.extend([f for f in family if f != median[0]])
            else:
                selected.append(family[0])
                removed.extend(family[1:])

    return selected, removed


def remove_length_metric_duplicates(cols):
    """
    Collapse redundant text length families.

    Examples collapsed:

    txt_description_char_len
    txt_description_word_count
    txt_description_line_count
    txt_description_avg_words_per_line

    Rule:
    keep word_count + line_count
    drop char_len and avg_words_per_line
    """

    keep = []
    removed = []

    groups = {}

    for col in cols:

        if not col.startswith("txt_"):
            keep.append(col)
            continue

        base = re.sub(
            r"_(char_len|chars|length|word_count|line_count|avg_words_per_line)$",
            "",
            col
        )

        groups.setdefault(base, []).append(col)

    for base, feats in groups.items():

        chosen = []

        for f in feats:
            if f.endswith("word_count"):
                chosen.append(f)

        for f in feats:
            if f.endswith("line_count"):
                chosen.append(f)

        # fallback if neither exists
        if not chosen:
            chosen = [feats[0]]

        keep.extend(chosen)

        for f in feats:
            if f not in chosen:
                removed.append(f)

    return keep, removed

def remove_redundant_features(df: pd.DataFrame, key_cols: List[str]) -> pd.DataFrame:
    """
    Improved feature redundancy elimination.

    Steps:
    1. Remove constant / near-constant features
    2. Remove perfect duplicates (Spearman correlation ≈ 1.0)
    3. Remove highly correlated clusters (Spearman rho > 0.95)
    4. Select representative feature using variance + missingness penalty
    """

    from scipy.cluster.hierarchy import linkage, fcluster
    from scipy.spatial.distance import squareform

    performance_cols = [c for c in df.columns if c.startswith(("top@", "mrr_", "map_", "rank_"))]
    exclude_cols = set(key_cols + performance_cols)

    numeric_cols = [
    c for c in df.select_dtypes(include=[np.number]).columns
    if c not in exclude_cols
    ]

    print(f"\nStarting redundancy elimination with {len(numeric_cols)} numeric features")

    removed_features = {
        "statistical_expansions": [],
        "length_duplicates": [],
        "constant": [],
        "duplicates": [],
        "high_correlation": []
    }

# ---------------------------------------------------
# NEW STEP 0: Collapse statistical feature families
# ---------------------------------------------------

    numeric_cols, removed_exp = collapse_statistical_feature_families(numeric_cols)
    removed_features["statistical_expansions"].extend(removed_exp)

    print(f"Collapsed {len(removed_exp)} statistical expansion features")
    print(f"Remaining features: {len(numeric_cols)}")

    # ---------------------------------------------------
    # NEW STEP 0.5: Remove obvious length duplicates
    # ---------------------------------------------------

    numeric_cols, removed_len = remove_length_metric_duplicates(numeric_cols)
    removed_features["length_duplicates"].extend(removed_len)

    print(f"Removed {len(removed_len)} length duplicate features")
    print(f"Remaining features: {len(numeric_cols)}")

    # ---------------------------------------------------
    # STEP 1: Remove constant / near-constant features
    # ---------------------------------------------------

    print("Step 1: Removing constant / near-constant features")

    variance_floor = 0.001

    keep_cols = []

    for col in numeric_cols:

        n_unique = df[col].nunique()
        std_val = df[col].std()
        var_val = df[col].var()

        if n_unique < 5 or std_val < 0.01 or var_val < variance_floor:
            removed_features["constant"].append(col)
        else:
            keep_cols.append(col)

    numeric_cols = keep_cols

    print(f"  Removed {len(removed_features['constant'])} low-variance features")
    print(f"  Remaining features: {len(numeric_cols)}")

    # ---------------------------------------------------
    # STEP 2: Remove duplicate features
    # ---------------------------------------------------

    print("Step 2: Removing duplicate features (Spearman ≈ 1.0)")

    corr_matrix = df[numeric_cols].corr(method="spearman").abs()

    to_remove = set()

    for i in range(len(corr_matrix)):
        for j in range(i + 1, len(corr_matrix)):
            if corr_matrix.iloc[i, j] >= 0.999:
                f1 = corr_matrix.index[i]
                f2 = corr_matrix.columns[j]

                if f1 not in to_remove:
                    to_remove.add(f2)
                    removed_features["duplicates"].append(f2)

    numeric_cols = [c for c in numeric_cols if c not in to_remove]

    print(f"  Removed {len(removed_features['duplicates'])} duplicates")
    print(f"  Remaining features: {len(numeric_cols)}")

    # ---------------------------------------------------
    # STEP 3: Cluster highly correlated features
    # ---------------------------------------------------

    print("Step 3: Clustering highly correlated features (ρ > 0.95)")

    spearman_corr = df[numeric_cols].corr(method="spearman").abs()

    distance_matrix = 1 - spearman_corr

    condensed_dist = squareform(distance_matrix, checks=False)

    linkage_matrix = linkage(condensed_dist, method="average")

    clusters = fcluster(linkage_matrix, t=0.05, criterion="distance")

    cluster_df = pd.DataFrame({
        "feature": numeric_cols,
        "cluster": clusters,
        "variance": df[numeric_cols].var(),
        "missingness": df[numeric_cols].isna().mean()
    })

    cluster_sizes = cluster_df["cluster"].value_counts()

    print("\nCluster size distribution:")

    for size, count in cluster_sizes.value_counts().sort_index().items():
        print(f"  size {size}: {count} clusters")

    selected_features = []

    for cluster_id in cluster_df["cluster"].unique():

        cluster_feats = cluster_df[cluster_df["cluster"] == cluster_id]

        if len(cluster_feats) == 1:

            selected_features.append(cluster_feats["feature"].iloc[0])

        else:

            # representative score: variance × (1 - missingness)

            cluster_feats["score"] = (
                cluster_feats["variance"] *
                (1 - cluster_feats["missingness"])
            )

            best = cluster_feats.sort_values("score", ascending=False).iloc[0]["feature"]

            selected_features.append(best)

            removed = cluster_feats[cluster_feats["feature"] != best]["feature"].tolist()

            removed_features["high_correlation"].extend(removed)

    numeric_cols = selected_features

    print(f"\nRemoved {len(removed_features['high_correlation'])} highly correlated features")
    print(f"Remaining numeric features: {len(numeric_cols)}")

    # ---------------------------------------------------
    # Build final dataset
    # ---------------------------------------------------

    final_cols = key_cols + performance_cols + numeric_cols

    non_numeric_cols = [
        c for c in df.columns
        if c not in df.select_dtypes(include=[np.number]).columns
    ]

    for col in non_numeric_cols:
        if col not in final_cols:
            final_cols.append(col)

    # ---------------------------------------------------
    # Summary table
    # ---------------------------------------------------

    print("\nRedundancy elimination summary")

    summary = {
        "Initial features": len(df.select_dtypes(include=[np.number]).columns),
        "Statistical expansions removed": len(removed_features["statistical_expansions"]),
        "Length duplicates removed": len(removed_features["length_duplicates"]),
        "Constant removed": len(removed_features["constant"]),
        "Duplicates removed": len(removed_features["duplicates"]),
        "High correlation removed": len(removed_features["high_correlation"]),
        "Final numeric features": len(numeric_cols)
    }

    for k, v in summary.items():
        print(f"  {k}: {v}")

    # ---------------------------------------------------
    # Save removal log
    # ---------------------------------------------------

    removed_log_file = DATA_DIR / "removed_features_log.txt"

    with open(removed_log_file, "w") as f:

        f.write("FEATURE REDUNDANCY ELIMINATION LOG\n")
        f.write("=" * 60 + "\n\n")

        for key in removed_features:
            f.write(f"{key.upper()} ({len(removed_features[key])})\n")
            for feat in sorted(removed_features[key]):
                f.write(f"  - {feat}\n")
            f.write("\n")

    print(f"\nSaved removal log: {removed_log_file}")

    return df[final_cols]


# 6.11 Standardize numeric features
print("\n6.11 Standardizing numeric features...")
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

# Exclude from scaling
exclude_from_scaling = (
    key_cols + 
    [c for c in numeric_cols if c.startswith(("mrr_", "top@", "rank_", "cat_", "cat__", "fg_cat_", "fg_cat__", "concept__"))] +
    [c for c in numeric_cols if c.endswith("_is_missing") or c.endswith("__is_missing")] +
    [c for c in numeric_cols if "_has_" in c or c.startswith("has_")] +
    [c for c in numeric_cols if c in ["fine_grained_confidence", "embedding_cluster"]]
)

cols_to_scale = [c for c in numeric_cols if c not in exclude_from_scaling]

print(f"  Total numeric columns: {len(numeric_cols)}")
print(f"  Columns to scale: {len(cols_to_scale)}")
print(f"  Excluded from scaling: {len(exclude_from_scaling)}")

if ENABLE_STANDARDIZATION and cols_to_scale:
    print(f"\n  Standardization: ENABLED")
    
    # Store original values and statistics before standardization
    original_stats = {}
    for col in cols_to_scale:
        # Store original values as a new column
        df[f"{col}__orig"] = df[col].copy()
        
        # Calculate and store statistics
        original_stats[col] = {
            'mean': float(df[col].mean()),
            'std': float(df[col].std()),
            'min': float(df[col].min()),
            'median': float(df[col].median()),
            'max': float(df[col].max())
        }
        
        # Add statistics as constant columns (same value for all rows)
        df[f"{col}__orig_mean"] = original_stats[col]['mean']
        df[f"{col}__orig_std"] = original_stats[col]['std']
        df[f"{col}__orig_min"] = original_stats[col]['min']
        df[f"{col}__orig_median"] = original_stats[col]['median']
        df[f"{col}__orig_max"] = original_stats[col]['max']
    
    # Now standardize the original columns
    scaler = StandardScaler()
    df[cols_to_scale] = scaler.fit_transform(df[cols_to_scale])
    
    SCALER_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(SCALER_FILE, "wb") as f:
        pickle.dump({'scaler': scaler, 'columns': cols_to_scale, 'original_stats': original_stats}, f)
    print(f"  Saved scaler to: {SCALER_FILE}")
    print(f"  Added {len(cols_to_scale)} original value columns and {len(cols_to_scale) * 5} statistics columns")
elif not ENABLE_STANDARDIZATION:
    print(f"\n  Standardization: DISABLED")
else:
    print(f"\n  Standardization: SKIPPED (no columns to scale)")

df = remove_redundant_features(df, key_cols)

# 6.12 Final NaN handling
print("\n6.12 Final NaN handling...")
derived_feature_patterns = ['_density', '_avg_', '_ratio', '_gap', '_product', '_composite']
for col in df.columns:
    if any(pattern in col for pattern in derived_feature_patterns):
        if df[col].isna().any():
            median_val = df[col].median()
            if pd.isna(median_val):
                median_val = 0.0
            df[col] = df[col].fillna(median_val)

remaining_nan_cols = df.select_dtypes(include=[np.number]).columns[df.select_dtypes(include=[np.number]).isna().any()].tolist()
if remaining_nan_cols:
    print(f"  Filling remaining NaNs with 0 in {len(remaining_nan_cols)} columns")
    df[remaining_nan_cols] = df[remaining_nan_cols].fillna(0.0)

# 6.13 Separate features from performance metrics
print("\n6.13 Separating features from performance metrics...")
PERFORMANCE_COLS = [
    c for c in df.columns
    if c.startswith(("top@", "mrr_", "map_"))
]
X = df.drop(columns=PERFORMANCE_COLS, errors="ignore")
print(f"  Performance columns identified: {len(PERFORMANCE_COLS)}")
print(f"  Feature matrix X shape: {X.shape}")
if PERFORMANCE_COLS:
    print(f"  Performance columns: {PERFORMANCE_COLS[:5]}{'...' if len(PERFORMANCE_COLS) > 5 else ''}")

# Save preprocessed dataset
print(f"\nPreprocessed (rich) shape: {df.shape}")
OUT_PREP.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUT_PREP, index=False)
print(f"Saved preprocessed dataset: {OUT_PREP}")

# -----------------------------
# 7. Summary statistics
# -----------------------------
print("\n" + "=" * 60)
print("PREPROCESSING SUMMARY")
print("=" * 60)
print(f"Final dataset: {df.shape[0]} rows × {df.shape[1]} columns")
print(f"\nColumn breakdown:")
print(f"  Key columns: {len(key_cols)}")
print(f"  Performance metrics: {len([c for c in df.columns if c.startswith('mrr_') or c.startswith('top@')])}")
print(f"  Category features: {len([c for c in df.columns if c.startswith('cat_') or c.startswith('cat__')])}")
print(f"  Fine-grained category features: {len([c for c in df.columns if c.startswith('fg_cat_') or c.startswith('fg_cat__')])}")
print(f"  Text-derived features: {len([c for c in df.columns if c.startswith('txt_')])}")
print(f"  Concept features: {len([c for c in df.columns if c.startswith('concept')])}")
print(f"  Bee results features: {len([c for c in df.columns if c in ['has_OB', 'has_EB', 'has_S2R', 'missing_OB', 'missing_EB', 'missing_S2R', 'num_OB', 'num_EB', 'num_S2R']])}")

# NEW: Enhanced feature statistics
print(f"\n  Enhanced semantic features: {len([c for c in df.columns if 'semantic_' in c or 'embedding_' in c])}")
print(f"  Enhanced exception features: {len([c for c in df.columns if 'exc_cat_' in c or 'has_exc_' in c or 'exception_user' in c])}")
print(f"  LLM interaction features: {len([c for c in df.columns if any(x in c for x in ['_gap', '_product', '_composite', 'consistency'])])}")
print(f"  Ambiguity encoding features: {len([c for c in df.columns if 'ambiguity' in c])}")
print(f"  Concept network features: {len([c for c in df.columns if 'concept_network_' in c])}")

print("\nFiles created:")
print(f"  1. {OUT_FULL} - Full dataset with all original columns")
if ENABLE_STANDARDIZATION:
    print(f"  2. {OUT_PREP} - Preprocessed dataset with standardized features")
    print(f"  3. {SCALER_FILE} - StandardScaler for feature scaling")
else:
    print(f"  2. {OUT_PREP} - Preprocessed dataset with raw (unscaled) features")

print("\n" + "=" * 60)
print("Enhanced preprocessing complete!")
