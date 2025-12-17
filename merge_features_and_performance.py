import re
import json
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional
from sklearn.preprocessing import StandardScaler

DATA_DIR = Path(".")

FEATURES_FILE = DATA_DIR / "bug_features_v2.csv"
RATINGS_FILE  = DATA_DIR / "gemini_bug_ratings.csv"
CATEG_FILE    = DATA_DIR / "gemini_bug_categorization.csv"
PERF_FILE     = DATA_DIR / "tool_comparison_summary.csv"

OUT_FULL = DATA_DIR / "experimentA_full_dataset.csv"
OUT_PREP = DATA_DIR / "experimentA_preprocessed_rich.csv"
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
    """
    Extract lightweight numeric features from free text with normalization.
    Keeps signal for outlier mining without requiring embeddings.
    """
    s = series.fillna("").astype(str)

    # basic lengths
    char_len = s.str.len()
    word_counts = s.apply(lambda t: len(_word_re.findall(t)))
    line_counts = s.apply(lambda t: t.count("\n") + (1 if t else 0))

    # Normalized features (avoid division by zero)
    avg_word_len = char_len / word_counts.replace(0, np.nan)
    avg_words_per_line = word_counts / line_counts.replace(0, np.nan)

    # simple uncertainty / hedging markers - with normalization
    lower = s.str.lower()
    hedge_markers = [
        "maybe", "might", "likely", "possibly", "unclear", "unsure",
        "seems", "appears", "probably", "could", "cannot", "can't",
        "unknown", "approximately", "guess"
    ]
    hedge_count = sum(lower.str.count(re.escape(m)) for m in hedge_markers)
    hedge_density = hedge_count / word_counts.replace(0, np.nan)

    # structural cues - normalized by length
    has_code_like = lower.str.contains(
        r"\b(stack trace|exception|nullpointer|assert|traceback|line \d+)\b", 
        regex=True
    ).astype(int)
    question_density = s.str.count(r"\?") / word_counts.replace(0, np.nan)
    exclaim_density = s.str.count(r"!") / word_counts.replace(0, np.nan)
    digit_density = s.str.count(r"\d") / char_len.replace(0, np.nan)

    # diversity proxy: unique word ratio
    def uniq_ratio(t: str) -> float:
        toks = _word_re.findall(t.lower())
        if not toks:
            return np.nan  # Properly indicate missing instead of 0.0
        return len(set(toks)) / len(toks)

    uniq_word_ratio = s.apply(uniq_ratio)

    # Sentence complexity: average words per sentence
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
    """
    Parse a 'concepts' field that may be:
    - comma separated
    - semicolon separated
    - JSON list
    - free text
    Returns a clean list of concept tokens.
    """
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

    # split on common delimiters
    parts = re.split(r"[;,|\n]+", raw)
    parts = [re.sub(r"\s+", " ", p.strip().lower()) for p in parts]
    parts = [p for p in parts if p]
    return parts

def multi_hot_from_concepts(df: pd.DataFrame, col: str, prefix: str, min_freq: int = 30) -> pd.DataFrame:
    """
    Create multi-hot concept flags, keeping only concepts that appear at least min_freq times.
    Also keep a 'num_concepts' and 'concepts_is_missing' feature.
    Increased min_freq to reduce sparsity and improve generalization.
    """
    concepts_lists = df[col].apply(parse_concepts_to_list) if col in df.columns else pd.Series([[]] * len(df))

    # counts for vocab
    vocab_counts: Dict[str, int] = {}
    for lst in concepts_lists:
        for c in set(lst):
            vocab_counts[c] = vocab_counts.get(c, 0) + 1

    vocab = sorted([c for c, cnt in vocab_counts.items() if cnt >= min_freq])
    
    print(f"Concept vocabulary: {len(vocab)} concepts (min_freq={min_freq})")
    if len(vocab) > 0:
        print(f"  Most common: {sorted(vocab_counts.items(), key=lambda x: -x[1])[:5]}")
    
    out = pd.DataFrame(index=df.index)
    out[f"{prefix}_num_concepts"] = concepts_lists.apply(len)
    out[f"{prefix}_is_missing"] = concepts_lists.apply(lambda x: 1 if len(x) == 0 else 0)

    # multi-hot
    for c in vocab:
        clean_name = re.sub(r'[^a-z0-9_]+', '_', c)[:60]
        out[f"{prefix}__{clean_name}"] = concepts_lists.apply(lambda lst: int(c in set(lst)))

    return out

def one_hot_topk_categories(df: pd.DataFrame, col: str, prefix: str, min_freq: int = 20) -> pd.DataFrame:
    """
    One-hot categories but keep rare categories grouped as 'other' instead of dropping them.
    Also keep missingness. Increased min_freq to reduce noise from rare categories.
    """
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

# -----------------------------
# 1. Load
# -----------------------------
print("=" * 60)
print("Loading data...")
features_df = pd.read_csv(FEATURES_FILE)
ratings_df  = pd.read_csv(RATINGS_FILE)
categ_df    = pd.read_csv(CATEG_FILE)
perf_df     = pd.read_csv(PERF_FILE)

print(f"  Features: {features_df.shape}")
print(f"  Ratings:  {ratings_df.shape}")
print(f"  Categories: {categ_df.shape}")
print(f"  Performance: {perf_df.shape}")

# -----------------------------
# 2. Parse IDs
# -----------------------------
print("\nParsing bug IDs...")
features_df = split_id(features_df)
ratings_df  = split_id(ratings_df)
categ_df    = split_id(categ_df)

# -----------------------------
# 3. Compute per-tool MRR from rank
# -----------------------------
print("\nComputing MRR from ranks...")
perf_df = perf_df.copy()
perf_df["mrr"] = np.where(perf_df["rank"].notna(), 1.0 / perf_df["rank"], 0.0)

perf_subset = perf_df[["project", "bug_id", "tool", "mrr", "top@1", "top@5"]].copy()

# -----------------------------
# 4. Pivot performance wide
# -----------------------------
print("Pivoting performance metrics...")
perf_wide = perf_subset.pivot_table(
    index=["project", "bug_id"],
    columns="tool",
    values=["mrr", "top@1", "top@5"],
    aggfunc="first"
)
perf_wide.columns = [f"{metric}_{tool}" for metric, tool in perf_wide.columns]
perf_wide = perf_wide.reset_index()

# -----------------------------
# 5. Merge all sources
# -----------------------------
print("\nMerging all data sources...")
merged = features_df.merge(perf_wide, on=["project", "bug_id"], how="left")

# Keep raw text columns (do NOT drop them here)
drop_from_ratings = ["description_length"]  # keep title if present, it may be useful
ratings_for_merge = ratings_df.drop(columns=drop_from_ratings, errors="ignore")

merged = merged.merge(
    ratings_for_merge,
    on=["project", "bug_id"],
    how="left",
    suffixes=("", "_ratings")
)

categ_for_merge = categ_df.copy()
merged = merged.merge(
    categ_for_merge,
    on=["project", "bug_id"],
    how="left",
    suffixes=("", "_categ")
)

# Save full dataset
print(f"\nFull merged shape: {merged.shape}")
merged.to_csv(OUT_FULL, index=False)
print(f"Saved full dataset: {OUT_FULL}")

# -----------------------------
# 6. Rich preprocessing for outlier discovery
# -----------------------------
print("\n" + "=" * 60)
print("Starting rich preprocessing...")
df = merged.copy()

# 6.1 Treat missing performance as failure, but keep missingness flags too
print("\n6.1 Processing performance metrics...")
perf_cols = [c for c in df.columns if c.startswith("mrr_") or c.startswith("top@")]
df = add_missingness_indicators(df, perf_cols)
df[perf_cols] = df[perf_cols].fillna(0.0)
print(f"  Performance columns: {len(perf_cols)}")

# 6.2 Encode category in a way that preserves rare signals
print("\n6.2 Encoding categories...")
for cat_col in ["category", "category_categ", "bug_category", "bug_category_categ"]:
    if cat_col in df.columns:
        cat_oh = one_hot_topk_categories(df, cat_col, prefix="cat", min_freq=20)
        df = pd.concat([df, cat_oh], axis=1)
        # keep raw category too in OUT_FULL, but drop from modeling view
        df.drop(columns=[cat_col], inplace=True)
        break

# 6.3 Extract numeric features from text instead of dropping it
print("\n6.3 Extracting text statistics...")
text_candidates = [
    "title", "description", "reasoning", "reasoning_ratings",
    "likely_impacted_code_concepts"
]
for col in text_candidates:
    if col in df.columns:
        print(f"  Processing text column: {col}")
        stats = text_stats(df[col], prefix=f"txt_{col}")
        df = pd.concat([df, stats], axis=1)

# 6.4 Convert "likely_impacted_code_concepts" into multi-hot flags
print("\n6.4 Creating concept multi-hot encoding...")
if "likely_impacted_code_concepts" in df.columns:
    concept_feats = multi_hot_from_concepts(
        df,
        col="likely_impacted_code_concepts",
        prefix="concept",
        min_freq=30  # Increased from 5 to reduce sparsity
    )
    df = pd.concat([df, concept_feats], axis=1)

# 6.5 Keep original text columns in OUT_FULL, but drop them from OUT_PREP modeling view
print("\n6.5 Dropping raw text columns from preprocessed dataset...")
drop_raw_text_cols = [
    "title", "description", "reasoning", "reasoning_ratings", "likely_impacted_code_concepts"
]
df.drop(columns=drop_raw_text_cols, inplace=True, errors="ignore")

# 6.6 Make sure key columns exist and are first
print("\n6.6 Reordering columns...")
key_cols = [c for c in ["project", "bug_id", "id"] if c in df.columns]
other_cols = [c for c in df.columns if c not in key_cols]
df = df[key_cols + other_cols]

# 6.7 Standardize numeric features for outlier detection
print("\n6.7 Standardizing numeric features...")
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

# Exclude from scaling:
# - IDs and keys
# - Binary flags (one-hot, multi-hot, missingness indicators)
# - Performance targets (we want to keep these interpretable)
exclude_from_scaling = (
    key_cols + 
    [c for c in numeric_cols if c.startswith(("mrr_", "top@", "cat_", "cat__", "concept__"))] +
    [c for c in numeric_cols if c.endswith("_is_missing") or c.endswith("__is_missing")] +
    [c for c in numeric_cols if "_has_" in c]  # Binary indicators
)

cols_to_scale = [c for c in numeric_cols if c not in exclude_from_scaling]

print(f"  Total numeric columns: {len(numeric_cols)}")
print(f"  Columns to scale: {len(cols_to_scale)}")
print(f"  Excluded from scaling: {len(exclude_from_scaling)}")

if cols_to_scale:
    scaler = StandardScaler()
    df[cols_to_scale] = scaler.fit_transform(df[cols_to_scale])
    
    # Save scaler for potential future use
    with open(SCALER_FILE, "wb") as f:
        pickle.dump({'scaler': scaler, 'columns': cols_to_scale}, f)
    print(f"  Saved scaler to: {SCALER_FILE}")

# 6.8 Final cleanup: fill any remaining NaNs from derived features
print("\n6.8 Final NaN handling...")
# For derived features (ratios, densities), NaN means the denominator was 0
# Fill these with 0 or median depending on the feature type
derived_feature_patterns = ['_density', '_avg_', '_ratio']
for col in df.columns:
    if any(pattern in col for pattern in derived_feature_patterns):
        if df[col].isna().any():
            # Use median for derived features (more robust than 0)
            median_val = df[col].median()
            if pd.isna(median_val):
                median_val = 0.0
            df[col] = df[col].fillna(median_val)
            print(f"  Filled NaNs in {col} with median: {median_val:.4f}")

# Any remaining NaNs in numeric columns -> fill with 0
remaining_nan_cols = df.select_dtypes(include=[np.number]).columns[df.select_dtypes(include=[np.number]).isna().any()].tolist()
if remaining_nan_cols:
    print(f"  Filling remaining NaNs with 0 in: {remaining_nan_cols}")
    df[remaining_nan_cols] = df[remaining_nan_cols].fillna(0.0)

# Save preprocessed dataset
print(f"\nPreprocessed (rich) shape: {df.shape}")
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
print(f"  Text-derived features: {len([c for c in df.columns if c.startswith('txt_')])}")
print(f"  Concept features: {len([c for c in df.columns if c.startswith('concept')])}")
print(f"  Other features: {len([c for c in df.columns if not any(c.startswith(p) for p in ['mrr_', 'top@', 'cat', 'txt_', 'concept']) and c not in key_cols])}")

print("\nFiles created:")
print(f"  1. {OUT_FULL} - Full dataset with all original columns")
print(f"  2. {OUT_PREP} - Preprocessed dataset ready for modeling")
print(f"  3. {SCALER_FILE} - StandardScaler for feature scaling")

print("\n" + "=" * 60)
print("Preprocessing complete!")
print("=" * 60)