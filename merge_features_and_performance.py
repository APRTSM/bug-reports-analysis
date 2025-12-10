import pandas as pd
import numpy as np
from pathlib import Path

# ============================
# CONFIG
# ============================

DATA_DIR = Path(".")  # change this if your CSVs live elsewhere

FEATURES_FILE = DATA_DIR / "bug_features_v2.csv"
RATINGS_FILE  = DATA_DIR / "gemini_bug_ratings.csv"
CATEG_FILE    = DATA_DIR / "gemini_bug_categorization.csv"
PERF_FILE     = DATA_DIR / "tool_comparison_summary.csv"

OUT_FULL = DATA_DIR / "experimentA_full_dataset.csv"
OUT_PREP = DATA_DIR / "experimentA_preprocessed.csv"


# ============================
# 1. LOAD DATA
# ============================

features_df = pd.read_csv(FEATURES_FILE)
ratings_df  = pd.read_csv(RATINGS_FILE)
categ_df    = pd.read_csv(CATEG_FILE)
perf_df     = pd.read_csv(PERF_FILE)

print("Loaded:")
print(" - features_df:", features_df.shape)
print(" - ratings_df :", ratings_df.shape)
print(" - categ_df   :", categ_df.shape)
print(" - perf_df    :", perf_df.shape)


# ============================
# 2. COMPUTE MRR PER (BUG, TOOL)
# ============================

# rank is NaN when not detected → MRR = 0
perf_df["mrr"] = np.where(perf_df["rank"].notna(), 1.0 / perf_df["rank"], 0.0)

# Keep only relevant columns for pivot
perf_subset = perf_df[["project", "bug_id", "tool", "mrr", "top@1", "top@5"]].copy()


# ============================
# 3. PIVOT TOOL PERFORMANCE WIDE
# ============================

perf_wide = perf_subset.pivot_table(
    index=["project", "bug_id"],
    columns="tool",
    values=["mrr", "top@1", "top@5"],
    aggfunc="first"   # there should be at most one row per (bug, tool)
)

# Flatten MultiIndex columns → mrr_AgentFL, top@1_Locus, ...
perf_wide.columns = [f"{metric}_{tool}" for metric, tool in perf_wide.columns]
perf_wide = perf_wide.reset_index()

print("perf_wide shape:", perf_wide.shape)


# ============================
# 4. ALIGN KEYS ACROSS TABLES
# ============================

# All three bug-level tables use "id" as bug identifier; perf uses "bug_id"
for df in [features_df, ratings_df, categ_df]:
    df["bug_id"] = df["id"]

# Convert bug_id to string for consistent merging across all dataframes
# This prevents type mismatch errors (object vs int64)
features_df["bug_id"] = features_df["bug_id"].astype(str)
ratings_df["bug_id"] = ratings_df["bug_id"].astype(str)
categ_df["bug_id"] = categ_df["bug_id"].astype(str)
perf_wide["bug_id"] = perf_wide["bug_id"].astype(str)


# ============================
# 5. MERGE: FEATURES + PERFORMANCE
# ============================

merged = features_df.merge(
    perf_wide,
    on="bug_id",
    how="left"    # keep all bugs from features_df
)

# Add project column if missing in features_df
if "project" not in merged.columns and "project" in perf_wide.columns:
    merged["project"] = merged["project"]  # no-op; kept for clarity

print("After merging perf:", merged.shape)


# ============================
# 6. MERGE: GEMINI RATINGS
# ============================
# ratings_df has some overlapping columns; we typically keep the feature-like ones.
# Drop raw text columns we don't need from ratings before merging.
drop_from_ratings = ["title", "description_length"]
ratings_for_merge = ratings_df.drop(columns=drop_from_ratings, errors="ignore")

merged = merged.merge(
    ratings_for_merge,
    on="bug_id",
    how="left",
    suffixes=("", "_ratings")
)

print("After merging ratings:", merged.shape)


# ============================
# 7. MERGE: GEMINI CATEGORIZATION
# ============================
# Keep category + confidence (and optionally reasoning if you still want it).
drop_from_categ = ["title"]  # description_length might duplicate others, so we can drop or keep
categ_for_merge = categ_df.drop(columns=drop_from_categ, errors="ignore")

merged = merged.merge(
    categ_for_merge,
    on="bug_id",
    how="left",
    suffixes=("", "_categ")
)

print("After merging categorization:", merged.shape)


# ============================
# 8. SAVE FULL (RAW) MERGED DATASET
# ============================

merged.to_csv(OUT_FULL, index=False)
print(f"Saved full merged dataset to: {OUT_FULL}  shape={merged.shape}")


# ============================
# 9. PREPROCESSING FOR EXPERIMENT A
# ============================

df = merged.copy()

# 9.1 Treat missing performance as failure: MRR = 0, Top@k = 0
perf_cols = [c for c in df.columns if c.startswith("mrr_") or c.startswith("top@")]
df[perf_cols] = df[perf_cols].fillna(0.0)

# 9.2 One-hot encode category (from categorization file)
if "category" in df.columns:
    df = df.join(pd.get_dummies(df["category"], prefix="cat"))
    df.drop(columns=["category"], inplace=True)

# 9.3 Drop long-text / high-cardinality columns you don't want in numeric analyses
# You can adjust this list based on your actual columns.
drop_text_cols = [
    "title", "reasoning",          # from categorization
    "reasoning_ratings",           # if exists via suffix
    "likely_impacted_code_concepts",
    "id_x", "id_y", "id",          # redundant IDs if generated
]
df.drop(columns=drop_text_cols, inplace=True, errors="ignore")

# Optional: ensure bug_id and project are at the front
key_cols = []
if "bug_id" in df.columns: key_cols.append("bug_id")
if "project" in df.columns: key_cols.append("project")
other_cols = [c for c in df.columns if c not in key_cols]
df = df[key_cols + other_cols]

# ============================
# 10. SAVE PREPROCESSED DATASET
# ============================

df.to_csv(OUT_PREP, index=False)
print(f"Saved preprocessed dataset to: {OUT_PREP}  shape={df.shape}")

print("Done.")
