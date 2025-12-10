import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = Path(".")

FEATURES_FILE = DATA_DIR / "bug_features_v2.csv"
RATINGS_FILE  = DATA_DIR / "gemini_bug_ratings.csv"
CATEG_FILE    = DATA_DIR / "gemini_bug_categorization.csv"
PERF_FILE     = DATA_DIR / "tool_comparison_summary.csv"

OUT_FULL = DATA_DIR / "experimentA_full_dataset.csv"
OUT_PREP = DATA_DIR / "experimentA_preprocessed.csv"

# 1. Load
features_df = pd.read_csv(FEATURES_FILE)
ratings_df  = pd.read_csv(RATINGS_FILE)
categ_df    = pd.read_csv(CATEG_FILE)
perf_df     = pd.read_csv(PERF_FILE)

# 2. Parse project & bug_id from "id" like "Chart-1"
def split_id(df):
    proj_bug = df["id"].str.split("-", n=1, expand=True)
    df["project"] = proj_bug[0]
    df["bug_id"]  = proj_bug[1].astype(int)
    return df

features_df = split_id(features_df)
ratings_df  = split_id(ratings_df)
categ_df    = split_id(categ_df)

# 3. Compute MRR from rank
perf_df["mrr"] = np.where(perf_df["rank"].notna(), 1.0 / perf_df["rank"], 0.0)

perf_subset = perf_df[["project", "bug_id", "tool", "mrr", "top@1", "top@5"]].copy()

# 4. Pivot performance wide
perf_wide = perf_subset.pivot_table(
    index=["project", "bug_id"],
    columns="tool",
    values=["mrr", "top@1", "top@5"],
    aggfunc="first"
)
perf_wide.columns = [f"{metric}_{tool}" for metric, tool in perf_wide.columns]
perf_wide = perf_wide.reset_index()

# 5. Merge features + performance on (project, bug_id)
merged = features_df.merge(
    perf_wide,
    on=["project", "bug_id"],
    how="left"
)

# 6. Merge ratings and categorization (also on project + bug_id)
drop_from_ratings = ["title", "description_length"]
ratings_for_merge = ratings_df.drop(columns=drop_from_ratings, errors="ignore")

merged = merged.merge(
    ratings_for_merge,
    on=["project", "bug_id"],
    how="left",
    suffixes=("", "_ratings")
)

drop_from_categ = ["title"]
categ_for_merge = categ_df.drop(columns=drop_from_categ, errors="ignore")

merged = merged.merge(
    categ_for_merge,
    on=["project", "bug_id"],
    how="left",
    suffixes=("", "_categ")
)

# 7. Save full merged dataset
merged.to_csv(OUT_FULL, index=False)
print("Full merged shape:", merged.shape)

# 8. Preprocess: treat missing perf as failure, one-hot category, drop text cols
df = merged.copy()

perf_cols = [c for c in df.columns if c.startswith("mrr_") or c.startswith("top@")]
df[perf_cols] = df[perf_cols].fillna(0.0)

if "category" in df.columns:
    df = df.join(pd.get_dummies(df["category"], prefix="cat"))
    df.drop(columns=["category"], inplace=True)

drop_text_cols = [
    "title", "reasoning", "reasoning_ratings",
    "likely_impacted_code_concepts",
]
df.drop(columns=drop_text_cols, inplace=True, errors="ignore")

# Optional: reorder
key_cols = [c for c in ["project", "bug_id"] if c in df.columns]
other_cols = [c for c in df.columns if c not in key_cols]
df = df[key_cols + other_cols]

df.to_csv(OUT_PREP, index=False)
print("Preprocessed shape:", df.shape)
print("Saved:", OUT_PREP)
