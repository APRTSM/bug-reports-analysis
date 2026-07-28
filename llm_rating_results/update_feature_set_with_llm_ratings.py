"""
Replace old Gemini-derived LLM features in final_feature_set_bug_reports.csv
with features from the new aggregated multi-rater LLM ratings.

Old features removed (all derived from Gemini's single-rater output):
  reasoning_composite, actionability, clarity, expected_observed_alignment,
  technical_completeness, repair_difficulty, ambiguity, ambiguity_type_count,
  concept_network_concept_breadth, txt_likely_impacted_code_concepts_digit_density,
  txt_reasoning_word_count, txt_reasoning_avg_sentence_len, txt_reasoning_digit_density,
  txt_root_cause_guess_word_count, txt_root_cause_guess_digit_density,
  hidden_s2r_present, contradiction_present

New features added (from aggregated_features.csv):
  percent_agreement, cohens_kappa  — inter-rater agreement metrics
  {dim}                            — raw aggregated ratings (11 dimensions, 1–5 scale)
  z_{dim}                          — z-scored ratings across raters (11 dimensions)
  repair_difficulty, ambiguity_count, hidden_s2r_present  — derived booleans/ordinals

Engineered composites (from dimension scores):
  technical_completeness  — (technical_context/5 + root_cause_evidence/3) / 2
  reasoning_composite     — reasoning_quality / 5  (normalized; hidden_s2r excluded, saturated)
  quality_composite       — mean of non-saturated quality dims (specificity, reproducibility,
                            reasoning_quality, repair_readiness), normalized to 0–1
  quality_std             — std of same dims (spread / disagreement across dimensions)
  specificity_repro_gap   — specificity − reproducibility (clear but hard to reproduce?)
  spec_reasoning_product  — specificity × reasoning_quality (both informative → synergy)
  causal_consistency      — 1 − |clipped(num_causal_markers_z) − z_reasoning_quality|/2
                            (do textual causal markers align with LLM reasoning rating?)
  technical_consistency   — 1 − |technical_indicators/2 − technical_context/5|
                            (do code/stacktrace signals align with LLM technical rating?)

Item-level composites (from per-bug JSON final_checklist; items chosen for low ceiling rate
and clear theoretical motivation — not selected post-hoc on correlation with APR outcomes):
  causal_reasoning_score  — mean(Q1,Q2,Q3,Q5): non-saturated reasoning items; Q4 excluded
                            (97% ceiling: "internally coherent" is trivially true)
  repair_feasibility      — mean(P2,P5): fix location identifiable + fix implementable from
                            report alone — the two direct APR-readiness items in RepairReadiness
  repro_capability        — mean(R4,S2): reproduction steps complete + concrete ordered steps
  p5_fix_feasible         — P5 alone (bool): "a developer could implement a plausible fix using
                            only the information in this report" — most direct APR signal by design
"""

import glob
import json
import os
import numpy as np
import pandas as pd
import re


OLD_LLM_COLS = [
    "reasoning_composite",
    "actionability",
    "clarity",
    "expected_observed_alignment",
    "technical_completeness",
    "repair_difficulty",
    "ambiguity",
    "ambiguity_type_count",
    "concept_network_concept_breadth",
    "txt_likely_impacted_code_concepts_digit_density",
    "txt_reasoning_word_count",
    "txt_reasoning_avg_sentence_len",
    "txt_reasoning_digit_density",
    "txt_root_cause_guess_word_count",
    "txt_root_cause_guess_digit_density",
    "hidden_s2r_present",
    "contradiction_present",
]


def camel_to_snake(name: str) -> str:
    return re.sub(r"(?<!^)(?=[A-Z])", "_", name).lower()


def expand_aggregated(agg: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in agg.iterrows():
        raw = json.loads(row["aggregated_raw"])
        zscore = json.loads(row["aggregated_zscore"])
        derived = json.loads(row["aggregated_derived"])

        entry = {
            "bug_id": row["bug_id"],
            "percent_agreement": row["percent_agreement"],
            "cohens_kappa": row["cohens_kappa"],
        }

        # raw ratings: Actionability → actionability, etc.
        for key, val in raw.items():
            entry[camel_to_snake(key)] = val

        # z-scored ratings: Actionability → z_actionability, etc.
        for key, val in zscore.items():
            entry[f"z_{camel_to_snake(key)}"] = val

        # derived fields (skip raw duplicates: root_cause_evidence, impact_scope)
        for key in ("repair_difficulty", "ambiguity_count", "hidden_s2r_present"):
            entry[key] = derived[key]

        rows.append(entry)

    return pd.DataFrame(rows)


def extract_item_composites(json_dir: str) -> pd.DataFrame:
    """Read per-bug JSONs and compute item-level composite features.

    Items chosen for low ceiling rate (35–58%) and clear theoretical motivation;
    Q4 (97% ceiling) excluded from causal_reasoning_score.
    """
    rows = []
    for path in glob.glob(os.path.join(json_dir, "*.json")):
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        if "bug" not in data:
            continue

        fc = data.get("final_checklist", {})

        def item(dim, code):
            return int(fc.get(dim, {}).get(code, {}).get("value", False))

        q1 = item("ReasoningQuality", "Q1")
        q2 = item("ReasoningQuality", "Q2")
        q3 = item("ReasoningQuality", "Q3")
        q5 = item("ReasoningQuality", "Q5")
        p2 = item("RepairReadiness", "P2")
        p5 = item("RepairReadiness", "P5")
        r4 = item("Reproducibility", "R4")
        s2 = item("Specificity", "S2")

        rows.append({
            "bug_id": data["bug"]["bug_id"],
            "causal_reasoning_score": (q1 + q2 + q3 + q5) / 4.0,
            "repair_feasibility":     (p2 + p5) / 2.0,
            "repro_capability":       (r4 + s2) / 2.0,
            "p5_fix_feasible":        p5,
        })

    return pd.DataFrame(rows)


def create_llm_composite_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adapted from the old create_llm_interaction_features /
    create_semantic_consistency_features pipeline.

    Skipped vs old version:
      - clarity/actionability interactions (both saturated ~90-94% at ceiling)
      - hidden_s2r interactions (saturated at 89%)
      - encode_ambiguity_types (no per-type breakdown in new data)
      - create_concept_network_features (no raw concept list in new data)

    New dimension mapping vs old:
      technical_depth          → technical_context  (0–5)
      expected_observed_align  → root_cause_evidence (0–3)
      causal_reasoning_quality → reasoning_quality   (0–5)
    """
    out = pd.DataFrame(index=df.index)

    # --- Composite scores (analogues of old technical_completeness / reasoning_composite) ---

    # Normalize each to 0–1 before averaging so different max scales don't bias
    tc_norm = df["technical_context"] / 5.0
    rc_norm = df["root_cause_evidence"] / 3.0
    out["technical_completeness"] = (tc_norm + rc_norm) / 2.0

    out["reasoning_composite"] = df["reasoning_quality"] / 5.0

    # Quality composite: mean of the four non-saturated quality dims, normalized to 0–1
    quality_dims = ["specificity", "reproducibility", "reasoning_quality", "repair_readiness"]
    quality_norm = df[quality_dims].div([5, 5, 5, 5])
    out["quality_composite"] = quality_norm.mean(axis=1)
    out["quality_std"] = quality_norm.std(axis=1)

    # --- Interaction terms ---

    # Gap: specific enough to reproduce? (positive = easy to understand, hard to repro)
    out["specificity_repro_gap"] = df["specificity"] - df["reproducibility"]

    # Synergy: high specificity + high reasoning quality → very informative report
    out["spec_reasoning_product"] = df["specificity"] * df["reasoning_quality"]

    # --- Semantic consistency features ---

    # Causal consistency: do textual causal markers align with LLM reasoning rating?
    # num_causal_markers and z_reasoning_quality are both z-scored, so clip to [-3, 3]
    # then measure absolute gap, mapped to [0, 1] (1 = perfectly consistent)
    causal_z = df["num_causal_markers"].clip(-3, 3)
    reasoning_z = df["z_reasoning_quality"].clip(-3, 3)
    out["causal_consistency"] = 1.0 - (causal_z - reasoning_z).abs() / 6.0

    # Technical consistency: do structural signals (code, stacktrace) match LLM technical rating?
    tech_indicators = (df["has_code"].astype(float) + df["has_stacktrace"].astype(float)) / 2.0
    out["technical_consistency"] = 1.0 - (tech_indicators - df["technical_context"] / 5.0).abs()

    print(f"  Created {len(out.columns)} composite/interaction features: {list(out.columns)}")
    return out


def main():
    root_dir = os.path.dirname(os.path.abspath(__file__))
    feat_path = os.path.join(root_dir, "full_feature_preproccessed_fixed", "final_feature_set_bug_reports.csv")
    agg_path = os.path.join(root_dir, "llm_bug_rating", "data", "results_v2", "csv", "aggregated_features.csv")

    feat = pd.read_csv(feat_path)
    agg = pd.read_csv(agg_path)

    # Drop old LLM columns (only those that exist)
    to_drop = [c for c in OLD_LLM_COLS if c in feat.columns]
    feat = feat.drop(columns=to_drop)
    print(f"Dropped {len(to_drop)} old LLM columns: {to_drop}")

    # Expand aggregated features
    new_cols = expand_aggregated(agg)
    print(f"New columns to add: {list(new_cols.columns[1:])}")

    # Drop any columns that are already in feat from a previous run of this script
    # (avoids pandas adding _x/_y suffixes on merge)
    stale = [c for c in new_cols.columns if c != "bug_id" and c in feat.columns]
    if stale:
        feat = feat.drop(columns=stale)
        print(f"Dropped {len(stale)} stale columns from previous run: {stale}")

    # Also drop composite features that will be re-engineered below
    composite_cols = [
        "technical_completeness", "reasoning_composite", "quality_composite",
        "quality_std", "specificity_repro_gap", "spec_reasoning_product",
        "causal_consistency", "technical_consistency",
        # item-level composites
        "causal_reasoning_score", "repair_feasibility", "repro_capability", "p5_fix_feasible",
    ]
    stale_composites = [c for c in composite_cols if c in feat.columns]
    if stale_composites:
        feat = feat.drop(columns=stale_composites)
        print(f"Dropped {len(stale_composites)} stale composite columns: {stale_composites}")

    # Merge on bug ID (feat uses 'id', agg uses 'bug_id')
    # Rename agg's bug_id before merge to avoid collision with feat's bug_id column
    new_cols = new_cols.rename(columns={"bug_id": "_agg_bug_id"})
    merged = feat.merge(new_cols, left_on="id", right_on="_agg_bug_id", how="left")
    merged = merged.drop(columns=["_agg_bug_id"])

    missing = merged["percent_agreement"].isna().sum()
    if missing:
        print(f"Warning: {missing} rows have no matching LLM rating")

    # Engineer composite / interaction features from the new LLM ratings
    print("\nEngineering composite features...")
    composites = create_llm_composite_features(merged)
    merged = pd.concat([merged, composites], axis=1)

    # Item-level composites — read from per-bug JSONs
    json_dir = agg_path.replace("csv/aggregated_features.csv", "json")
    print("\nExtracting item-level composites from per-bug JSONs...")
    item_comps = extract_item_composites(json_dir)
    print(f"  Found {len(item_comps)} bugs with item-level data")
    item_comps = item_comps.rename(columns={"bug_id": "_item_bug_id"})
    merged = merged.merge(item_comps, left_on="id", right_on="_item_bug_id", how="left")
    merged = merged.drop(columns=["_item_bug_id"])
    missing_items = merged["p5_fix_feasible"].isna().sum()
    if missing_items:
        print(f"  Warning: {missing_items} rows have no item-level data")

    merged.to_csv(feat_path, index=False)
    print(f"\nSaved {len(merged)} rows, {len(merged.columns)} columns → {feat_path}")


if __name__ == "__main__":
    main()
