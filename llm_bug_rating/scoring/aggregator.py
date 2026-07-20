from __future__ import annotations
from typing import Dict, Any, List
import statistics

# Dimensions where more True items = worse quality; invert before z-scoring
# RepairReadiness is NOT inverted here — more repair-ready = better (higher z = easier to fix).
# Its derived counterpart repair_difficulty IS inverted separately in the derived block below.
_INVERTED_DIMS = {"Ambiguity"}

# Dimensions to expose as booleans in derived scores
_BOOLEAN_DIMS = {"HiddenReproducibility"}


def aggregate_scores(
    checklist_results: Dict[str, Dict[str, Dict[str, Any]]],
    mapping: Dict[str, List[str]] | None = None,
) -> Dict[str, Any]:
    """Aggregate checklist boolean values into per-dimension sums and derived scores.

    checklist_results: dimension -> item -> {value, confidence, reason}

    Returns:
        raw      — raw True-item counts per dimension
        derived  — semantically labelled scores (repair_difficulty inverted, etc.)
        zscore   — z-score over corrected (sign-flipped where needed) values
        total_possible_per_dimension — max score per dimension
    """
    raw_values: Dict[str, int] = {}
    for dim, items in checklist_results.items():
        raw_values[dim] = sum(1 if items[k]["value"] else 0 for k in items)

    # --- derived scores with correct semantics ---
    derived: Dict[str, Any] = {}

    if "RepairReadiness" in raw_values:
        max_p = len(checklist_results["RepairReadiness"])
        derived["repair_difficulty"] = max_p - raw_values["RepairReadiness"]

    if "Ambiguity" in raw_values:
        derived["ambiguity_count"] = raw_values["Ambiguity"]

    if "HiddenReproducibility" in raw_values:
        derived["hidden_s2r_present"] = raw_values["HiddenReproducibility"] == 1

    if "RootCauseEvidence" in raw_values:
        derived["root_cause_evidence"] = raw_values["RootCauseEvidence"]

    if "ImpactScope" in raw_values:
        derived["impact_scope"] = raw_values["ImpactScope"]

    # --- z-scores over sign-corrected values so all axes point "higher = better" ---
    corrected: Dict[str, float] = {}
    for dim, val in raw_values.items():
        if dim in _INVERTED_DIMS:
            max_v = len(checklist_results[dim])
            corrected[dim] = float(max_v - val)
        elif dim in _BOOLEAN_DIMS:
            corrected[dim] = float(val)
        else:
            corrected[dim] = float(val)

    vals_list = list(corrected.values())
    if len(vals_list) > 1:
        mean = statistics.mean(vals_list)
        stdev = statistics.pstdev(vals_list)
        z_scores = {d: ((v - mean) / stdev) if stdev > 0 else 0.0 for d, v in corrected.items()}
    else:
        z_scores = {d: 0.0 for d in corrected}

    return {
        "raw": raw_values,
        "derived": derived,
        "zscore": z_scores,
        "total_possible_per_dimension": {d: len(checklist_results[d]) for d in checklist_results},
    }
