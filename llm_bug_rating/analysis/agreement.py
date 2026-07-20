from __future__ import annotations
from typing import Dict, Any, Tuple
import math


def per_item_agreement(j1: Dict[str, Dict[str, Any]], j2: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Compute agreement per checklist item and confidence differences.

    j1/j2 shapes: dimension -> item -> {value, confidence, reason}
    """
    result = {}
    for dim in j1:
        result[dim] = {}
        for item in j1[dim]:
            v1 = bool(j1[dim][item]["value"])
            v2 = bool(j2[dim][item]["value"])
            c1 = float(j1[dim][item]["confidence"])
            c2 = float(j2[dim][item]["confidence"])
            agree = v1 == v2
            result[dim][item] = {
                "agreement": agree,
                "confidence_diff": abs(c1 - c2),
                "judge1_value": v1,
                "judge2_value": v2,
                "judge1_confidence": c1,
                "judge2_confidence": c2,
            }
    return result


def percent_agreement(per_item: Dict[str, Dict[str, Any]]) -> float:
    total = 0
    agrees = 0
    for dim in per_item:
        for item in per_item[dim]:
            total += 1
            if per_item[dim][item]["agreement"]:
                agrees += 1
    return agrees / total if total > 0 else 0.0


def cohens_kappa(j1: Dict[str, Dict[str, Any]], j2: Dict[str, Dict[str, Any]]) -> float:
    """Compute Cohen's Kappa treating each checklist item as a binary observation.

    Returns kappa in [-1,1].
    """
    # flatten
    obs = []
    for dim in j1:
        for item in j1[dim]:
            v1 = 1 if j1[dim][item]["value"] else 0
            v2 = 1 if j2[dim][item]["value"] else 0
            obs.append((v1, v2))

    n = len(obs)
    if n == 0:
        return 0.0

    # observed agreement
    agree = sum(1 for a, b in obs if a == b)
    p0 = agree / n

    # marginals
    p1_yes = sum(a for a, _ in obs) / n
    p2_yes = sum(b for _, b in obs) / n
    pe = p1_yes * p2_yes + (1 - p1_yes) * (1 - p2_yes)

    if pe == 1:
        return 1.0
    kappa = (p0 - pe) / (1 - pe) if (1 - pe) != 0 else 0.0
    return kappa
