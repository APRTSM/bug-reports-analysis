from __future__ import annotations
from typing import Dict, Any
from dataclasses import dataclass
import json


@dataclass
class AdjudicationResult:
    final_value: bool
    rationale: str
    selected: str  # 'judge1' | 'judge2' | 'both_partial'


def adjudicate_item(criterion: str, bug: Dict[str, str], j1_reason: Dict[str, Any], j2_reason: Dict[str, Any]) -> AdjudicationResult:
    """Simple adjudicator that picks the judge with higher confidence; if close, marks partially correct.

    This is a determinisitic fallback adjudicator. In production, this should call an adjudicator LLM.
    """
    # compare confidences
    c1 = float(j1_reason.get("confidence", 0))
    c2 = float(j2_reason.get("confidence", 0))

    if abs(c1 - c2) < 0.05:
        # partial
        # choose True if either judge has True
        final = bool(j1_reason.get("value", False)) or bool(j2_reason.get("value", False))
        return AdjudicationResult(final_value=final, rationale="Close confidences; merged truth by OR", selected="both_partial")

    if c1 > c2:
        return AdjudicationResult(final_value=bool(j1_reason.get("value", False)), rationale=f"Selected judge1 (higher confidence: {c1} > {c2})", selected="judge1")
    else:
        return AdjudicationResult(final_value=bool(j2_reason.get("value", False)), rationale=f"Selected judge2 (higher confidence: {c2} > {c1})", selected="judge2")
