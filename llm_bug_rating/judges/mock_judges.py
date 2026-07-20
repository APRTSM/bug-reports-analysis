"""Mock judge implementations that produce deterministic, structured outputs.

These are deterministic heuristics-based judges suitable for offline runs and tests.
Real API-backed judges should subclass the same interface and produce identical JSON shapes.
"""
from __future__ import annotations
from typing import Dict, Any
import re
import json
from .base import Judge, ChecklistItemResult


def _keyword_score(text: str, keywords: list[str]) -> float:
    text = text.lower()
    hits = sum(1 for k in keywords if k.lower() in text)
    # scale to [0,1] with diminishing returns
    return min(1.0, hits / max(1, len(keywords)))


class MockJudge:
    def __init__(self, name: str):
        self.name = name

    def evaluate(self, bug: Dict[str, str], checklist: Dict[str, Dict[str, str]]) -> Dict[str, Dict[str, ChecklistItemResult]]:
        summary = bug.get("summary", "") or ""
        description = bug.get("description", "") or ""
        text = (summary + "\n" + description).lower()

        # simple heuristics per checklist item using keywords
        out: Dict[str, Dict[str, ChecklistItemResult]] = {}

        for dim, items in checklist.items():
            out[dim] = {}
            for code, prompt in items.items():
                keywords = []
                # derive keywords heuristically from prompt words
                tokens = re.findall(r"[a-zA-Z]{3,}", prompt)
                keywords = tokens[:5]
                score = _keyword_score(text, keywords)
                # heuristic boolean threshold
                value = score >= 0.25
                # confidence maps score to [0.5, 0.99]
                confidence = 0.5 + 0.49 * score
                reason = f"Heuristic: matched {int(score*100)}% of keywords from '{prompt}'"
                out[dim][code] = ChecklistItemResult(value=value, confidence=round(confidence, 3), reason=reason)

        return out


def judge_factory(spec: Dict[str, Any]) -> Judge:
    judge_type = spec.get("type", "mock")

    if judge_type == "openai":
        from .llm_judges import OpenAIJudge
        return OpenAIJudge(spec)
    if judge_type == "anthropic":
        from .llm_judges import AnthropicJudge
        return AnthropicJudge(spec)
    if judge_type == "gemini":
        from .llm_judges import GeminiJudge
        return GeminiJudge(spec)

    # default: mock (no API calls)
    return MockJudge(spec.get("name", "mock-judge"))
