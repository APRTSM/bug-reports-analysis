from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Protocol


@dataclass
class ChecklistItemResult:
    value: bool
    confidence: float
    reason: str


class Judge(Protocol):
    name: str

    def evaluate(self, bug: Dict[str, str], checklist: Dict[str, Dict[str, str]]) -> Dict[str, Dict[str, ChecklistItemResult]]:
        ...
