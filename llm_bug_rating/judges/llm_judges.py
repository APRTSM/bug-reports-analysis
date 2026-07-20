from __future__ import annotations
import json
import os
from typing import Dict, Any
from json_repair import repair_json
from .base import ChecklistItemResult

_PROMPT_PATH = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "prompts", "checklist_prompt.txt"))


def _load_prompt() -> str:
    with open(_PROMPT_PATH, encoding="utf-8") as f:
        return f.read()


def _format_checklist(checklist: Dict[str, Dict[str, str]]) -> str:
    lines = []
    for dim, items in checklist.items():
        lines.append(f"{dim}:")
        for code, text in items.items():
            lines.append(f"  {code}: {text}")
    return "\n".join(lines)


def _parse_response(raw: str, checklist: Dict[str, Dict[str, str]]) -> Dict[str, Dict[str, ChecklistItemResult]]:
    """Parse LLM JSON response into ChecklistItemResult objects.

    Strips markdown code fences if present. Falls back gracefully per item
    if the model omits a key — value=False, confidence=0.0.
    """
    text = raw.strip()
    if text.startswith("```"):
        parts = text.split("```")
        text = parts[1] if len(parts) > 1 else text
        if text.startswith("json"):
            text = text[4:]
        text = text.strip()

    try:
        data = json.loads(text, strict=False)
    except json.JSONDecodeError:
        try:
            data = json.loads(repair_json(text), strict=False)
        except Exception as e:
            raise ValueError(f"LLM returned invalid JSON (repair also failed): {e}\n---\n{raw[:500]}") from e

    result: Dict[str, Dict[str, ChecklistItemResult]] = {}
    for dim, items in checklist.items():
        result[dim] = {}
        dim_data = data.get(dim, {})
        for code in items:
            item_data = dim_data.get(code, {})
            result[dim][code] = ChecklistItemResult(
                value=bool(item_data.get("value", False)),
                confidence=float(item_data.get("confidence", 0.0)),
                reason=str(item_data.get("reason", "Missing from model response")),
            )
    return result


class OpenAIJudge:
    def __init__(self, spec: Dict[str, Any]):
        import openai  # pip install openai; set OPENAI_API_KEY
        self.name = spec["name"]
        self.model = spec["model_version"]
        self.temperature = spec.get("temperature", 0.1)
        self.top_p = spec.get("top_p", 1.0)
        self.client = openai.OpenAI()

    def evaluate(self, bug: Dict[str, str], checklist: Dict[str, Dict[str, str]]) -> Dict[str, Dict[str, ChecklistItemResult]]:
        prompt = _load_prompt().format(
            checklist=_format_checklist(checklist),
            summary=bug.get("summary", ""),
            description=bug.get("description", ""),
        )
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            top_p=self.top_p,
        )
        return _parse_response(response.choices[0].message.content, checklist)


class AnthropicJudge:
    def __init__(self, spec: Dict[str, Any]):
        import anthropic  # pip install anthropic; set ANTHROPIC_API_KEY
        self.name = spec["name"]
        self.model = spec["model_version"]
        self.temperature = spec.get("temperature", 0.1)
        self.top_p = spec.get("top_p", 1.0)
        self.client = anthropic.Anthropic()

    def evaluate(self, bug: Dict[str, str], checklist: Dict[str, Dict[str, str]]) -> Dict[str, Dict[str, ChecklistItemResult]]:
        prompt = _load_prompt().format(
            checklist=_format_checklist(checklist),
            summary=bug.get("summary", ""),
            description=bug.get("description", ""),
        )
        response = self.client.messages.create(
            model=self.model,
            max_tokens=4096,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
        )
        return _parse_response(response.content[0].text, checklist)


class GeminiJudge:
    def __init__(self, spec: Dict[str, Any]):
        import google.generativeai as genai  # pip install google-generativeai; set GOOGLE_API_KEY
        self.name = spec["name"]
        self.generation_config = genai.types.GenerationConfig(
            temperature=spec.get("temperature", 0.1),
            top_p=spec.get("top_p", 1.0),
        )
        self.model = genai.GenerativeModel(spec["model_version"])

    def evaluate(self, bug: Dict[str, str], checklist: Dict[str, Dict[str, str]]) -> Dict[str, Dict[str, ChecklistItemResult]]:
        prompt = _load_prompt().format(
            checklist=_format_checklist(checklist),
            summary=bug.get("summary", ""),
            description=bug.get("description", ""),
        )
        response = self.model.generate_content(prompt, generation_config=self.generation_config)
        return _parse_response(response.text, checklist)
