"""
s2r_quality.py

Lightweight, Euler-inspired evaluation of Steps To Reproduce (S2Rs).

- Identify S2R sentences in a bug report
- Split them into individual steps
- Evaluate each step as High-quality / Ambiguous / Vocabulary-mismatch
  using only text-based heuristics (no app execution model).

Usage example (from another file):

    from s2r_quality import evaluate_s2r_block

    text = bug["description"]
    s2r_result = evaluate_s2r_block(text)
    print(s2r_result.overall_score, s2r_result.steps)

"""

from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional
import re
import os
import glob
import json
import pandas as pd

import spacy

# Load spaCy model once (reuse across calls)
# Make sure you have: python -m spacy download en_core_web_sm
print("Loading spaCy model...")
nlp = spacy.load("en_core_web_sm", exclude=["ner", "lemmatizer", "textcat"])
print("spaCy model loaded.")


# ---------------------------------------------------------------------------
# Configuration: verbs, vague terms, UI-ish words
# ---------------------------------------------------------------------------

STEP_VERBS = {
    "click", "tap", "press", "hit", "select", "choose", "pick",
    "open", "close", "enter", "type", "fill", "insert", "set",
    "hover", "scroll", "drag", "drop", "check", "uncheck",
    "enable", "disable", "switch", "navigate", "go",
}

VAGUE_TERMS = {
    "sometimes", "maybe", "often", "rarely", "occasionally",
    "from time to time", "every now and then", "randomly",
    "does not work properly", "weird", "strange", "odd",
}

UI_TERMS = {
    "button", "menu", "link", "checkbox", "check box", "radio",
    "field", "textbox", "text field", "dropdown", "drop-down",
    "list", "tab", "panel", "dialog", "popup", "pop-up",
    "screen", "page", "window", "toolbar", "icon",
}


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class StepEvaluation:
    step_text: str
    has_action: bool
    has_object: bool
    has_ui_term: bool
    is_imperative: bool
    has_vague_terms: bool
    quality_label: str          # "HQ" / "AS" / "VM"
    issues: List[str]

@dataclass
class S2RBlockEvaluation:
    """All S2R evaluations for a single bug report."""
    raw_s2r_sentences: List[str]
    steps: List[StepEvaluation]
    overall_score: float        # 0.0–1.0 average of per-step scores


# ---------------------------------------------------------------------------
# 1. Identify S2R sentences
# ---------------------------------------------------------------------------

def identify_s2r_sentences(text: str) -> List[str]:
    """
    Very lightweight S2R sentence detection.

    Heuristics:
      - sentence contains a step verb (click, tap, open, ...)
      - OR looks like a numbered / bulleted step

    This is the "cheap substitute" for Euler's sequence labelling model.
    """
    doc = nlp(text)
    candidates: List[str] = []

    for sent in doc.sents:
        s = sent.text.strip()
        if not s:
            continue

        lower = s.lower()

        # obvious numbered/bulleted patterns: "1.", "2)", "- ", "* "
        if re.match(r"^\s*(\d+\.|-|\*|\d+\))", s):
            candidates.append(s)
            continue

        # if sentence contains a known step verb, treat as potential S2R
        if any(v in lower for v in STEP_VERBS):
            candidates.append(s)
            continue

    return candidates


# ---------------------------------------------------------------------------
# 2. Split S2R sentences into individual steps
# ---------------------------------------------------------------------------

def split_into_individual_steps(sentence: str) -> List[str]:
    """
    Split an S2R sentence into simpler, atomic steps.

    Heuristics:
      - split on ' and ', ' then ', ' afterwards ', commas in long sentences.
    """
    # First split on ' then ' / ' and ' to get rough segments
    parts = re.split(r"\b(?:then|and)\b", sentence, flags=re.IGNORECASE)
    steps: List[str] = []

    for part in parts:
        p = part.strip(" .,:;")
        if not p:
            continue
        # If still too long, optionally split on commas
        if len(p.split()) > 18:
            subparts = [sp.strip(" .,:;") for sp in p.split(",") if sp.strip()]
            steps.extend(subparts)
        else:
            steps.append(p)

    # Deduplicate small junk
    uniq = []
    for s in steps:
        if s and s not in uniq:
            uniq.append(s)
    return uniq


# ---------------------------------------------------------------------------
# 3. Evaluate one step (Euler-inspired, text-only)
# ---------------------------------------------------------------------------

def evaluate_single_step(step_text: str) -> StepEvaluation:
    """
    Rough textual evaluation of a single S2R step.

    Labels:
      - VM (Vocabulary Mismatch): no clear action verb found.
      - AS (Ambiguous Step): action exists but vague / missing object / vague words.
      - HQ (High Quality): has action, object-ish phrase, and some UI hint, no strong vagueness.
    """
    issues: List[str] = []
    text = step_text.strip()
    doc = nlp(text)

    # Detect action (verb) using spaCy POS tags + STEP_VERBS list
    action_tokens = [t for t in doc if t.pos_ == "VERB" and t.lemma_.lower() in STEP_VERBS]
    has_action = len(action_tokens) > 0

    # Any noun / pronoun / proper noun after the first verb -> treat as object-ish
    has_object = False
    if has_action:
        first_verb_idx = min(t.i for t in action_tokens)
        for t in doc:
            if t.i <= first_verb_idx:
                continue
            if t.pos_ in {"NOUN", "PROPN", "PRON"}:
                has_object = True
                break

    # UI-ish terms
    lower = text.lower()
    has_ui_term = any(u in lower for u in UI_TERMS)

    # Vague language
    has_vague_terms = any(v in lower for v in VAGUE_TERMS)

    # Imperative: sentence starting with verb (or "then"/"and" + verb)
    is_imperative = False
    if len(doc) > 0:
        first = doc[0]
        if first.pos_ == "VERB":
            is_imperative = True
        elif first.lower_ in {"then", "and"} and len(doc) > 1 and doc[1].pos_ == "VERB":
            is_imperative = True

    # Decide quality label
    if not has_action:
        quality_label = "VM"
        issues.append("No clear action verb (vocabulary mismatch).")
    else:
        # we have an action, but is step well-specified?
        if not has_object:
            issues.append("Action present but no clear object (what to act on).")
        if not has_ui_term:
            issues.append("No obvious UI target (button/field/menu/etc.) mentioned.")
        if has_vague_terms:
            issues.append("Contains vague terms (sometimes/maybe/etc.).")

        if has_object and has_ui_term and not has_vague_terms:
            quality_label = "HQ"
        else:
            quality_label = "AS"  # ambiguous / underspecified

    return StepEvaluation(
        step_text=text,
        has_action=has_action,
        has_object=has_object,
        has_ui_term=has_ui_term,
        is_imperative=is_imperative,
        has_vague_terms=has_vague_terms,
        quality_label=quality_label,
        issues=issues,
    )


# ---------------------------------------------------------------------------
# 4. Evaluate all S2Rs in a bug description
# ---------------------------------------------------------------------------

def evaluate_s2r_block(description_text: str) -> S2RBlockEvaluation:
    """
    High-level entry point:

      description_text -> S2R sentences -> individual steps -> evaluations.

    Returns a S2RBlockEvaluation you can:
      - dump to JSON
      - convert to features
      - join with your bug_features CSV.
    """
    s2r_sents = identify_s2r_sentences(description_text)
    all_steps: List[StepEvaluation] = []

    for sent in s2r_sents:
        steps = split_into_individual_steps(sent)
        for step in steps:
            ev = evaluate_single_step(step)
            all_steps.append(ev)

    # Simple overall score: HQ=1.0, AS=0.5, VM=0.0
    if all_steps:
        score_map = {"HQ": 1.0, "AS": 0.5, "VM": 0.0}
        overall = sum(score_map[s.quality_label] for s in all_steps) / len(all_steps)
    else:
        overall = 0.0

    return S2RBlockEvaluation(
        raw_s2r_sentences=s2r_sents,
        steps=all_steps,
        overall_score=overall,
    )


# ---------------------------------------------------------------------------
# Main: Process all Defects4J bug reports
# ---------------------------------------------------------------------------

def read_json(path):
    """Load JSON or JSONL; return a list of dicts."""
    with open(path, "r", encoding="utf-8") as f:
        text = f.read().strip()
        if not text:
            return []
        if "\n" in text and not text.lstrip().startswith("{"):
            return [json.loads(line) for line in text.splitlines() if line.strip()]
        obj = json.loads(text)
        return [obj] if isinstance(obj, dict) else obj


def process_bug_reports(data_dir: str = "bug_reports/Defects4J", output_file: str = "s2r_correctness_results.csv"):
    """
    Process all bug reports in Defects4J directory and save results to CSV.
    """
    json_files = glob.glob(os.path.join(data_dir, "**/*.json"), recursive=True)
    print(f"Found {len(json_files)} JSON files under {data_dir!r}")
    
    try:
        from tqdm import tqdm
        json_files = tqdm(json_files, desc="Processing bug reports")
    except ImportError:
        pass
    
    rows = []
    
    for path in json_files:
        try:
            # Parse bug report
            name = os.path.splitext(os.path.basename(path))[0]
            for obj in read_json(path):
                title = (obj.get("title") or obj.get("summary") or "").strip()
                desc = (obj.get("description") or "").strip()
                if not (title or desc):
                    continue
                bid = str(obj.get("id") or obj.get("bug_id") or name)
                
                # Evaluate S2R correctness
                result = evaluate_s2r_block(desc)
                
                # Extract features from result
                row = {
                    "id": bid,
                    "overall_score": result.overall_score,
                    "num_s2r_sentences": len(result.raw_s2r_sentences),
                    "num_steps": len(result.steps),
                    "num_hq_steps": sum(1 for s in result.steps if s.quality_label == "HQ"),
                    "num_as_steps": sum(1 for s in result.steps if s.quality_label == "AS"),
                    "num_vm_steps": sum(1 for s in result.steps if s.quality_label == "VM"),
                    "steps_with_action": sum(1 for s in result.steps if s.has_action),
                    "steps_with_object": sum(1 for s in result.steps if s.has_object),
                    "steps_with_ui_term": sum(1 for s in result.steps if s.has_ui_term),
                    "steps_imperative": sum(1 for s in result.steps if s.is_imperative),
                    "steps_with_vague_terms": sum(1 for s in result.steps if s.has_vague_terms),
                }
                
                # Add detailed step information as JSON strings (optional)
                row["step_details"] = json.dumps([asdict(s) for s in result.steps])
                row["s2r_sentences"] = json.dumps(result.raw_s2r_sentences)
                
                rows.append(row)
                
        except Exception as e:
            print(f"Error processing {path}: {e}")
            continue
    
    # Save to CSV
    df = pd.DataFrame(rows)
    df.to_csv(output_file, index=False)
    print(f"\nSaved {len(df)} bug report evaluations to {output_file}")
    
    # Print summary statistics
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    if len(df) > 0:
        print(f"  Total bug reports processed: {len(df)}")
        print(f"  Reports with S2R sentences: {(df['num_s2r_sentences'] > 0).sum()} ({(df['num_s2r_sentences'] > 0).mean()*100:.1f}%)")
        print(f"  Reports with steps: {(df['num_steps'] > 0).sum()} ({(df['num_steps'] > 0).mean()*100:.1f}%)")
        print(f"  Average overall score: {df['overall_score'].mean():.3f}")
        print(f"  Average number of steps per report: {df['num_steps'].mean():.2f}")
        print(f"  Average HQ steps: {df['num_hq_steps'].mean():.2f}")
        print(f"  Average AS steps: {df['num_as_steps'].mean():.2f}")
        print(f"  Average VM steps: {df['num_vm_steps'].mean():.2f}")
    print("="*80)


if __name__ == "__main__":
    process_bug_reports()
