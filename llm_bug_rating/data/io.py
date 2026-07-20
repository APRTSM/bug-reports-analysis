from __future__ import annotations
from typing import List, Dict, Any
import csv
import os
import json


def read_bug_json(path: str) -> List[Dict[str, str]]:
    with open(path, encoding="utf-8") as f:
        records = json.load(f)
    rows = []
    for r in records:
        issue_id = r.get("issue_id", "")
        project = issue_id.split("-")[0] if "-" in issue_id else ""
        rows.append({
            "bug_id": issue_id,
            "project": project,
            "summary": r.get("title", ""),
            "description": r.get("description", "") or "",
        })
    return rows


def read_bug_csv(path: str) -> List[Dict[str, str]]:
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append({k: (v or "") for k, v in r.items()})
    return rows


def ensure_dirs(paths: List[str]) -> None:
    for p in paths:
        os.makedirs(p, exist_ok=True)


def write_json(path: str, obj: Any) -> None:
    ensure_dirs([os.path.dirname(path)])
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def write_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    ensure_dirs([os.path.dirname(path)])
    if not rows:
        return
    keys = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: json.dumps(v) if isinstance(v, (dict, list)) else v for k, v in r.items()})
