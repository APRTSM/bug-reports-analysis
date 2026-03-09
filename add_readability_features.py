"""
add_readability_features.py
----------------------------
Adds 7 readability scores (flesch, fog, lix, kincaid, ari, coleman_liau, smog)
to final_feature_set.csv by reading the raw bug report text from the same
defects4j_xml source files used by extract_bug_features.py.

Usage
-----
  python add_readability_features.py \
      --csv      final_feature_set.csv \
      --data_dir defects4j_xml \
      --output   final_feature_set.csv        # overwrites in-place (safe: backup is made)

Requirements
------------
  pip install textstat pandas
"""

import argparse
import glob
import json
import os
import re
import shutil
import xml.etree.ElementTree as ET

import pandas as pd
import textstat

# ---------------------------------------------------------------------------
# Config / CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--csv",      default="final_feature_set.csv",
                   help="Path to existing feature CSV")
    p.add_argument("--data_dir", default="defects4j_xml",
                   help="Directory containing .xml (or .json) bug files")
    p.add_argument("--output",   default="final_feature_set_readability.csv",
                   help="Output path (default: new file, not overwrite)")
    p.add_argument("--min_len",  type=int, default=100,
                   help="Min description length for reliable readability scores")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Text loading  (mirrors extract_bug_features.py logic exactly)
# ---------------------------------------------------------------------------

def read_xml(path):
    rows = []
    try:
        tree = ET.parse(path)
        root = tree.getroot()
    except ET.ParseError as e:
        print(f"  [WARN] Cannot parse {path}: {e}")
        return rows

    name = os.path.splitext(os.path.basename(path))[0]

    for bug_elem in root.iter("bug"):
        bug_id_raw = bug_elem.get("id", "")
        title = ""
        desc  = ""
        bug_info = bug_elem.find("buginformation")
        if bug_info is not None:
            s = bug_info.find("summary")
            d = bug_info.find("description")
            if s is not None and s.text:
                title = s.text.strip()
            if d is not None and d.text:
                desc = d.text.strip()

        # Reconstruct canonical id (Project-N)
        if "_" in name:
            project, num = name.rsplit("_", 1)
            bug_id = f"{project}-{num}"
        elif bug_id_raw:
            bug_id = f"{name}-{bug_id_raw}"
        else:
            bug_id = name

        rows.append({"id": bug_id, "summary": title, "description": desc})
    return rows


def read_json(path):
    rows = []
    name = os.path.splitext(os.path.basename(path))[0]
    try:
        with open(path, encoding="utf-8") as f:
            text = f.read().strip()
        if not text:
            return rows
        if "\n" in text and not text.lstrip().startswith("{"):
            objs = [json.loads(l) for l in text.splitlines() if l.strip()]
        else:
            obj = json.loads(text)
            objs = obj if isinstance(obj, list) else [obj]
    except Exception as e:
        print(f"  [WARN] Cannot parse {path}: {e}")
        return rows

    for obj in objs:
        title = (obj.get("title") or obj.get("summary") or "").strip()
        desc  = (obj.get("description") or "").strip()
        bid   = str(obj.get("id") or obj.get("bug_id") or name)
        rows.append({"id": bid, "summary": title, "description": desc})
    return rows


def load_all_bugs(data_dir):
    xml_files  = glob.glob(os.path.join(data_dir, "*.xml"))
    json_files = glob.glob(os.path.join(data_dir, "**/*.json"), recursive=True)

    bugs = {}
    for f in xml_files:
        for row in read_xml(f):
            bugs[row["id"]] = row
    for f in json_files:
        for row in read_json(f):
            if row["id"] not in bugs:          # XML takes precedence
                bugs[row["id"]] = row

    print(f"  Loaded {len(bugs)} bug records from {data_dir!r}")
    return bugs


# ---------------------------------------------------------------------------
# Readability computation
# ---------------------------------------------------------------------------

READABILITY_COLS = [
    "flesch", "fog", "lix", "kincaid", "ari", "coleman_liau", "smog"
]


def compute_readability(text, min_len):
    """Return dict of 7 readability scores; NaN if text is too short."""
    if not text or len(text) < min_len:
        return {c: float("nan") for c in READABILITY_COLS}

    return {
        "flesch":       textstat.flesch_reading_ease(text),
        "fog":          textstat.gunning_fog(text),
        "lix":          textstat.lix(text),
        "kincaid":      textstat.flesch_kincaid_grade(text),
        "ari":          textstat.automated_readability_index(text),
        "coleman_liau": textstat.coleman_liau_index(text),
        "smog":         textstat.smog_index(text),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    # ---- Load CSV ----
    df = pd.read_csv(args.csv)
    print(f"Loaded CSV: {len(df)} rows, columns: {list(df.columns[:5])} ...")

    # ---- Load raw bug texts ----
    bugs = load_all_bugs(args.data_dir)

    # ---- Compute readability per row ----
    records = []
    missing = 0
    for _, row in df.iterrows():
        bug_id = str(row["id"])
        bug    = bugs.get(bug_id)

        if bug is None:
            missing += 1
            records.append({c: float("nan") for c in READABILITY_COLS})
            continue

        text = (bug["summary"] + " " + bug["description"]).strip()
        records.append(compute_readability(text, args.min_len))

    if missing:
        print(f"  [WARN] {missing} bug IDs not found in data_dir "
              f"(readability will be NaN for those rows)")

    # ---- Merge into dataframe ----
    # Drop columns if they already exist (safe re-run)
    df.drop(columns=[c for c in READABILITY_COLS if c in df.columns],
            inplace=True, errors="ignore")

    readability_df = pd.DataFrame(records, index=df.index)
    df = pd.concat([df, readability_df], axis=1)

    # ---- Save ----
    # Backup original if writing to same path
    if os.path.abspath(args.output) == os.path.abspath(args.csv):
        backup = args.csv + ".bak"
        shutil.copy(args.csv, backup)
        print(f"  Backup saved to {backup}")

    df.to_csv(args.output, index=False)
    print(f"\nDone. Output written to: {args.output}")
    print(f"New columns added: {READABILITY_COLS}")

    # Quick sanity check
    non_nan = readability_df["flesch"].notna().sum()
    print(f"Readability computed for {non_nan}/{len(df)} bugs "
          f"(rest NaN — text too short or ID not found)")


if __name__ == "__main__":
    main()