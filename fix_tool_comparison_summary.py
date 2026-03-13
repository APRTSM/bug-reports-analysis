"""
Fix and filter `tool_comparison_summary.csv`:

1. **Fix BRaIn bug_id values**:
   - Rows for tool == "BRaIn" currently have `bug_id` like "Time-4" or "Mockito-32".
   - We normalize these to the plain numeric ID (e.g., 4, 32) while keeping the `project`
     column as-is ("Time", "Mockito").

2. **Filter rows to only bugs that exist in `defects4j_xml`**:
   - We derive the set of valid (project, bug_id) pairs from filenames in `defects4j_xml`,
     which have the form `<Project>_<n>.xml` (e.g., "Time_4.xml", "Mockito_32.xml").
   - Any row whose (project, bug_id) pair is not present in this set is removed.

The script makes a one-time in-place update to `tool_comparison_summary.csv`, after first
writing a backup copy `tool_comparison_summary_original.csv` in the same directory.
"""

import csv
import re
from pathlib import Path

BASE_DIR = Path(".")
SUMMARY_FILE = BASE_DIR / "tool_comparison_summary.csv"
BACKUP_FILE = BASE_DIR / "tool_comparison_summary_original.csv"
DEFECTS4J_XML_DIR = BASE_DIR / "defects4j_xml"


def load_valid_bug_ids():
    """
    Scan `defects4j_xml` and return a set of (project, bug_id_int) pairs
    derived from filenames like 'Time_4.xml', 'Mockito_32.xml'.
    """
    valid = set()

    if not DEFECTS4J_XML_DIR.exists():
        raise FileNotFoundError(f"{DEFECTS4J_XML_DIR} does not exist")

    for path in DEFECTS4J_XML_DIR.iterdir():
        if not path.name.endswith(".xml"):
            continue

        # Expect pattern <Project>_<n>.xml
        m = re.match(r"^([A-Za-z0-9]+)_([0-9]+)\.xml$", path.name)
        if not m:
            # Skip any unexpected names silently
            continue

        project = m.group(1)
        bug_num = int(m.group(2))
        valid.add((project, bug_num))

    return valid


def normalize_brain_bug_id(project: str, bug_id: str) -> str:
    """
    For BRaIn rows, bug_id is sometimes like 'Time-4' or 'Mockito-32'.
    If bug_id matches '<Project>-<n>' (case-sensitive project match),
    return just the numeric part as a string. Otherwise, return unchanged.
    """
    if not bug_id:
        return bug_id

    # e.g., project='Time', bug_id='Time-4'
    pattern = rf"^{re.escape(project)}-([0-9]+)$"
    m = re.match(pattern, bug_id)
    if m:
        return m.group(1)

    return bug_id


def main():
    if not SUMMARY_FILE.exists():
        raise FileNotFoundError(f"{SUMMARY_FILE} not found")

    print("Loading valid (project, bug_id) pairs from defects4j_xml/ ...")
    valid_pairs = load_valid_bug_ids()
    print(f"  Found {len(valid_pairs)} unique bugs in defects4j_xml")

    # Read original CSV
    print(f"\nReading {SUMMARY_FILE} ...")
    with SUMMARY_FILE.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = reader.fieldnames or []

    print(f"  Loaded {len(rows)} rows")

    # Back up original file
    if not BACKUP_FILE.exists():
        BACKUP_FILE.write_bytes(SUMMARY_FILE.read_bytes())
        print(f"\nBackup written to {BACKUP_FILE}")
    else:
        print(f"\nBackup already exists at {BACKUP_FILE} (not overwritten)")

    fixed_rows = []
    dropped_missing_xml = 0
    fixed_brain_ids = 0
    dropped_bad_bugid = 0

    for row in rows:
        project = row.get("project", "")
        bug_id_raw = row.get("bug_id", "")
        tool = row.get("tool", "")

        # Normalize BRaIn bug_id if needed
        if tool == "BRaIn":
            new_bug_id = normalize_brain_bug_id(project, bug_id_raw)
            if new_bug_id != bug_id_raw:
                fixed_brain_ids += 1
                row["bug_id"] = new_bug_id
                bug_id_raw = new_bug_id

        # Parse bug_id as int
        try:
            bug_num = int(str(bug_id_raw))
        except (TypeError, ValueError):
            dropped_bad_bugid += 1
            continue

        # Keep only if this bug exists in defects4j_xml
        if (project, bug_num) not in valid_pairs:
            dropped_missing_xml += 1
            continue

        fixed_rows.append(row)

    print("\nSummary of changes:")
    print(f"  BRaIn bug_id values normalized: {fixed_brain_ids}")
    print(f"  Rows dropped due to invalid/non-numeric bug_id: {dropped_bad_bugid}")
    print(f"  Rows dropped because bug not in defects4j_xml: {dropped_missing_xml}")
    print(f"  Rows kept: {len(fixed_rows)}")

    # Write updated CSV in-place
    print(f"\nWriting updated CSV to {SUMMARY_FILE} ...")
    with SUMMARY_FILE.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(fixed_rows)

    print("Done.")


if __name__ == "__main__":
    main()

