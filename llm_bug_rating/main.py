from __future__ import annotations
import json
from typing import Dict, Any
import argparse
import os
from llm_bug_rating.data import io as data_io
from llm_bug_rating.judges import mock_judges
from llm_bug_rating.adjudication import adjudicator
from llm_bug_rating.scoring import aggregator
from llm_bug_rating.analysis import agreement


def load_config(root: str = None, checklist_file: str = "checklists.json") -> Dict[str, Any]:
    root = root or os.path.dirname(__file__)
    checklist_path = (
        checklist_file if os.path.isabs(checklist_file)
        else os.path.join(root, "config", checklist_file)
    )
    with open(checklist_path, "r", encoding="utf-8") as f:
        checklist = json.load(f)
    with open(os.path.join(root, "config/settings.json"), "r", encoding="utf-8") as f:
        settings = json.load(f)
    return {"checklist": checklist, "settings": settings}


def serialize_judge_out(jout: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    out = {}
    for dim, items in jout.items():
        out[dim] = {}
        for k, v in items.items():
            # ChecklistItemResult -> dict
            out[dim][k] = {"value": bool(v.value), "confidence": float(v.confidence), "reason": v.reason}
    return out


def process(input_path: str, out_dir: str, use_json: bool = False, checklist_file: str = "checklists.json"):
    pkg_root = os.path.dirname(__file__)
    cfg = load_config(pkg_root, checklist_file)
    checklist = cfg["checklist"]["dimensions"]
    settings = cfg["settings"]

    # Anchor relative output paths to the package root so they don't depend on cwd
    for key in ("json_dir", "csv_dir", "logs_dir"):
        p = settings["output"][key]
        if not os.path.isabs(p):
            settings["output"][key] = os.path.join(pkg_root, p)

    # --out overrides the base results directory, preserving the json/csv/logs substructure
    if out_dir and out_dir != "data/results":
        out_abs = out_dir if os.path.isabs(out_dir) else os.path.join(pkg_root, out_dir)
        settings["output"]["json_dir"] = os.path.join(out_abs, "json")
        settings["output"]["csv_dir"]  = os.path.join(out_abs, "csv")
        settings["output"]["logs_dir"] = os.path.join(out_abs, "logs")

    if use_json:
        rows = data_io.read_bug_json(input_path)
    else:
        rows = data_io.read_bug_csv(input_path)
    data_io.ensure_dirs([settings["output"]["json_dir"], settings["output"]["csv_dir"], settings["output"]["logs_dir"]])

    judge1 = mock_judges.judge_factory(settings["models"]["judge1"])
    judge2 = mock_judges.judge_factory(settings["models"]["judge2"])

    results = []
    per_bug_jsons = []

    total = len(rows)
    for i, r in enumerate(rows, 1):
        bug = {"bug_id": r.get("bug_id", ""), "project": r.get("project", ""), "summary": r.get("summary", ""), "description": r.get("description", "")}
        json_path = os.path.join(settings["output"]["json_dir"], f"{bug['bug_id']}.json")

        # Resume: if output already exists, load it and skip re-evaluation
        if os.path.exists(json_path):
            print(f"[{i}/{total}] {bug['bug_id']} (skipped — already processed)", flush=True)
            with open(json_path, encoding="utf-8") as _f:
                existing = json.load(_f)
            results.append({
                "bug_id":             existing["bug"]["bug_id"],
                "project":            existing["bug"]["project"],
                "percent_agreement":  existing["percent_agreement"],
                "cohens_kappa":       existing["cohens_kappa"],
                "aggregated_raw":     existing["aggregated"]["raw"],
                "aggregated_derived": existing["aggregated"]["derived"],
                "aggregated_zscore":  existing["aggregated"]["zscore"],
            })
            per_bug_jsons.append(json_path)
            continue

        print(f"[{i}/{total}] {bug['bug_id']}", flush=True)

        j1_raw = judge1.evaluate(bug, checklist)
        j2_raw = judge2.evaluate(bug, checklist)

        j1 = serialize_judge_out(j1_raw)
        j2 = serialize_judge_out(j2_raw)

        per_item = agreement.per_item_agreement(j1, j2)
        pct = agreement.percent_agreement(per_item)
        kappa = agreement.cohens_kappa(j1, j2)

        # adjudicate where disagreement
        adjudications = {}
        final_combined = {}
        for dim in j1:
            final_combined[dim] = {}
            adjudications[dim] = {}
            for item in j1[dim]:
                if per_item[dim][item]["agreement"]:
                    # keep either value
                    final_combined[dim][item] = {"value": j1[dim][item]["value"], "confidence": max(j1[dim][item]["confidence"], j2[dim][item]["confidence"]), "reason": "agreement"}
                    adjudications[dim][item] = {"selected": "agreement"}
                else:
                    adj = adjudicator.adjudicate_item(item, bug, j1[dim][item], j2[dim][item])
                    final_combined[dim][item] = {"value": adj.final_value, "confidence": max(j1[dim][item]["confidence"], j2[dim][item]["confidence"]), "reason": adj.rationale}
                    adjudications[dim][item] = {"selected": adj.selected, "rationale": adj.rationale}

        aggregated = aggregator.aggregate_scores(final_combined)

        out_record = {
            "bug_id": bug["bug_id"],
            "project": bug["project"],
            "percent_agreement": pct,
            "cohens_kappa": kappa,
            "aggregated_raw": aggregated["raw"],
            "aggregated_derived": aggregated["derived"],
            "aggregated_zscore": aggregated["zscore"],
        }

        results.append(out_record)

        bug_json = {
            "bug": bug,
            "judge1": j1,
            "judge2": j2,
            "per_item_agreement": per_item,
            "percent_agreement": pct,
            "cohens_kappa": kappa,
            "adjudication": adjudications,
            "final_checklist": final_combined,
            "aggregated": aggregated,
            "prompt_tracking": {"prompt_version": cfg["checklist"]["prompt_version"], "models": settings["models"]}
        }

        data_io.write_json(json_path, bug_json)
        per_bug_jsons.append(json_path)

    # write summary csv and json
    summary_csv = os.path.join(settings["output"]["csv_dir"], "aggregated_features.csv")
    data_io.write_csv(summary_csv, results)
    data_io.write_json(os.path.join(settings["output"]["json_dir"], "summary.json"), {"count": len(results), "rows": results})

    print(f"Processed {len(results)} bugs. Outputs in {settings['output']['json_dir']} and {settings['output']['csv_dir']}")


def main():
    parser = argparse.ArgumentParser(description="LLM-based checklist bug report rater (mocked)")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--csv", help="Input CSV with bug_id,project,summary,description")
    group.add_argument("--json", help="Input JSON array (defects4j_cleaned_issues.json format)")
    parser.add_argument("--out", required=False, default="data/results", help="Output base directory (overrides config paths; creates json/, csv/, logs/ inside)")
    parser.add_argument("--checklist", required=False, default="checklists.json", help="Checklist filename (relative to config/) or absolute path")
    args = parser.parse_args()
    if args.json:
        process(args.json, args.out, use_json=True, checklist_file=args.checklist)
    else:
        process(args.csv, args.out, use_json=False, checklist_file=args.checklist)


if __name__ == "__main__":
    main()
