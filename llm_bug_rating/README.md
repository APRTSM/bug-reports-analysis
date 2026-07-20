# llm_bug_rating

Checklist-based LLM assessment framework (prototype, mock judges).

Usage (example):

```bash
python -m llm_bug_rating.main --csv path/to/bugs.csv
```

Input CSV columns: `bug_id,project,summary,description`

Outputs are written under `data/results/json` and `data/results/csv` as configured in `config/settings.json`.

This repository uses deterministic mock judges. Replace with real judge implementations by creating a judge that follows the same interface in `llm_bug_rating.judges`.
