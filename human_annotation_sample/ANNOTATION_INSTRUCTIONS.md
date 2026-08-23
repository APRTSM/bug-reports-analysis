# Human Annotation Instructions

You are reviewing a sample of 87 bug reports (randomly drawn from a population of 835, seed=42, matching Cochran's sample size for 95% confidence / 10% margin of error) using the identical checklist given to the two LLM judges (GPT-5.4-mini, Claude Haiku) that rated the full dataset. This sample is for validating LLM-vs-human agreement on the same checklist.

## Task (same framing given to the LLM judges)

> You are given a bug report (summary and description) and a checklist of binary items. For each checklist item, judge **TRUE** or **FALSE** based on the report text alone -- do not guess at anything not stated or reasonably inferable from the report. Optionally record a confidence (0.0-1.0) and a short note justifying your judgment.

## Files

- `bug_reports_for_annotation.csv` -- the 87 sampled bug reports (bug_id, project, title, description)
- `annotation_checklist_template.csv` -- one row per (bug, checklist item); fill in `human_value` (TRUE/FALSE), and optionally `human_confidence` and `human_notes`, for each row
- `sampled_bug_ids.txt` -- plain list of the 87 sampled bug IDs

For each bug, read the title + description in `bug_reports_for_annotation.csv`, then go through all 48 checklist items below for that `bug_id` in `annotation_checklist_template.csv`.

## Checklist (9 dimensions, 48 items -- identical to `llm_bug_rating/config/checklists_v2.json`)

### Actionability

| Code | Item |
|---|---|
| A1 | The report describes observable faulty behavior with enough detail to distinguish it from normal or expected behavior. |
| A2 | The affected class, method, API, or code path is specifically identified — not just a general feature area or subsystem. |
| A3 | The report includes concrete debugging artifacts (stack traces, error messages, code snippets, or logs) that directly implicate the fault location. |
| A4 | A developer could immediately attempt to reproduce or investigate the bug using only the information in this report, without needing clarification. |
| A5 | The report pinpoints a specific module, class, or component precisely enough that a developer would know where to start reading code. |

### Clarity

| Code | Item |
|---|---|
| C1 | The report is written with enough precision that a developer unfamiliar with the codebase could understand the failure scenario without guessing. |
| C2 | The problem is described precisely enough that a developer could write a failing test case without asking for clarification. |
| C3 | The sequence of events or conditions leading to the failure is presented unambiguously — a reader cannot construct multiple different interpretations. |
| C4 | All referenced entities, components, or behaviors are identified precisely enough that a developer could locate them in the codebase without prior knowledge of this specific bug. |
| C5 | Technical terminology is used accurately and consistently such that a developer would not be misled about the nature or location of the fault. |

### Specificity

| Code | Item |
|---|---|
| S1 | Specific inputs, conditions, or triggering events are described concretely enough to be directly used in a test case. |
| S2 | Concrete, ordered reproduction steps are provided that could be followed without prior knowledge of the system. |
| S3 | Relevant environment or version information is included and specific enough to rule out version-related ambiguity. |
| S4 | Specific outputs, failures, exceptions, or error messages are reported verbatim or with enough detail to identify the exact failure mode. |
| S5 | The report contains details that uniquely identify this issue and distinguish it from superficially similar bugs. |

### Reproducibility

| Code | Item |
|---|---|
| R1 | The expected behavior is explicitly stated in a way that serves as a concrete correctness criterion. |
| R2 | The observed behavior is explicitly stated with enough detail to recognize the failure when encountered. |
| R3 | The discrepancy between expected and observed behavior is unambiguously identifiable from the report alone. |
| R4 | The reported reproduction steps are complete and sufficient to attempt reproduction without additional investigation. |
| R5 | Required environmental conditions or dependencies are documented specifically enough to set up the reproduction environment. |

### ReasoningQuality

| Code | Item |
|---|---|
| Q1 | The report explicitly proposes a specific potential cause or explanation — not just a symptom description. |
| Q2 | The proposed explanation is directly and specifically linked to observed symptoms with a plausible causal mechanism. |
| Q3 | The report provides concrete evidence (code references, traces, logs, or data) that supports the proposed explanation. |
| Q4 | The explanation is internally coherent and logically consistent — it does not contradict other information in the report. |
| Q5 | The explanation references specific technical artifacts (exact log lines, specific method names, concrete code locations, or configuration values). |

### TechnicalContext

| Code | Item |
|---|---|
| T1 | Specific classes, methods, files, or APIs are named — not just described in general terms. |
| T2 | The affected subsystem or module is identifiable precisely enough to locate the relevant source files. |
| T3 | Technical artifacts such as stack traces, logs, screenshots, or code snippets are included and directly relevant to the failure. |
| T4 | External dependencies, libraries, services, or integrations are referenced by name and version when they are relevant to the bug. |
| T5 | Environment details (OS, browser, framework version, hardware, runtime, etc.) are provided with enough specificity to reconstruct the affected environment. |

### Ambiguity

| Code | Item |
|---|---|
| M1 | The report contains vague or underspecified references that a developer would need to resolve before investigating. |
| M2 | Inputs, parameters, or conditions are insufficiently described to attempt reproduction. |
| M3 | Expected behavior is unclear or underspecified to the point that a developer could not determine what the correct behavior should be. |
| M4 | Environment or version information that would materially affect reproduction is missing. |
| M5 | The report contains contradictory or inconsistent statements that would mislead a developer. |

### RepairReadiness

| Code | Item |
|---|---|
| P1 | The faulty behavior is described precisely enough to determine what the correct behavior should be and what code change would achieve it. |
| P2 | The specific component, class, method, or code location requiring a fix is identifiable directly from the report. |
| P3 | The expected correct behavior is stated concretely enough to serve as an acceptance criterion for the fix. |
| P4 | Relevant constraints, edge cases, or conditions that would affect the fix are documented explicitly. |
| P5 | A developer could implement a plausible fix using only the information in this report, without further investigation or clarification. |

### RootCauseEvidence

| Code | Item |
|---|---|
| RC1 | A specific potential root cause is explicitly proposed in the report — not merely suggested or loosely implied. |
| RC2 | The proposed root cause references a specific code construct, operation, API call, data condition, or logic error by name. |
| RC3 | The causal chain from the root cause to the observed failure is articulated clearly enough that a developer could verify it. |

### HiddenReproducibility

| Code | Item |
|---|---|
| H1 | The report describes a specific, concrete scenario from which complete and unambiguous reproduction steps can be directly derived — the steps must be fully determinable, not merely suggested or partially implied. |

### ImpactScope

| Code | Item |
|---|---|
| I1 | A specific subsystem, module, class, or code concept likely affected by the bug is identifiable from the report — not just a broad functional area. |
| I2 | The report distinguishes between where the symptom manifests and where the underlying defect likely resides. |
| I3 | Multiple impacted areas or cross-cutting concerns are identified with enough specificity to understand the scope of a fix. |
| I4 | The report names three or more distinct technical concepts, classes, methods, or components that are relevant to understanding or fixing the bug. |
