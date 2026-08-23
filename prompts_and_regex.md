# Replication Package — Prompts, Categories, and Regex Patterns

---

## Regex Patterns (`extract_bug_features.py`)

All patterns are applied to the concatenated bug report text (`summary + " " + description`) unless noted otherwise.

### XML Preprocessing

| Purpose | Pattern | Flags |
|---|---|---|
| Strip non-printable control characters before XML parsing | `[\x00-\x08\x0B-\x0C\x0E-\x1F]` | — |

### Structural Flags

#### `has_stacktrace`
Detects the presence of a Java/JVM stack trace or exception block.

```
(Exception|Error|Throwable|
 at\s+[a-zA-Z0-9_.]+\([^)]*\)|
 ^\s*at\s+|
 Caused\s+by:|
 Stack\s+trace:)
```
Flags: `MULTILINE | IGNORECASE`

#### `has_code`
Detects inline code blocks or programming language syntax.

```
```|
(?:public|private|protected|static)\s+(?:class|void|int|String)|
(?:^|\n)(?:def|class|import|from\s+\w+\s+import)\s+|
(?:^|\n)\s*function\s+\w+\s*\(|
#include\s*<|
(?:^|\n)\s{4,}\w+.*[;{]
```
Flags: `MULTILINE`

#### `has_patch`
Detects unified diff / patch content.

```
diff --git|\.patch\b|commit [0-9a-f]{6,40}\b|
^\+{3}\s+|^-{3}\s+|
^@@.*@@
```
Flags: `MULTILINE`

#### `has_enumeration`
Detects numbered or bulleted lists.

```
(?:^|\n)\s*(?:\d+[\.\)]|[-*])\s+[A-Z\w]
```
Flags: `MULTILINE`

---

### Ambiguity Score

Vague phrase patterns matched case-insensitively on lowercased text:

| Pattern | Matches |
|---|---|
| `i\s+think` | "i think" |
| `not\s+sure` | "not sure" |
| `may\s+be` | "may be" |
| `could\s+be` | "could be" |

Token extraction (for vague term counting): `\b[a-zA-Z]+\b`

---

### Exception / Stack Trace Semantics

| Purpose | Pattern | Flags |
|---|---|---|
| Extract exception type names | `\b([A-Za-z_][A-Za-z0-9_]*(?:Exception\|Error\|Throwable))\b` | — |
| Count stack frame lines | `^\s*at\s+.+` | `MULTILINE` |
| Count `Caused by:` chains | `Caused by:` | — |
| Extract frame package names (library vs. user) | `at\s+([a-zA-Z0-9_.]+)\.` | — |

---

### Context Features

| Purpose | Pattern |
|---|---|
| Version strings (e.g., `1.2.3`, `v2.0`) | `\b(v?\d+\.\d+(?:\.\d+)*(?:_\d+)?)\b` |

---

### Causal and Temporal Markers

Single-word markers are matched with word boundaries (`\b<word>\b`); multi-word phrases use `re.escape` with exact string matching on lowercased text.

**Causal markers:** `because`, `since`, `so that`, `due to`, `therefore`, `hence`, `thus`, `consequently`, `as a result`, `in order to`, `if`, `when`, `whenever`, `unless`, `until`, `after`, `before`, `once`, `while`

**Temporal markers:** `then`, `next`, `afterwards`, `later`, `finally`, `subsequently`, `eventually`, `initially`, `first`, `second`, `third`

---

### Sentence Boundary Approximation

Used as a fallback when spaCy sentence count is unavailable:

```
[.!?]+
```
Applied via `re.split`; result length minus 1 gives the approximate sentence count.

---

### Completeness Score — Steps-to-Reproduce (S2R) Fallback

Used when `bee_results.jsonl` is not available. Any match on lowercased text counts as evidence of S2R:

| Pattern | Example match |
|---|---|
| `step\s+\d+` | "step 1", "step 2" |
| `step\s+[1-9]` | "step 3" |
| `reproduce` | "reproduce the bug" |
| `reproduction` | "reproduction case" |
| `steps:` | "Steps:" |
| `steps\s+to` | "steps to reproduce" |
| `how\s+to\s+reproduce` | "how to reproduce" |
| `reproduction\s+steps` | "reproduction steps" |

### Completeness Score — Expected/Observed Behavior Fallback

**Expected behavior patterns** (any match → `has_expected = True`):

| Pattern | Example match |
|---|---|
| `expected` | "expected output" |
| `should\s+(?:be\|have\|do\|show\|display)` | "should be null" |
| `but\s+(?:got\|received\|observed\|actual)` | "but got an error" |
| `actual\s+(?:result\|output\|behavior)` | "actual result is" |
| `instead\s+(?:of\|I\|we)` | "instead of returning" |

**Observed behavior patterns** (any match → `has_observed = True`):

Literal keyword scan (no regex) on lowercased text: `observed`, `actual`, `got`, `received`, `instead`

`has_expected_observed = has_expected AND has_observed` → contributes +1.0 to completeness score.


---

## LLM Bug Rating Pipeline (`llm_bug_rating/`)

### Overview

This is the **current source** of every rating-related column in `final_feature_set_bug_reports.csv` (`percent_agreement`, `cohens_kappa`, the 11 raw dimension scores, their `z_*` counterparts, and the derived composites).

This pipeline evaluates bug reports against a fixed **checklist of binary (true/false) items**, grouped into 9 quality dimensions, using two independent LLM judges plus an adjudication step for disagreements.

- **Judge 1:** GPT-5.4-mini (`gpt-5.4-mini-2026-03-17`, temperature 0.1)
- **Judge 2:** Claude Haiku (`claude-haiku-4-5-20251001`, temperature 0.1)
- **Adjudicator (configured):** Gemini 3.1 Flash Lite (`gemini-3.1-flash-lite`) — see the caveat under [Adjudication](#adjudication) below; the adjudication logic actually invoked in `main.py` is a deterministic confidence-comparison fallback, not an LLM call.
- Config: `llm_bug_rating/config/settings.json`. Checklist: `llm_bug_rating/config/checklists_v2.json` (`prompt_version: "v2"`).

### Prompt (`llm_bug_rating/prompts/checklist_prompt.txt`)

```
You are given a bug report (summary and description) and a checklist of binary items.
For each checklist item, return a JSON object with the fields:
 - value: boolean (true/false)
 - confidence: float between 0.0 and 1.0
 - reason: concise explanation for the judgment

Constraints:
- Use deterministic behavior (temperature=0.1).
- Output valid JSON only (no surrounding commentary).
- Use short, factual reasons referencing the report.

Checklist:
{checklist}

Bug Report:
Summary:
{summary}

Description:
{description}

Produce a JSON mapping from dimension -> item -> {"value", "confidence", "reason"}.
```

Both judges receive the identical prompt (same template, same checklist, same bug text) — only the underlying model differs. `{checklist}` is the full dimension → item → description listing below, formatted as plain text (`Dimension:\n  CODE: description text`).

### Checklist (`llm_bug_rating/config/checklists_v2.json`, v2)

9 dimensions, 48 binary items total (5+5+5+5+5+5+5+5 = 40 across the eight 5-item dimensions, +3 RootCauseEvidence, +1 HiddenReproducibility, +4 ImpactScope). Each dimension's raw score (the CSV's `actionability`, `clarity`, etc.) is the **count of items judged `true`** (0–5 for most dimensions, 0–3 for `RootCauseEvidence`, 0–1 for `HiddenReproducibility`, 0–4 for `ImpactScope`).

| Dimension → CSV column | Items |
|---|---|
| **Actionability** → `actionability` | A1: observable faulty behavior with enough detail to distinguish from expected behavior · A2: affected class/method/API/code path specifically identified · A3: concrete debugging artifacts (traces, errors, snippets, logs) implicating the fault location · A4: developer could immediately attempt to reproduce/investigate without clarification · A5: pinpoints a specific module/class/component precisely enough to know where to start reading code |
| **Clarity** → `clarity` | C1: precise enough for an unfamiliar developer to understand without guessing · C2: precise enough to write a failing test case without clarification · C3: sequence of events/conditions presented unambiguously (no multiple interpretations) · C4: referenced entities identified precisely enough to locate in the codebase · C5: technical terminology used accurately/consistently, not misleading about nature or location of the fault |
| **Specificity** → `specificity` | S1: specific inputs/conditions/triggers concrete enough for a test case · S2: concrete, ordered reproduction steps · S3: environment/version info specific enough to rule out version ambiguity · S4: specific outputs/failures/exceptions/error messages verbatim or detailed enough to identify the exact failure mode · S5: details uniquely distinguish this issue from superficially similar bugs |
| **Reproducibility** → `reproducibility` | R1: expected behavior explicitly stated as a concrete correctness criterion · R2: observed behavior explicitly stated with enough detail to recognize the failure · R3: discrepancy between expected/observed unambiguously identifiable · R4: reproduction steps complete and sufficient without additional investigation · R5: required environmental conditions/dependencies documented specifically enough to set up reproduction |
| **ReasoningQuality** → `reasoning_quality` | Q1: explicitly proposes a specific potential cause (not just symptoms) · Q2: proposed explanation directly/specifically linked to symptoms with a plausible mechanism · Q3: concrete evidence (code refs, traces, logs, data) supporting the explanation · Q4: internally coherent and logically consistent · Q5: references specific technical artifacts (exact log lines, method names, code locations, config values) |
| **TechnicalContext** → `technical_context` | T1: specific classes/methods/files/APIs named, not just general terms · T2: affected subsystem/module identifiable enough to locate relevant source files · T3: technical artifacts (traces, logs, screenshots, snippets) included and directly relevant · T4: external dependencies/libraries/services referenced by name+version when relevant · T5: environment details (OS, browser, framework version, hardware, runtime) specific enough to reconstruct the environment |
| **Ambiguity** → `ambiguity` (inverted for `z_ambiguity` — see below) | M1: vague/underspecified references needing resolution before investigating · M2: inputs/parameters/conditions insufficiently described to attempt reproduction · M3: expected behavior unclear/underspecified · M4: environment/version info materially affecting reproduction is missing · M5: contradictory or inconsistent statements that would mislead a developer |
| **RepairReadiness** → `repair_readiness` (also feeds `repair_difficulty`, `repair_feasibility`, `p5_fix_feasible`) | P1: faulty behavior precise enough to determine correct behavior and what change would achieve it · P2: specific component/class/method/location requiring a fix identifiable directly · P3: expected correct behavior concrete enough to serve as an acceptance criterion · P4: relevant constraints/edge cases/conditions affecting the fix documented explicitly · P5: developer could implement a plausible fix using only this report ("most direct APR signal by design" per `update_feature_set_with_llm_ratings.py`) |
| **RootCauseEvidence** → `root_cause_evidence` | RC1: specific potential root cause explicitly proposed, not merely implied · RC2: proposed root cause references a specific code construct/operation/API/data condition/logic error by name · RC3: causal chain from root cause to observed failure articulated clearly enough to verify |
| **HiddenReproducibility** → `hidden_reproducibility` | H1: a specific, concrete scenario from which complete and unambiguous reproduction steps can be *directly derived* — must be fully determinable, not merely suggested |
| **ImpactScope** → `impact_scope` | I1: a specific subsystem/module/class/code concept likely affected is identifiable, not just a broad functional area · I2: distinguishes where the symptom manifests from where the underlying defect likely resides · I3: multiple impacted areas/cross-cutting concerns identified with enough specificity to understand fix scope · I4: names three or more distinct technical concepts/classes/methods/components relevant to understanding or fixing the bug |

(`llm_bug_rating/config/checklists.json`, the v1 checklist, exists alongside `checklists_v2.json` but is superseded — `settings.json` pins `prompt_version: "v2"`.)

### Aggregation (`llm_bug_rating/scoring/aggregator.py`)

For each bug, per dimension:

- **`raw`** — count of items judged `true` (this is the CSV's raw column, e.g. `actionability`, `clarity`).
- **`derived`** — semantically relabelled scores:
  - `repair_difficulty` = (max items in RepairReadiness) − raw RepairReadiness count (inverted: more repair-ready ⇒ lower difficulty)
  - `ambiguity_count` = raw Ambiguity count
  - `hidden_s2r_present` = `True` iff the single HiddenReproducibility item is true
  - `root_cause_evidence`, `impact_scope` = raw counts (pass-through)
- **`zscore`** — ⚠️ **not a per-feature, cross-bug standardization.** For each bug independently, the 9 dimension raw counts (with `Ambiguity` sign-flipped so "higher = better" on every axis) are treated as a 9-value sample, and each dimension's z-score is computed *relative to that same bug's other 8 dimension scores* (`(value − mean_of_this_bugs_9_dimensions) / stdev_of_this_bugs_9_dimensions`). It answers "did this report score unusually well/poorly on dimension X *relative to how it scored on its own other dimensions*", not "how does this report's dimension-X score compare to other bug reports' dimension-X scores." This is why `z_actionability` correlates only ~0.3–0.7 (not ~1.0) with raw `actionability` in the final data — confirmed empirically during the 2026-07 redundancy analysis. Keep this in mind when interpreting or writing up the `z_*` columns.

Item-level composites (`causal_reasoning_score`, `repair_feasibility`, `repro_capability`, `p5_fix_feasible`) are computed separately in `update_feature_set_with_llm_ratings.py` directly from specific checklist items in the per-bug JSON (`final_checklist`), not from the aggregator's dimension sums — see that script's docstring for the exact item selections and rationale (chosen for low ceiling/saturation rate).

### Agreement (`llm_bug_rating/analysis/agreement.py`)

Computed per bug across all 48 checklist items, judge1 vs. judge2:

- **`percent_agreement`** — fraction of the 48 items where both judges gave the same boolean value.
- **`cohens_kappa`** — Cohen's κ treating each of the 48 items as one binary observation (standard 2×2 formula, corrected for chance agreement from each judge's marginal "true" rate).

### Adjudication (`llm_bug_rating/adjudication/adjudicator.py`)

Only invoked for items where judge1 and judge2 disagree (`main.py` skips adjudication entirely on agreement, keeping either judge's value).

⚠️ **Discrepancy worth knowing:** `config/settings.json` configures an `adjudicator` model (Gemini 3.1 Flash Lite), but `adjudicator.py`'s own docstring calls it "a deterministic fallback adjudicator... In production, this should call an adjudicator LLM" — and `main.py` calls exactly this deterministic function, not a Gemini API call. The actual adjudication rule used to produce the current data is:

- If judges' confidences differ by < 0.05: take the **OR** of both judges' boolean values ("close confidences; merged truth by OR").
- Otherwise: take the value from whichever judge reported **higher confidence**.

So despite the settings file listing a third model, the pipeline that produced `final_feature_set_bug_reports.csv`'s ratings is effectively **2-judge with a rule-based tiebreaker**, not a true 3-model adjudicated ensemble. Worth correcting in any methodology write-up.

### Input / invocation

Reads bug reports via `--csv bug_id,project,summary,description` or `--json` (array of `{issue_id, title, description}`, e.g. `apr-tool-comparison/defects4j/defects4j_cleaned_issues.json` — see the note on that file's known Mockito data-quality issue, fixed in `defects4j_xml/` this session but not necessarily in that external copy). Resumable: if a bug's output JSON already exists in `data/results_v2/json/`, it's skipped and the cached result reused.

## Script 1: `gemini_bug_categorization_overall.py`

### Purpose

Performs coarse-grained categorization of all bug reports. Each report is assigned to exactly one of 9 categories.

### Prompt

```
You are an expert software engineer categorizing bug reports. Please analyze the following bug report and categorize it into ONE of the 9 predefined categories.

Bug Report Title: {title}

Bug Report Description:
{description}

Bug Report Categories:
1. Configuration Issue: {description}
2. Network Issue: {description}
...
9. Test Code-Related Issue: {description}

Your Output:
Return ONLY a JSON object with EXACTLY the following three fields:
{
  "category": "<one of the 9 categories above, exact string>",
  "confidence": <integer 1-5>,
  "reasoning": "<max 2 sentences explaining the choice>"
}

Strict Rules:
- Output MUST be valid JSON.
- NO markdown. NO backticks. NO extra text before or after the JSON.
- The "category" value MUST match EXACTLY one of the categories listed above.
- Choose the MOST RELEVANT category even if multiple could apply.
- If the bug is ambiguous, select the category that best fits the core issue and reflect uncertainty using a lower confidence score (1–2).
- Do NOT use any category as a default or fallback.
- Do NOT add additional fields, comments, or explanations.
```

### Output Fields

| Field | Type | Description |
|---|---|---|
| `id` | string | Bug report identifier (e.g., `Lang-19`) |
| `title` | string | Bug report summary/title |
| `description_length` | integer | Character length of the description |
| `category` | string | Assigned coarse-grained category (one of 9) |
| `confidence` | integer 1–5 | Model's confidence in the assignment |
| `reasoning` | string | Up to 2 sentences explaining the choice |

### Categories

| # | Category | Definition |
|---|---|---|
| 1 | Configuration Issue | Bugs related to build configuration files, incorrect directory or file paths in XML or manifest artifacts, and external libraries needing updates. |
| 2 | Network Issue | Bugs caused by connection or server problems, unexpected server shutdowns, or improper use of communication protocols. |
| 3 | Database-Related Issue | Problems involving the connection between the main application and a database. |
| 4 | GUI-Related Issue | Stylistic errors (layouts, colors, padding, buttons) and unexpected failures appearing to users as unusual error messages. |
| 5 | Performance Issue | Issues related to memory overuse, energy leaks, and methods causing endless loops. |
| 6 | Permission/Deprecation Issue | Issues involving deprecated method calls or APIs, unused API permissions, and unexpected behavior when external API methods are called. |
| 7 | Security Issue | Vulnerability and security-related problems, including the need to reload parameters or remove unused permissions. |
| 8 | Functional Issue | Bugs in core application logic causing unexpected behavior, exceptions, problems with return values, and unexpected crashes due to logic errors. |
| 9 | Test Code-Related Issue | Bugs appearing in test code, including problems with running, fixing, or updating test cases, and intermittent tests. |

---

## Script 2: `fine_grained_gemini_catg.py`

### Purpose

Performs fine-grained sub-categorization of bug reports previously classified as **Functional Issue** by Script 1. Each such report is assigned to one of 18 sub-categories spanning three fault families: Logic, Memory, and Concurrency.

### Prompt

```
You are an expert software engineer categorizing bug reports. Please analyze the following bug report and categorize it into ONE of the {N} predefined categories.

Bug Report Title: {title}

Bug Report Description:
{description}

Bug Report Categories:
1. Exception handling: {description}
2. Missing case: {description}
...
18. Other (Concurrency): {description}

Your Output:
Return ONLY a JSON object with EXACTLY the following three fields:
{
  "category": "<one of the categories above, exact string>",
  "confidence": <integer 1-5>,
  "reasoning": "<max 2 sentences explaining the choice>"
}

Strict Rules:
- Output MUST be valid JSON.
- NO markdown. NO backticks. NO extra text before or after the JSON.
- The "category" value MUST match EXACTLY one of the categories listed above.
- Choose the MOST RELEVANT category even if multiple could apply.
- If the bug is ambiguous, select the category that best fits the core issue and reflect uncertainty using a lower confidence score (1–2).
- Do NOT use any category as a default or fallback.
- Do NOT add additional fields, comments, or explanations.
```

### Output Fields

| Field | Type | Description |
|---|---|---|
| `id` | string | Bug report identifier |
| `title` | string | Bug report summary/title |
| `description_length` | integer | Character length of the description |
| `category` | string | Assigned fine-grained sub-category (one of 18) |
| `confidence` | integer 1–5 | Model's confidence in the assignment |
| `reasoning` | string | Up to 2 sentences explaining the choice |

### Sub-Categories

#### Logic Faults

| Category | Definition |
|---|---|
| Exception handling | Missing or improper exception handling, including uncaught or incorrectly handled exceptions. |
| Missing case | Faults due to unhandled input cases, often caused by incomplete conditional logic. |
| Processing | Incorrect implementation logic such as miscalculations, wrong method output, or flawed algorithmic steps. |
| Typo | Ambiguous naming or typographical errors in identifiers, SQL statements, URLs, or file paths. |
| Dependency | Unexpected behavior caused by changes in external libraries, frameworks, or underlying systems. |
| Other (Logic) | Semantic or logic-related faults not covered by the above categories. |

#### Memory Faults

| Category | Definition |
|---|---|
| Buffer overflow | Buffer overflows excluding numeric overflows. |
| Null pointer dereference | Dereferencing of null pointers. |
| Uninitialized memory read | Accessing memory that has not been properly initialized. |
| Memory leak | Failure to release allocated memory. |
| Dangling pointer | Accessing memory through pointers that reference deallocated objects. |
| Double free | Multiple deallocations of the same memory region. |
| Other (Memory) | Memory-related bugs not covered by the above categories. |

#### Concurrency Faults

| Category | Definition |
|---|---|
| Order violation | Incorrect or missing synchronization leading to improper execution order across threads. |
| Race condition | Concurrent access to shared resources without proper synchronization. |
| Atomicity violation | Violations of assumed atomic execution due to missing constraints on operation interleavings. |
| Deadlock | Two or more threads waiting indefinitely for each other to release resources. |
| Other (Concurrency) | Concurrency-related bugs not covered by the above categories. |

---


