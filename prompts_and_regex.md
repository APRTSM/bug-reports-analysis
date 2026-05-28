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

## Gemini LLM Annotation Scripts

This document describes the three Python scripts used to annotate Defects4J bug reports via the Google Gemini API. Each script reads bug reports from XML files (`defects4j_xml/`), issues a structured prompt to Gemini, and writes results incrementally to a CSV file. All scripts use `gemini-2.0-flash` or `gemini-2.5-flash` as the primary model with automatic fallback.

---

## Overview of Scripts

| Script | Input | Output CSV | Purpose |
|---|---|---|---|
| `gemini_bug_categorization_overall.py` | `defects4j_xml/*.xml` | `gemini_bug_categorization.csv` | Assigns each bug report one of 9 coarse-grained categories |
| `fine_grained_gemini_catg.py` | `defects4j_xml/*.xml` + `gemini_bug_categorization.csv` | `fine_grained_gemini_categorization.csv` | Assigns fine-grained sub-category to bugs classified as "Functional Issue" |
| `gemini_bug_ratings.py` | `defects4j_xml/*.xml` | `gemini_bug_ratings.csv` | Rates each bug report across 13 quality and content dimensions |

---

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

## Script 3: `gemini_bug_ratings.py`

### Purpose

Rates each bug report across 13 quality and content dimensions, producing numeric scores, boolean flags, and free-text annotations. All bug reports are processed regardless of category.

### Prompt

```
You are an expert software engineer evaluating a bug report. Please analyze the following bug report and provide ratings in JSON format.
Bug Report Title: {title}
Bug Report Description:
{description}
Please evaluate this bug report on the following dimensions and return ONLY a valid JSON object with these exact fields:
{{
    "actionability": <integer 0-5>,
    // How actionable is this bug report for a developer attempting to fix it?
    // 0 = No actionable information; cannot determine what to fix or where to start.
    // 1 = Minimal context; bug is vaguely described with no reproduction path.
    // 2 = Some context but missing key details (e.g., no steps, unclear component).
    // 3 = Moderately actionable; affected component is identifiable but steps are incomplete.
    // 4 = Mostly actionable; reproduction path is present but minor details are missing.
    // 5 = Fully actionable; all necessary context (component, steps, environment, expected behavior) is present.

    "clarity": <integer 0-5>,
    // How unambiguous and logically structured is the bug description?
    // 0 = Incoherent, contradictory, or completely unintelligible prose.
    // 1 = Difficult to follow; key information is buried or inconsistently described.
    // 2 = Partially clear; the general issue is guessable but reasoning is muddled.
    // 3 = Mostly clear; problem is understandable but some ambiguity remains.
    // 4 = Clear and well-structured; minor phrasing issues only.
    // 5 = Immediately comprehensible; problem, context, and behavior are unambiguous.

    "specificity": <integer 0-5>,
    // How specific and precise are the details provided (versions, inputs, steps, environment)?
    // 0 = No specific details whatsoever; entirely generic description.
    // 1 = Very few specifics; missing versions, inputs, and environment information.
    // 2 = Some specifics present but incomplete (e.g., version mentioned, steps missing).
    // 3 = Moderate specificity; key details present but not exhaustive.
    // 4 = Highly specific; most relevant details (versions, inputs, steps) are provided.
    // 5 = Exhaustively specific; all relevant technical details are precisely stated.

    "expected_observed_alignment": <integer 0-5>,
    // How clearly does the report distinguish what was expected versus what actually occurred?
    // 0 = No distinction made; expected and observed behavior are not described.
    // 1 = One side described (e.g., only observed behavior, no expectation stated).
    // 2 = Both present but conflated or inconsistently described.
    // 3 = Both described but the contrast is implicit rather than explicit.
    // 4 = Clear distinction with minor gaps in either expected or observed description.
    // 5 = Explicit, precise contrast between expected and observed behavior.

    "root_cause_guess": "<string>",
    // Your best guess at the root cause based on the report content.
    // Examples: "null pointer dereference", "type mismatch", "race condition",
    // "off-by-one error", "configuration issue", "unhandled exception", "memory leak".
    // Use "unknown" if the report provides insufficient information.

    "technical_depth": <integer 0-5>,
    // How technically detailed is the report in terms of diagnostic evidence?
    // 0 = No technical content; purely narrative with no code, traces, or technical context.
    // 1 = Minimal technical content; at most a brief mention of a class or method name.
    // 2 = Some technical content; e.g., an exception type mentioned but no stack trace.
    // 3 = Moderate depth; stack trace or code snippet present but incomplete.
    // 4 = Good depth; stack trace and/or code present with relevant technical context.
    // 5 = Comprehensive; full stack trace, code snippet, environment details, and version info all present.

    "ambiguity_types": ["<string>", ...],
    // List all ambiguity types present in the report. Use empty list [] if none.
    // Choose from: "missing steps", "vague inputs", "unclear error messages",
    // "missing context", "unclear reproduction", "missing environment info",
    // "contradictory information", "unclear expected behavior", "vague component reference".

    "hidden_s2r_present": <boolean>,
    // Are there implicit steps to reproduce embedded in the narrative
    // (i.e., reproducible sequence is inferable but not explicitly listed as steps)?

    "causal_reasoning_quality": <integer 0-5>,
    // How well does the reporter explain cause-and-effect relationships?
    // 0 = No causal explanation; symptoms reported with no reasoning.
    // 1 = Weak causal hint; vague connection implied but not articulated.
    // 2 = Partial reasoning; cause is suggested but not logically connected to effect.
    // 3 = Moderate reasoning; causal chain is present but incomplete or imprecise.
    // 4 = Good reasoning; cause and effect are clearly linked with supporting evidence.
    // 5 = Strong reasoning; coherent, complete causal explanation with evidence.

    "contradiction_present": <boolean>,
    // Are there any internal contradictions or conflicting statements in the report
    // (e.g., says bug occurs always but later says it is intermittent)?

    "repair_difficulty": <integer 0-5>,
    // How difficult would this bug be to fix based solely on the information provided?
    // 0 = Trivial; fix is immediately obvious from the report (e.g., typo, config value).
    // 1 = Easy; root cause is clear and fix is straightforward.
    // 2 = Moderate; root cause is identifiable but fix requires some investigation.
    // 3 = Difficult; root cause is unclear or fix requires significant code changes.
    // 4 = Very difficult; report is vague, root cause is deeply buried, or fix is complex.
    // 5 = Intractable from report alone; insufficient information to diagnose or fix.

    "likely_impacted_code_concepts": ["<string>", ...]
    // List of code concepts or subsystems likely affected based on report content.
    // Examples: "JSON parsing", "UI rendering", "database queries", "authentication",
    // "file I/O", "concurrency", "memory management", "API integration", "type conversion".
}}

Important:
- Return ONLY the JSON object, no markdown formatting, no code blocks, no explanations
- All integer fields must be integers (0-5)
- All boolean fields must be true/false (lowercase)
- All string fields must be valid strings
- Arrays must be valid JSON arrays
- If a field cannot be determined, use reasonable defaults (0 for integers, false for booleans, empty string/array for strings/arrays)
```

### Output Fields

| Field | Type | Scale / Values | Description |
|---|---|---|---|
| `id` | string | — | Bug report identifier |
| `title` | string | — | Bug report summary/title |
| `description` | string | — | Full bug report description text |
| `description_length` | integer | — | Character length of the description |
| `actionability` | integer | 0–5 | How actionable is the report? Can a developer immediately start fixing it? |
| `clarity` | integer | 0–5 | How clear and well-written is the description? |
| `specificity` | integer | 0–5 | How specific are the details (versions, steps, inputs, etc.)? |
| `expected_observed_alignment` | integer | 0–5 | How well does the report describe expected vs. observed behavior? |
| `root_cause_guess` | string | free text | Model's best guess at the root cause (e.g., "null pointer", "race condition"). |
| `technical_depth` | integer | 0–5 | How technically detailed is the report (code, stack traces, technical context)? |
| `ambiguity_types` | string[] | see below | List of ambiguity types present; empty if none. |
| `hidden_s2r_present` | boolean | true/false | Are implicit steps to reproduce hidden in the description (not explicitly listed)? |
| `causal_reasoning_quality` | integer | 0–5 | How well does the reporter explain cause-and-effect relationships? |
| `contradiction_present` | boolean | true/false | Are there any contradictions or conflicting information in the report? |
| `repair_difficulty` | integer | 0–5 | Estimated difficulty to fix based on available information (0 = very easy, 5 = very difficult). |
| `likely_impacted_code_concepts` | string[] | free text | Code concepts likely impacted (e.g., "JSON parsing", "authentication"). |

#### Ambiguity Type Vocabulary

The model is instructed to draw from (but not limited to) these labels:

- `missing steps`
- `vague inputs`
- `unclear error messages`
- `missing context`
- `unclear reproduction`
- `missing environment info`

---

## Common Implementation Notes

**Model selection.** All scripts attempt models in priority order, falling back automatically if a model is unavailable or quota-exhausted:
1. `gemini-2.0-flash`
2. `gemini-2.5-flash`
3. `gemini-2.0-flash-lite`
4. `gemini-2.5-pro`
5. `gemini-pro-latest`

**Rate limiting.** Scripts impose a configurable inter-call delay (`API_DELAY`): 2 seconds for the categorization scripts and 5 seconds for the ratings script. Exponential backoff and respect for Retry-After headers are implemented for 429 errors.

**Incremental writes.** Each script writes one row to the output CSV immediately after processing, allowing safe interruption and resume without data loss.

**Input format.** Bug reports are read from `defects4j_xml/*.xml`. Bug IDs are normalized from the filename (e.g., `Lang_19.xml` → `Lang-19`).

**Response parsing.** All scripts strip markdown code fences if present before JSON parsing. Invalid or unrecognized category strings fall back to a default (`Other (Logic)` for fine-grained, `Functional Issue` for coarse-grained).