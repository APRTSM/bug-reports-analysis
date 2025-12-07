import os, re, glob, json
import pandas as pd
import numpy as np

# ---- NLP libs ----
import spacy
from textstat import textstat
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import nltk

# Download NLTK data once at startup
nltk.download("punkt", quiet=True)

# ----------------- CONFIG -------------------
DATA_DIR = "bug_reports/Defects4J"
OUTPUT_FILE = "bug_features_v2.csv"
EMBED_MODEL = "all-MiniLM-L6-v2"

# Minimum text length for reliable readability metrics
MIN_TEXT_LENGTH = 100

# ----------------- INIT MODELS --------------
print("Loading spaCy ...")
nlp = spacy.load("en_core_web_sm", exclude=["ner", "lemmatizer", "textcat"])
print("Loading sentence transformer model ...")
model = SentenceTransformer(EMBED_MODEL)

# ----------------- HELPERS ------------------
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

def parse_bug_json(path):
    rows = []
    name = os.path.splitext(os.path.basename(path))[0]
    for obj in read_json(path):
        title = (obj.get("title") or obj.get("summary") or "").strip()
        desc  = (obj.get("description") or "").strip()
        if not (title or desc):
            continue
        bid = str(obj.get("id") or obj.get("bug_id") or name)
        rows.append({"id": bid, "summary": title, "description": desc})
    return rows

# ============================================================================
# IMPROVED S2R DETECTION (Based on Paper's Approach)
# ============================================================================

def identify_s2r_sentences(text: str):
    """
    Identify sentences that describe steps to reproduce.
    
    IMPROVEMENT: Uses multiple patterns and sentence-level analysis
    instead of just checking if the phrase exists anywhere.
    
    Based on Euler's approach but simplified (without neural model).
    """
    doc = nlp(text)
    sentences = [sent.text.strip() for sent in doc.sents]
    
    s2r_sentences = []
    # Use list to preserve order (though order doesn't matter for matching)
    s2r_indicators = [
        # Direct S2R headers
        r"(?i)^steps?\s+to\s+reproduce",
        r"(?i)^reproduction\s+steps?",
        r"(?i)^how\s+to\s+reproduce",
        r"(?i)^repro\s+steps?",
        
        # Conditional patterns (like paper's examples)
        r"(?i)^when\s+(I|you|one)",
        r"(?i)^if\s+(I|you|one)",
        r"(?i)^after\s+(I|you|one)",
        
        # Imperative patterns
        r"(?i)^(open|click|tap|press|select|enter|type|run|start|go|navigate)",
        
        # Enumerated steps
        r"^\d+[\.\)]\s+",  # 1. or 1)
        r"^[-*]\s+\w",     # - or * 
    ]
    
    in_s2r_section = False
    
    for i, sent in enumerate(sentences):
        # Check if this starts a S2R section
        for pattern in s2r_indicators:
            if re.match(pattern, sent.strip()):
                s2r_sentences.append(sent)
                in_s2r_section = True
                break
        else:
            # If we're in a S2R section and this looks like a continuation
            if in_s2r_section:
                # Check if it's still describing steps (imperative, enumerated, etc.)
                if (re.match(r"^\d+[\.\)]", sent.strip()) or 
                   re.match(r"^[-*]", sent.strip()) or
                   re.match(r"(?i)^(then|next|after|finally)", sent.strip())):
                    s2r_sentences.append(sent)
                else:
                    # Exit S2R section if we hit non-step content
                    in_s2r_section = False
    
    return s2r_sentences


def extract_individual_steps(text: str):
    """
    Extract individual steps from S2R sentences using dependency parsing.
    
    IMPROVEMENT: Actually extracts individual actions like the paper does,
    following the [action] [object] [preposition] [object2] format.
    """
    s2r_sentences = identify_s2r_sentences(text)
    
    if not s2r_sentences:
        return []
    
    individual_steps = []
    
    for sent in s2r_sentences:
        doc = nlp(sent)
        
        # Find the main verb (action)
        for token in doc:
            if token.pos_ == "VERB":
                action = token.lemma_
                obj = None
                obj2 = None
                prep = None
                
                # Find direct object
                for child in token.children:
                    if child.dep_ == "dobj":
                        obj = child.text
                        
                        # Find prepositional object
                        for grandchild in child.children:
                            if grandchild.dep_ == "prep":
                                prep = grandchild.text
                                for ggchild in grandchild.children:
                                    if ggchild.dep_ == "pobj":
                                        obj2 = ggchild.text
                
                # Only add if we found at least an action
                if action:
                    step = {
                        "action": action,
                        "object": obj,
                        "preposition": prep,
                        "object2": obj2,
                        "raw_sentence": sent
                    }
                    individual_steps.append(step)
                    break  # Only get first verb per sentence
    
    return individual_steps


def count_individual_steps(text: str):
    """Count the number of individual steps identified."""
    steps = extract_individual_steps(text)
    return len(steps)


def has_reproduction_steps_advanced(text: str):
    """
    Advanced detection of reproduction steps.
    
    IMPROVEMENT: Not just checking for keyword, but actually identifying
    S2R sentences. If we find S2R sentences (via headers, enumeration, or imperatives),
    that's sufficient evidence of steps.
    """
    s2r_sentences = identify_s2r_sentences(text)
    
    # If we found any S2R sentences, that's evidence of steps
    # This is less strict than requiring imperative verbs
    if s2r_sentences:
        return True
    
    # Also check for enumerated items anywhere in the text (not just sentence starts)
    # This catches cases like "1. Do this\n2. Do that" even if not at sentence start
    if re.search(r"(?:^|\n)\s*(?:\d+[\.\)]|[-*])\s+[A-Z\w]", text, re.MULTILINE):
        return True
    
    return False


# ============================================================================
# IMPROVED STRUCTURAL FLAGS
# ============================================================================

def extract_structural_flags_v2(text: str):
    """
    Enhanced structural flag extraction with better patterns.
    """
    return {
        # Stacktrace: More comprehensive patterns
        "has_stacktrace": bool(re.search(
            r"(Exception|Error|Throwable|"
            r"at\s+[a-zA-Z0-9_.]+\([^)]*\)|"
            r"^\s*at\s+|"
            r"Caused\s+by:|"
            r"Stack\s+trace:)",
            text,
            re.MULTILINE | re.IGNORECASE
        )),
        
        # Steps: Use the advanced detection
        "has_steps": has_reproduction_steps_advanced(text),
        
        # Number of individual steps identified
        "num_steps": count_individual_steps(text),
        
        # Code: More specific patterns (fixed from original)
        "has_code": bool(re.search(
            r"```|"  # Markdown code blocks
            r"(?:public|private|protected|static)\s+(?:class|void|int|String)|"  # Java
            r"(?:^|\n)(?:def|class|import|from\s+\w+\s+import)\s+|"  # Python
            r"(?:^|\n)\s*function\s+\w+\s*\(|"  # JavaScript
            r"#include\s*<|"  # C/C++
            r"(?:^|\n)\s{4,}\w+.*[;{]",  # Indented code
            text,
            re.MULTILINE
        )),
        
        # Patch: Git diffs or patches
        "has_patch": bool(re.search(
            r"diff --git|\.patch\b|commit [0-9a-f]{6,40}\b|"
            r"^\+{3}\s+|^-{3}\s+|"
            r"^@@.*@@",
            text,
            re.MULTILINE
        )),
        
        # Enumeration: More specific (fixed from original)
        "has_enumeration": bool(re.search(
            r"(?:^|\n)\s*(?:\d+[\.\)]|[-*])\s+[A-Z\w]",
            text,
            re.MULTILINE
        ))
    }


# ============================================================================
# ENHANCED QUALITY METRICS
# ============================================================================

def ambiguity_score_v2(text: str):
    """
    Enhanced ambiguity detection.
    
    IMPROVEMENT: Considers context and adds more vague terms.
    """
    vague_terms = {
        "maybe", "sometimes", "appears", "seems", "probably",
        "might", "could", "apparently", "likely", "unsure",
        "perhaps", "possibly", "unclear", "uncertain", "somehow",
        "something", "somewhere", "somewhat", "kind of", "sort of"
    }
    
    # Also check for vague phrases
    vague_phrases = [
        r"i\s+think",
        r"not\s+sure",
        r"may\s+be",
        r"could\s+be"
    ]
    
    tokens = re.findall(r"\b[a-zA-Z]+\b", text.lower())
    if not tokens:
        return 0.0
    
    vague_count = sum(tok in vague_terms for tok in tokens)
    
    # Add phrase matches
    for phrase in vague_phrases:
        vague_count += len(re.findall(phrase, text.lower()))
    
    return vague_count / len(tokens)


def semantic_redundancy(text: str):
    """Mean sentence similarity within a description (higher => more repetitive)."""
    sents = [s.text.strip() for s in nlp(text).sents if s.text.strip()]
    if len(sents) < 2:
        return 0.0
    
    try:
        embs = model.encode(sents, normalize_embeddings=True)
        sims = cosine_similarity(embs)
        upper = sims[np.triu_indices(len(sims), k=1)]
        return float(np.mean(upper)) if upper.size else 0.0
    except Exception as e:
        print(f"Warning: Redundancy calculation failed: {e}")
        return 0.0


# EXTRA HIGH-LEVEL FEATURES (causality, context, error semantics, behavior)
# ============================================================================

def extract_causal_features(text: str, n_sentences=None):
    """
    Capture causal and temporal connectives in the report.

    These are cheap proxies for how much the reporter explains
    when/why things happen (useful for LLM-based reasoning).
    """
    lower = text.lower()

    causal_markers = [
        "because", "since", "so that", "due to", "therefore", "hence",
        "thus", "consequently", "as a result", "in order to", "if",
        "when", "whenever", "unless", "until", "after", "before", "once",
        "while",
    ]
    temporal_markers = [
        "then", "next", "afterwards", "later", "finally", "subsequently",
        "eventually", "initially", "first", "second", "third",
    ]

    def _count_phrases(text_lower: str, phrases):
        count = 0
        for p in phrases:
            if " " in p:
                # multi-word phrase
                count += len(re.findall(re.escape(p), text_lower))
            else:
                count += len(re.findall(r"\b" + re.escape(p) + r"\b", text_lower))
        return count

    num_causal = _count_phrases(lower, causal_markers)
    num_temporal = _count_phrases(lower, temporal_markers)

    if not n_sentences:
        # Fallback: approximate number of sentences
        approx = len(re.split(r"[.!?]+", text)) - 1
        n_sentences = approx if approx > 0 else 1

    return {
        "num_causal_markers": num_causal,
        "num_temporal_markers": num_temporal,
        "causal_density": float(num_causal) / n_sentences if n_sentences else 0.0,
        "temporal_density": float(num_temporal) / n_sentences if n_sentences else 0.0,
    }


def extract_context_features(text: str):
    """
    Execution / environment context richness.

    We look for OS, browser, and environment mentions plus explicit
    version-like strings. This is still text-only but often highly
    predictive of FL difficulty.
    """
    lower = text.lower()

    os_terms = [
        "windows", "linux", "ubuntu", "debian", "fedora", "red hat",
        "mac os", "macos", "os x", "android", "ios",
    ]
    browser_terms = [
        "chrome", "firefox", "safari", "edge", "internet explorer",
        "ie11", "ie 11", "opera",
    ]
    env_terms = [
        "production", "staging", "qa", "test environment",
        "development environment", "dev environment",
        "local environment", "localhost", "server", "client",
        "backend", "front-end", "frontend",
    ]

    num_os = sum(1 for t in os_terms if t in lower)
    num_browser = sum(1 for t in browser_terms if t in lower)
    num_env = sum(1 for t in env_terms if t in lower)

    # Very loose "version-like" pattern (v1.2.3, 1.8.0_271, etc.)
    version_pattern = r"\b(v?\d+\.\d+(?:\.\d+)*(?:_\d+)?)\b"
    num_versions = len(re.findall(version_pattern, text))

    return {
        "num_os_mentions": num_os,
        "num_browser_mentions": num_browser,
        "num_env_mentions": num_env,
        "num_versions": num_versions,
        "has_os_info": bool(num_os),
        "has_version_info": bool(num_versions),
    }


def parse_stacktrace_semantics(text: str):
    """
    Extract a few cheap semantic features from stacktraces if present.

    This is still regex-based and purely textual, but gives you:
      - primary exception type
      - number of distinct exception types
      - stacktrace depth (number of frames)
      - number of 'Caused by' chains
    """
    # ExceptionLikeNameException / Error / Throwable
    exception_pattern = r"\b([A-Za-z_][A-Za-z0-9_]*(?:Exception|Error|Throwable))\b"
    exception_types = [m.group(1) for m in re.finditer(exception_pattern, text)]
    unique_exceptions = sorted(set(exception_types))
    primary_exception = unique_exceptions[0] if unique_exceptions else None

    # Frame lines: typical "at package.Class.method(File.java:123)"
    frame_pattern = r"^\s*at\s+.+"
    num_frames = len(re.findall(frame_pattern, text, flags=re.MULTILINE))

    num_caused_by = len(re.findall(r"Caused by:", text))

    return {
        "primary_exception_type": primary_exception,
        "num_exception_types": len(unique_exceptions),
        "stacktrace_depth": num_frames,
        "num_caused_by": num_caused_by,
    }


def extract_behavior_features(text: str, n_sentences=None):
    """
    Count behavior-describing verbs such as 'crash', 'hang', 'freeze',
    'render', 'save', 'load', etc.

    This is useful to distinguish behavior-rich reports from vague ones.
    """
    behavior_verbs = {
        "crash", "hang", "freeze", "fail", "throw", "error", "break",
        "render", "display", "show", "save", "load", "open", "close",
        "restart", "reload", "timeout", "time out", "terminate",
        "shut", "shut down", "exit", "return", "calculate", "compute",
        "parse", "compile", "build", "deploy", "upload", "download",
    }

    doc = nlp(text)
    behavior_count = 0
    for token in doc:
        if token.pos_ == "VERB" and token.lemma_.lower() in behavior_verbs:
            behavior_count += 1

    # We don't want to re-count sentences if we already know them
    if not n_sentences:
        n_sentences = sum(1 for _ in doc.sents) or 1

    return {
        "behavior_verb_count": behavior_count,
        "behavior_verb_density": float(behavior_count) / n_sentences if n_sentences else 0.0,
    }


def compute_modality_features(text: str, n_sentences=None):
    """
    Simple intent / obligation proxy based on modal verbs.

    Higher modal density often means clearer expectations ('should do X'),
    which can help both IRFL and LLM-based tools.
    """
    lower = text.lower()
    positive_modals = [
        "should", "must", "have to", "need to", "needs to",
    ]
    negative_modals = [
        "should not", "shouldn't", "must not", "mustn't",
        "cannot", "can't",
    ]

    def _count_multiword(text_lower: str, phrases):
        count = 0
        for p in phrases:
            count += len(re.findall(re.escape(p), text_lower))
        return count

    pos_count = _count_multiword(lower, positive_modals)
    neg_count = _count_multiword(lower, negative_modals)
    total_modals = pos_count + neg_count

    if not n_sentences:
        approx = len(re.split(r"[.!?]+", text)) - 1
        n_sentences = approx if approx > 0 else 1

    return {
        "num_modal_verbs": total_modals,
        "num_negative_modals": neg_count,
        "modal_density": float(total_modals) / n_sentences if n_sentences else 0.0,
    }



# ============================================================================
# STEP QUALITY ASSESSMENT (Simplified version of Euler's approach)
# ============================================================================

def assess_step_quality(steps: list):
    """
    Assess the quality of individual steps.
    
    This is a SIMPLIFIED version of what Euler does (without app execution).
    We check for common quality issues based on textual analysis.
    
    Returns quality metrics for the steps.
    """
    if not steps:
        return {
            "avg_step_length": 0,
            "steps_with_objects": 0,
            "steps_with_specific_actions": 0,
            "potential_ambiguous_steps": 0,
            "potential_vague_steps": 0
        }
    
    specific_actions = {
        "click", "tap", "press", "select", "enter", "type", "open",
        "close", "navigate", "scroll", "swipe", "drag", "create",
        "delete", "save", "load", "run", "execute", "build"
    }
    
    generic_actions = {
        "do", "make", "get", "see", "find", "fix", "check", "use", "try"
    }
    
    generic_objects = {
        "it", "this", "that", "thing", "button", "field", "item", "element"
    }
    
    step_lengths = []
    steps_with_objects = 0
    steps_with_specific = 0
    ambiguous = 0
    vague = 0
    
    for step in steps:
        raw = step.get("raw_sentence", "")
        step_lengths.append(len(raw.split()))
        
        action = step.get("action", "").lower()
        obj = step.get("object", "")
        
        # Check if step has an object
        if obj:
            steps_with_objects += 1
            
            # Check if object is generic
            if obj.lower() in generic_objects:
                ambiguous += 1
        else:
            vague += 1
        
        # Check if action is specific
        if action in specific_actions:
            steps_with_specific += 1
        elif action in generic_actions:
            vague += 1
    
    return {
        "avg_step_length": np.mean(step_lengths) if step_lengths else 0,
        "steps_with_objects": steps_with_objects,
        "steps_with_specific_actions": steps_with_specific,
        "potential_ambiguous_steps": ambiguous,
        "potential_vague_steps": vague,
        "completeness_score": steps_with_objects / len(steps) if steps else 0,
        "specificity_score": steps_with_specific / len(steps) if steps else 0
    }


def compute_syntactic_features(text: str):
    """Token/sentence counts via spaCy + readability via textstat."""
    doc = nlp(text)
    n_tokens = sum(1 for t in doc if not t.is_space)
    n_words = sum(1 for t in doc if t.is_alpha)
    n_sentences = sum(1 for _ in doc.sents)

    # Only compute readability for longer texts
    flesch = smog = gunning = coleman = ari = None
    if len(text) >= MIN_TEXT_LENGTH:
        try:
            flesch = textstat.flesch_reading_ease(text)
            smog = textstat.smog_index(text)
            gunning = textstat.gunning_fog(text)
            coleman = textstat.coleman_liau_index(text)
            ari = textstat.automated_readability_index(text)
        except (ValueError, ZeroDivisionError) as e:
            print(f"Warning: Readability calculation failed: {e}")

    return {
        "n_tokens": n_tokens,
        "n_words": n_words,
        "n_sentences": n_sentences,
        "flesch_reading_ease": flesch,
        "smog_index": smog,
        "gunning_fog": gunning,
        "coleman_liau_index": coleman,
        "automated_readability_index": ari,
    }


# ----------------- MAIN ---------------------
def main():
    json_files = glob.glob(os.path.join(DATA_DIR, "**/*.json"), recursive=True)
    print(f"Found {len(json_files)} JSON files under {DATA_DIR!r}")
    
    try:
        from tqdm import tqdm
        json_files = tqdm(json_files, desc="Processing files")
    except ImportError:
        pass
    
    rows = []

    for path in json_files:
        try:
            for bug in parse_bug_json(path):
                text = (bug["summary"] + " " + bug["description"]).strip()

                               # Syntactic features
                syn = compute_syntactic_features(text)
                
                # Improved structural flags (use full text for consistency)
                flags = extract_structural_flags_v2(text)
                
                # Extract individual steps (use full text for consistency)
                individual_steps = extract_individual_steps(text)
                
                # Assess step quality
                step_quality = assess_step_quality(individual_steps)
                
                # Semantic features (embedding + redundancy, ambiguity, S2R correctness)
                # Use full text for consistency
                emb = model.encode(text, normalize_embeddings=True)
                
                if emb.shape[0] != 384:
                    print(f"Warning: Unexpected embedding dimension for {bug['id']}: {emb.shape}")
                
                redund = semantic_redundancy(text)
                ambig = ambiguity_score_v2(text)

                # New higher-level features (causality, context, stacktrace semantics, behavior, modality)
                # Use full text for consistency
                causal_feats = extract_causal_features(text, syn.get("n_sentences"))
                context_feats = extract_context_features(text)
                stacktrace_feats = (
                    parse_stacktrace_semantics(text)
                    if flags.get("has_stacktrace")
                    else {
                        "primary_exception_type": None,
                        "num_exception_types": 0,
                        "stacktrace_depth": 0,
                        "num_caused_by": 0,
                    }
                )
                behavior_feats = extract_behavior_features(text, syn.get("n_sentences"))
                modality_feats = compute_modality_features(text, syn.get("n_sentences"))

                rows.append({
                    "id": bug["id"],
                    "summary_chars": len(bug["summary"]),
                    "description_chars": len(bug["description"]),
                    **syn,
                    **flags,
                    **step_quality,
                    **causal_feats,
                    **context_feats,
                    **stacktrace_feats,
                    **behavior_feats,
                    **modality_feats,
                    "redundancy": redund,
                    "ambiguity": ambig,
                    "embedding": json.dumps(emb.tolist()),
                })

        except Exception as e:
            print(f"Error processing {path}: {e}")
            continue

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"\nSaved {len(df)} bug feature rows to {OUTPUT_FILE}")
    
    # Print summary statistics
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    print(f"  Average description length: {df['description_chars'].mean():.0f} chars")
    print(f"  Reports with steps (advanced detection): {df['has_steps'].sum()} ({df['has_steps'].mean()*100:.1f}%)")
    print(f"  Average number of individual steps: {df['num_steps'].mean():.2f}")
    print(f"  Reports with code: {df['has_code'].sum()} ({df['has_code'].mean()*100:.1f}%)")
    print(f"  Reports with stacktraces: {df['has_stacktrace'].sum()} ({df['has_stacktrace'].mean()*100:.1f}%)")
    print(f"\n  Average step completeness score: {df['completeness_score'].mean():.2f}")
    print(f"  Average step specificity score: {df['specificity_score'].mean():.2f}")
    print("="*80)

if __name__ == "__main__":
    main()