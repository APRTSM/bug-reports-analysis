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
        
        # Code: More specific patterns
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
    print(f"  Reports with code: {df['has_code'].sum()} ({df['has_code'].mean()*100:.1f}%)")
    print(f"  Reports with stacktraces: {df['has_stacktrace'].sum()} ({df['has_stacktrace'].mean()*100:.1f}%)")
    
    # Only print these if the columns exist
    if 'completeness_score' in df.columns:
        print(f"\n  Average step completeness score: {df['completeness_score'].mean():.2f}")
    if 'specificity_score' in df.columns:
        print(f"  Average step specificity score: {df['specificity_score'].mean():.2f}")
    
    print("="*80)

if __name__ == "__main__":
    main()