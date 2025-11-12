import os, re, glob, json
import pandas as pd
import numpy as np

# ---- NLP libs ----
import spacy
from textstat import textstat
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# ----------------- CONFIG -------------------
DATA_DIR = "bug_reports/Defects4J"   # <-- parent folder containing .json files (searched recursively)
OUTPUT_FILE = "bug_features.csv"
EMBED_MODEL = "all-MiniLM-L6-v2"

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
        # naive JSONL detection: multiple lines, each object-like
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

def extract_structural_flags(text: str):
    return {
        "has_stacktrace": bool(re.search(r"(Exception|Error|at\s+[a-zA-Z0-9_.]+\()", text)),
        "has_steps": bool(re.search(r"(?i)(steps?\s+to\s+reproduce|how\s+to\s+reproduce|reproduction\s+steps?)", text)),
        "has_code": bool(re.search(r"```|public |class |void |;|{|}\s*$", text)),
        "has_patch": bool(re.search(r"diff --git|\.patch\b|commit [0-9a-f]{6,40}\b", text)),
        "has_enumeration": bool(re.search(r"(?:^|\n)\s*(?:\d+\.|-|\*)\s+\S+", text))
    }

def ambiguity_score(text: str):
    vague_terms = {"maybe","sometimes","appears","seems","probably","might","could","apparently","likely","unsure"}
    tokens = re.findall(r"\b[a-zA-Z]+\b", text.lower())
    vague = sum(tok in vague_terms for tok in tokens)
    return vague / max(len(tokens), 1)

def semantic_redundancy(text: str):
    """Mean sentence similarity within a description (higher => more repetitive)."""
    import nltk
    nltk.download("punkt", quiet=True)
    sents = [s.text.strip() for s in nlp(text).sents if s.text.strip()]
    if len(sents) < 2:
        return 0.0
    embs = model.encode(sents, normalize_embeddings=True)
    sims = cosine_similarity(embs)
    upper = sims[np.triu_indices(len(sims), k=1)]
    return float(np.mean(upper)) if upper.size else 0.0

def steps_correctness(text: str):
    """Heuristic: fraction of sentences that look imperative/actionable."""
    import nltk
    nltk.download("punkt", quiet=True)
    sents = [s.text.strip() for s in nlp(text).sents if s.text.strip()]
    if not sents:
        return 0.0
    imperative_verbs = r"open|click|run|go|select|press|choose|enter|build|compile|execute|start|install|update"
    imp = sum(bool(re.match(rf"^\s*(?:\d+\.\s*)?(?:{imperative_verbs})\b", s.lower())) for s in sents)
    return imp / len(sents)

def compute_syntactic_features(text: str):
    """Token/sentence counts via spaCy + readability via textstat."""
    doc = nlp(text)
    n_tokens = sum(1 for t in doc if not t.is_space)
    n_words = sum(1 for t in doc if t.is_alpha)
    n_sentences = sum(1 for _ in doc.sents)

    # textstat expects raw text
    try:
        flesch = textstat.flesch_reading_ease(text)
        smog = textstat.smog_index(text)
        gunning = textstat.gunning_fog(text)
        coleman = textstat.coleman_liau_index(text)
        ari = textstat.automated_readability_index(text)
    except Exception:
        flesch = smog = gunning = coleman = ari = None

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
    rows = []

    for path in json_files:
        for bug in parse_bug_json(path):
            text = (bug["summary"] + " " + bug["description"]).strip()

            # Syntactic
            syn = compute_syntactic_features(text)
            flags = extract_structural_flags(text)

            # Semantic
            emb = model.encode(text, normalize_embeddings=True)
            redund = semantic_redundancy(bug["description"])
            ambig  = ambiguity_score(bug["description"])
            stepok = steps_correctness(bug["description"])

            rows.append({
                "id": bug["id"],
                "summary_chars": len(bug["summary"]),
                "description_chars": len(bug["description"]),
                **syn,
                **flags,
                "redundancy": redund,
                "ambiguity": ambig,
                "steps_correctness": stepok,
                "embedding": json.dumps(emb.tolist())
            })

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"Saved {len(df)} bug feature rows to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
