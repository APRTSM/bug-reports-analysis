import os, re, glob, json
import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np

# ---- NLP libs ----
import spacy
from textstat import textstat
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import nltk

# Download NLTK data once at startup
nltk.download("punkt", quiet=True)

# ----------------- CONFIG -------------------
DATA_DIR = "defects4j_xml"
OUTPUT_FILE = "bug_features_enhanced_fixed.csv"
EMBED_MODEL = "all-MiniLM-L6-v2"
USE_XML = True
BEE_RESULTS_FILE = "bee_results.jsonl"

# Minimum text length for reliable readability metrics
MIN_TEXT_LENGTH = 100

# NEW: Embedding enhancement settings
NUM_SEMANTIC_CLUSTERS = 10  # Number of clusters for embedding-based grouping
PCA_COMPONENTS = 50  # Components to keep for clustering
MIN_SENTENCES_FOR_DIVERSITY = 2  # Minimum sentences needed for diversity metrics

# ----------------- INIT MODELS --------------
print("Loading spaCy ...")
nlp = spacy.load("en_core_web_sm", exclude=["ner", "lemmatizer", "textcat"])
print("Loading sentence transformer model ...")
model = SentenceTransformer(EMBED_MODEL)

# We'll fit PCA and KMeans after collecting all embeddings
global_embeddings = []
global_bug_ids = []

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

def read_xml(path):
    """Load XML bug report; return a list of dicts with bug information."""
    try:
        with open(path, 'rb') as f:
            raw_content = f.read()
        
        try:
            content = raw_content.decode('utf-8')
        except UnicodeDecodeError:
            content = raw_content.decode('latin-1', errors='replace')
        
        content = re.sub(r'[\x00-\x08\x0B-\x0C\x0E-\x1F]', '', content)
        root = ET.fromstring(content)
        
        bugs = []
        if root.tag == 'bugRepository':
            bug_elements = root.findall('.//bug')
        elif root.tag == 'bug':
            bug_elements = [root]
        else:
            bug_elements = []
        
        for bug_elem in bug_elements:
            bug_id = bug_elem.get('id', '')
            bug_info = bug_elem.find('buginformation')
            
            if bug_info is not None:
                summary_elem = bug_info.find('summary')
                desc_elem = bug_info.find('description')
                
                title = ""
                if summary_elem is not None and summary_elem.text:
                    title = summary_elem.text.strip()
                    title = title.replace('&lt;', '<').replace('&gt;', '>').replace('&amp;', '&')
                
                desc = ""
                if desc_elem is not None and desc_elem.text:
                    desc = desc_elem.text.strip()
                    desc = desc.replace('&lt;', '<').replace('&gt;', '>').replace('&amp;', '&')
                
                name = os.path.splitext(os.path.basename(path))[0]
                if '_' in name:
                    project, num = name.rsplit('_', 1)
                    bug_id = f"{project}-{num}"
                elif bug_id:
                    project = name
                    bug_id = f"{project}-{bug_id}"
                else:
                    bug_id = name
                
                bugs.append({
                    "id": bug_id,
                    "title": title,
                    "description": desc,
                    "opendate": bug_elem.get('opendate', ''),
                    "fixdate": bug_elem.get('fixdate', ''),
                    "resolution": bug_elem.get('resolution', '')
                })
        
        return bugs if bugs else []
    except ET.ParseError as e:
        print(f"Error parsing XML file {path}: {e}")
        return []
    except Exception as e:
        print(f"Error reading XML file {path}: {e}")
        return []

def parse_bug_file(path):
    """Parse bug report from either JSON or XML file."""
    rows = []
    name = os.path.splitext(os.path.basename(path))[0]
    
    if USE_XML and path.endswith('.xml'):
        bugs = read_xml(path)
        for bug in bugs:
            title = bug.get("title", "").strip()
            desc = bug.get("description", "").strip()
            if not (title or desc):
                continue
            bid = bug.get("id", name)
            rows.append({"id": str(bid), "summary": title, "description": desc})
    else:
        for obj in read_json(path):
            title = (obj.get("title") or obj.get("summary") or "").strip()
            desc  = (obj.get("description") or "").strip()
            if not (title or desc):
                continue
            bid = str(obj.get("id") or obj.get("bug_id") or name)
            rows.append({"id": bid, "summary": title, "description": desc})
    
    return rows


# ============================================================================
# STRUCTURAL FLAGS
# ============================================================================

def extract_structural_flags_v2(text: str):
    """Enhanced structural flag extraction with better patterns."""
    return {
        "has_stacktrace": bool(re.search(
            r"(Exception|Error|Throwable|"
            r"at\s+[a-zA-Z0-9_.]+\([^)]*\)|"
            r"^\s*at\s+|"
            r"Caused\s+by:|"
            r"Stack\s+trace:)",
            text,
            re.MULTILINE | re.IGNORECASE
        )),
        
        "has_code": bool(re.search(
            r"```|"
            r"(?:public|private|protected|static)\s+(?:class|void|int|String)|"
            r"(?:^|\n)(?:def|class|import|from\s+\w+\s+import)\s+|"
            r"(?:^|\n)\s*function\s+\w+\s*\(|"
            r"#include\s*<|"
            r"(?:^|\n)\s{4,}\w+.*[;{]",
            text,
            re.MULTILINE
        )),
        
        "has_patch": bool(re.search(
            r"diff --git|\.patch\b|commit [0-9a-f]{6,40}\b|"
            r"^\+{3}\s+|^-{3}\s+|"
            r"^@@.*@@",
            text,
            re.MULTILINE
        )),
        
        "has_enumeration": bool(re.search(
            r"(?:^|\n)\s*(?:\d+[\.\)]|[-*])\s+[A-Z\w]",
            text,
            re.MULTILINE
        ))
    }


# ============================================================================
# QUALITY METRICS
# ============================================================================

def ambiguity_score_v2(text: str):
    """Enhanced ambiguity detection."""
    vague_terms = {
        "maybe", "sometimes", "appears", "seems", "probably",
        "might", "could", "apparently", "likely", "unsure",
        "perhaps", "possibly", "unclear", "uncertain", "somehow",
        "something", "somewhere", "somewhat", "kind of", "sort of"
    }
    
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


# ============================================================================
# NEW: ENHANCED EMBEDDING FEATURES
# ============================================================================

def extract_semantic_diversity_features(text: str):
    """
    Measure semantic richness and diversity using sentence-level embeddings.
    Returns entropy, spread, and coherence metrics.
    """
    sents = [s.text.strip() for s in nlp(text).sents if s.text.strip()]
    
    if len(sents) < MIN_SENTENCES_FOR_DIVERSITY:
        return {
            "semantic_entropy": 0.0,
            "semantic_spread_pc1": 0.0,
            "semantic_spread_pc2": 0.0,
            "semantic_coherence": 1.0,  # Single sentence is perfectly coherent
            #"num_semantic_sentences": len(sents)
        }
    
    try:
        # Get sentence embeddings
        sent_embeddings = model.encode(sents, normalize_embeddings=True)
        
        # Apply PCA to find principal components
        n_components = min(len(sents), 10)
        pca = PCA(n_components=n_components)
        transformed = pca.fit_transform(sent_embeddings)
        
        # Semantic entropy: diversity of semantic content
        # Higher entropy = more diverse topics covered
        var_ratios = pca.explained_variance_ratio_
        epsilon = 1e-10  # Avoid log(0)
        semantic_entropy = -np.sum(var_ratios * np.log(var_ratios + epsilon))
        
        # Semantic spread: how much variance along principal components
        # Higher spread = sentences cover wider semantic space
        spread_pc1 = np.std(transformed[:, 0]) if transformed.shape[1] > 0 else 0.0
        spread_pc2 = np.std(transformed[:, 1]) if transformed.shape[1] > 1 else 0.0
        
        # Semantic coherence: inverse of average distance between sentences
        # Higher coherence = sentences are more similar to each other
        pairwise_sims = cosine_similarity(sent_embeddings)
        # Get upper triangle (exclude diagonal)
        upper_tri = pairwise_sims[np.triu_indices(len(pairwise_sims), k=1)]
        avg_similarity = np.mean(upper_tri) if upper_tri.size > 0 else 1.0
        
        return {
            "semantic_entropy": float(semantic_entropy),
            "semantic_spread_pc1": float(spread_pc1),
            "semantic_spread_pc2": float(spread_pc2),
            "semantic_coherence": float(avg_similarity),
            "num_semantic_sentences": len(sents)
        }
        
    except Exception as e:
        print(f"Warning: Semantic diversity calculation failed: {e}")
        return {
            "semantic_entropy": 0.0,
            "semantic_spread_pc1": 0.0,
            "semantic_spread_pc2": 0.0,
            "semantic_coherence": 0.0,
            "num_semantic_sentences": len(sents)
        }


def extract_embedding_summary_statistics(embedding):
    """
    Extract summary statistics from the full document embedding.
    These can capture overall semantic properties.
    """
    # L2 norm (magnitude of embedding)
    norm = np.linalg.norm(embedding)
    
    # Statistics on embedding dimensions
    mean_val = np.mean(embedding)
    std_val = np.std(embedding)
    min_val = np.min(embedding)
    max_val = np.max(embedding)
    
    # Number of near-zero dimensions (sparsity indicator)
    near_zero_count = np.sum(np.abs(embedding) < 0.01)
    sparsity_ratio = near_zero_count / len(embedding)
    
    # Skewness proxy: ratio of positive to negative dimensions
    pos_count = np.sum(embedding > 0)
    neg_count = np.sum(embedding < 0)
    pos_neg_ratio = pos_count / (neg_count + 1)  # Avoid division by zero
    
    return {
        "embedding_norm": float(norm),
        "embedding_mean": float(mean_val),
        "embedding_std": float(std_val),
        "embedding_min": float(min_val),
        "embedding_max": float(max_val),
        "embedding_sparsity": float(sparsity_ratio),
        "embedding_pos_neg_ratio": float(pos_neg_ratio)
    }


# ============================================================================
# ENHANCED EXCEPTION SEMANTICS
# ============================================================================

def parse_stacktrace_semantics_enhanced(text: str):
    """
    Enhanced exception semantic extraction with categorization.
    """
    # Basic exception extraction
    exception_pattern = r"\b([A-Za-z_][A-Za-z0-9_]*(?:Exception|Error|Throwable))\b"
    exception_types = [m.group(1) for m in re.finditer(exception_pattern, text)]
    unique_exceptions = sorted(set(exception_types))
    primary_exception = unique_exceptions[0] if unique_exceptions else None
    
    # Frame lines
    frame_pattern = r"^\s*at\s+.+"
    num_frames = len(re.findall(frame_pattern, text, flags=re.MULTILINE))
    
    num_caused_by = len(re.findall(r"Caused by:", text))
    
    # NEW: Exception categorization
    exception_categories = {
        'null_pointer': ['NullPointerException', 'NullReferenceException', 'NullPointer'],
        'type_error': ['ClassCastException', 'TypeError', 'TypeException', 'TypeMismatch'],
        'index_error': ['IndexOutOfBoundsException', 'ArrayIndexOutOfBoundsException', 'IndexError'],
        'io_error': ['IOException', 'FileNotFoundException', 'SocketException', 'EOFException'],
        'runtime_error': ['RuntimeException', 'IllegalStateException', 'IllegalArgumentException'],
        'concurrency': ['ConcurrentModificationException', 'DeadlockException', 'InterruptedException'],
        'assertion': ['AssertionError', 'AssertionFailedException'],
        'arithmetic': ['ArithmeticException', 'DivisionByZero', 'NumberFormatException']
    }
    
    exception_category_counts = {}
    for cat, exc_types in exception_categories.items():
        count = sum(1 for exc in exception_types if any(exc_type in exc for exc_type in exc_types))
        exception_category_counts[f"exc_cat_{cat}"] = count
    
    # Has at least one exception in each category
    for cat, exc_types in exception_categories.items():
        has_cat = any(any(exc_type in exc for exc_type in exc_types) for exc in exception_types)
        exception_category_counts[f"has_exc_{cat}"] = int(has_cat)
    
    # NEW: Exception source analysis (library vs user code)
    frame_pattern_detailed = r"at\s+([a-zA-Z0-9_.]+)\."
    frames = re.findall(frame_pattern_detailed, text)
    
    # Common library packages
    library_packages = [
        'java.', 'javax.', 'sun.', 'com.sun.',
        'org.apache.', 'org.springframework.', 'com.google.',
        'org.junit.', 'org.testng.',
        'scala.', 'kotlin.'
    ]
    
    library_frames = sum(1 for f in frames if any(f.startswith(pkg) for pkg in library_packages))
    user_frames = len(frames) - library_frames
    
    return {
        "primary_exception_type": primary_exception,
        "num_exception_types": len(unique_exceptions),
        "stacktrace_depth": num_frames,
        "num_caused_by": num_caused_by,
        **exception_category_counts,
        "exception_library_frames": library_frames,
        "exception_user_frames": user_frames,
        "exception_user_frame_ratio": user_frames / len(frames) if frames else 0.0,
        "exception_avg_frame_package_depth": np.mean([f.count('.') for f in frames]) if frames else 0.0
    }


# ============================================================================
# EXISTING FEATURE FUNCTIONS (keeping for compatibility)
# ============================================================================

def load_bee_results_dict(jsonl_path: str) -> dict:
    """Load bee_results.jsonl and return a dictionary mapping bug_id to stats."""
    bee_dict = {}
    
    if not os.path.exists(jsonl_path):
        print(f"  Warning: {jsonl_path} not found, completeness_score will use fallback methods")
        return bee_dict
    
    try:
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    bug_id = record.get('bug_id')
                    stats = record.get('stats', {})
                    
                    if bug_id and stats:
                        bee_dict[bug_id] = {
                            'has_S2R': stats.get('has_S2R', False),
                            'has_OB': stats.get('has_OB', False),
                            'has_EB': stats.get('has_EB', False),
                        }
                except json.JSONDecodeError:
                    continue
    except Exception as e:
        print(f"  Warning: Failed to load bee_results: {e}")
    
    return bee_dict


def completeness_score(text: str, flags: dict, bug_id: str = None, bee_results: dict = None):
    """Calculate completeness score based on presence of key bug report elements."""
    score = 0.0
    
    if len(text.strip()) > 50:
        score += 1.0
    
    if flags.get("has_stacktrace", False):
        score += 1.0
    
    if flags.get("has_code", False):
        score += 1.0
    
    has_steps = False
    if bee_results and bug_id and bug_id in bee_results:
        has_steps = bee_results[bug_id].get('has_S2R', False)
    else:
        lower = text.lower()
        step_indicators = [
            r"step\s+\d+", r"step\s+[1-9]", r"reproduce", r"reproduction",
            r"steps:", r"steps\s+to", r"how\s+to\s+reproduce", r"reproduction\s+steps"
        ]
        has_steps = any(re.search(pattern, lower) for pattern in step_indicators)
    
    if has_steps:
        score += 1.0
    
    has_expected_observed = False
    if bee_results and bug_id and bug_id in bee_results:
        bee_stats = bee_results[bug_id]
        has_expected_observed = bee_stats.get('has_OB', False) and bee_stats.get('has_EB', False)
    else:
        lower = text.lower()
        expected_patterns = [
            r"expected", r"should\s+(?:be|have|do|show|display)",
            r"but\s+(?:got|received|observed|actual)",
            r"actual\s+(?:result|output|behavior)", r"instead\s+(?:of|I|we)"
        ]
        observed_patterns = [r"observed", r"actual", r"got", r"received", r"instead"]
        has_expected = any(re.search(pattern, lower) for pattern in expected_patterns)
        has_observed = any(re.search(pattern, lower) for pattern in observed_patterns)
        has_expected_observed = has_expected and has_observed
    
    if has_expected_observed:
        score += 1.0
    
    return score


def extract_causal_features(text: str, n_sentences=None):
    """Capture causal and temporal connectives in the report."""
    lower = text.lower()

    causal_markers = [
        "because", "since", "so that", "due to", "therefore", "hence",
        "thus", "consequently", "as a result", "in order to", "if",
        "when", "whenever", "unless", "until", "after", "before", "once", "while",
    ]
    temporal_markers = [
        "then", "next", "afterwards", "later", "finally", "subsequently",
        "eventually", "initially", "first", "second", "third",
    ]

    def _count_phrases(text_lower: str, phrases):
        count = 0
        for p in phrases:
            if " " in p:
                count += len(re.findall(re.escape(p), text_lower))
            else:
                count += len(re.findall(r"\b" + re.escape(p) + r"\b", text_lower))
        return count

    num_causal = _count_phrases(lower, causal_markers)
    num_temporal = _count_phrases(lower, temporal_markers)

    if not n_sentences:
        approx = len(re.split(r"[.!?]+", text)) - 1
        n_sentences = approx if approx > 0 else 1

    return {
        "num_causal_markers": num_causal,
        "num_temporal_markers": num_temporal,
        "causal_density": float(num_causal) / n_sentences if n_sentences else 0.0,
        "temporal_density": float(num_temporal) / n_sentences if n_sentences else 0.0,
    }


def extract_context_features(text: str):
    """Execution / environment context richness."""
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


def extract_behavior_features(text: str, n_sentences=None):
    """Count behavior-describing verbs."""
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

    if not n_sentences:
        n_sentences = sum(1 for _ in doc.sents) or 1

    return {
        "behavior_verb_count": behavior_count,
        "behavior_verb_density": float(behavior_count) / n_sentences if n_sentences else 0.0,
    }


def compute_modality_features(text: str, n_sentences=None):
    """Simple intent / obligation proxy based on modal verbs."""
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
        #"n_tokens": n_tokens,
        #"n_words": n_words,
        "n_sentences": n_sentences,
        #"flesch_reading_ease": flesch,
        #"smog_index": smog,
        "gunning_fog": gunning,
        #"coleman_liau_index": coleman,
        #"automated_readability_index": ari,
    }


# ============================================================================
# CLUSTERING (SECOND PASS)
# ============================================================================

def fit_embedding_clusters(embeddings_list, n_clusters=NUM_SEMANTIC_CLUSTERS):
    """
    Fit PCA and KMeans on all collected embeddings.
    Returns fitted models.
    """
    print(f"\nFitting semantic clusters on {len(embeddings_list)} bug reports...")
    
    if len(embeddings_list) < n_clusters:
        print(f"  Warning: Only {len(embeddings_list)} bugs, reducing clusters to {len(embeddings_list)}")
        n_clusters = len(embeddings_list)
    
    embeddings_array = np.array(embeddings_list)
    
    # PCA for dimensionality reduction
    pca = PCA(n_components=min(PCA_COMPONENTS, len(embeddings_list), embeddings_array.shape[1]))
    reduced = pca.fit_transform(embeddings_array)
    
    # Ensure float64 dtype for KMeans (sklearn expects double precision)
    reduced = reduced.astype(np.float64)
    
    print(f"  PCA: reduced from {embeddings_array.shape[1]} to {reduced.shape[1]} dimensions")
    print(f"  Explained variance: {pca.explained_variance_ratio_[:5].sum():.2%} (first 5 components)")
    
    # KMeans clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(reduced)
    
    # Cluster distribution
    unique, counts = np.unique(clusters, return_counts=True)
    print(f"  Cluster distribution: {dict(zip(unique, counts))}")
    
    return pca, kmeans, reduced


def assign_cluster_features(embedding, pca, kmeans, reduced_embeddings=None):
    """
    Assign cluster membership and distance features to a single embedding.
    """
    # Reduce dimensionality
    reduced = pca.transform([embedding])
    
    # Ensure float64 dtype for KMeans (sklearn expects double precision)
    reduced = reduced.astype(np.float64)
    
    # Predict cluster
    cluster = kmeans.predict(reduced)[0]
    
    # Distance to cluster center
    cluster_center = kmeans.cluster_centers_[cluster]
    distance_to_center = np.linalg.norm(reduced[0] - cluster_center)
    
    # Distance to nearest other cluster center (separation)
    all_distances = np.linalg.norm(kmeans.cluster_centers_ - reduced[0], axis=1)
    sorted_distances = np.sort(all_distances)
    distance_to_nearest_other = sorted_distances[1] if len(sorted_distances) > 1 else 0.0
    
    return {
        "embedding_cluster": int(cluster),
        "embedding_cluster_distance": float(distance_to_center),
        "embedding_cluster_separation": float(distance_to_nearest_other),
        "embedding_cluster_size": int(np.sum(kmeans.labels_ == cluster))
    }


# ----------------- MAIN ---------------------
def main():
    # Load bee_results if available
    print("Loading bee_results...")
    bee_results = load_bee_results_dict(BEE_RESULTS_FILE)
    if bee_results:
        print(f"  Loaded bee_results for {len(bee_results)} bugs")
    else:
        print("  No bee_results found, will use fallback regex patterns")
    
    # Find files
    if USE_XML:
        files = glob.glob(os.path.join(DATA_DIR, "*.xml"))
        print(f"Found {len(files)} XML files under {DATA_DIR!r}")
    else:
        files = glob.glob(os.path.join(DATA_DIR, "**/*.json"), recursive=True)
        print(f"Found {len(files)} JSON files under {DATA_DIR!r}")
    
    try:
        from tqdm import tqdm
        files = tqdm(files, desc="Processing files (Pass 1: Feature extraction)")
    except ImportError:
        pass
    
    rows = []

    # PASS 1: Extract all features and collect embeddings
    print("\n" + "="*80)
    print("PASS 1: Extracting features and collecting embeddings")
    print("="*80)
    
    for path in files:
        try:
            for bug in parse_bug_file(path):
                text = (bug["summary"] + " " + bug["description"]).strip()

                # Syntactic features
                syn = compute_syntactic_features(text)
                
                # Structural flags
                flags = extract_structural_flags_v2(text)
                
                # Document embedding (for clustering later)
                emb = model.encode(text, normalize_embeddings=True)
                
                if emb.shape[0] != 384:
                    print(f"Warning: Unexpected embedding dimension for {bug['id']}: {emb.shape}")
                
                # Store for clustering
                global_embeddings.append(emb)
                global_bug_ids.append(bug["id"])
                
                # Semantic features (existing)
                redund = semantic_redundancy(text)
                ambig = ambiguity_score_v2(text)
                
                # NEW: Semantic diversity features
                diversity_feats = extract_semantic_diversity_features(text)
                
                # NEW: Embedding summary statistics
                embedding_stats = extract_embedding_summary_statistics(emb)
                
                # Other features (existing)
                causal_feats = extract_causal_features(text, syn.get("n_sentences"))
                context_feats = extract_context_features(text)
                
                # Enhanced stacktrace features
                if flags.get("has_stacktrace"):
                    stacktrace_feats = parse_stacktrace_semantics_enhanced(text)
                else:
                    # Default values
                    stacktrace_feats = {
                        "primary_exception_type": None,
                        "num_exception_types": 0,
                        "stacktrace_depth": 0,
                        "num_caused_by": 0,
                        "exc_cat_null_pointer": 0,
                        "exc_cat_type_error": 0,
                        "exc_cat_index_error": 0,
                        "exc_cat_io_error": 0,
                        "exc_cat_runtime_error": 0,
                        "exc_cat_concurrency": 0,
                        #"exc_cat_assertion": 0,
                        "exc_cat_arithmetic": 0,
                        "has_exc_null_pointer": 0,
                        "has_exc_type_error": 0,
                        "has_exc_index_error": 0,
                        "has_exc_io_error": 0,
                        "has_exc_runtime_error": 0,
                        "has_exc_concurrency": 0,
                        "has_exc_assertion": 0,
                        "has_exc_arithmetic": 0,
                        "exception_library_frames": 0,
                        "exception_user_frames": 0,
                        "exception_user_frame_ratio": 0.0,
                        "exception_avg_frame_package_depth": 0.0
                    }
                
                behavior_feats = extract_behavior_features(text, syn.get("n_sentences"))
                modality_feats = compute_modality_features(text, syn.get("n_sentences"))
                
                # Completeness score
                complete_score = completeness_score(text, flags, bug_id=bug["id"], bee_results=bee_results)

                # Combine all features (without clustering yet)
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
                    **diversity_feats,
                    **embedding_stats,
                    "completeness_score": complete_score,
                    "embedding": json.dumps(emb.tolist()),
                })

        except Exception as e:
            print(f"Error processing {path}: {e}")
            continue

    # PASS 2: Fit clustering models
    print("\n" + "="*80)
    print("PASS 2: Fitting semantic clustering models")
    print("="*80)
    
    if len(global_embeddings) > 0:
        pca, kmeans, reduced = fit_embedding_clusters(global_embeddings)
        
        # Assign cluster features to each bug
        print("\nAssigning cluster features to bug reports...")
        for i, row in enumerate(rows):
            cluster_feats = assign_cluster_features(
                global_embeddings[i], 
                pca, 
                kmeans
            )
            row.update(cluster_feats)
    else:
        print("Warning: No embeddings collected, skipping clustering")

    # PASS 3: Save results
    print("\n" + "="*80)
    print("PASS 3: Saving results")
    print("="*80)
    
    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"\nSaved {len(df)} bug feature rows to {OUTPUT_FILE}")
    
    # Summary statistics
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    print(f"  Total bugs processed: {len(df)}")
    print(f"  Average description length: {df['description_chars'].mean():.0f} chars")
    print(f"  Reports with code: {df['has_code'].sum()} ({df['has_code'].mean()*100:.1f}%)")
    print(f"  Reports with stacktraces: {df['has_stacktrace'].sum()} ({df['has_stacktrace'].mean()*100:.1f}%)")
    
    if 'completeness_score' in df.columns:
        print(f"\n  Average completeness score: {df['completeness_score'].mean():.2f}")
    
    # NEW: Semantic feature statistics
    print(f"\n  Semantic diversity metrics:")
    print(f"    Average semantic entropy: {df['semantic_entropy'].mean():.3f}")
    print(f"    Average semantic coherence: {df['semantic_coherence'].mean():.3f}")
    print(f"    Average embedding norm: {df['embedding_norm'].mean():.3f}")
    
    if 'embedding_cluster' in df.columns:
        print(f"\n  Clustering results:")
        print(f"    Number of clusters: {df['embedding_cluster'].nunique()}")
        cluster_counts = df['embedding_cluster'].value_counts().sort_index()
        print(f"    Cluster sizes: {dict(cluster_counts)}")
    
    # Exception category statistics
    exc_cat_cols = [c for c in df.columns if c.startswith('exc_cat_')]
    if exc_cat_cols:
        print(f"\n  Exception categories:")
        for col in exc_cat_cols:
            cat_name = col.replace('exc_cat_', '')
            count = df[col].sum()
            pct = (count / len(df)) * 100
            print(f"    {cat_name}: {count} ({pct:.1f}%)")
    
    print("="*80)
    print("\nFeature extraction complete!")
    print(f"Enhanced features include:")
    print(f"  • Semantic diversity (entropy, spread, coherence)")
    print(f"  • Embedding summary statistics (norm, sparsity, distribution)")
    print(f"  • Semantic clustering ({NUM_SEMANTIC_CLUSTERS} clusters)")
    print(f"  • Enhanced exception categorization (8 categories)")
    print(f"  • Exception source analysis (library vs user code)")
    print("="*80)

if __name__ == "__main__":
    main()