"""
gemini_bug_ratings.py

Uses Google Gemini API to evaluate bug reports on multiple dimensions:
- Actionability, clarity, specificity, expected/observed alignment
- Root cause guess, technical depth, ambiguity types
- Hidden S2R presence, causal reasoning quality, contradictions
- Bug type taxonomy, repair difficulty, impacted code concepts

Usage:
    python gemini_bug_ratings.py

Requires:
    - Google Gemini API key set as environment variable GEMINI_API_KEY
    - Or set in the script directly (not recommended for production)
"""

import os
import glob
import json
import re
import xml.etree.ElementTree as ET
import pandas as pd
from typing import Dict, Any, Optional
import time

try:
    import google.generativeai as genai
except ImportError:
    print("Error: google-generativeai package not installed.")
    print("Install with: pip install google-generativeai")
    exit(1)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DATA_DIR = "defects4j_xml"  # Changed from "bug_reports/Defects4J" to use XML files
OUTPUT_FILE = "gemini_bug_ratings.csv"
USE_XML = True  # Set to True to use XML files, False to use JSON files
# Try these model names in order (fallback if one doesn't work)
# Note: Model names should match what's available in your API
GEMINI_MODELS = [
    #"gemini-2.0-flash",     # Fast and (usually) has free-tier quota
    "gemini-2.5-flash",     # Alternative flash model
    "gemini-pro-latest",    # Fallback
]
GEMINI_MODEL = GEMINI_MODELS[0]

# Get API key from environment variable or set here (not recommended)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    print("Warning: GEMINI_API_KEY environment variable not set.")
    print("Please set it with: export GEMINI_API_KEY='your-api-key'")
    print("Or set it directly in the script (not recommended for production)")
    # Uncomment and set your key here if needed:
    GEMINI_API_KEY = "AIzaSyCk_jxNfBN7o3_Vn5F8Hfa5dzv1aR20KQY"

# Rate limiting: delay between API calls (seconds)
# Increased to avoid hitting free tier limits
API_DELAY = 5.0  # Increased delay for free tier

# Maximum retries for failed API calls
MAX_RETRIES = 5

# Maximum wait time for rate limit errors (seconds)
MAX_RATE_LIMIT_WAIT = 300  # 5 minutes


# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------

def read_json(path: str) -> list:
    """Load JSON or JSONL; return a list of dicts."""
    with open(path, "r", encoding="utf-8") as f:
        text = f.read().strip()
        if not text:
            return []
        if "\n" in text and not text.lstrip().startswith("{"):
            return [json.loads(line) for line in text.splitlines() if line.strip()]
        obj = json.loads(text)
        return [obj] if isinstance(obj, dict) else obj

def read_xml(path: str) -> list:
    """Load XML bug report; return a list of dicts with bug information."""
    try:
        # Read file content first to handle encoding issues
        with open(path, 'rb') as f:
            raw_content = f.read()
        
        # Try to decode with UTF-8, fallback to latin-1 for special characters
        try:
            content = raw_content.decode('utf-8')
        except UnicodeDecodeError:
            content = raw_content.decode('latin-1', errors='replace')
        
        # Replace problematic characters that might break XML parsing
        content = re.sub(r'[\x00-\x08\x0B-\x0C\x0E-\x1F]', '', content)
        
        # Parse from string instead of file
        root = ET.fromstring(content)
        
        bugs = []
        # Handle both <bugRepository><bug> and direct <bug> structures
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
                
                # Extract text and handle HTML entities
                title = ""
                if summary_elem is not None and summary_elem.text:
                    title = summary_elem.text.strip()
                    # Decode HTML entities
                    title = title.replace('&lt;', '<').replace('&gt;', '>').replace('&amp;', '&')
                
                desc = ""
                if desc_elem is not None and desc_elem.text:
                    desc = desc_elem.text.strip()
                    # Decode HTML entities
                    desc = desc.replace('&lt;', '<').replace('&gt;', '>').replace('&amp;', '&')
                
                # Extract bug ID from filename - always use filename format for clarity
                # Format: "Lang_19" -> "Lang-19", "Time_4" -> "Time-4"
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
                    "summary": title,  # Alias for compatibility
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

def read_bug_file(path: str) -> list:
    """Read bug report from either JSON or XML file."""
    if USE_XML and path.endswith('.xml'):
        return read_xml(path)
    else:
        return read_json(path)


def create_rating_prompt(title: str, description: str) -> str:
    """Create a prompt for Gemini to rate the bug report."""
    prompt = f"""You are an expert software engineer evaluating a bug report. Please analyze the following bug report and provide ratings in JSON format.

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
"""
    return prompt


def parse_gemini_response(response_text: str) -> Optional[Dict[str, Any]]:
    """Parse Gemini's response and extract the JSON rating."""
    # Try to extract JSON from the response
    # Gemini might wrap it in markdown code blocks
    text = response_text.strip()
    
    # Remove markdown code blocks if present
    if text.startswith("```json"):
        text = text[7:]  # Remove ```json
    elif text.startswith("```"):
        text = text[3:]   # Remove ```
    
    if text.endswith("```"):
        text = text[:-3]
    
    text = text.strip()
    
    try:
        rating = json.loads(text)
        return rating
    except json.JSONDecodeError as e:
        print(f"  Warning: Failed to parse JSON response: {e}")
        print(f"  Response text: {text[:200]}...")
        return None


def extract_retry_delay(error_msg: str) -> Optional[float]:
    """Extract retry delay from rate limit error message."""
    import re
    # Look for "Please retry in X.XXs" pattern
    match = re.search(r'retry in ([\d.]+)s', error_msg, re.IGNORECASE)
    if match:
        return float(match.group(1))
    
    # Look for retry_delay { seconds: X } pattern
    match = re.search(r'seconds:\s*(\d+)', error_msg)
    if match:
        return float(match.group(1))
    
    return None


def call_gemini_api(prompt: str, retries: int = MAX_RETRIES) -> Optional[str]:
    """Call Gemini API with retry logic and model fallback."""
    quota_exhausted_models = set()  # Track models that have exhausted quota
    
    # Try each model in order if one fails
    for model_name in GEMINI_MODELS:
        # Skip models that have exhausted quota
        if model_name in quota_exhausted_models:
            continue
            
        for attempt in range(retries):
            try:
                model = genai.GenerativeModel(model_name)
                response = model.generate_content(prompt)
                # If we used a different model, update the global
                global GEMINI_MODEL
                if model_name != GEMINI_MODEL:
                    print(f"  Using model: {model_name}")
                    GEMINI_MODEL = model_name
                return response.text
            except Exception as e:
                error_msg = str(e)
                
                # Check for rate limit errors (429)
                if "429" in error_msg or "quota" in error_msg.lower() or "rate limit" in error_msg.lower():
                    # Check if it's a daily quota exhaustion (limit: 0)
                    if "limit: 0" in error_msg or "quota exceeded" in error_msg.lower():
                        print(f"  ⚠ Daily quota exhausted for {model_name}. Skipping this model.")
                        quota_exhausted_models.add(model_name)
                        break  # Try next model
                    
                    retry_delay = extract_retry_delay(error_msg)
                    if retry_delay:
                        wait_time = min(retry_delay + 5, MAX_RATE_LIMIT_WAIT)  # Add buffer, cap at max
                        print(f"  Rate limit exceeded. Waiting {wait_time:.1f}s before retry...")
                        time.sleep(wait_time)
                        continue  # Retry with same model
                    else:
                        # No retry delay specified, use exponential backoff
                        wait_time = min((attempt + 1) * 10, MAX_RATE_LIMIT_WAIT)
                        print(f"  Rate limit exceeded (attempt {attempt + 1}/{retries}). Waiting {wait_time}s...")
                        time.sleep(wait_time)
                        continue
                
                # If it's a 404/model not found error, try next model
                if "404" in error_msg or "not found" in error_msg.lower() or "not supported" in error_msg.lower():
                    if model_name != GEMINI_MODELS[-1]:  # Not the last model
                        print(f"  Model {model_name} not available, trying next model...")
                        break  # Break inner loop, try next model
                    else:
                        # Last model failed, return None
                        print(f"  Error: All models failed. Last error: {e}")
                        return None
                elif attempt < retries - 1:
                    wait_time = (attempt + 1) * 2  # Exponential backoff
                    print(f"  API call failed (attempt {attempt + 1}/{retries}), retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    print(f"  Error calling Gemini API with {model_name}: {e}")
                    if model_name == GEMINI_MODELS[-1]:
                        # All models exhausted
                        if quota_exhausted_models:
                            print(f"  ❌ All available models have exhausted quota: {', '.join(quota_exhausted_models)}")
                            print(f"  💡 Please wait and try again later, or upgrade your API plan.")
                        return None
                    break  # Try next model
    
    # If we get here, all models failed
    if quota_exhausted_models:
        print(f"  ❌ All models exhausted quota. Please wait and try again later.")
    return None


def rate_bug_report(title: str, description: str) -> Optional[Dict[str, Any]]:
    """Rate a single bug report using Gemini API."""
    prompt = create_rating_prompt(title, description)
    response = call_gemini_api(prompt)
    
    if not response:
        return None
    
    rating = parse_gemini_response(response)
    return rating


# ---------------------------------------------------------------------------
# Main Processing Function
# ---------------------------------------------------------------------------

def list_available_models():
    """List available Gemini models for debugging."""
    try:
        models = genai.list_models()
        print("\nAvailable models:")
        available = []
        for model in models:
            if 'generateContent' in model.supported_generation_methods:
                # Remove 'models/' prefix if present
                model_name = model.name.replace('models/', '')
                print(f"  - {model.name}")
                available.append(model_name)
        return available
    except Exception as e:
        print(f"Could not list models: {e}")
        return []


def get_best_available_model():
    """Automatically select the best available model from the API."""
    available = list_available_models()
    if not available:
        return None
    
    # Prefer these models in order (flash models are cheaper/faster, better for free tier)
    preferred = [
        "gemini-2.0-flash",      # Fast and efficient, better for rate limits
        "gemini-2.5-flash",      # Alternative flash model
        "gemini-2.0-flash-lite", # Lightest option
        "gemini-2.5-pro",        # Pro model (more expensive)
        "gemini-pro-latest",     # Fallback
    ]
    
    # Find first preferred model that's available
    for pref in preferred:
        # Check exact match or if any available model contains the preferred name
        for avail in available:
            if pref in avail or avail == pref:
                return avail
    
    # If no preferred model found, return first available
    return available[0] if available else None


def get_processed_ids(output_file: str) -> set:
    """Get set of already processed bug report IDs from existing CSV."""
    if not os.path.exists(output_file):
        return set()
    try:
        df = pd.read_csv(output_file)
        return set(df['id'].astype(str).tolist())
    except Exception as e:
        print(f"Warning: Could not read existing CSV: {e}")
        return set()


def write_row_to_csv(row: dict, output_file: str, is_first_row: bool = False):
    """Write a single row to CSV file, appending if file exists."""
    # Convert list columns to JSON strings for CSV compatibility
    for col in ["ambiguity_types", "likely_impacted_code_concepts"]:
        if col in row and isinstance(row[col], list):
            row[col] = json.dumps(row[col])
    
    # Create DataFrame with single row
    df = pd.DataFrame([row])
    
    # Write header if new file, append if existing
    df.to_csv(output_file, mode='a' if not is_first_row else 'w', 
              header=is_first_row, index=False)


def process_bug_reports(data_dir: str = DATA_DIR, output_file: str = OUTPUT_FILE):
    """
    Process all bug reports in Defects4J directory and get Gemini ratings.
    Saves results incrementally to CSV to prevent data loss.
    """
    # Initialize Gemini
    if not GEMINI_API_KEY:
        print("Error: GEMINI_API_KEY not set. Cannot proceed.")
        return
    
    genai.configure(api_key=GEMINI_API_KEY)
    print(f"Initialized Gemini API")
    
    # Auto-detect best available model
    print("\nChecking available models...")
    best_model = get_best_available_model()
    if best_model:
        global GEMINI_MODEL, GEMINI_MODELS
        GEMINI_MODEL = best_model
        # Update model list to start with the best available
        if best_model not in GEMINI_MODELS:
            GEMINI_MODELS.insert(0, best_model)
        print(f"\nUsing model: {GEMINI_MODEL}")
        print(f"Will try models in order: {', '.join(GEMINI_MODELS[:3])}...")
    else:
        print(f"\nWarning: Could not auto-detect model. Will try: {', '.join(GEMINI_MODELS)}")
    
    # Check for existing results to resume from
    processed_ids = get_processed_ids(output_file)
    if processed_ids:
        print(f"\nFound {len(processed_ids)} already processed reports. Will skip them.")
        is_first_row = False
    else:
        is_first_row = True
    
    # Find all files (XML or JSON based on USE_XML)
    if USE_XML:
        files = glob.glob(os.path.join(data_dir, "*.xml"))
        print(f"Found {len(files)} XML files under {data_dir!r}")
    else:
        files = glob.glob(os.path.join(data_dir, "**/*.json"), recursive=True)
        print(f"Found {len(files)} JSON files under {data_dir!r}")
    
    try:
        from tqdm import tqdm
        files = tqdm(files, desc="Processing bug reports")
    except ImportError:
        pass
    
    failed_count = 0
    success_count = 0
    skipped_count = 0
    
    for path in files:
        try:
            # Parse bug report
            name = os.path.splitext(os.path.basename(path))[0]
            for obj in read_bug_file(path):
                title = (obj.get("title") or obj.get("summary") or "").strip()
                desc = (obj.get("description") or "").strip()
                if not (title or desc):
                    continue
                bid = str(obj.get("id") or obj.get("bug_id") or name)
                
                # Skip if already processed
                if bid in processed_ids:
                    skipped_count += 1
                    continue
                
                # Get Gemini rating
                print(f"Rating bug report: {bid}")
                rating = rate_bug_report(title, desc)
                
                # Add delay after EVERY API call (successful or not) to respect rate limits
                time.sleep(API_DELAY)
                
                if rating:
                    # Combine bug report info with rating
                    row = {
                        "id": bid,
                        "title": title,
                        "description": desc,
                        "description_length": len(desc),
                        **rating
                    }
                    # Write immediately to CSV
                    write_row_to_csv(row, output_file, is_first_row)
                    is_first_row = False
                    processed_ids.add(bid)  # Track as processed
                    success_count += 1
                    print(f"  ✓ Successfully rated {bid} (saved to CSV)")
                else:
                    print(f"  ✗ Failed to rate {bid}")
                    failed_count += 1
                    # Still add a row with null values
                    row = {
                        "id": bid,
                        "title": title,
                        "description": desc,
                        "description_length": len(desc),
                        "actionability": None,
                        "clarity": None,
                        "specificity": None,
                        "expected_observed_alignment": None,
                        "root_cause_guess": None,
                        "technical_depth": None,
                        "ambiguity_types": None,
                        "hidden_s2r_present": None,
                        "causal_reasoning_quality": None,
                        "contradiction_present": None,
                        "bug_type_taxonomy": None,
                        "repair_difficulty": None,
                        "likely_impacted_code_concepts": None,
                    }
                    # Write failed row to CSV too
                    write_row_to_csv(row, output_file, is_first_row)
                    is_first_row = False
                    processed_ids.add(bid)  # Track as processed
                
                # Rate limiting
                time.sleep(API_DELAY)
                
        except Exception as e:
            print(f"Error processing {path}: {e}")
            failed_count += 1
            continue
    
    # Final summary
    total_processed = success_count + failed_count + skipped_count
    print(f"\n{'='*80}")
    print("PROCESSING COMPLETE")
    print(f"{'='*80}")
    print(f"  Total processed: {total_processed}")
    print(f"  Successfully rated: {success_count}")
    print(f"  Failed ratings: {failed_count}")
    print(f"  Skipped (already processed): {skipped_count}")
    print(f"  Results saved to: {output_file}")
    
    # Print summary statistics from CSV
    if os.path.exists(output_file):
        try:
            df = pd.read_csv(output_file)
            print("\n" + "="*80)
            print("SUMMARY STATISTICS")
            print("="*80)
            if len(df) > 0:
                numeric_cols = ["actionability", "clarity", "specificity", "expected_observed_alignment", 
                               "technical_depth", "causal_reasoning_quality", "repair_difficulty"]
                for col in numeric_cols:
                    if col in df.columns:
                        mean_val = df[col].mean()
                        if pd.notna(mean_val):
                            print(f"  Average {col}: {mean_val:.2f}")
                
                if "bug_type_taxonomy" in df.columns:
                    print(f"\n  Bug type distribution:")
                    type_counts = df["bug_type_taxonomy"].value_counts()
                    for bug_type, count in type_counts.items():
                        if pd.notna(bug_type):
                            print(f"    {bug_type}: {count} ({count/len(df)*100:.1f}%)")
                
                if "hidden_s2r_present" in df.columns:
                    hidden_count = df["hidden_s2r_present"].sum()
                    print(f"\n  Reports with hidden S2R: {hidden_count} ({hidden_count/len(df)*100:.1f}%)")
                
                if "contradiction_present" in df.columns:
                    contradiction_count = df["contradiction_present"].sum()
                    print(f"\n  Reports with contradictions: {contradiction_count} ({contradiction_count/len(df)*100:.1f}%)")
            print("="*80)
        except Exception as e:
            print(f"Warning: Could not generate summary statistics: {e}")
    
    # Print summary statistics
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    if len(df) > 0:
        numeric_cols = ["actionability", "clarity", "specificity", "expected_observed_alignment", 
                       "technical_depth", "causal_reasoning_quality", "repair_difficulty"]
        for col in numeric_cols:
            if col in df.columns:
                mean_val = df[col].mean()
                if pd.notna(mean_val):
                    print(f"  Average {col}: {mean_val:.2f}")
        
        if "bug_type_taxonomy" in df.columns:
            print(f"\n  Bug type distribution:")
            type_counts = df["bug_type_taxonomy"].value_counts()
            for bug_type, count in type_counts.items():
                if pd.notna(bug_type):
                    print(f"    {bug_type}: {count} ({count/len(df)*100:.1f}%)")
        
        if "hidden_s2r_present" in df.columns:
            hidden_count = df["hidden_s2r_present"].sum()
            print(f"\n  Reports with hidden S2R: {hidden_count} ({hidden_count/len(df)*100:.1f}%)")
        
        if "contradiction_present" in df.columns:
            contradiction_count = df["contradiction_present"].sum()
            print(f"\n  Reports with contradictions: {contradiction_count} ({contradiction_count/len(df)*100:.1f}%)")
    print("="*80)


if __name__ == "__main__":
    process_bug_reports()

