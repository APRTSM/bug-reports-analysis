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

DATA_DIR = "bug_reports/Defects4J"
OUTPUT_FILE = "gemini_bug_ratings.csv"
# Try these model names in order (fallback if one doesn't work)
# Note: Model names should match what's available in your API
GEMINI_MODELS = [
    "gemini-2.5-pro",           # Latest pro model
    "gemini-2.0-flash",        # Fast and efficient
    "gemini-pro-latest",        # Fallback to latest pro
    "gemini-2.5-flash",         # Alternative flash model
]
GEMINI_MODEL = GEMINI_MODELS[0]  # Start with gemini-2.5-pro

# Get API key from environment variable or set here (not recommended)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    print("Warning: GEMINI_API_KEY environment variable not set.")
    print("Please set it with: export GEMINI_API_KEY='your-api-key'")
    print("Or set it directly in the script (not recommended for production)")
    # Uncomment and set your key here if needed:
    # GEMINI_API_KEY = "your-api-key-here"

# Rate limiting: delay between API calls (seconds)
# Increased to avoid hitting free tier limits
API_DELAY = 2.0

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


def create_rating_prompt(title: str, description: str) -> str:
    """Create a prompt for Gemini to rate the bug report."""
    prompt = f"""You are an expert software engineer evaluating a bug report. Please analyze the following bug report and provide ratings in JSON format.

Bug Report Title: {title}

Bug Report Description:
{description}

Please evaluate this bug report on the following dimensions and return ONLY a valid JSON object with these exact fields:

{{
  "actionability": <integer 0-5>,  // How actionable is this bug report? Can a developer immediately start fixing it?
  "clarity": <integer 0-5>,  // How clear and well-written is the description?
  "specificity": <integer 0-5>,  // How specific are the details (versions, steps, inputs, etc.)?
  "expected_observed_alignment": <integer 0-5>,  // How well does it describe what was expected vs what was observed?
  "root_cause_guess": "<string>",  // Your best guess at the root cause (e.g., "null pointer", "type mismatch", "race condition", "configuration issue", etc.)
  "technical_depth": <integer 0-5>,  // How technically detailed is the report? Does it include code, stack traces, technical context?
  "ambiguity_types": ["<string>", ...],  // List of ambiguity types present: "missing steps", "vague inputs", "unclear error messages", "missing context", "unclear reproduction", "missing environment info", etc. (empty list if none)
  "hidden_s2r_present": <boolean>,  // Are there implicit steps to reproduce hidden in the description (not explicitly listed)?
  "causal_reasoning_quality": <integer 0-5>,  // How well does the reporter explain cause-and-effect relationships?
  "contradiction_present": <boolean>,  // Are there any contradictions or conflicting information in the report?
  "repair_difficulty": <integer 0-5>,  // How difficult would it be to fix this bug based on the information provided? (0=very easy, 5=very difficult)
  "likely_impacted_code_concepts": ["<string>", ...]  // List of code concepts likely impacted (e.g., "JSON parsing", "UI rendering", "database queries", "authentication", etc.)
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
    # Try each model in order if one fails
    for model_name in GEMINI_MODELS:
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
                        return None
                    break  # Try next model
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
    
    # Find all JSON files
    json_files = glob.glob(os.path.join(data_dir, "**/*.json"), recursive=True)
    print(f"Found {len(json_files)} JSON files under {data_dir!r}")
    
    try:
        from tqdm import tqdm
        json_files = tqdm(json_files, desc="Processing bug reports")
    except ImportError:
        pass
    
    failed_count = 0
    success_count = 0
    skipped_count = 0
    
    for path in json_files:
        try:
            # Parse bug report
            name = os.path.splitext(os.path.basename(path))[0]
            for obj in read_json(path):
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
                
                if rating:
                    # Combine bug report info with rating
                    row = {
                        "id": bid,
                        "title": title,
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

