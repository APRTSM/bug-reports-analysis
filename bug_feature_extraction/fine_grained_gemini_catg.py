"""
fine_grained_gemini_catg.py

Uses Google Gemini API to perform fine-grained categorization of bug reports that are
already categorized as "Functional Issue" into one of 18 sub-categories:
- Logic categories: Exception handling, Missing case, Processing, Typo, Dependency, Other (Logic)
- Memory categories: Buffer overflow, Null pointer dereference, Uninitialized memory read, 
  Memory leak, Dangling pointer, Double free, Other (Memory)
- Concurrency categories: Order violation, Race condition, Atomicity violation, Deadlock, Other (Concurrency)

This script reads gemini_bug_categorization.csv to filter for only bug reports with 
category="Functional Issue" and then performs fine-grained categorization on those.

Usage:
    python fine_grained_gemini_catg.py

Requires:
    - Google Gemini API key set as environment variable GEMINI_API_KEY
    - gemini_bug_categorization.csv file with functional issue categories
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

import os as _os
_ROOT_DIR = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
DATA_DIR = _os.path.join(_ROOT_DIR, "defects4j_xml")  # Changed from "bug_reports/Defects4J" to use XML files
# Written into / read from gemini_ratings/ so it lands where
# tool_feature_analysis/merge_features_and_performance.py expects it, and lines up with
# gemini_bug_categorization_overall.py's output (must be run first).
_os.makedirs(_os.path.join(_ROOT_DIR, "gemini_ratings"), exist_ok=True)
OUTPUT_FILE = _os.path.join(_ROOT_DIR, "gemini_ratings", "fine_grained_gemini_categorization.csv")  # Different output file
FUNCTIONAL_ISSUES_CSV = _os.path.join(_ROOT_DIR, "gemini_ratings", "gemini_bug_categorization.csv")  # CSV with functional issue categories
USE_XML = True  # Set to True to use XML files, False to use JSON files

# Try these model names in order (fallback if one doesn't work)
GEMINI_MODELS = [
    "gemini-2.0-flash",      # Fast and efficient, better for rate limits
    "gemini-2.5-flash",      # Alternative flash model
    "gemini-2.0-flash-lite", # Lightest option
    "gemini-2.5-pro",        # Pro model (more expensive)
    "gemini-pro-latest",     # Fallback
]
GEMINI_MODEL = GEMINI_MODELS[0]  # Start with gemini-2.0-flash

# Get API key from environment variable or set here (not recommended)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    print("Warning: GEMINI_API_KEY environment variable not set.")
    print("Please set it with: export GEMINI_API_KEY='your-api-key'")
    # Uncomment and set your key here if needed:
    GEMINI_API_KEY = "AIzaSyC-J9-kYUpfiu_8ENKlm50oXm2AYkK1Edk"

# Rate limiting: delay between API calls (seconds)
API_DELAY = 2.0

# Maximum retries for failed API calls
MAX_RETRIES = 5

# Maximum wait time for rate limit errors (seconds)
MAX_RATE_LIMIT_WAIT = 300  # 5 minutes

# Bug type categories and their descriptions
BUG_CATEGORIES = {
    "Exception handling": "Missing or improper exception handling, including uncaught or incorrectly handled exceptions.",
    "Missing case": "Faults due to unhandled input cases, often caused by incomplete conditional logic.",
    "Processing": "Incorrect implementation logic such as miscalculations, wrong method output, or flawed algorithmic steps.",
    "Typo": "Ambiguous naming or typographical errors in identifiers, SQL statements, URLs, or file paths.",
    "Dependency": "Unexpected behavior caused by changes in external libraries, frameworks, or underlying systems.",
    "Other (Logic)": "Semantic or logic-related faults not covered by the above categories.",

    "Buffer overflow": "Buffer overflows excluding numeric overflows.",
    "Null pointer dereference": "Dereferencing of null pointers.",
    "Uninitialized memory read": "Accessing memory that has not been properly initialized.",
    "Memory leak": "Failure to release allocated memory.",
    "Dangling pointer": "Accessing memory through pointers that reference deallocated objects.",
    "Double free": "Multiple deallocations of the same memory region.",
    "Other (Memory)": "Memory-related bugs not covered by the above categories.",

    "Order violation": "Incorrect or missing synchronization leading to improper execution order across threads.",
    "Race condition": "Concurrent access to shared resources without proper synchronization.",
    "Atomicity violation": "Violations of assumed atomic execution due to missing constraints on operation interleavings.",
    "Deadlock": "Two or more threads waiting indefinitely for each other to release resources.",
    "Other (Concurrency)": "Concurrency-related bugs not covered by the above categories."
}



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


def create_categorization_prompt(title: str, description: str) -> str:
    """Create a prompt for Gemini to categorize the bug report."""
    
    # Build category descriptions
    categories_text = "\n".join([
        f"{i+1}. {cat}: {desc}"
        for i, (cat, desc) in enumerate(BUG_CATEGORIES.items())
    ])
    
    # Count categories for accurate prompt
    num_categories = len(BUG_CATEGORIES)
    
    # Escape curly braces for f-string
    json_example = """{
  "category": "<one of the categories above, exact string>",
  "confidence": <integer 1-5>,
  "reasoning": "<max 2 sentences explaining the choice>"
}"""
    
    prompt = f"""You are an expert software engineer categorizing bug reports. Please analyze the following bug report and categorize it into ONE of the {num_categories} predefined categories.

Bug Report Title: {title}

Bug Report Description:
{description}

Bug Report Categories:
{categories_text}

Your Output:
Return ONLY a JSON object with EXACTLY the following three fields:
{json_example}

Strict Rules:
- Output MUST be valid JSON.
- NO markdown. NO backticks. NO extra text before or after the JSON.
- The "category" value MUST match EXACTLY one of the categories listed above.
- Choose the MOST RELEVANT category even if multiple could apply.
- If the bug is ambiguous, select the category that best fits the core issue and reflect uncertainty using a lower confidence score (1–2).
- Do NOT use any category as a default or fallback.
- Do NOT add additional fields, comments, or explanations.
"""
    return prompt


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


def parse_gemini_response(response_text: str) -> Optional[Dict[str, Any]]:
    """Parse Gemini's response and extract the JSON categorization."""
    # Try to extract JSON from the response
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
        result = json.loads(text)
        # Validate category
        if "category" in result:
            category = result["category"]
            if category not in BUG_CATEGORIES:
                print(f"  Warning: Invalid category '{category}', using 'Other (Logic)' as default")
                result["category"] = "default: Other (Logic)"
        return result
    except json.JSONDecodeError as e:
        print(f"  Warning: Failed to parse JSON response: {e}")
        print(f"  Response text: {text[:200]}...")
        return None


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


def categorize_bug_report(title: str, description: str) -> Optional[Dict[str, Any]]:
    """Categorize a single bug report using Gemini API."""
    prompt = create_categorization_prompt(title, description)
    response = call_gemini_api(prompt)
    
    if not response:
        return None
    
    categorization = parse_gemini_response(response)
    return categorization


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

def get_functional_issue_ids(csv_file: str) -> set:
    """Get set of bug IDs that are categorized as 'Functional Issue'."""
    if not os.path.exists(csv_file):
        print(f"Warning: {csv_file} not found. Will process all bug reports.")
        return None  # None means process all
    
    try:
        df = pd.read_csv(csv_file)
        if "category" not in df.columns or "id" not in df.columns:
            print(f"Warning: {csv_file} missing required columns. Will process all bug reports.")
            return None
        
        functional_ids = set(df[df["category"] == "Functional Issue"]["id"].astype(str).tolist())
        print(f"Found {len(functional_ids)} bug reports categorized as 'Functional Issue' in {csv_file}")
        return functional_ids
    except Exception as e:
        print(f"Warning: Could not read {csv_file}: {e}. Will process all bug reports.")
        return None


def write_row_to_csv(row: dict, output_file: str, is_first_row: bool = False):
    """Write a single row to CSV file, appending if file exists."""
    # Create DataFrame with single row
    df = pd.DataFrame([row])
    
    # Write header if new file, append if existing
    df.to_csv(output_file, mode='a' if not is_first_row else 'w', 
              header=is_first_row, index=False)


# ---------------------------------------------------------------------------
# Main Processing Function
# ---------------------------------------------------------------------------

def process_bug_reports(data_dir: str = DATA_DIR, output_file: str = OUTPUT_FILE):
    """
    Process all bug reports in Defects4J directory and categorize them.
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
    
    # Get functional issue IDs to filter by
    functional_issue_ids = get_functional_issue_ids(FUNCTIONAL_ISSUES_CSV)
    
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
    
    # Filter to only functional issues if specified
    if functional_issue_ids is not None:
        print(f"Filtering to only process {len(functional_issue_ids)} functional issue bug reports...")
    
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
                
                # Skip if not a functional issue (when filtering is enabled)
                if functional_issue_ids is not None and bid not in functional_issue_ids:
                    skipped_count += 1
                    continue
                
                # Skip if already processed
                if bid in processed_ids:
                    skipped_count += 1
                    continue
                
                # Categorize bug report
                print(f"Categorizing bug report: {bid}")
                categorization = categorize_bug_report(title, desc)
                
                if categorization:
                    # Combine bug report info with categorization
                    row = {
                        "id": bid,
                        "title": title,
                        "description_length": len(desc),
                        "category": categorization.get("category", "Other (Logic)"),
                        "confidence": categorization.get("confidence", 3),
                        "reasoning": categorization.get("reasoning", ""),
                    }
                    # Write immediately to CSV
                    write_row_to_csv(row, output_file, is_first_row)
                    is_first_row = False
                    processed_ids.add(bid)  # Track as processed
                    success_count += 1
                    print(f"  ✓ Categorized {bid} as: {row['category']} (confidence: {row['confidence']})")
                else:
                    print(f"  ✗ Failed to categorize {bid}")
                    failed_count += 1
                    # Still add a row with default values
                    row = {
                        "id": bid,
                        "title": title,
                        "description_length": len(desc),
                        "category": "Other (Logic)",  # Default category
                        "confidence": None,
                        "reasoning": "Failed to categorize",
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
    print(f"  Successfully categorized: {success_count}")
    print(f"  Failed categorizations: {failed_count}")
    print(f"  Skipped (already processed): {skipped_count}")
    print(f"  Results saved to: {output_file}")
    
    # Print summary statistics from CSV
    if os.path.exists(output_file):
        try:
            df = pd.read_csv(output_file)
            print("\n" + "="*80)
            print("CATEGORIZATION SUMMARY")
            print("="*80)
            if len(df) > 0:
                print(f"\n  Total bug reports categorized: {len(df)}")
                
                if "category" in df.columns:
                    print(f"\n  Category distribution:")
                    category_counts = df["category"].value_counts()
                    for category, count in category_counts.items():
                        if pd.notna(category):
                            print(f"    {category}: {count} ({count/len(df)*100:.1f}%)")
                
                if "confidence" in df.columns:
                    mean_confidence = df["confidence"].mean()
                    if pd.notna(mean_confidence):
                        print(f"\n  Average confidence: {mean_confidence:.2f}")
            print("="*80)
        except Exception as e:
            print(f"Warning: Could not generate summary statistics: {e}")


if __name__ == "__main__":
    process_bug_reports()

