# XML Migration Summary

## Overview
All scripts have been updated to read bug reports from XML files in `defects4j_xml/` instead of JSON files in `bug_reports/Defects4J/`.

## Updated Scripts

### 1. `bug_feature_extraction/extract_bug_features.py`
- **Changed**: `DATA_DIR = "defects4j_xml"` (was `"bug_reports/Defects4J"`)
- **Added**: `USE_XML = True` flag
- **Added**: `read_xml()` function to parse XML bug reports
- **Added**: `parse_bug_file()` function that handles both XML and JSON
- **Updated**: File glob pattern to search for `*.xml` files when `USE_XML = True`

### 2. `bug_feature_extraction/gemini_bug_ratings.py`
- **Changed**: `DATA_DIR = "defects4j_xml"`
- **Added**: `USE_XML = True` flag
- **Added**: `read_xml()` and `read_bug_file()` functions
- **Updated**: File processing loop to use `read_bug_file()` instead of `read_json()`

### 3. `bug_feature_extraction/gemini_bug_categorization_overall.py`
- **Changed**: `DATA_DIR = "defects4j_xml"`
- **Added**: `USE_XML = True` flag
- **Added**: `read_xml()` and `read_bug_file()` functions
- **Updated**: File processing loop to use `read_bug_file()` instead of `read_json()`

### 4. `fine_grained_gemini_catg.py`
- **Changed**: `DATA_DIR = "defects4j_xml"`
- **Added**: `USE_XML = True` flag
- **Added**: `read_xml()` and `read_bug_file()` functions
- **Updated**: File processing loop to use `read_bug_file()` instead of `read_json()`

## XML Structure

The XML files follow this structure:
```xml
<bugRepository>
  <bug id="..." opendate="..." fixdate="..." resolution="...">
    <buginformation>
      <summary>Bug title/summary</summary>
      <description>Bug description text</description>
    </buginformation>
    <fixedFiles>
      <file>...</file>
    </fixedFiles>
  </bug>
</bugRepository>
```

## XML Parsing Details

The `read_xml()` function:
1. Parses the XML file using `xml.etree.ElementTree`
2. Extracts bug information from `<buginformation>` element
3. Gets title from `<summary>` element
4. Gets description from `<description>` element
5. Extracts bug ID from the `id` attribute or filename
6. Handles HTML entities (`&lt;`, `&gt;`, `&amp;`)
7. Returns a list of dictionaries with keys: `id`, `title`, `summary`, `description`, `opendate`, `fixdate`, `resolution`

## Backward Compatibility

All scripts maintain backward compatibility:
- If `USE_XML = False`, they will still read JSON files from the original location
- The `read_json()` function is preserved for JSON file support
- The `read_bug_file()` function automatically chooses XML or JSON based on file extension and `USE_XML` flag

## Usage

To use XML files (default):
- Set `USE_XML = True` in each script
- Ensure XML files are in `defects4j_xml/` directory
- Run scripts as normal

To switch back to JSON files:
- Set `USE_XML = False` in each script
- Ensure JSON files are in `bug_reports/Defects4J/` directory

## Testing

The XML parsing has been tested and verified to:
- Correctly extract bug IDs from XML attributes or filenames
- Extract title and description text
- Handle HTML entities
- Work with the existing feature extraction pipeline

