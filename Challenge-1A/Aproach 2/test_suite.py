#!/usr/bin/env python3
"""
Test script to validate the outline extractor on all training files
"""

import json
import os
from outline_extractor import OutlineExtractor
from Code import extract_pdf_data

def test_file(pdf_file, expected_json):
    """Test a single PDF file against expected output"""
    print(f"\n=== Testing {pdf_file} ===")
    
    # Extract segments
    segments_data = extract_pdf_data(pdf_file, f"temp_{pdf_file}_segments.json")
    
    # Load expected data
    with open(expected_json, 'r') as f:
        expected = json.load(f)
    
    # Extract outline
    extractor = OutlineExtractor()
    result = extractor.extract_outline(segments_data)
    
    # Compare results
    print(f"Expected title: '{expected.get('title', '')}' (length: {len(expected.get('title', ''))})")
    print(f"Actual title:   '{result['title']}' (length: {len(result['title'])})")
    
    expected_headings = len(expected.get('outline', []))
    actual_headings = len(result['outline'])
    print(f"Expected headings: {expected_headings}")
    print(f"Actual headings:   {actual_headings}")
    
    # Count by level
    expected_levels = {}
    for item in expected.get('outline', []):
        level = item.get('level', 'unknown')
        if level in ['H1', 'H2', 'H3']:  # Only count relevant levels
            expected_levels[level] = expected_levels.get(level, 0) + 1
    
    actual_levels = {}
    for item in result['outline']:
        level = item['level']
        actual_levels[level] = actual_levels.get(level, 0) + 1
    
    print(f"Expected levels: {expected_levels}")
    print(f"Actual levels:   {actual_levels}")
    
    # Clean up temp file
    temp_file = f"temp_{pdf_file}_segments.json"
    if os.path.exists(temp_file):
        os.remove(temp_file)
    
    return result

def main():
    """Test all training files"""
    test_files = [
        ("file01.pdf", "file01.json"),
        ("file02.pdf", "file02.json"),
        ("file03.pdf", "file03.json"),
        ("file04.pdf", "file04.json"),
        ("file05.pdf", "file05.json")
    ]
    
    print("PDF Structured Outline Extractor - Test Suite")
    print("=" * 50)
    
    results = {}
    for pdf_file, json_file in test_files:
        if os.path.exists(pdf_file) and os.path.exists(json_file):
            results[pdf_file] = test_file(pdf_file, json_file)
        else:
            print(f"\nSkipping {pdf_file} - files not found")
    
    print(f"\n=== SUMMARY ===")
    print(f"Tested {len(results)} files successfully")
    for pdf_file, result in results.items():
        print(f"{pdf_file}: Title='{result['title'][:50]}...', Headings={len(result['outline'])}")

if __name__ == "__main__":
    main()
