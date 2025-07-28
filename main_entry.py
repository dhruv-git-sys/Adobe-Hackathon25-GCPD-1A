#!/usr/bin/env python3
"""
Docker entry point for Adobe Hackathon Challenge 1A
Processes all PDFs from /app/input and generates JSON files in /app/output
"""

import os
from pathlib import Path
from main_extractor import process_main

def main():
    # Simple: Try Docker paths first, fallback to local paths
    if Path("/app/input").exists():
        input_dir = Path("/app/input")
        output_dir = Path("/app/output")
        print("Using Docker paths")
    else:
        input_dir = Path("input")
        output_dir = Path("output")
        print("Using local paths")
    
    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all PDF files
    pdf_files = list(input_dir.glob("*.pdf"))
    
    if not pdf_files:
        print(f"No PDF files found in {input_dir}")
        return
    
    print(f"Processing {len(pdf_files)} PDF file(s)...")
    
    for pdf_file in pdf_files:
        try:
            output_file = output_dir / f"{pdf_file.stem}.json"
            print(f"Processing: {pdf_file.name}")
            process_main(str(pdf_file), str(output_file))
        except Exception as e:
            print(f"Error processing {pdf_file.name}: {e}")

if __name__ == "__main__":
    main()
