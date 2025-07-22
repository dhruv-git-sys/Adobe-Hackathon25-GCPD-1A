#
# Final Hybrid Script for "Connecting the Dots" - Round 1A
#
# This script uses a 3-step "waterfall" approach for maximum accuracy.
#
# Required libraries:
# pip install PyMuPDF beautifulsoup4 lxml
#

import fitz  # PyMuPDF
import json
import os
import re
from bs4 import BeautifulSoup

# --- Helper function for Heuristic Method ---
def extract_text_blocks(doc):
    """A balanced text extractor for the heuristic method."""
    all_blocks = []
    for page_num, page in enumerate(doc):
        page_blocks = page.get_text("dict", sort=True).get("blocks", [])
        for block in page_blocks:
            if block.get("type") == 0:
                for line in block.get("lines", []):
                    spans = line.get("spans", [])
                    full_line_text = " ".join(s.get("text", "") for s in spans).strip()
                    if not full_line_text or not spans: continue
                    first_span = spans[0]
                    all_blocks.append({
                        "text": full_line_text,
                        "size": round(first_span.get("size", 0)),
                        "font": first_span.get("font", ""),
                        "flags": first_span.get("flags", 0),
                        "page": page_num + 1
                    })
    return all_blocks

# --- The Unified Heuristic "Brain" (Plan C) ---

def score_block_intelligently(block):
    """
    A unified, intelligent scoring function that understands resumes, forms, and articles.
    """
    text = block['text'].strip()
    text_lower = text.lower()
    score = 0

    # 1. Basic disqualifiers for noise
    if len(text) < 2 or len(text) > 300:
        return 0

    # 2. Strong signals for RESUME/SECTION headings (give a high, confident score)
    RESUME_HEADINGS = {
        "experience", "education", "skills", "projects", "summary",
        "objective", "certifications", "languages", "achievements", "publications"
    }
    if text_lower.replace(':', '') in RESUME_HEADINGS:
        return 25  # High score for definitive section titles

    # Bonus for all-caps titles, common in resumes and reports
    if text.isupper() and len(text.split()) < 5:
        score += 8

    # 3. Penalize FORM-LIKE content heavily
    FORM_STOP_WORDS = {"name", "age", "date", "s.no", "signature", "address", "phone", "relationship", "sex"}
    if text_lower in FORM_STOP_WORDS:
        score -= 20 # Huge penalty
    # Penalize lines that are just a number and a dot
    if re.fullmatch(r'\d+\s*\.', text):
        score -= 20 # Huge penalty

    # 4. Standard heading style scoring
    score += block['size']
    if block['flags'] & 16:  # is_bold
        score += 4

    # 5. Reward structured heading patterns like "1.1 Introduction"
    if re.match(r'^((\d+\.)+\d*|[A-Z]\.)\s', text):
        score += 5

    return max(0, score)

def build_outline_from_heuristics(text_blocks):
    """Builds the outline using our new, unified scoring brain."""
    for block in text_blocks:
        block['score'] = score_block_intelligently(block)

    candidates = [b for b in text_blocks if b['score'] > 14]
    if not candidates: return []

    heading_styles = sorted(list(set((c['size'], c['flags'] & 16) for c in candidates)), reverse=True)
    style_to_level = {style: f"H{i+1}" for i, style in enumerate(heading_styles[:3])}

    outline = []
    for block in candidates:
        style_key = (block['size'], block['flags'] & 16)
        if style_key in style_to_level:
            outline.append({"level": style_to_level[style_key], "text": block["text"], "page": block["page"]})
    return outline


# --- HTML Method Function (Plan B) ---

def extract_outline_via_html(doc):
    """Extracts outline using HTML conversion."""
    outline = []
    for page_num, page in enumerate(doc):
        try: html_content = page.get_text("html")
        except: continue
        soup = BeautifulSoup(html_content, "lxml")
        heading_tags = soup.find_all(["h1", "h2", "h3", "h4"])
        for tag in heading_tags:
            text = tag.get_text(strip=True)
            if text and len(text) > 2:
                outline.append({"level": tag.name.upper(), "text": text, "page": page_num + 1})
    return outline


# --- The Master Orchestrator ---

def process_pdf(pdf_path):
    """Runs the full 3-step hybrid "waterfall" strategy."""
    try: doc = fitz.open(pdf_path)
    except Exception as e: return {"title": f"Error opening file: {e}", "outline": []}

    # Plan A: Built-in Table of Contents (The Golden Ticket)
    toc = doc.get_toc()
    if toc:
        print(f"  ✅ Strategy: Built-in ToC")
        outline = [{"level": f"H{lvl}", "text": title, "page": page} for lvl, title, page in toc]
        title = doc.metadata.get('title', outline[0]['text'] if outline else "").strip()
        doc.close()
        return {"title": title if title else os.path.basename(pdf_path), "outline": outline}

    # Plan B: HTML Analysis (The Smart Analyst)
    outline = extract_outline_via_html(doc)
    if outline:
        print(f"  ✅ Strategy: HTML Analysis")
        title = doc.metadata.get('title', outline[0]['text']).strip()
        doc.close()
        return {"title": title if title else os.path.basename(pdf_path), "outline": outline}

    # Plan C: Unified Heuristic Fallback (The Detective)
    print(f"  ⚠ No ToC or HTML structure found. Using Unified Heuristic Brain.")
    text_blocks = extract_text_blocks(doc)
    if not text_blocks:
        doc.close()
        return {"title": "Empty or Scanned Document", "outline": []}

    outline = build_outline_from_heuristics(text_blocks)
    
    # Final title detection for heuristic method
    title = doc.metadata.get('title', '').strip()
    if not title and outline:
        # Find the H1 with the highest score, or the first item
        h1_candidates = [b for b in outline if b['level'] == 'H1']
        title = h1_candidates[0]['text'] if h1_candidates else outline[0]['text']
    elif not title:
        # Absolute fallback: largest text on first page
        first_page_blocks = [b for b in text_blocks if b['page'] == 1]
        title = max(first_page_blocks, key=lambda b: b['size'])['text'] if first_page_blocks else os.path.basename(pdf_path)

    doc.close()
    return {"title": title, "outline": outline}


# --- Main Execution Block ---
def main(pdf):
	return process_pdf(pdf)