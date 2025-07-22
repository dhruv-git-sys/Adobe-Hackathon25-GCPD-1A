import fitz  # PyMuPDF
import json
import os
import re


def extract_text_blocks(doc):
    """
    The workhorse function. It reads a PDF, reassembles broken text,
    and pulls out all the necessary info (text, style, position) for our brain to analyze.
    """
    processed_data = []
    for page_num, page in enumerate(doc):
        # Get text blocks without the layout parameter
        blocks = page.get_text("dict")["blocks"]
        for block in blocks:
            if block["type"] == 0:  # This means it's a text block
                for line in block["lines"]:
                    # Skip tiny or purely decorative text
                    if not line["spans"] or len(line["spans"][0]["text"].strip()) < 2:
                        continue

                    # Reassemble the full line of text
                    full_text = " ".join(s["text"] for s in line["spans"]).strip()
                    first_span = line["spans"][0]

                    processed_data.append({
                        "text": full_text,
                        "size": round(first_span["size"]),
                        "font": first_span["font"],
                        "flags": first_span["flags"],
                        "color": first_span["color"],
                        "bbox": line["bbox"],
                        "page": page_num + 1,
                    })
    return processed_data


def score_headings_by_style(text_blocks):
    """
    The 'Style & Context Brain'. Now with a 'Form Field Filter' to be smarter!
    """
    # A set of common form/table words to ignore. This is our "bouncer list".
    FORM_STOP_WORDS = {
        "name", "age", "date", "s.no", "signature", "address", "phone",
        "relationship", "sex", "male", "female", "page"
    }

    # First, find the most common style (size, bold) - that's our body text.
    style_counts = {}
    for block in text_blocks:
        is_bold = block['flags'] & 16
        style_key = (block['size'], is_bold)
        style_counts[style_key] = style_counts.get(style_key, 0) + 1

    body_style = max(style_counts, key=style_counts.get) if style_counts else (0, False)

    # Now, score everything with our new, smarter rules!
    for block in text_blocks:
        score = 0
        text_lower = block['text'].lower()
        is_bold = block['flags'] & 16
        style_key = (block['size'], is_bold)

        # --- NEW: The Form Field Filter ---
        # 1. Reject if it's a common form word
        if text_lower in FORM_STOP_WORDS:
            block['score'] = 0
            continue  # Skip to the next block

        # 2. Reject if it's just a number (like "1." or "2.")
        if re.fullmatch(r'[\d\s\.]+', text_lower):
            block['score'] = 0
            continue

        # --- Original Scoring Logic (still important!) ---
        if style_key == body_style: score -= 5
        score += block['size']
        if is_bold: score += 5
        if re.match(r'^((\d+\.)+\d*|[A-Z]\.|[IVX]+)\s', block['text']): score += 5

        # 3. Penalize very short text that isn't a numbered heading
        if len(text_lower.split()) < 2 and not re.match(r'^\d', block['text']):
            score -= 5

        block['score'] = max(0, score)

    return text_blocks


def build_final_outline(scored_blocks):
    """
    Takes the scored text and builds the final, clean JSON outline.
    This part decides what makes the cut and assigns H1, H2, H3 levels.
    """
    # A heading's score should be clearly higher than average body text (score > 12 is a good start)
    candidates = [b for b in scored_blocks if b['score'] > 12]

    if not candidates: return []

    # Find the unique heading styles and rank them to create H1, H2, H3...
    heading_styles = sorted(list(set((c['size'], c['flags'] & 16) for c in candidates)), reverse=True)
    style_to_level = {style: f"H{i + 1}" for i, style in enumerate(heading_styles[:3])}

    outline = []
    for block in candidates:
        style_key = (block['size'], block['flags'] & 16)
        if style_key in style_to_level:
            outline.append({
                "level": style_to_level[style_key],
                "text": block["text"],
                "page": block["page"],
            })
    return outline


def find_best_title(doc, outline, text_blocks):
    """
    Our title detective. It tries three methods to find the most accurate title.
    """
    # Method 1: The PDF's own metadata (the best source!)
    if doc.metadata and doc.metadata['title']:
        return doc.metadata['title']

    # Method 2: The first H1 we found in the outline
    if outline:
        return outline[0]['text']

    # Method 3 (Fallback): The biggest text on the first page
    first_page_blocks = [b for b in text_blocks if b['page'] == 1]
    if not first_page_blocks: return "Untitled Document"
    return max(first_page_blocks, key=lambda b: b['size'])['text']


def process_pdf_like_a_pro(pdf_path):
    """
    This is the master function that runs the whole show for one PDF.
    """
    doc = fitz.open(pdf_path)

    # 🚀 The Silver Bullet: Check for a built-in Table of Contents first. It's 99% accurate.
    toc = doc.get_toc()
    if toc:
        ##print(f"✅ Found a built-in ToC for {os.path.basename(pdf_path)}. Easy win!")
        outline = [{"level": f"H{lvl}", "text": title, "page": page} for lvl, title, page in toc]
        title = find_best_title(doc, outline, [])
        doc.close()
        return {"title": title, "outline": outline}

    # fallback to our custom brain if no ToC is found
    ##print(f"🧠 No ToC found for {os.path.basename(pdf_path)}. Using our custom brain!")
    text_blocks = extract_text_blocks(doc)
    if not text_blocks:
        doc.close()
        return {"title": "Empty or Scanned Document", "outline": []}

    scored_blocks = score_headings_by_style(text_blocks)
    outline = build_final_outline(scored_blocks)
    title = find_best_title(doc, outline, text_blocks)

    doc.close()
    return {"title": title, "outline": outline}


def main(pdf):
	return process_pdf_like_a_pro(pdf)