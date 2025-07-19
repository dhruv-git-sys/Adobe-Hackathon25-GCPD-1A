import fitz  # PyMuPDF
import json
import os
import re


def extract_and_reassemble_text(doc):
    """
    Extracts text, reassembles lines to fix fragmentation, and keeps vertical position for sorting.
    """
    processed_data = []
    for page_num, page in enumerate(doc):
        blocks = page.get_text("dict", sort=True)["blocks"]
        for block in blocks:
            if block["type"] == 0:
                for line in block["lines"]:
                    y0 = line["bbox"][1]  # Get vertical position for sorting
                    spans = sorted(line["spans"], key=lambda s: s["bbox"][0])
                    if not spans: continue

                    full_text = " ".join(span["text"] for span in spans).strip()
                    if not full_text: continue

                    first_span = spans[0]
                    processed_data.append({
                        "text": full_text,
                        "size": round(first_span["size"]),
                        "font": first_span["font"],
                        "is_bold": "bold" in first_span["font"].lower(),
                        "page": page_num + 1,
                        "y0": y0  # Keep y0 only for sorting the final outline
                    })
    return processed_data


def calculate_heading_score(item):
    """
    The stable and reliable scoring function.
    """
    text = item["text"]
    score = 0

    # Basic sanity checks
    if len(text) < 3 or len(text) > 200: return 0
    if re.fullmatch(r'[\d\.\s]+', text): return 0

    # Main scoring based on text properties
    score += item["size"]
    if item["is_bold"]: score += 4
    if re.match(r'^((\d+\.)+\d*|[A-Z]\.|[IVX]+)\s', text): score += 5
    if text.endswith(('.', ':', ',')): score -= 2

    return max(0, score)


def generate_outline_from_heuristics(text_data):
    """Generates outline using the stable scoring heuristic."""
    for item in text_data:
        item['score'] = calculate_heading_score(item)

    # Use a dynamic threshold based on the scores
    avg_score = sum(item['score'] for item in text_data) / len(text_data) if text_data else 10
    # A heading's score should be significantly above average and above a minimum font size
    heading_candidates = [item for item in text_data if item['score'] > avg_score and item['score'] > 12]

    if not heading_candidates: return []

    heading_styles = sorted(list(set((item['size'], item['is_bold']) for item in heading_candidates)), reverse=True)
    style_to_level = {style: f"H{i + 1}" for i, style in enumerate(heading_styles[:4])}

    outline = []
    for item in heading_candidates:
        style = (item['size'], item['is_bold'])
        if style in style_to_level:
            outline.append({
                "level": style_to_level[style],
                "text": item['text'],
                "page": item['page'],
                "y0": item['y0']
            })

    # Sort the final outline for perfect reading order
    outline.sort(key=lambda x: (x['page'], x['y0']))
    for item in outline:
        del item['y0']  # Clean up the temporary key

    return outline


def generate_outline_from_toc(doc):
    """Generates outline from the PDF's built-in ToC (most reliable method)."""
    toc = doc.get_toc()
    if not toc: return None
    outline = [{"level": f"H{level}", "text": title.strip(), "page": page} for level, title, page in toc]
    return outline if outline else None


def find_title(doc, text_data, outline):
    """Finds the document title using a multi-step approach."""
    title = doc.metadata.get('title', '').strip()
    if title and len(title) > 5: return title

    if outline:
        first_h1 = next((item['text'] for item in outline if item['level'] == 'H1'), None)
        if first_h1 and len(first_h1) > 5: return first_h1

    first_page_text = [item for item in text_data if item["page"] == 1]
    if not first_page_text: return os.path.basename(doc.name)

    max_size = max(item['size'] for item in first_page_text)
    potential_titles = [item['text'] for item in first_page_text if item['size'] == max_size]
    return " ".join(potential_titles)


def process_pdf(pdf_path):
    """Master function to process a single PDF with a cleaner, stable workflow."""
    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        return {"title": f"Error opening {os.path.basename(pdf_path)}", "outline": []}

    # Gracefully handle scanned/empty PDFs
    if doc.page_count == 0 or (doc.page_count > 0 and not doc[0].get_text()):
        doc.close()
        return {"title": f"{os.path.basename(pdf_path)} (unsupported format or empty)", "outline": []}

    # Tier 1: Attempt to get outline from embedded Table of Contents
    outline = generate_outline_from_toc(doc)

    # Tier 2: If ToC fails, fall back to heuristic analysis
    if not outline:
        text_data = extract_and_reassemble_text(doc)
        outline = generate_outline_from_heuristics(text_data)
        title = find_title(doc, text_data, outline)
    else:
        # If ToC worked, we still need to find a title
        title = find_title(doc, [], outline)  # Pass empty text_data as it's not needed

    doc.close()
    return {"title": title, "outline": outline}



def main(pdf):
	result = process_pdf(pdf)
	return (result)