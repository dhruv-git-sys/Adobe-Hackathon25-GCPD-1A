import re
import json
import fitz  # PyMuPDF
from pymupdf4llm import to_markdown


def pdf_to_outline_json(pdf_path: str) -> dict:
    """
    Converts a PDF to an outline JSON structure by first converting pages to Markdown,
    then extracting headings (H1-H6) with their levels and page numbers.

    Args:
        pdf_path: Path to the input PDF file.

    Returns:
        A dict with 'title' and 'outline' keys. 'title' is inferred from the first H1,
        and 'outline' is a list of dicts with 'level', 'text', and 'page'.
    """
    # Open the PDF document
    doc = fitz.open(pdf_path)
    # Convert to markdown by chunks per page
    chunks = to_markdown(doc, page_chunks=True)

    outline = []
    title = None

    # Iterate through each page chunk
    for chunk in chunks:
        page_num = chunk['metadata'].get('page', None)
        text = chunk.get('text', '')
        # Split into lines and parse markdown headings
        for line in text.splitlines():
            match = re.match(r'^(#{1,6})\s+(.*)', line)
            if match:
                hashes, heading_text = match.groups()
                level = f"H{len(hashes)}"
                # Set title if first H1
                if title is None and level == 'H1':
                    title = heading_text.strip()
                outline.append({
                    'level': level,
                    'text': heading_text.strip(),
                    'page': page_num
                })

    # Fallback for title if no H1 found
    if title is None and doc.metadata.get('title'):
        title = doc.metadata['title']
    elif title is None:
        title = ''

    result = {
        'title': title,
        'outline': outline
    }
    return result


def main(pdf):
	return pdf_to_outline_json(pdf)
