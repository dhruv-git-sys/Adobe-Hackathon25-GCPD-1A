# Code.py (Updated with Bbox Merging)
import fitz
import json
import sys

def merge_bboxes(bbox1, bbox2):
    """Creates a new bounding box that encloses two other bounding boxes."""
    x0 = min(bbox1[0], bbox2[0])
    y0 = min(bbox1[1], bbox2[1])
    x1 = max(bbox1[2], bbox2[2])
    y1 = max(bbox1[3], bbox2[3])
    return [x0, y0, x1, y1]

def extract_pdf_data(pdf_path, out_path=None):
    """
    Extracts and intelligently merges text spans from a PDF, updating
    the bounding box to enclose the entire merged segment.
    """
    document = fitz.open(pdf_path)
    pages_data = []
    segment_id_counter = 0

    for page in document:
        page_dict = page.get_text("dict")
        page_segments = []
        current_segment = None

        for block in page_dict.get("blocks", []):
            if block.get("type") == 0:  # Text blocks
                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        text = span.get("text", "").strip()
                        if not text:
                            continue

                        # Define a key for merging based on style
                        style_key = (span['font'], round(span['size']), span['flags'])

                        if current_segment and current_segment["style_key"] == style_key:
                            # --- MERGE with current segment ---
                            current_segment["text"] += " " + text
                            # Merge the bounding boxes
                            current_segment["bbox"] = merge_bboxes(current_segment["bbox"], span["bbox"])
                        else:
                            # --- FINALIZE the previous segment and START a new one ---
                            if current_segment:
                                page_segments.append(current_segment)

                            # Start a new segment
                            current_segment = {
                                "id": segment_id_counter,
                                "style_key": style_key, # Store key for comparison
                                "text": text,
                                "font": span["font"],
                                "size": span["size"],
                                "flags": span["flags"],
                                "color": span["color"],
                                "ascender": span.get("ascender"),
                                "descender": span.get("descender"),
                                "origin": span["origin"],
                                "bbox": span["bbox"]
                            }
                            segment_id_counter += 1
        
        # Don't forget the very last segment on the page
        if current_segment:
            page_segments.append(current_segment)
            
        pages_data.append(page_segments)
    
    document.close()

    # Clean up temporary style_key before saving
    for page in pages_data:
        for segment in page:
            del segment["style_key"]

    if pages_data and out_path:
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(pages_data, f, indent=2)

    print(f"Extracted and merged {segment_id_counter} segments from {len(pages_data)} pages.")
    return pages_data

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python Code.py input.pdf output.json")
        sys.exit(1)
    
    input_pdf = sys.argv[1]
    output_json = sys.argv[2]
    extract_pdf_data(input_pdf, output_json)