# my.py (Updated)
import sys
import json
from collections import defaultdict
from pdf2segment import extract_pdf_data

def extract_outline_final(segmented_data):
    """
    Extracts a structured outline by deeply analyzing style patterns and
    structural context (e.g., inline vs. standalone text).
    """
    # --- Phase 1: Enhanced Style Profiling ---

    # 1a. Flatten data, group by style, and capture standalone status
    style_buckets = defaultdict(list)
    total_segments = 0
    all_segments_by_id = {} # Use a dictionary for fast lookups

    for page_num, page in enumerate(segmented_data, 1):
        lines = defaultdict(list)
        for seg in page:
            if seg.get('text', '').strip():
                line_key = (round(seg['bbox'][1], 0), round(seg['bbox'][3], 0))
                lines[line_key].append(seg)

        for line_key, segs_in_line in lines.items():
            is_standalone = len(segs_in_line) == 1
            for seg in segs_in_line:
                total_segments += 1
                sig = f"{seg['font']}|{round(seg['size'], 2)}|{seg['flags']}"
                seg_meta = {
                    'id': seg['id'], # Capture the unique ID
                    'text': seg['text'].strip(), 'page': page_num,
                    'font_size': round(seg['size'], 2), 'style_sig': sig,
                    'is_standalone': is_standalone
                }
                style_buckets[sig].append(seg_meta)
                all_segments_by_id[seg['id']] = seg_meta

    if not total_segments:
        return {"title": "", "outline": []}

    # 1b. Filter for true "heading" styles using structural heuristics
    heading_styles = []
    for sig, segments in style_buckets.items():
        standalone_count = sum(1 for s in segments if s['is_standalone'])
        standalone_ratio = standalone_count / len(segments)
        if standalone_ratio < 0.85:
            continue

        frequency = len(segments) / total_segments
        if frequency > 0.15:
            continue

        avg_len = sum(len(s['text']) for s in segments) / len(segments)
        if avg_len > 150:
            continue
        
        heading_styles.append({'sig': sig, 'size': segments[0]['font_size'], 'segments': segments})

    # --- Phase 2: Dynamic Hierarchy Assignment ---
    
    if not heading_styles:
        return {"title": "", "outline": []}

    heading_styles.sort(key=lambda s: s['size'], reverse=True)

    style_map = {}
    title_style_sig = None
    
    if len(heading_styles) > 1:
        top_style = heading_styles[0]
        is_title_unique = len(top_style['segments']) < 5 and len(top_style['segments']) < len(heading_styles[1]['segments'])
        
        if is_title_unique and top_style['segments'][0]['page'] <= 2:
            title_style_sig = top_style['sig']
            style_map[title_style_sig] = 'Title'
            heading_levels_map = {f'H{i+1}': s for i, s in enumerate(heading_styles[1:4])}
        else:
            heading_levels_map = {f'H{i+1}': s for i, s in enumerate(heading_styles[0:3])}
    else:
        heading_levels_map = {f'H{i+1}': s for i, s in enumerate(heading_styles[0:3])}
        
    for level, style_info in heading_levels_map.items():
        style_map[style_info['sig']] = level

    # --- Phase 3: Outline Assembly ---

    # 3a. Collect all items matching our heading styles
    all_outline_items = []
    for sig, segments in style_buckets.items():
        if sig in style_map:
            level = style_map[sig]
            for seg in segments:
                all_outline_items.append({**seg, 'level': level})
    
    # 3b. BUG FIX: Sort by the unique ID to restore original document order
    all_outline_items.sort(key=lambda x: x['id'])
    
    title = ""
    final_outline = []
    
    if title_style_sig:
        title_candidates = [item for item in all_outline_items if item['style_sig'] == title_style_sig]
        if title_candidates:
            title = " ".join(item['text'] for item in title_candidates)

    for item in all_outline_items:
        if item['level'] != 'Title':
            final_outline.append({
                "level": item['level'], "text": item['text'], "page": item['page']
            })

    if not title and final_outline and final_outline[0]['level'] == 'H1':
        title = final_outline.pop(0)['text']

    return {"title": title, "outline": final_outline}


def process_main(input_path, output_path):
    """
    Main processing function that handles both PDF and JSON inputs,
    extracts outline, and saves to output file.
    
    Args:
        input_path (str): Path to input PDF or JSON file
        output_path (str): Path to output JSON file
    
    Returns:
        dict: The extracted outline data
    """
    # Check if input is PDF or JSON
    if input_path.lower().endswith('.pdf'):
        # Extract PDF data directly without creating intermediate file
        print(f"📄 Extracting PDF content from '{input_path}'...")
        try:
            segmented_data = extract_pdf_data(input_path)
            print("✅ PDF extraction completed successfully")
        except Exception as e:
            print(f"Error extracting PDF '{input_path}': {e}")
            raise e
    else:
        # Read JSON file (backward compatibility)
        print(f"📂 Reading segmented data from '{input_path}'...")
        try:
            with open(input_path, 'r', encoding='utf-8') as f:
                segmented_data = json.load(f)
            print("✅ JSON file loaded successfully")
        except Exception as e:
            print(f"Error reading input file '{input_path}': {e}")
            raise e
    
    print("🔍 Analyzing document structure...")
    final_outline = extract_outline_final(segmented_data)

    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(final_outline, f, indent=2, ensure_ascii=False)
        print(f"✅ Outline successfully extracted and saved to '{output_path}'")
    except Exception as e:
        print(f"Error writing to output file '{output_path}': {e}")
        raise e
    
    return final_outline


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python my.py <input.pdf|segmented_input.json> <output.json>")
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2]

    try:
        process_main(input_path, output_path)
    except Exception as e:
        print(f"Processing failed: {e}")
        sys.exit(1)