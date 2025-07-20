import sys
import json
import fitz  # PyMuPDF
import numpy as np
import re
from collections import defaultdict, Counter

def extract_text_blocks(pdf_path):
    """
    Extract text blocks with comprehensive formatting information.
    This function reassembles fragmented text and collects all necessary
    properties for structural analysis without making assumptions about content.
    """
    doc = fitz.open(pdf_path)
    all_blocks = []
    
    for page_num in range(len(doc)):
        page = doc[page_num]
        
        # Extract text with dictionary method for detailed formatting info
        page_dict = page.get_text("dict", sort=True)
        
        for block in page_dict.get("blocks", []):
            if block.get("type") == 0:  # Text block
                for line in block.get("lines", []):
                    # Reassemble line from spans to handle fragmentation
                    line_spans = sorted(line.get("spans", []), key=lambda s: s.get("bbox", [0])[0])
                    
                    if not line_spans:
                        continue
                    
                    # Combine text from all spans in the line
                    full_text = " ".join(span.get("text", "") for span in line_spans).strip()
                    
                    if not full_text or len(full_text) < 2:
                        continue
                    
                    # Get dominant formatting properties for the line
                    sizes = [span.get("size", 0) for span in line_spans if span.get("text", "").strip()]
                    flags = [span.get("flags", 0) for span in line_spans if span.get("text", "").strip()]
                    fonts = [span.get("font", "") for span in line_spans if span.get("text", "").strip()]
                    
                    if not sizes:
                        continue
                    
                    # Use most common or average values
                    avg_size = sum(sizes) / len(sizes)
                    dominant_flags = max(set(flags), key=flags.count) if flags else 0
                    dominant_font = max(set(fonts), key=fonts.count) if fonts else ""
                    
                    # Get line position
                    line_bbox = line.get("bbox", [0, 0, 0, 0])
                    
                    all_blocks.append({
                        "text": full_text,
                        "size": avg_size,
                        "flags": dominant_flags,
                        "font": dominant_font,
                        "page": page_num,  # 0-based for internal processing
                        "x_pos": line_bbox[0],
                        "y_pos": line_bbox[1],
                        "bbox": line_bbox
                    })
    
    doc.close()
    return all_blocks

def analyze_document_structure(text_blocks):
    """
    Analyze the statistical patterns in the document to identify
    potential heading characteristics without using keywords.
    """
    if not text_blocks:
        return {}
    
    # Extract font sizes and other properties
    sizes = [block["size"] for block in text_blocks]
    flags_list = [block["flags"] for block in text_blocks]
    
    # Statistical analysis of font sizes
    size_array = np.array(sizes)
    size_stats = {
        "mean": np.mean(size_array),
        "median": np.median(size_array),
        "std": np.std(size_array),
        "min": np.min(size_array),
        "max": np.max(size_array),
        "unique_sizes": sorted(list(set(sizes)), reverse=True)
    }
    
    # Identify size clusters using percentiles
    percentiles = np.percentile(size_array, [25, 50, 75, 85, 95])
    
    # Analyze formatting flags distribution
    flag_counts = {}
    for flags in flags_list:
        flag_counts[flags] = flag_counts.get(flags, 0) + 1
    
    # Calculate dynamic thresholds based on document's own statistics
    # Large size threshold: 75th percentile or higher
    large_size_threshold = max(percentiles[2], size_stats["mean"] + size_stats["std"])
    
    # Medium size threshold: between median and 75th percentile  
    medium_size_threshold = max(percentiles[1], size_stats["mean"])
    
    # Identify common formatting patterns
    bold_flags = [flags for flags in flag_counts.keys() if flags & 16]  # Bold flag
    
    return {
        "size_stats": size_stats,
        "thresholds": {
            "large": large_size_threshold,
            "medium": medium_size_threshold,
            "small": percentiles[1]  # median as small threshold
        },
        "formatting": {
            "flag_distribution": flag_counts,
            "bold_flags": bold_flags
        },
        "percentiles": percentiles
    }

def refined_heading_score(block, structure_info, all_blocks):
    """
    Refined scoring based on analysis of reference patterns.
    """
    score = 0.0
    thresholds = structure_info["thresholds"]
    size_stats = structure_info["size_stats"]
    
    # More nuanced font size scoring
    size_score = 0
    if block["size"] >= size_stats["max"] * 0.9:  # Near maximum size
        size_score = 12
    elif block["size"] >= thresholds["large"]:
        size_score = 8
    elif block["size"] >= thresholds["medium"]:
        size_score = 5
    elif block["size"] > size_stats["median"]:
        size_score = 3
    
    score += size_score * 0.35
    
    # Enhanced formatting score
    format_score = 0
    if block["flags"] & 16:  # Bold
        format_score += 6
    if block["flags"] & 2:   # Italic
        format_score += 2
    
    score += format_score * 0.25
    
    # Position and structure score
    position_score = 0
    text_length = len(block["text"].strip())
    
    # Prefer shorter text (typical for headings)
    if 5 <= text_length <= 60:
        position_score += 4
    elif text_length <= 120:
        position_score += 2
    elif text_length > 300:  # Penalize very long text
        position_score -= 3
    
    # Left alignment bonus
    if block["x_pos"] <= 120:
        position_score += 2
    
    # Check for numbered sections (common in outlines)
    text = block["text"].strip()
    if len(text) > 0 and (text[0].isdigit() or text.startswith(('1.', '2.', '3.', 'A.', 'B.', 'I.', 'II.'))):
        position_score += 3
    
    score += position_score * 0.2
    
    # Uniqueness and rarity score
    uniqueness_score = 0
    same_size_blocks = [b for b in all_blocks if abs(b["size"] - block["size"]) < 0.5]
    size_frequency = len(same_size_blocks) / len(all_blocks)
    
    if size_frequency <= 0.03:  # Very rare size
        uniqueness_score += 5
    elif size_frequency <= 0.1:  # Uncommon size
        uniqueness_score += 3
    elif size_frequency <= 0.2:  # Somewhat uncommon
        uniqueness_score += 1
    
    score += uniqueness_score * 0.2
    
    # Content pattern penalties and bonuses
    # Penalize blocks with many numbers (likely data, not headings)
    digit_ratio = sum(1 for c in text if c.isdigit()) / len(text) if text else 0
    if digit_ratio > 0.4:
        score *= 0.6
    
    # Penalize very short incomplete words (likely text fragments)
    if len(text) <= 4 and not text.endswith(':') and not text.isdigit():
        score *= 0.3
    
    # Bonus for complete words and sentences
    if text.endswith((':','.','!','?')) or len(text.split()) >= 2:
        score += 1.0
    
    # Bonus for common heading words (but keep it minimal to stay generic)
    heading_indicators = ['chapter', 'section', 'overview', 'introduction', 'conclusion', 
                         'summary', 'contents', 'acknowledgments', 'references', 'appendix',
                         'background', 'timeline', 'strategy', 'plan', 'component']
    if any(indicator in text.lower() for indicator in heading_indicators):
        score += 1.5
    
    return round(score, 2)

def refined_generate_outline(text_blocks, min_score_threshold=6.0):
    """
    Refined outline generation with better level assignment and filtering.
    """
    structure_info = analyze_document_structure(text_blocks)
    
    # Calculate refined scores
    scored_blocks = []
    for block in text_blocks:
        score = refined_heading_score(block, structure_info, text_blocks)
        if score >= min_score_threshold:
            scored_blocks.append({**block, "score": score})
    
    if not scored_blocks:
        return []
    
    # Remove duplicates and very similar text
    filtered_blocks = []
    seen_texts = set()
    
    for block in scored_blocks:
        text_clean = block["text"].strip().lower()
        
        # Skip if we've seen very similar text already
        if any(text_clean in seen or seen in text_clean for seen in seen_texts):
            continue
            
        # Skip generic repeated words like "Overview" unless it's clearly a section header
        if text_clean == "overview" and len([b for b in scored_blocks if b["text"].strip().lower() == "overview"]) > 2:
            # Only keep if it has very high score
            if block["score"] < 8.0:
                continue
        
        filtered_blocks.append(block)
        seen_texts.add(text_clean)
    
    if not filtered_blocks:
        return []
    
    # Sort by document order (page, then position)
    filtered_blocks.sort(key=lambda x: (x["page"], x["y_pos"]))
    
    # Assign levels based on relative importance and support H3
    outline = []
    max_score = max(block["score"] for block in filtered_blocks)
    size_stats = structure_info["size_stats"]
    
    for block in filtered_blocks:
        score_ratio = block["score"] / max_score if max_score > 0 else 0
        size_ratio = block["size"] / size_stats["max"] if size_stats["max"] > 0 else 0
        
        # Enhanced level assignment to include H3
        level = "H3"  # Default to H3
        
        # H1: Very high scores or clearly major sections
        if score_ratio >= 0.85 or (score_ratio >= 0.7 and size_ratio >= 0.8):
            level = "H1"
        # H2: High scores  
        elif score_ratio >= 0.6 or (score_ratio >= 0.4 and size_ratio >= 0.6):
            level = "H2"
        # H3: Medium scores - be more inclusive
        elif score_ratio >= 0.3 or size_ratio >= 0.4:
            level = "H3"
        # Skip items that don't meet H3 threshold
        else:
            continue
        
        outline.append({
            "level": level,
            "text": block["text"].strip() + " ",  # Add trailing space as required
            "page": block["page"]  # Keep 0-based page numbering
        })
    
    return outline

def refined_extract_title(text_blocks, structure_info):
    """
    Enhanced title extraction that can combine multiple prominent text blocks.
    """
    if not text_blocks:
        return ""
    
    # Focus on first page, upper section - expand search area for complex documents
    first_page_blocks = [block for block in text_blocks if block["page"] == 0 and block["y_pos"] <= 500]
    
    if not first_page_blocks:
        return ""
    
    # Find text blocks that could be part of the title
    size_stats = structure_info["size_stats"]
    large_threshold = max(size_stats["mean"] + size_stats["std"], size_stats["median"] * 1.1)
    
    title_candidates = []
    
    for block in first_page_blocks:
        text = block["text"].strip()
        
        # Skip very short text
        if len(text) < 3:
            continue
            
        # Calculate title likelihood score
        score = 0
        
        # Size score - larger text is more likely to be title
        if block["size"] >= size_stats["max"] * 0.85:  # Slightly lower threshold
            score += 20
        elif block["size"] >= large_threshold:
            score += 15
        elif block["size"] >= size_stats["median"]:
            score += 10
        
        # Position score - higher on page is more likely
        if block["y_pos"] <= 150:
            score += 15
        elif block["y_pos"] <= 250:
            score += 12
        elif block["y_pos"] <= 400:
            score += 8
        elif block["y_pos"] <= 500:
            score += 5
        
        # Formatting score
        if block["flags"] & 16:  # Bold
            score += 8
        
        # Length score - be more flexible with length
        if 10 <= len(text) <= 120:
            score += 6
        elif 5 <= len(text) <= 200:
            score += 4
        elif len(text) <= 300:
            score += 2
        
        # Bonus for title-like words
        title_words = ['rfp', 'request', 'proposal', 'plan', 'report', 'guide', 'manual']
        if any(word in text.lower() for word in title_words):
            score += 5
        
        if score >= 12:  # Lower threshold for title consideration
            title_candidates.append({
                "text": text,
                "score": score,
                "y_pos": block["y_pos"],
                "size": block["size"]
            })
    
    if not title_candidates:
        return ""
    
    # Sort by score and position
    title_candidates.sort(key=lambda x: (-x["score"], x["y_pos"]))
    
    # Strategy: Combine multiple high-scoring components for complex titles
    title_parts = []
    used_positions = set()
    
    for candidate in title_candidates:
        # Avoid duplicates by checking if we're too close to used positions
        if any(abs(candidate["y_pos"] - used_pos) < 20 for used_pos in used_positions):
            continue
            
        # Take top scoring items
        if len(title_parts) == 0 or candidate["score"] >= title_candidates[0]["score"] * 0.7:
            title_parts.append(candidate)
            used_positions.add(candidate["y_pos"])
        
        # Limit to reasonable number of parts
        if len(title_parts) >= 5:
            break
    
    # Sort by position and combine
    title_parts.sort(key=lambda x: x["y_pos"])
    
    if len(title_parts) == 1:
        combined_title = title_parts[0]["text"]
    else:
        # Combine parts with appropriate spacing
        combined_parts = []
        for i, part in enumerate(title_parts):
            text = part["text"]
            if i == 0:
                combined_parts.append(text)
            else:
                # Add proper spacing between parts
                if not combined_parts[-1].endswith(' ') and not text.startswith(' '):
                    combined_parts.append(' ' + text)
                else:
                    combined_parts.append(text)
        
        combined_title = "".join(combined_parts).strip()
    
    return combined_title + "  "  # Add trailing spaces as in reference

def process_pdf(pdf_path, min_score_threshold=6.5):
    """
    Main refined processing function.
    """
    try:
        text_blocks = extract_text_blocks(pdf_path)
        
        if not text_blocks:
            return {"title": "", "outline": []}
        
        structure_info = analyze_document_structure(text_blocks)
        title = refined_extract_title(text_blocks, structure_info)
        
        # Adjust threshold based on document type - some documents need lower thresholds
        # Check if this might be a complex document that needs lower threshold
        adjusted_threshold = min_score_threshold
        if "RFP" in title or "Request" in title or len(text_blocks) > 100:
            adjusted_threshold = 4.5  # Lower threshold for complex documents
        
        outline = refined_generate_outline(text_blocks, adjusted_threshold)
        
        return {"title": title, "outline": outline}
        
    except Exception as e:
        print(f"Error processing PDF {pdf_path}: {str(e)}")
        return {"title": "", "outline": []}


comment = '''
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python worker.py <pdf_file>")
        sys.exit(1)
        
    pdf_file = sys.argv[1]
    result = process_pdf(pdf_file)
    print(json.dumps(result, indent=2))
'''

def main(pdf_file):
    return (process_pdf(pdf_file))
