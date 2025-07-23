# PyMuPDF
import fitz
from json import dump

class Segment:
    """Represents a contiguous block of text with uniform formatting"""
    
    def __init__(self, size, flags, bidi, char_flags, font, color, alpha, ascender, descender, text):
        self.size = size                    # font size
        self.flags = flags                  # formatting flags (bold, italic, etc.)
        self.bidi = bidi                    # bidirectional text direction
        self.char_flags = char_flags        # character-level flags
        self.font = font                    # font family name
        self.color = color                  # text color
        self.alpha = alpha                  # transparency level
        self.ascender = ascender            # font ascender metric
        self.descender = descender          # font descender metric
        self.text = text.strip()            # cleaned text content
        self.len = len(self.text)           # text length
        self.link = None                    # default link value
    
    def __repr__(self):
        return str(self.__dict__)           # return instance dictionary

def extract_pdf_data(pdf_path, out_path=None):
    """Extract PDF content as structured list of page segments"""
    document = fitz.open(pdf_path)          # open PDF document
    pages_segments = []                     # list of page segment lists
    
    for page in document:                   # iterate through pages
        page_dict = page.get_text("dict")   # get page structure
        page_segments = []                  # segments for current page
        current_segment = None              # track current segment for merging
        
        # Single-pass processing with immediate segment creation/merging
        for block in page_dict.get("blocks", []):  # direct access, no filtering needed
            if block.get("type") != 0:      # skip non-text blocks early
                continue
                
            for line in block.get("lines", []):  # process lines directly
                for span in line.get("spans", []):  # process spans directly
                    text = span.get("text", "").strip()  # extract and clean text immediately
                    if not text:            # skip empty spans immediately
                        continue
                    
                    # Extract formatting once, use tuple for faster comparison
                    fmt = (span.get('size', 0), span.get('flags', 0), span.get('bidi', 0),
                           span.get('char_flags', 0), span.get('font', ''), span.get('color', 0),
                           span.get('alpha', 255), span.get('ascender', 0), span.get('descender', 0))
                    
                    # Fast merging check using tuple comparison
                    if (current_segment and current_segment.size == fmt[0] and 
                        current_segment.flags == fmt[1] and current_segment.font == fmt[4] and
                        current_segment.color == fmt[5] and current_segment.bidi == fmt[2] and
                        current_segment.char_flags == fmt[3] and current_segment.alpha == fmt[6] and
                        current_segment.ascender == fmt[7] and current_segment.descender == fmt[8]):
                        # Merge with existing segment
                        current_segment.text += " " + text
                        current_segment.len = len(current_segment.text)
                    else:
                        # Finalize previous segment if exists
                        if current_segment:
                            page_segments.append(current_segment)
                        # Create new segment directly with tuple unpacking
                        current_segment = Segment(*fmt, text)
        
        # Don't forget the last segment
        if current_segment:
            page_segments.append(current_segment)
            
        pages_segments.append(page_segments)    # add page segments to result
    
    document.close()                        # clean up document

    print(f"Extracted {len(pages_segments)} pages")  # print summary
    if pages_segments and out_path:  # Only save to file if out_path is provided
        with open(out_path, 'w') as f:
            dump(pages_segments, f, indent=2, default = lambda x: x.__dict__)

    return pages_segments                   # return list of page segment lists