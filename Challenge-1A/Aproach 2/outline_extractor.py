import json
import pickle
from model_trainer import OutlinePredictor

class OutlineExtractor:
    """
    Main class for extracting structured outlines from PDF segments
    """
    
    def __init__(self, model_path='outline_model.pkl'):
        self.predictor = OutlinePredictor()
        self.predictor.load_model(model_path)
    
    def extract_outline(self, segments_data):
        """
        Extract structured outline from PDF segments
        
        Args:
            segments_data: List of page segments (output from extract_pdf_data)
            
        Returns:
            dict: Structured outline with title and heading hierarchy
        """
        title = ""
        outline = []
        
        # Process all segments to find structure
        for page_idx, page_segments in enumerate(segments_data):
            for segment_idx, segment in enumerate(page_segments):
                text = segment.text.strip()  # Use attribute access for Segment objects
                
                if not text:
                    continue
                
                # Predict segment type
                prediction, confidence = self.predictor.predict_segment(
                    segment, page_idx, segment_idx, segments_data
                )
                
                # Only consider high-confidence predictions
                if confidence < 0.4:  # Lowered threshold for better recall
                    continue
                
                if prediction == 'title' and not title:
                    title = text
                elif prediction in ['H1', 'H2', 'H3']:
                    outline.append({
                        "level": prediction,
                        "text": text,
                        "page": page_idx
                    })
        
        return {
            "title": title,
            "outline": outline
        }

def main():
    """
    Main function to process PDF and extract outline
    """
    import sys
    from Code import extract_pdf_data
    
    if len(sys.argv) != 3:
        print("Usage: python outline_extractor.py <input.pdf> <output.json>")
        sys.exit(1)
    
    pdf_path = sys.argv[1]
    output_path = sys.argv[2]
    
    try:
        print(f"Processing {pdf_path}...")
        
        # Extract segments from PDF
        segments_data = extract_pdf_data(pdf_path, "temp_segments.json")
        
        # Extract outline structure
        extractor = OutlineExtractor()
        outline = extractor.extract_outline(segments_data)
        
        # Save result
        with open(output_path, 'w') as f:
            json.dump(outline, f, indent=2)
        
        print(f"Outline extracted and saved to {output_path}")
        print(f"Title: {outline['title']}")
        print(f"Found {len(outline['outline'])} headings")
        
    except Exception as e:
        print(f"Error processing PDF: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
