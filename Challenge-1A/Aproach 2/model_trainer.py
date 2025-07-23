import json
import re
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import pickle
import os
from Code import extract_pdf_data  # Import the PDF extraction function

class OutlinePredictor:
    """
    Predicts document structure (title, H1, H2, H3) from PDF segments
    """
    
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        self.feature_names = []
        self.label_mapping = {
            'title': 0,
            'H1': 1, 
            'H2': 2,
            'H3': 3,
            'ignore': 4  # For H4, paragraphs, and other content we skip
        }
        self.reverse_mapping = {v: k for k, v in self.label_mapping.items()}
        
    def extract_features(self, segment, page_idx, segment_idx, all_segments_in_doc):
        """Extract features from a single segment"""
        features = []
        
        # Handle both dict and Segment object formats
        if hasattr(segment, '__dict__'):
            # Segment object - convert to dict
            seg_dict = segment.__dict__
        else:
            # Already a dict
            seg_dict = segment
        
        # Basic formatting features
        features.append(seg_dict.get('size', 0))
        features.append(seg_dict.get('flags', 0))
        features.append(seg_dict.get('char_flags', 0))
        features.append(seg_dict.get('color', 0))
        features.append(seg_dict.get('alpha', 255))
        features.append(seg_dict.get('ascender', 0))
        features.append(seg_dict.get('descender', 0))
        
        # Text-based features
        text = seg_dict.get('text', '')
        features.append(len(text))
        features.append(len(text.split()))  # word count
        features.append(1 if text.isupper() else 0)  # all caps
        features.append(1 if text.istitle() else 0)  # title case
        features.append(text.count(':'))  # colon count (often in headings)
        
        # Position features
        features.append(page_idx)  # page number
        features.append(segment_idx)  # position in page
        
        # Relative size features (compared to document)
        all_sizes = []
        for page_segs in all_segments_in_doc:
            for s in page_segs:
                if hasattr(s, '__dict__'):
                    size = s.__dict__.get('size', 0)
                else:
                    size = s.get('size', 0)
                if size > 0:
                    all_sizes.append(size)
                    
        if all_sizes:
            max_size = max(all_sizes)
            avg_size = np.mean(all_sizes)
            features.append(seg_dict.get('size', 0) / max_size if max_size > 0 else 0)
            features.append(seg_dict.get('size', 0) / avg_size if avg_size > 0 else 0)
        else:
            features.append(0)
            features.append(0)
            
        # Font-based features
        font = seg_dict.get('font', '')
        features.append(1 if 'bold' in font.lower() else 0)
        features.append(1 if 'italic' in font.lower() else 0)
        
        # Boolean feature for first segment (likely title)
        features.append(1 if page_idx == 0 and segment_idx == 0 else 0)
        
        # Text pattern features
        features.append(1 if re.search(r'^[0-9]+\.', text.strip()) else 0)  # numbered
        features.append(1 if text.strip().endswith(':') else 0)  # ends with colon
        
        return features
    
    def load_training_data(self):
        """Load and prepare training data from PDF segments and expected JSON"""
        training_files = [
            ('file01.pdf', 'file01.json'),
            ('file02.pdf', 'file02.json'), 
            ('file03.pdf', 'file03.json'),
            ('file04.pdf', 'file04.json'),
            ('file05.pdf', 'file05.json')
        ]
        
        X = []
        y = []
        
        for pdf_file, expected_file in training_files:
            if not os.path.exists(pdf_file) or not os.path.exists(expected_file):
                print(f"Warning: Missing files {pdf_file} or {expected_file}")
                continue
                
            print(f"Processing {pdf_file}...")
            # Extract segments directly from PDF using extract_pdf_data
            segments_data = extract_pdf_data(pdf_file)  # No output file needed
            
            # Load expected structure
            with open(expected_file, 'r') as f:
                expected_data = json.load(f)
            
            # Create label mapping
            labels = self.create_labels(segments_data, expected_data)
            
            # Extract features for each segment
            for page_idx, page_segments in enumerate(segments_data):
                for segment_idx, segment in enumerate(page_segments):
                    if hasattr(segment, 'text') and segment.text.strip():  # Use Segment object attributes
                        features = self.extract_features(segment, page_idx, segment_idx, segments_data)
                        label_key = f"{page_idx}_{segment_idx}"
                        
                        X.append(features)
                        y.append(labels.get(label_key, self.label_mapping['ignore']))
        
        self.feature_names = [
            'size', 'flags', 'char_flags', 'color', 'alpha', 'ascender', 'descender',
            'text_length', 'word_count', 'is_upper', 'is_title', 'colon_count',
            'page_idx', 'segment_idx', 'relative_size_max', 'relative_size_avg',
            'is_bold_font', 'is_italic_font', 'is_first_segment', 'is_numbered', 'ends_with_colon'
        ]
        
        return np.array(X), np.array(y)
    
    def create_labels(self, segments_data, expected_data):
        """Create labels by matching segments with expected outline"""
        labels = {}
        
        # Get title
        title_text = expected_data.get('title', '').strip()
        
        # Get outline items  
        outline_items = expected_data.get('outline', [])
        
        # Match segments to expected structure
        for page_idx, page_segments in enumerate(segments_data):
            for segment_idx, segment in enumerate(page_segments):
                text = segment.text.strip() if hasattr(segment, 'text') else segment.get('text', '').strip()  # Handle both Segment objects and dicts
                label_key = f"{page_idx}_{segment_idx}"
                
                if not text:
                    continue
                
                # Check if this is the title
                if title_text and self.text_similarity(text, title_text) > 0.8:
                    labels[label_key] = self.label_mapping['title']
                    continue
                
                # Check if this matches any outline item (only H1, H2, H3)
                matched = False
                for item in outline_items:
                    item_text = item.get('text', '').strip()
                    item_level = item.get('level', '')
                    
                    # Only consider H1, H2, H3 levels
                    if item_level in ['H1', 'H2', 'H3'] and item_text and self.text_similarity(text, item_text) > 0.8:
                        labels[label_key] = self.label_mapping[item_level]
                        matched = True
                        break
                
                if not matched:
                    labels[label_key] = self.label_mapping['ignore']
        
        return labels
    
    def text_similarity(self, text1, text2):
        """Simple text similarity based on common words"""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0
        
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union if union > 0 else 0
    
    def train(self):
        """Train the prediction model"""
        print("Loading training data...")
        X, y = self.load_training_data()
        
        if len(X) == 0:
            raise ValueError("No training data found!")
        
        print(f"Training on {len(X)} samples with {len(self.feature_names)} features")
        print(f"Label distribution: {np.bincount(y)}")
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train model
        self.model.fit(X_scaled, y)
        
        # Print feature importance
        feature_importance = sorted(zip(self.feature_names, self.model.feature_importances_), 
                                  key=lambda x: x[1], reverse=True)
        print("\nTop 10 Most Important Features:")
        for feature, importance in feature_importance[:10]:
            print(f"{feature}: {importance:.4f}")
        
        return self.model
    
    def predict_segment(self, segment, page_idx, segment_idx, all_segments_in_doc):
        """Predict the class of a single segment"""
        features = self.extract_features(segment, page_idx, segment_idx, all_segments_in_doc)
        features_scaled = self.scaler.transform([features])
        prediction = self.model.predict(features_scaled)[0]
        confidence = max(self.model.predict_proba(features_scaled)[0])
        
        return self.reverse_mapping[prediction], confidence
    
    def save_model(self, filepath):
        """Save the trained model"""
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'label_mapping': self.label_mapping,
            'reverse_mapping': self.reverse_mapping
        }
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load a trained model"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.feature_names = model_data['feature_names']
        self.label_mapping = model_data['label_mapping']
        self.reverse_mapping = model_data['reverse_mapping']
        print(f"Model loaded from {filepath}")

if __name__ == "__main__":
    predictor = OutlinePredictor()
    
    try:
        # Train the model
        predictor.train()
        
        # Save the model
        predictor.save_model('outline_model.pkl')
        
        print("Training completed successfully!")
        
    except Exception as e:
        print(f"Training failed: {e}")
