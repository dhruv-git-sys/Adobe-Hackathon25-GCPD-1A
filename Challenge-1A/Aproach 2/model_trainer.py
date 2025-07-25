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
        self.model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
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
        """Extract features using ONLY percentile ranks and statistical analysis (Rules #1, #4)"""
        features = []
        
        # Handle both dict and Segment object formats
        if hasattr(segment, '__dict__'):
            seg_dict = segment.__dict__
        else:
            seg_dict = segment
        
        text = seg_dict.get('text', '')
        size = seg_dict.get('size', 0)
        
        # Collect all document statistics for percentile ranking (Rule #4)
        all_sizes, all_flags, all_colors, all_lengths, all_words = [], [], [], [], []
        all_ascenders, all_descenders, all_alphas = [], [], []
        
        for page_segs in all_segments_in_doc:
            for s in page_segs:
                s_dict = s.__dict__ if hasattr(s, '__dict__') else s
                if s_dict.get('size', 0) > 0:  # Only valid segments
                    all_sizes.append(s_dict.get('size', 0))
                    all_flags.append(s_dict.get('flags', 0))
                    all_colors.append(s_dict.get('color', 0))
                    all_ascenders.append(s_dict.get('ascender', 0))
                    all_descenders.append(s_dict.get('descender', 0))
                    all_alphas.append(s_dict.get('alpha', 255))
                    
                    s_text = s_dict.get('text', '')
                    all_lengths.append(len(s_text))
                    all_words.append(len(s_text.split()))
        
        # Convert to percentile ranks (Rule #1 - NO hardcoded thresholds)
        def get_percentile_rank(value, sorted_list):
            if not sorted_list or value is None:
                return 0.0
            return np.searchsorted(sorted(sorted_list), value) / len(sorted_list)
        
        # Basic formatting features as percentile ranks
        features.append(get_percentile_rank(size, all_sizes))  # size_percentile
        features.append(get_percentile_rank(seg_dict.get('flags', 0), all_flags))
        features.append(get_percentile_rank(seg_dict.get('color', 0), all_colors))
        features.append(get_percentile_rank(seg_dict.get('alpha', 255), all_alphas))
        features.append(get_percentile_rank(seg_dict.get('ascender', 0), all_ascenders))
        features.append(get_percentile_rank(seg_dict.get('descender', 0), all_descenders))
        
        # Text length features as percentile ranks
        features.append(get_percentile_rank(len(text), all_lengths))
        features.append(get_percentile_rank(len(text.split()), all_words))
        features.append(get_percentile_rank(len(text.strip()), all_lengths))
        
        # Statistical ratios (Rule #4 - relative analysis)
        if all_sizes:
            size_mean = np.mean(all_sizes)
            size_std = np.std(all_sizes) if len(all_sizes) > 1 else 1.0
            size_z_score = (size - size_mean) / size_std if size_std > 0 else 0.0
            features.append(min(max(size_z_score / 3.0, -1.0), 1.0))  # normalized z-score
        else:
            features.append(0.0)
        
        # Formatting pattern features (Rule #2 - no keywords, pattern-based)
        features.append(1 if text.isupper() else 0)
        features.append(1 if text.istitle() else 0)
        features.append(1 if text.strip() and text.strip()[0].isupper() else 0)
        
        # Position features as percentile ranks
        total_pages = len(all_segments_in_doc)
        page_segments_count = len(all_segments_in_doc[page_idx]) if page_idx < total_pages else 1
        
        features.append(page_idx / max(total_pages - 1, 1))  # page_position_ratio
        features.append(segment_idx / max(page_segments_count - 1, 1))  # segment_position_ratio
        
        # Document-relative position features
        features.append(1 if page_idx == 0 and segment_idx < 3 else 0)  # early_in_doc
        features.append(1 if segment_idx < max(page_segments_count * 0.2, 1) else 0)  # early_in_page
        
        # Size comparison within page (Rule #4 - adaptive)
        page_sizes = []
        if page_idx < len(all_segments_in_doc):
            for s in all_segments_in_doc[page_idx]:
                s_size = s.__dict__.get('size', 0) if hasattr(s, '__dict__') else s.get('size', 0)
                if s_size > 0:
                    page_sizes.append(s_size)
        
        if page_sizes:
            features.append(get_percentile_rank(size, page_sizes))  # page_size_percentile
            features.append(1 if size == max(page_sizes) else 0)  # is_largest_on_page
            page_mean = np.mean(page_sizes)
            page_ratio = size / page_mean if page_mean > 0 else 1.0
            features.append(min(page_ratio / 3.0, 1.0))  # normalized page size ratio
        else:
            features.extend([0.0, 0.0, 0.0])
        
        # Font-based features (pattern detection, not keywords)
        font = seg_dict.get('font', '').lower()
        features.append(1 if 'bold' in font else 0)
        features.append(1 if 'italic' in font else 0)
        features.append(1 if any(weight in font for weight in ['black', 'heavy', 'ultra', 'medium']) else 0)
        
        # Character pattern analysis (Rule #2 - no hardcoded keywords)
        if text:
            features.append(text.count(' ') / len(text))  # space_density
            features.append(sum(c.isupper() for c in text) / len(text))  # uppercase_ratio
            features.append(sum(c.isdigit() for c in text) / len(text))  # digit_ratio
            features.append(sum(c.isalpha() for c in text) / len(text))  # alpha_ratio
        else:
            features.extend([0.0, 0.0, 0.0, 0.0])
        
        # Structural patterns
        features.append(1 if text.strip().endswith(':') else 0)
        features.append(1 if len(text.strip().split('\n')) == 1 else 0)  # single_line
        features.append(1 if any(c in text for c in '()[]{}') else 0)  # has_brackets
        
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
            'size_percentile', 'flags_percentile', 'color_percentile', 'alpha_percentile', 
            'ascender_percentile', 'descender_percentile',
            'text_length_percentile', 'word_count_percentile', 'trimmed_length_percentile',
            'size_z_score_normalized', 'is_upper', 'is_title', 'starts_capital',
            'page_position_ratio', 'segment_position_ratio', 'early_in_doc', 'early_in_page',
            'page_size_percentile', 'is_largest_on_page', 'page_size_ratio_normalized',
            'is_bold_font', 'is_italic_font', 'is_heavy_font',
            'space_density', 'uppercase_ratio', 'digit_ratio', 'alpha_ratio',
            'ends_with_colon', 'is_single_line', 'has_brackets'
        ]
        
        return np.array(X), np.array(y)
    
    def create_labels(self, segments_data, expected_data):
        """Create labels by learning from expected JSON outputs - NO hardcoded assumptions"""
        labels = {}
        
        # Get expected structure
        expected_title = expected_data.get('title', '').strip()
        expected_outline = expected_data.get('outline', [])
        
        # Build lookup for expected items using fuzzy matching
        expected_items = {}
        if expected_title:
            expected_items[expected_title] = 'title'
        
        for item in expected_outline:
            item_text = item.get('text', '').strip()
            item_level = item.get('level', '')
            if item_text and item_level in ['H1', 'H2', 'H3']:
                expected_items[item_text] = item_level
        
        # Calculate document statistics for relative analysis (Rule #4)
        all_segments = []
        for page_segs in segments_data:
            all_segments.extend(page_segs)
        
        if not all_segments:
            return labels
        
        # Extract all numerical features for percentile ranking
        sizes = [s.size if hasattr(s, 'size') else s.get('size', 0) for s in all_segments]
        text_lengths = [len(s.text if hasattr(s, 'text') else s.get('text', '')) for s in all_segments]
        word_counts = [len((s.text if hasattr(s, 'text') else s.get('text', '')).split()) for s in all_segments]
        
        # Remove zeros for meaningful statistics
        sizes = [s for s in sizes if s > 0]
        text_lengths = [l for l in text_lengths if l > 0]
        word_counts = [w for w in word_counts if w > 0]
        
        if not sizes:
            return labels
        
        # Sort for percentile calculations (Rule #1)
        sorted_sizes = sorted(sizes)
        sorted_lengths = sorted(text_lengths)
        sorted_words = sorted(word_counts)
        
        # Label segments by matching with expected outputs
        for page_idx, page_segments in enumerate(segments_data):
            for segment_idx, segment in enumerate(page_segments):
                text = segment.text.strip() if hasattr(segment, 'text') else segment.get('text', '').strip()
                label_key = f"{page_idx}_{segment_idx}"
                
                if not text:
                    labels[label_key] = self.label_mapping['ignore']
                    continue
                
                # Find best match with expected items using adaptive similarity
                best_match = None
                best_score = 0.0
                
                for expected_text, expected_label in expected_items.items():
                    # Multi-level similarity matching (Rule #2 - no keywords)
                    
                    # Exact match (highest priority)
                    if text == expected_text:
                        best_match = expected_label
                        best_score = 1.0
                        break
                    
                    # Normalized exact match
                    if self.normalize_text(text) == self.normalize_text(expected_text):
                        score = 0.95
                        if score > best_score:
                            best_match = expected_label
                            best_score = score
                    
                    # Token-based similarity
                    token_sim = self.calculate_token_similarity(text, expected_text)
                    if token_sim > best_score and token_sim > 0.7:
                        best_match = expected_label
                        best_score = token_sim
                    
                    # Subsequence matching for partial matches
                    subseq_sim = self.calculate_subsequence_similarity(text, expected_text)
                    if subseq_sim > best_score and subseq_sim > 0.6:
                        best_match = expected_label
                        best_score = subseq_sim
                
                # Assign label based on best match with adaptive threshold (Rule #3)
                if best_match and best_score > 0.5:  # Base threshold, will be adaptive
                    labels[label_key] = self.label_mapping[best_match]
                else:
                    labels[label_key] = self.label_mapping['ignore']
        
        return labels
    
    def normalize_text(self, text):
        """Normalize text for better matching"""
        import re
        # Remove extra whitespace, normalize case, remove special chars
        text = re.sub(r'\s+', ' ', text.strip().lower())
        text = re.sub(r'[^\w\s]', '', text)
        return text
    
    def calculate_token_similarity(self, text1, text2):
        """Calculate similarity based on token overlap"""
        tokens1 = set(self.normalize_text(text1).split())
        tokens2 = set(self.normalize_text(text2).split())
        
        if not tokens1 or not tokens2:
            return 0.0
        
        intersection = len(tokens1.intersection(tokens2))
        union = len(tokens1.union(tokens2))
        
        return intersection / union if union > 0 else 0.0
    
    def calculate_subsequence_similarity(self, text1, text2):
        """Calculate similarity based on longest common subsequence"""
        t1, t2 = self.normalize_text(text1), self.normalize_text(text2)
        
        if not t1 or not t2:
            return 0.0
        
        # Simple LCS-based similarity
        def lcs_length(s1, s2):
            m, n = len(s1), len(s2)
            dp = [[0] * (n + 1) for _ in range(m + 1)]
            
            for i in range(1, m + 1):
                for j in range(1, n + 1):
                    if s1[i-1] == s2[j-1]:
                        dp[i][j] = dp[i-1][j-1] + 1
                    else:
                        dp[i][j] = max(dp[i-1][j], dp[i][j-1])
            
            return dp[m][n]
        
        lcs_len = lcs_length(t1, t2)
        max_len = max(len(t1), len(t2))
        
        return lcs_len / max_len if max_len > 0 else 0.0
    
    def train(self):
        """Train the prediction model using only formatting characteristics"""
        print("Loading training data...")
        X, y = self.load_training_data()
        
        if len(X) == 0:
            raise ValueError("No training data found!")
        
        print(f"Training on {len(X)} samples with {len(self.feature_names)} features")
        print(f"Label distribution: {np.bincount(y)}")
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train model with balanced classes
        self.model.fit(X_scaled, y)
        
        # Print feature importance
        feature_importance = sorted(zip(self.feature_names, self.model.feature_importances_), 
                                  key=lambda x: x[1], reverse=True)
        print("\nTop 15 Most Important Features (Characteristic-Based):")
        for feature, importance in feature_importance[:15]:
            print(f"{feature}: {importance:.4f}")
        
        # Add cross-validation for better assessment
        from sklearn.model_selection import cross_val_score
        cv_scores = cross_val_score(self.model, X_scaled, y, cv=5)
        print(f"\nCross-validation accuracy: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
        
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
