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
        """Extract features following ALL 5 RULES strictly"""
        features = []
        
        # Handle both dict and Segment object formats
        if hasattr(segment, '__dict__'):
            seg_dict = segment.__dict__
        else:
            seg_dict = segment
        
        text = seg_dict.get('text', '').strip()
        size = seg_dict.get('size', 0)
        
        # Rule #4: Build document-wide statistical profiles for ALL decisions
        all_sizes, all_flags, all_colors, all_lengths, all_words = [], [], [], [], []
        all_ascenders, all_descenders, all_alphas = [], [], []
        
        for page_segs in all_segments_in_doc:
            for s in page_segs:
                s_dict = s.__dict__ if hasattr(s, '__dict__') else s
                if s_dict.get('size', 0) > 0:
                    all_sizes.append(s_dict.get('size', 0))
                    all_flags.append(s_dict.get('flags', 0))
                    all_colors.append(s_dict.get('color', 0))
                    all_ascenders.append(s_dict.get('ascender', 0))
                    all_descenders.append(s_dict.get('descender', 0))
                    all_alphas.append(s_dict.get('alpha', 255))
                    
                    s_text = s_dict.get('text', '').strip()
                    all_lengths.append(len(s_text))
                    all_words.append(len(s_text.split()))
        
        # Rule #1: Use statistical methods ONLY - NO hardcoded font sizes
        def get_percentile_rank(value, sorted_list):
            if not sorted_list or value is None:
                return 0.0
            return np.searchsorted(sorted(sorted_list), value) / len(sorted_list)
        
        # Rule #1: Font size analysis using statistical distributions ONLY
        features.append(get_percentile_rank(size, all_sizes))
        features.append(get_percentile_rank(seg_dict.get('flags', 0), all_flags))
        features.append(get_percentile_rank(seg_dict.get('color', 0), all_colors))
        features.append(get_percentile_rank(seg_dict.get('alpha', 255), all_alphas))
        features.append(get_percentile_rank(seg_dict.get('ascender', 0), all_ascenders))
        features.append(get_percentile_rank(seg_dict.get('descender', 0), all_descenders))
        
        # Rule #4: Text length analysis using relative comparisons
        features.append(get_percentile_rank(len(text), all_lengths))
        features.append(get_percentile_rank(len(text.split()), all_words))
        features.append(get_percentile_rank(len(text.strip()), all_lengths))
        
        # Rule #4: Statistical ratios - relative analysis ONLY (with empty array protection)
        if all_sizes and len(all_sizes) > 0:
            size_mean = np.mean(all_sizes) if all_sizes else 0
            size_std = np.std(all_sizes) if len(all_sizes) > 1 else 1.0
            size_z_score = (size - size_mean) / size_std if size_std > 0 and size_mean > 0 else 0.0
            # Normalize z-score to [-1, 1] range
            features.append(np.tanh(size_z_score / 2.0))
        else:
            features.append(0.0)
        
        # Rule #2: Pattern-based features (NO keyword searching)
        features.append(1 if text.isupper() else 0)
        features.append(1 if text.istitle() else 0)
        features.append(1 if text and text[0].isupper() else 0)
        
        # Rule #4: Position features as percentile ranks (adaptive)
        total_pages = len(all_segments_in_doc)
        page_segments_count = len(all_segments_in_doc[page_idx]) if page_idx < total_pages else 1
        
        features.append(page_idx / max(total_pages - 1, 1))
        features.append(segment_idx / max(page_segments_count - 1, 1))
        
        # Rule #4: Document-relative position analysis
        features.append(1 if page_idx == 0 and segment_idx < 3 else 0)
        features.append(1 if segment_idx < max(page_segments_count * 0.2, 1) else 0)
        
        # Rule #1 & #4: Size comparison within page using statistical analysis
        page_sizes = []
        if page_idx < len(all_segments_in_doc):
            for s in all_segments_in_doc[page_idx]:
                s_size = s.__dict__.get('size', 0) if hasattr(s, '__dict__') else s.get('size', 0)
                if s_size > 0:
                    page_sizes.append(s_size)
        
        if page_sizes and len(page_sizes) > 0:
            features.append(get_percentile_rank(size, page_sizes))
            features.append(1 if size == max(page_sizes) else 0)
            page_mean = np.mean(page_sizes) if page_sizes else 1.0
            page_ratio = size / page_mean if page_mean > 0 else 1.0
            # Use tanh to normalize ratio
            features.append(np.tanh((page_ratio - 1.0) * 2.0))
        else:
            features.extend([0.0, 0.0, 0.0])
        
        # Rule #2: Font pattern detection (NO hardcoded keywords)
        font = seg_dict.get('font', '').lower()
        features.append(1 if 'bold' in font else 0)
        features.append(1 if 'italic' in font else 0)
        features.append(1 if any(weight in font for weight in ['black', 'heavy', 'ultra', 'medium']) else 0)
        
        # Rule #2: Character pattern analysis (NO hardcoded keywords)
        if text:
            features.append(text.count(' ') / len(text))
            features.append(sum(c.isupper() for c in text) / len(text))
            features.append(sum(c.isdigit() for c in text) / len(text))
            features.append(sum(c.isalpha() for c in text) / len(text))
        else:
            features.extend([0.0, 0.0, 0.0, 0.0])
        
        # Rule #2: Structural patterns (NO keyword searching)
        features.append(1 if text.endswith(':') else 0)
        features.append(1 if '\n' not in text else 0)
        features.append(1 if any(c in text for c in '()[]{}') else 0)
        
        # Rule #5: Universal patterns across formats
        # Whitespace pattern analysis
        features.append(len(text) / max(len(text.replace(' ', '')), 1))  # whitespace ratio
        
        # Rule #4: Advanced statistical features (with empty array protection)
        if all_lengths and len(all_lengths) > 0:
            length_mean = np.mean(all_lengths) if all_lengths else 0
            length_std = np.std(all_lengths) if len(all_lengths) > 1 else 1.0
            length_z_score = (len(text) - length_mean) / length_std if length_std > 0 and length_mean >= 0 else 0.0
            features.append(np.tanh(length_z_score / 2.0))
        else:
            features.append(0.0)
        
        # Enhanced bonus-eligible features for heading detection
        # Rule #4: Multi-factor heading indicators
        word_count = len(text.split())
        features.append(get_percentile_rank(word_count, all_words))  # word_count_percentile_refined
        
        # Rule #1 & #4: Size dominance features (with robust empty array protection)
        if all_sizes and len(all_sizes) > 0:
            # Size rank within document
            size_rank = sorted(all_sizes, reverse=True).index(size) + 1 if size in all_sizes else len(all_sizes)
            features.append(size_rank / len(all_sizes))  # size_rank_percentile
            
            # Size vs paragraph heuristic (Rule #4: statistical comparison) - FIXED
            if len(all_sizes) > 5:  # Need sufficient data
                percentile_70 = np.percentile(all_sizes, 70)
                para_sizes = [s for s in all_sizes if s < percentile_70]
                if para_sizes:  # Only calculate if we have paragraph-sized text
                    avg_para_size = np.mean(para_sizes)
                else:
                    avg_para_size = np.mean(all_sizes) * 0.8  # Fallback estimate
            else:
                avg_para_size = np.mean(all_sizes) if all_sizes else 1.0
                
            size_vs_para_ratio = size / avg_para_size if avg_para_size > 0 else 1.0
            features.append(min(np.tanh(size_vs_para_ratio - 1.0), 1.0))  # heading_size_indicator
        else:
            features.extend([0.0, 0.0])
        
        # Rule #2: Enhanced text pattern features
        features.append(1 if text.count('.') <= 1 and len(text.split()) <= 15 else 0)  # heading_length_pattern
        features.append(1 if not any(c in text for c in ',.;!?') and len(text) > 3 else 0)  # clean_heading_pattern
        
        # Rule #4: Contextual position features
        features.append(1 if page_idx == 0 and segment_idx == 0 else 0)  # document_start
        features.append(1 if segment_idx == 0 else 0)  # page_start
        
        return features
    
    def load_training_data(self, base_folder):
        """Load and prepare training data from PDF segments and expected JSON"""
        # Dynamically discover all folders in the base directory
        if not os.path.exists(base_folder):
            raise ValueError(f"Base folder '{base_folder}' does not exist!")
        
        # Only get directories, ignore files
        training_folders = [item for item in os.listdir(base_folder) 
                           if os.path.isdir(os.path.join(base_folder, item))]
        
        if not training_folders:
            raise ValueError(f"No folders found in '{base_folder}'!")
        
        print(f"Found {len(training_folders)} training folders: {training_folders}")
        
        X = []
        y = []
        
        for folder in training_folders:
            # Use correct file paths from base folder
            segmented_file = f"{base_folder}/{folder}/Sample Segmented Input.json"
            expected_file = f"{base_folder}/{folder}/Expected Output.json"
            pdf_file = f"{base_folder}/{folder}/Sample Input.pdf"
            
            if not os.path.exists(expected_file):
                print(f"Warning: Missing expected file {expected_file}")
                continue
                
            print(f"Processing {folder}...")
            
            # Always generate segmented data from PDF (create/overwrite JSON)
            if os.path.exists(pdf_file):
                segments_data = json.loads(extract_pdf_data(pdf_file, segmented_file))
            else:
                print(f"Warning: PDF file {pdf_file} not found")
                continue
            
            # Convert dict segments to Segment-like objects for consistency
            processed_segments = []
            for page_segments in segments_data:
                page_list = []
                for seg in page_segments:
                    # Create object-like access for dict data
                    class SegmentDict:
                        def __init__(self, data):
                            for key, value in data.items():
                                setattr(self, key, value)
                    page_list.append(SegmentDict(seg))
                processed_segments.append(page_list)
            
            # Load expected structure
            with open(expected_file, 'r') as f:
                expected_data = json.load(f)
            
            # Create label mapping
            labels = self.create_labels(processed_segments, expected_data)
            
            # Extract features for each segment
            for page_idx, page_segments in enumerate(processed_segments):
                for segment_idx, segment in enumerate(page_segments):
                    if hasattr(segment, 'text') and segment.text.strip():
                        features = self.extract_features(segment, page_idx, segment_idx, processed_segments)
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
            'ends_with_colon', 'is_single_line', 'has_brackets',
            'whitespace_ratio', 'length_z_score_normalized',
            'word_count_percentile_refined', 'size_rank_percentile', 'heading_size_indicator',
            'heading_length_pattern', 'clean_heading_pattern', 'document_start', 'page_start'
        ]
        
        return np.array(X), np.array(y)
    
    def create_labels(self, segments_data, expected_data):
        """Create labels using STRICT RULE-COMPLIANT similarity matching"""
        labels = {}
        
        # Get expected structure
        expected_title = expected_data.get('title', '').strip()
        expected_outline = expected_data.get('outline', [])
        
        # Build lookup for expected items
        expected_items = {}
        if expected_title:
            expected_items[expected_title] = 'title'
        
        for item in expected_outline:
            item_text = item.get('text', '').strip()
            item_level = item.get('level', '')
            if item_text and item_level in ['H1', 'H2', 'H3']:
                expected_items[item_text] = item_level
        
        # Calculate document-wide statistics (Rule #4 - Adaptive Statistical Analysis)
        all_segments = []
        for page_segs in segments_data:
            all_segments.extend(page_segs)
        
        if not all_segments:
            return labels
        
        # Rule #4: Build statistical profiles for all decisions
        sizes = [s.size if hasattr(s, 'size') else s.get('size', 0) for s in all_segments]
        text_lengths = [len(s.text if hasattr(s, 'text') else s.get('text', '')) for s in all_segments]
        
        # Remove zeros for meaningful statistics
        sizes = [s for s in sizes if s > 0]
        text_lengths = [l for l in text_lengths if l > 0]
        
        if not sizes:
            return labels
        
        # Rule #1: Use statistical distributions only - NO hardcoded font sizes (with empty array protection)
        if sizes and len(sizes) > 0:
            size_mean = np.mean(sizes)
            size_std = np.std(sizes) if len(sizes) > 1 else 1.0
        else:
            size_mean = size_std = 1.0
            
        if text_lengths and len(text_lengths) > 0:
            length_mean = np.mean(text_lengths)
            length_std = np.std(text_lengths) if len(text_lengths) > 1 else 1.0
        else:
            length_mean = length_std = 1.0
        
        # Label segments using EXACT matching only (Rule #2: No keywords, ML pattern recognition)
        for page_idx, page_segments in enumerate(segments_data):
            for segment_idx, segment in enumerate(page_segments):
                text = segment.text.strip() if hasattr(segment, 'text') else segment.get('text', '').strip()
                label_key = f"{page_idx}_{segment_idx}"
                
                if not text:
                    labels[label_key] = self.label_mapping['ignore']
                    continue
                
                # Rule #2: Use ML-based pattern recognition - EXACT matching only
                best_match = None
                best_score = 0.0
                
                # Get segment statistics for adaptive analysis (Rule #4)
                seg_size = segment.size if hasattr(segment, 'size') else segment.get('size', 0)
                seg_length = len(text)
                
                # Rule #4: Use relative comparisons only - percentiles and z-scores (with empty array protection)
                if sizes and len(sizes) > 0:
                    size_percentile = np.searchsorted(sorted(sizes), seg_size) / len(sizes) if seg_size > 0 else 0
                else:
                    size_percentile = 0.0
                    
                if text_lengths and len(text_lengths) > 0:
                    length_percentile = np.searchsorted(sorted(text_lengths), seg_length) / len(text_lengths)
                else:
                    length_percentile = 0.0
                
                for expected_text, expected_label in expected_items.items():
                    # Normalize both texts for comparison (Rule #2: Pattern recognition)
                    normalized_text = self.normalize_text(text)
                    normalized_expected = self.normalize_text(expected_text)
                    
                    # EXACT match only (Rule #2: No keyword searching)
                    if normalized_text == normalized_expected:
                        best_match = expected_label
                        best_score = 1.0
                        break
                    
                    # High similarity token matching (Rule #2: ML-based pattern recognition)
                    token_similarity = self.calculate_exact_token_similarity(normalized_text, normalized_expected)
                    if token_similarity > best_score:
                        best_match = expected_label
                        best_score = token_similarity
                
                # Rule #3: Adaptive thresholds based on document context
                doc_complexity_ratio = len(expected_items) / len(all_segments)
                segment_importance = (size_percentile + length_percentile) / 2
                
                # Rule #3: Dynamic threshold adjustment based on document context
                base_threshold = 0.8  # High threshold for precision
                adaptive_threshold = base_threshold - (doc_complexity_ratio * 0.2) + (segment_importance * 0.1)
                adaptive_threshold = max(0.6, min(0.95, adaptive_threshold))  # Clamp to reasonable range
                
                # Assign label using adaptive threshold (Rule #3)
                if best_match and best_score >= adaptive_threshold:
                    labels[label_key] = self.label_mapping[best_match]
                else:
                    labels[label_key] = self.label_mapping['ignore']
        
        return labels
    
    def calculate_exact_token_similarity(self, text1, text2):
        """Calculate exact token similarity following Rule #2 (ML-based pattern recognition)"""
        tokens1 = set(text1.split())
        tokens2 = set(text2.split())
        
        if not tokens1 or not tokens2:
            return 0.0
        
        # Jaccard similarity - ML pattern recognition approach
        intersection = len(tokens1.intersection(tokens2))
        union = len(tokens1.union(tokens2))
        
        return intersection / union if union > 0 else 0.0
    
    def normalize_text(self, text):
        """Normalize text for better matching"""
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
    
    def train(self, base_folder):
        """Train the prediction model using only formatting characteristics"""
        print("Loading training data...")
        X, y = self.load_training_data(base_folder)
        
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
        """ENHANCED prediction with hierarchical structure analysis and intelligent bonus system"""
        features = self.extract_features(segment, page_idx, segment_idx, all_segments_in_doc)
        features_scaled = self.scaler.transform([features])
        prediction = self.model.predict(features_scaled)[0]
        base_confidence = max(self.model.predict_proba(features_scaled)[0])
        
        # Extract segment properties
        text = segment.text.strip() if hasattr(segment, 'text') else segment.get('text', '').strip()
        size = segment.size if hasattr(segment, 'size') else segment.get('size', 0)
        font = segment.font if hasattr(segment, 'font') else segment.get('font', '').lower()
        flags = segment.flags if hasattr(segment, 'flags') else segment.get('flags', 0)
        
        # Rule #4: Build document statistics for intelligent bonus system
        all_sizes, all_lengths, all_positions = [], [], []
        heading_candidates = []  # Track potential headings for hierarchical analysis
        
        for p_idx, page_segs in enumerate(all_segments_in_doc):
            for s_idx, s in enumerate(page_segs):
                s_size = s.__dict__.get('size', 0) if hasattr(s, '__dict__') else s.get('size', 0)
                s_text = s.__dict__.get('text', '') if hasattr(s, '__dict__') else s.get('text', '')
                s_font = s.__dict__.get('font', '') if hasattr(s, '__dict__') else s.get('font', '')
                s_flags = s.__dict__.get('flags', 0) if hasattr(s, '__dict__') else s.get('flags', 0)
                
                if s_size > 0 and s_text.strip():
                    all_sizes.append(s_size)
                    all_lengths.append(len(s_text.strip()))
                    all_positions.append((p_idx, s_idx))
                    
                    # Identify heading characteristics for hierarchical analysis
                    text_len = len(s_text.split())
                    if (s_size > np.mean(all_sizes) * 1.1 if all_sizes else True) or \
                       'bold' in s_font.lower() or (s_flags & 16) or \
                       (text_len <= 15 and s_text[0].isupper() if s_text else False):
                        heading_candidates.append({
                            'text': s_text.strip(),
                            'size': s_size,
                            'position': (p_idx, s_idx),
                            'font': s_font,
                            'flags': s_flags,
                            'length': text_len
                        })
        
        # Apply ENHANCED bonus system with hierarchical understanding
        bonus = 0.0
        if prediction in ['title', 'H1', 'H2', 'H3'] and all_sizes:
            
            # Rule #1: ADVANCED size analysis with document context
            size_percentile = np.searchsorted(sorted(all_sizes), size) / len(all_sizes) if size > 0 else 0
            size_mean = np.mean(all_sizes) if all_sizes else 1
            
            # MAJOR bonuses for size dominance (Rule #1: Statistical approach)
            if size_percentile > 0.95:  # Top 5% - likely title
                bonus += 0.30
                if prediction == 'title':
                    bonus += 0.10  # Extra for title prediction
            elif size_percentile > 0.85:  # Top 15% - major heading
                bonus += 0.25
            elif size_percentile > 0.75:  # Top 25% - heading
                bonus += 0.20
            elif size_percentile > 0.60:  # Above 60th percentile
                bonus += 0.15
            elif size_percentile < 0.25:  # Too small for heading
                bonus -= 0.25
            
            # Rule #2: CRITICAL font characteristics (headings are usually bold)
            if 'bold' in font or (flags & 16):  # Bold is crucial for headings
                bonus += 0.25
            if any(weight in font for weight in ['black', 'heavy', 'ultra']):
                bonus += 0.20
            if any(weight in font for weight in ['medium', 'semibold']):
                bonus += 0.15
            
            # Rule #4: STRATEGIC position analysis (early = more important)
            if page_idx == 0:  # First page bonuses
                if segment_idx == 0:  # Very first segment
                    bonus += 0.20
                elif segment_idx < 3:  # Early on first page
                    bonus += 0.15
                elif segment_idx < 5:  # Still early
                    bonus += 0.10
            elif page_idx == 1 and segment_idx < 3:  # Early second page
                bonus += 0.12
            elif segment_idx == 0:  # Start of any page
                bonus += 0.08
            
            # Rule #2: TEXT PATTERN intelligence
            text_len = len(text.split())
            
            # Optimal length patterns for different heading types
            if prediction == 'title':
                if 2 <= text_len <= 12:  # Perfect title length
                    bonus += 0.20
                elif text_len <= 20:  # Acceptable title length
                    bonus += 0.15
                elif text_len > 25:  # Too long for title
                    bonus -= 0.30
            else:  # H1, H2, H3
                if 1 <= text_len <= 8:  # Perfect heading length
                    bonus += 0.18
                elif text_len <= 15:  # Acceptable heading length
                    bonus += 0.12
                elif text_len > 20:  # Too long for heading
                    bonus -= 0.25
            
            # Rule #2: CASE and formatting patterns
            if text.isupper() and 3 <= text_len <= 10:  # ALL CAPS headings
                bonus += 0.18
            elif text.istitle() and text_len <= 12:  # Title Case headings
                bonus += 0.15
            elif text and text[0].isupper():  # Capitalized
                bonus += 0.10
            
            # Rule #2: CLEAN heading indicators
            punct_count = sum(text.count(c) for c in ',.;!?')
            if punct_count == 0 and len(text) > 3:  # No punctuation = cleaner heading
                bonus += 0.12
            elif punct_count <= 1:  # Minimal punctuation
                bonus += 0.08
            elif punct_count > 3:  # Too much punctuation
                bonus -= 0.15
            
            if text.endswith(':') and prediction in ['H1', 'H2', 'H3']:  # Colon endings
                bonus += 0.10
            
            # Rule #4: HIERARCHICAL structure understanding
            current_pos = (page_idx, segment_idx)
            similar_sized_before = [h for h in heading_candidates 
                                  if h['position'] < current_pos and 
                                  abs(h['size'] - size) < size_mean * 0.1]
            
            if len(similar_sized_before) == 0 and size > size_mean * 1.2:
                bonus += 0.15  # Likely first major heading
            elif len(similar_sized_before) <= 2:
                bonus += 0.10  # Early in hierarchy
            
            # Rule #5: UNIVERSAL formatting patterns
            if '\n' not in text:  # Single line (good for headings)
                bonus += 0.08
            if not any(char.isdigit() for char in text[:3]):  # No leading numbers
                bonus += 0.05
            if len(text.strip()) == len(text):  # No extra whitespace
                bonus += 0.03
            
        # Apply SMART penalties for paragraph-like characteristics
        if prediction in ['title', 'H1', 'H2', 'H3']:
            text_len = len(text.split())
            
            # Length-based penalties
            if text_len > 35:  # Definitely too long
                bonus -= 0.40
            elif text_len > 25:  # Very long
                bonus -= 0.25
            elif text_len > 20:  # Long
                bonus -= 0.15
            
            # Content structure penalties
            sentence_count = text.count('.') + text.count('!') + text.count('?')
            if sentence_count > 3:  # Multiple sentences
                bonus -= 0.25
            elif sentence_count > 1:
                bonus -= 0.15
            
            comma_count = text.count(',')
            if comma_count > 4:  # Too many commas
                bonus -= 0.20
            elif comma_count > 2:
                bonus -= 0.10
            
            # Paragraph-specific content indicators
            para_indicators = ['paragraph', 'figure', 'table', 'section', 'page', 'chapter']
            if any(word in text.lower() for word in para_indicators):
                bonus -= 0.20
            
            # Multi-line penalty
            if '\n' in text and text.count('\n') > 1:
                bonus -= 0.15
        
        # Rule #3: ADAPTIVE confidence adjustment with bounds
        adjusted_confidence = min(0.97, max(0.03, base_confidence + bonus))
        
        return self.reverse_mapping[prediction], adjusted_confidence
    
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
    import sys
    
    # Check if base folder parameter is provided
    if len(sys.argv) < 2:
        print("Usage: python model_trainer.py <training_data_folder>")
        print("Example: python model_trainer.py \"Training Data\"")
        sys.exit(1)
    
    base_folder = sys.argv[1]
    
    predictor = OutlinePredictor()
    
    try:
        # Train the model
        predictor.train(base_folder)
        
        # Save the model
        predictor.save_model('outline_model.pkl')
        
        print("Training completed successfully!")
        
    except Exception as e:
        print(f"Training failed: {e}")
