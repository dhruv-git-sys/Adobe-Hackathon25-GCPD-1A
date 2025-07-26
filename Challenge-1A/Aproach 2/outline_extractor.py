import json
from model_trainer import OutlinePredictor
from Code import extract_pdf_data
import sys

class OutlineExtractor:
    """
    OPTIMIZED class for fast outline extraction from PDF segments
    """
    
    def __init__(self, model_path='outline_model.pkl'):
        self.predictor = OutlinePredictor()
        self.predictor.load_model(model_path)
        self._doc_stats_cache = None  # Cache for document statistics
    
    def _compute_document_stats(self, segments_data):
        """
        PRE-COMPUTE and CACHE document statistics for faster processing
        """
        if self._doc_stats_cache is not None:
            return self._doc_stats_cache
        
        all_sizes, all_lengths, all_flags = [], [], []
        
        for page_segs in segments_data:
            for s in page_segs:
                if hasattr(s, 'text') and s.text.strip():
                    size = getattr(s, 'size', 0)
                    if size > 0:
                        all_sizes.append(size)
                        all_lengths.append(len(s.text.strip()))
                        all_flags.append(getattr(s, 'flags', 0))
        
        # Cache the computed statistics
        self._doc_stats_cache = {
            'sizes': all_sizes,
            'lengths': all_lengths,
            'flags': all_flags,
            'size_mean': sum(all_sizes) / len(all_sizes) if all_sizes else 1,
            'size_75th': sorted(all_sizes)[int(len(all_sizes) * 0.75)] if all_sizes else 1
        }
        
        return self._doc_stats_cache
    
    def extract_outline(self, segments_data):
        """
        OPTIMIZED outline extraction with detailed segment analysis logging
        
        Args:
            segments_data: List of page segments (output from extract_pdf_data)
            
        Returns:
            dict: Structured outline with title and heading hierarchy
        """
        if not segments_data:
            return {"title": "", "outline": []}

        print("\n🔍 STARTING SEGMENT-BY-SEGMENT ANALYSIS")
        print("=" * 80)
        
        # PRE-COMPUTE document statistics ONCE for massive speedup
        doc_stats = self._compute_document_stats(segments_data)
        
        title = ""
        heading_candidates = []
        processed_count = 0
        max_candidates = 50  # Limit for performance
        
        # FAST adaptive threshold calculation
        total_segments = sum(len(page) for page in segments_data)
        doc_complexity = total_segments / len(segments_data)
        
        if doc_complexity > 50:
            base_threshold = 0.55
        elif doc_complexity > 20:
            base_threshold = 0.50
        else:
            base_threshold = 0.45
        
        title_threshold = min(0.75, base_threshold + 0.15)
        heading_threshold = min(0.70, base_threshold + 0.10)
        
        print(f"📊 DOCUMENT STATS:")
        print(f"   Total segments: {total_segments}")
        print(f"   Pages: {len(segments_data)}")
        print(f"   Complexity: {doc_complexity:.1f}")
        print(f"   Title threshold: {title_threshold:.2f}")
        print(f"   Heading threshold: {heading_threshold:.2f}")
        print(f"   Size mean: {doc_stats['size_mean']:.1f}")
        print(f"   Size 75th percentile: {doc_stats['size_75th']:.1f}")
        
        # BATCH PROCESS with early termination for speed
        for page_idx, page_segments in enumerate(segments_data):
            if not page_segments:
                continue
                
            print(f"\n📄 PAGE {page_idx + 1} - Processing {len(page_segments)} segments:")
            print("-" * 60)
                
            # PERFORMANCE: Skip processing too many pages if we have enough candidates
            if len(heading_candidates) > max_candidates and page_idx > 20:
                print(f"⚠️  PERFORMANCE LIMIT: Skipping remaining pages (have {len(heading_candidates)} candidates)")
                break
                
            for segment_idx, segment in enumerate(page_segments):
                processed_count += 1
                
                # PERFORMANCE: Early exit if processing too many segments
                if processed_count > 2000:  # Limit for 10-second constraint
                    print(f"⚠️  PERFORMANCE LIMIT: Stopping at segment {processed_count}")
                    break
                    
                text = segment.text.strip()
                segment_info = f"[P{page_idx+1}S{segment_idx+1}] '{text[:30]}{'...' if len(text) > 30 else ''}'"
                
                # ULTRA-FAST pre-filtering
                text_len = len(text)
                if text_len <= 2:
                    print(f"❌ {segment_info} → EXCLUDED: Too short ({text_len} chars)")
                    continue
                elif text_len > 200:
                    print(f"❌ {segment_info} → EXCLUDED: Too long ({text_len} chars)")
                    continue
                elif not text.replace(' ', '').replace('.', '').replace(',', '').replace(':', '').replace('-', '').isalnum():
                    print(f"❌ {segment_info} → EXCLUDED: Non-alphanumeric content")
                    continue
                elif text.isdigit():
                    print(f"❌ {segment_info} → EXCLUDED: Pure number")
                    continue
                elif text.count('.') > 3:
                    print(f"❌ {segment_info} → EXCLUDED: Too many dots ({text.count('.')})")
                    continue
                
                word_count = len(text.split())
                if word_count > 25:
                    print(f"❌ {segment_info} → EXCLUDED: Too many words ({word_count})")
                    continue
                elif word_count == 0:
                    print(f"❌ {segment_info} → EXCLUDED: No words")
                    continue
                
                # FAST size-based pre-filtering using cached stats
                segment_size = getattr(segment, 'size', 0)
                if segment_size > 0 and segment_size < doc_stats['size_mean'] * 0.8:
                    print(f"❌ {segment_info} → EXCLUDED: Size too small ({segment_size:.1f} < {doc_stats['size_mean'] * 0.8:.1f})")
                    continue
                
                # Get ML prediction with confidence (this is the expensive part)
                print(f"🤖 {segment_info} → ANALYZING with ML...")
                prediction, confidence = self.predictor.predict_segment(
                    segment, page_idx, segment_idx, segments_data
                )
                
                # FAST threshold filtering
                required_threshold = title_threshold if prediction == 'title' else heading_threshold
                if confidence < required_threshold:
                    print(f"❌ {segment_info} → EXCLUDED: Low confidence ({prediction}={confidence:.3f} < {required_threshold:.3f})")
                    continue
                
                # Process valid candidates
                if prediction == 'title' and not title:
                    print(f"✅ {segment_info} → ACCEPTED AS TITLE (confidence={confidence:.3f})")
                    title = text
                elif prediction in ['H1', 'H2', 'H3']:
                    print(f"✅ {segment_info} → ACCEPTED AS {prediction} (confidence={confidence:.3f}, size={segment_size:.1f})")
                    heading_candidates.append({
                        "level": prediction,
                        "text": text,
                        "page": page_idx,
                        "confidence": confidence,
                        "size": segment_size,
                        "position": (page_idx, segment_idx)
                    })
                    
                    # PERFORMANCE: Stop if we have enough candidates
                    if len(heading_candidates) >= max_candidates:
                        print(f"⚠️  PERFORMANCE LIMIT: Reached {max_candidates} candidates, stopping page processing")
                        break
                else:
                    print(f"❌ {segment_info} → EXCLUDED: Unexpected prediction ({prediction})")
            
            # Break from outer loop too if we have enough
            if len(heading_candidates) >= max_candidates:
                break
        
        print(f"\n📊 EXTRACTION SUMMARY:")
        print(f"   Processed segments: {processed_count}")
        print(f"   Title found: {'✅' if title else '❌'} '{title[:40]}{'...' if len(title) > 40 else ''}'" if title else "   Title found: ❌")
        print(f"   Heading candidates: {len(heading_candidates)}")
        
        # FAST duplicate removal and validation
        print(f"\n🔄 POST-PROCESSING: Removing duplicates and validating...")
        validated_outline = self._fast_process_headings(heading_candidates)
        
        final_result = {
            "title": title,
            "outline": [
                {
                    "level": item['level'],
                    "text": item['text'],
                    "page": item['page'] + 1  # Convert to 1-based page numbering
                }
                for item in validated_outline
            ]
        }
        
        print(f"\n🎯 FINAL RESULT:")
        print(f"   Title: '{final_result['title']}'")
        print(f"   Final outline items: {len(final_result['outline'])}")
        for i, item in enumerate(final_result['outline'][:10]):  # Show first 10
            print(f"   [{i+1}] {item['level']}: '{item['text'][:40]}{'...' if len(item['text']) > 40 else ''}' (p.{item['page']})")
        if len(final_result['outline']) > 10:
            print(f"   ... and {len(final_result['outline']) - 10} more items")
        
        return final_result
    
    def _fast_process_headings(self, candidates):
        """
        ULTRA-FAST heading processing with optimized duplicate removal
        """
        if not candidates:
            print("   No candidates to process")
            return []
        
        print(f"   Input candidates: {len(candidates)}")
        
        # Sort by confidence first, then position (keep best candidates)
        candidates.sort(key=lambda x: (-x['confidence'], x['position']))
        
        # SUPER FAST duplicate removal using hash set
        seen_hashes = set()
        unique_candidates = []
        duplicates_removed = 0
        
        for candidate in candidates:
            # Ultra-fast text normalization and hashing
            text = candidate['text']
            normalized = ''.join(c.lower() for c in text if c.isalnum())[:50]  # Limit length
            text_hash = hash(normalized)
            
            # Skip duplicates instantly
            if text_hash in seen_hashes:
                duplicates_removed += 1
                continue
            
            seen_hashes.add(text_hash)
            
            # LIGHTNING FAST validation
            word_count = text.count(' ') + 1  # Faster than split()
            if word_count > 15:
                print(f"   🗑️  Filtered out: '{text[:30]}...' (too many words: {word_count})")
                continue
            elif len(text) <= 3:
                print(f"   🗑️  Filtered out: '{text}' (too short)")
                continue
            
            unique_candidates.append(candidate)
            print(f"   ✅ Kept: {candidate['level']} '{text[:30]}{'...' if len(text) > 30 else ''}' (conf={candidate['confidence']:.3f})")
            
            # PERFORMANCE: Limit results
            if len(unique_candidates) >= 25:
                print(f"   ⚠️  Performance limit reached: keeping top 25 candidates")
                break
        
        print(f"   Duplicates removed: {duplicates_removed}")
        print(f"   After deduplication: {len(unique_candidates)}")
        
        # FAST level optimization based on size and position
        print(f"   🔧 Optimizing heading levels...")
        level_changes = 0
        for candidate in unique_candidates:
            text = candidate['text']
            size = candidate.get('size', 0)
            confidence = candidate['confidence']
            original_level = candidate['level']
            
            # Quick pattern-based level assignment
            if confidence > 0.8 and (size > 15 or any(word in text.lower() for word in ['chapter', 'section'])):
                candidate['level'] = 'H1'
            elif len(text) > 60 or text.count(' ') > 10:
                candidate['level'] = 'H3'
            elif candidate['level'] not in ['H1', 'H2', 'H3']:
                candidate['level'] = 'H2'
            
            if original_level != candidate['level']:
                level_changes += 1
                print(f"   🔄 Level changed: '{text[:25]}...' {original_level} → {candidate['level']}")
        
        if level_changes == 0:
            print(f"   No level changes needed")
        else:
            print(f"   Level changes: {level_changes}")
        
        # Return top 20 candidates for performance
        final_count = min(20, len(unique_candidates))
        final_candidates = unique_candidates[:final_count]
        print(f"   Final candidates returned: {len(final_candidates)}")
        
        return final_candidates

def main(pdf_path, output_path):
    """
    OPTIMIZED main function with performance monitoring
    """
    import time
    start_time = time.time()
    
    try:
        print(f"Processing {pdf_path}...")
        
        # FAST PDF extraction
        segments_json = extract_pdf_data(pdf_path)
        segments_data = json.loads(segments_json)
        
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
        
        extraction_time = time.time() - start_time
        print(f"PDF extraction: {extraction_time:.2f}s")
        
        if not processed_segments:
            print("No segments extracted!")
            return
        
        # FAST outline extraction
        extractor = OutlineExtractor()
        outline = extractor.extract_outline(processed_segments)
        processing_time = time.time() - start_time - extraction_time
        print(f"Outline processing: {processing_time:.2f}s")
        
        # FAST JSON save
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(outline, f, indent=2, ensure_ascii=False)
        
        total_time = time.time() - start_time
        print(f"Total time: {total_time:.2f}s")
        print(f"Title: {outline['title'][:50]}{'...' if len(outline['title']) > 50 else ''}")
        print(f"Found {len(outline['outline'])} headings")
        
        # PERFORMANCE WARNING if too slow
        if total_time > 8.0:
            print(f"⚠️  WARNING: Processing took {total_time:.2f}s (target: <10s)")
        else:
            print(f"✅ PERFORMANCE: Processing completed in {total_time:.2f}s")
        
    except Exception as e:
        print(f"Error processing PDF: {e}")
        # Save empty result to avoid crashes
        empty_result = {"title": "", "outline": []}
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(empty_result, f, indent=2)
        return

if __name__ == "__main__":
    
    # Expect two args: input PDF path and output JSON path
    if len(sys.argv) != 3:
        print("Usage: python outline_extractor.py <input_pdf_path> <output_json_path>")
        sys.exit(1)

    input_pdf  = sys.argv[1]
    output_json = sys.argv[2]
    main(input_pdf, output_json)
