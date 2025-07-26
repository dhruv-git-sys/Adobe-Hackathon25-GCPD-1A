# Adobe India Hackathon 2025
> [**Connecting the Dots ...**](https://d8it4huxumps7.cloudfront.net/uploads/submissions_case/6874faecd848a_Adobe_India_Hackathon_-_Challenge_Doc.pdf)

## 👥 Team

**Team Name:** 🚀 _GCPD_  
**College:** 🎓 _The National Institute of Engineering, Mysore_

| Name                                                  |     Role    |
|-------------------------------------------------------|-------------|
| [Ankit Kumar Shah ](https://github.com/ankitkrshah30) |**👑 Leader** |
| [Dhruv Agrawal](https://github.com/dhruv-git-sys)     |⭐ Member     |
| [Arnav Sharma](https://github.com/ArnavSharma2908/)   |⭐ Member    |

---

## ✅ Round 1A – Structured Outline Extractor

### 🧠 Problem Statement

- Input: 📄 PDF file (≤ 50 pages)
- Output: 📝 JSON with `title`, `H1`, `H2`, `H3` headings + page numbers
- Must run offline in Docker (CPU-only, ≤10s, ≤200MB model) 💻
- JSON Example:
```json
{
  "title": "Understanding AI",
  "outline": [
    { "level": "H1", "text": "Intro", "page": 1 },
    { "level": "H2", "text": "What is AI?", "page": 2 },
    { "level": "H3", "text": "History", "page": 3 }
  ]
}
```

---

## 🛠 Proposed Solution

### Approach 💡

**ML-based Document Structure Extractor** using **formatting characteristics only** - truly language-agnostic system that works on any PDF regardless of language or content type.

#### 🎯 **Core Innovation: Rule-Compliant Design**

Our approach follows **5 strict design principles** for universal compatibility:

**Rule #1: No Hardcoded Font Sizes** 🚫
- Zero fixed checks like `if size > 12.0`
- Pure statistical approach: `size_percentile = np.searchsorted(sorted_sizes, current_size) / total_sizes`

**Rule #2: No Hardcoded Keywords** 🚫  
- No language-dependent words like "introduction", "chapter", "section"
- ML-based pattern recognition using formatting features only

**Rule #3: No Fixed Thresholds** 🚫
- Zero absolute checks like `confidence > 0.5`
- Adaptive thresholds tuned per document context

**Rule #4: Adaptive Statistical Analysis Only** 📊
- Percentiles, z-scores, and relative ratios exclusively
- Document-wide statistical profiles for all decisions

**Rule #5: Truly Generic Processing** 🌍
- Works on all PDF types: forms, manuals, reports, papers
- Handles complex layouts: multi-column, mixed content, any language

#### 🏗️ **System Architecture**

1. **`Code.py`** - PDF Segment Extraction
   - Extracts text segments with formatting metadata using PyMuPDF
   - **Segment Class Attributes**:
   ```
   `size`, `flags`, `bidi`, `char_flags`, `font`, `color`, `alpha`, `ascender`, `descender`, `text`, `len`, `link`
   ```
   - **Output Structure**: List of pages → List of segments per page → Segment objects with formatting data (OUTPUT[PAGE][SEGMENT])

2. **`model_trainer.py`** - ML Model Training  
   - Random Forest classifier with **37 percentile-based features**
   - **Features**:
   ```
   `size_percentile`, `flags_percentile`, `color_percentile`, `alpha_percentile`, `ascender_percentile`,
   `descender_percentile`, `text_length_percentile`, `word_count_percentile`, `trimmed_length_percentile`,
   `size_z_score_normalized`, `is_upper`, `is_title`, `starts_capital`, `page_position_ratio`, `segment_position_ratio`,
   `early_in_doc`, `early_in_page`, `page_size_percentile`, `is_largest_on_page`, `page_size_ratio_normalized`,
   `is_bold_font`, `is_italic_font`, `is_heavy_font`, `space_density`, `uppercase_ratio`, `digit_ratio`,
   `alpha_ratio`, `ends_with_colon`, `is_single_line`, `has_brackets`, `whitespace_ratio`,
   `length_z_score_normalized`, `word_count_percentile_refined`, `size_rank_percentile`, `heading_size_indicator`,
   `heading_length_pattern`, `clean_heading_pattern`, `document_start`, `page_start`
   ```
   - Enhanced bonus system with hierarchical structure analysis
   - **89.2% cross-validation accuracy**

3. **`outline_extractor.py`** - Optimized Pipeline
   - Document statistics caching for **sub-3s performance**
   - Early filtering and batch processing
   - Structured JSON output generation

4. **`outline_model.pkl`** - Trained Model File
   - Serialized Random Forest model with scaler and feature mappings
   - Links model_trainer.py training output to outline_extractor.py prediction pipeline
   - Contains model, scaler, feature_names, label_mapping, and reverse_mapping

#### 🎯 **Advanced Features**

- **37 Language-Agnostic Features**: Font percentiles, position ratios, formatting patterns
- **Intelligent Bonus System**: Hierarchical analysis with size dominance detection  
- **Performance Optimized**: Processes 50-page PDFs in under 3 seconds
- **Universal Compatibility**: Works with Chinese, Arabic, Hindi, English, Spanish - any language

#### 🌟 **Key Advantages**

✅ **Zero Language Dependencies** - Works with any language  
✅ **Pure Formatting Analysis** - No content-based assumptions  
✅ **High Performance** - Optimized for speed and accuracy  
✅ **Universal Patterns** - Adapts to any document type  
✅ **Docker Ready** - CPU-only, lightweight deployment

> **Note:** The Retrospective_Archive.zip file is intended for internal documentation of retrospective logs and is not part of the production pipeline.


---

### Libraries Used 📚

```python
# Core ML & Data Processing
scikit-learn>=1.0.0    # Random Forest classifier
numpy>=1.21.0          # Statistical analysis and percentile calculations

# PDF Processing  
PyMuPDF>=1.20.0        # PDF text extraction with formatting metadata

# Standard Libraries
json                   # JSON output formatting
os                     # File system operations
re                     # Text normalization (format-only)
```

#### 📊 **Training Data Used**

5 diverse document types for robust training:
- **file01**: Government form
- **file02**: Technical manual  
- **file03**: Business proposal
- **file04**: Educational document
- **file05**: Event announcement

---

### Installation/Setup ⚙️

```bash
# Clone repository
git clone <repository-url>
cd "Challenge-1A/Approach 2"

# Install dependencies
pip install PyMuPDF scikit-learn numpy

# Train the model (one-time setup)
python model_trainer.py "Training Data"

# Extract outline from any PDF
python outline_extractor.py input.pdf output.json
```

#### 🚀 **Quick Start Example**

```bash
# Process a sample document
python outline_extractor.py "Training Data/file01/Sample Input.pdf" result.json

# Output: Structured JSON with title and hierarchical headings
```


---

> Built with ❤️ by Team GCPD
