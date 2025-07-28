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

**Bonus/Penalty-Based System for Document Structure Extractor** using **formatting characteristics only** - truly language-agnostic system that works on any PDF regardless of language or content type.

#### 🎯 **Core Innovation: Rule-Compliant Design**

Our approach follows **5 strict design principles** for universal compatibility:

**Rule #1: No Hardcoded Font Sizes** 🚫
- Zero fixed checks like `if size > 12.0`
- Pure statistical approach using relative font size analysis

**Rule #2: No Hardcoded Keywords** 🚫  
- No language-dependent words like "introduction", "chapter", "section"
- Pattern recognition using formatting features only

**Rule #3: No Fixed Thresholds** 🚫
- Zero absolute checks like `confidence > 0.5`
- Adaptive thresholds tuned per document context

**Rule #4: Adaptive Statistical Analysis Only** 📊
- Relative ratios and document-wide statistical profiles
- Dynamic analysis based on document structure

**Rule #5: Truly Generic Processing** 🌍
- Works on all PDF types: forms, manuals, reports, papers
- Handles complex layouts: multi-column, mixed content, any language

#### 🏗️ **System Architecture**

1. **`pdf2segment.py`** - PDF Text Extraction & Segmentation
   - Extracts text segments with formatting metadata using PyMuPDF
   - Intelligently merges spans with same styling
   - **Key Attributes**: `font`, `size`, `flags`, `color`, `bbox`, `text`
   - Outputs structured segmented data for analysis

2. **`main_extractor.py`** - Outline Analysis & Extraction
   - **Phase 1**: Style profiling and standalone text detection
   - **Phase 2**: Dynamic hierarchy assignment based on font sizes
   - **Phase 3**: Structured outline generation with proper ordering
   - Handles both PDF and JSON inputs for flexibility

#### 🎯 **Key Features**

- **Style-Based Analysis**: Groups text by font, size, and formatting flags
- **Standalone Detection**: Identifies headings by their positioning context  
- **Dynamic Hierarchy**: Assigns H1/H2/H3 levels based on relative font sizes
- **Universal Compatibility**: Works with any language or document type

#### 🌟 **Key Advantages**

✅ **Zero Language Dependencies** - Works with any language  
✅ **Pure Formatting Analysis** - No content-based assumptions  
✅ **High Performance** - Fast processing without ML overhead  
✅ **Universal Patterns** - Adapts to any document type  
✅ **Docker Ready** - CPU-only, lightweight deployment


---

### Libraries Used 📚

```python
# PDF Processing  
PyMuPDF>=1.20.0        # PDF text extraction with formatting metadata

# Standard Libraries
json                   # JSON output formatting
sys                    # Command line arguments
collections            # Data structures (defaultdict)
```

---

### Installation/Setup ⚙️

#### 🐳 **Docker Setup (Recommended for Hackathon Submission)**

```bash
# Clone repository
git clone <repository-url>
cd Adobe-Hackathon25-GCPD

# Build Docker image (AMD64 compatible)
docker build --platform linux/amd64 -t mysolutionname:somerandomidentifier .

# Create input/output directories
mkdir input output

# Add PDF files to process
cp your-document.pdf input/

# Run batch processing (processes all PDFs in input/ directory)
docker run --rm -v "$(pwd)/input":/app/input -v "$(pwd)/output":/app/output --network none mysolutionname:somerandomidentifier

# Check results
ls output/
cat output/*.json
```

#### 💻 **Local Development Setup**

```bash
# Clone repository
git clone <repository-url>
cd Adobe-Hackathon25-GCPD

# Install dependencies
pip install PyMuPDF

# Extract outline from single PDF
python main_extractor.py input.pdf output.json
```

#### 🚀 **Quick Start Examples**

**Docker (Batch Processing):**
```bash
# Process multiple PDFs at once
docker run --rm -v "$(pwd)/input":/app/input -v "$(pwd)/output":/app/output --network none mysolutionname:somerandomidentifier
```

**Local (Single File):**
```bash
# Process a sample document
python main_extractor.py Sample.pdf output.json

# Output: Structured JSON with title and hierarchical headings
```

#### 📋 **Docker Requirements Met**
- ✅ **AMD64 Platform**: Compatible with `linux/amd64` architecture
- ✅ **Offline Processing**: No network dependencies (`--network none`)  
- ✅ **CPU-Only**: No GPU requirements
- ✅ **Lightweight**: Model size well under 200MB (PyMuPDF only)
- ✅ **Batch Processing**: Automatically processes all PDFs from `/app/input`
- ✅ **Standard Output**: Generates `filename.json` for each `filename.pdf`


---

> Built with ❤️ by Team GCPD
