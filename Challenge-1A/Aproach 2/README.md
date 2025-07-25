# PDF Document Structure Extractor

ML-based system for extracting structured outlines from PDF documents using formatting characteristics.

## Overview

Intelligent document analyzer using machine learning to identify titles and hierarchical headings (H1, H2, H3) from PDFs. Analyzes formatting features like font size, style, positioning, and text patterns without hardcoded rules.

## Components

### 1. `Code.py` - PDF Segment Extraction
Extracts text segments with formatting metadata using PyMuPDF.

### 2. `model_trainer.py` - ML Model Training
Trains Random Forest classifier using percentile-based features. Follows strict design principles:

**Rule #1: No Hardcoded Font Sizes**
- Avoid fixed checks like `if size > 12.0`
- Use statistical methods: `size_percentile = np.searchsorted(sorted_sizes, current_size) / total_sizes`
- Approach: Analyze font sizes using statistical distributions only

**Rule #2: No Hardcoded Keywords**
- Avoid searching for fixed words like "introduction", "chapter", "section"
- Use ML-based pattern recognition
- Approach: Classify based on features, not keywords

**Rule #3: No Fixed Thresholds**
- Avoid absolute checks like `confidence > 0.5`
- Use adaptive thresholds tuned per document
- Approach: Dynamically adjust thresholds based on document context

**Rule #4: Adaptive Statistical Analysis Only**
- Use only relative comparisons: percentiles, z-scores, ratios
- Approach: Build document-wide statistical profiles for all decisions

**Rule #5: Truly Generic Processing**
- Work on all PDF types—forms, manuals, reports, papers
- Handle complex layouts: multi-column, mixed content
- Approach: Detect universal patterns across formats

### 3. `outline_extractor.py` - Main Pipeline
Processes new PDFs: Segments → ML Prediction → Structured Outline.

## Usage

```bash
# Install dependencies
pip install PyMuPDF scikit-learn numpy

# Train model
python model_trainer.py

# Extract outline
python outline_extractor.py input.pdf output.json
```

## Training Data

5 diverse document types in `Training Data/`:
- **file01**: Government form
- **file02**: Technical manual  
- **file03**: Business proposal
- **file04**: Educational document
- **file05**: Event announcement

Each contains: PDF, extracted segments, expected output, actual predictions.

## Features

29 percentile-based features including:
- Font size/style percentile rankings
- Position ratios within document/page
- Text pattern analysis (case, density)
- Document-relative statistical measures

## Output

```json
{
  "title": "Document Title",
  "outline": [
    {"level": "H1", "text": "Heading", "page": 0}
  ]
}
```

## Model Details

- **Algorithm**: Random Forest (100 trees)
- **Classes**: title, H1, H2, H3, ignore
- **Training**: 5-fold cross-validation
- **Features**: Formatting-only, no content keywords

*Adobe Hackathon 2025 Challenge 1A - Approach 2*
