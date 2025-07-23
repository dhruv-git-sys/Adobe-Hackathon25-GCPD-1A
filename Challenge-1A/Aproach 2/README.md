# PDF Structured Outline Extractor

This project implements a machine learning-based solution for extracting structured outlines from PDF documents, identifying titles and hierarchical headings (H1, H2, H3).

## Files Overview

- **`Code.py`**: PDF text extraction using PyMuPDF, creates Segment objects with formatting information
- **`model_trainer.py`**: Machine learning model training pipeline using Random Forest classifier
- **`outline_extractor.py`**: Main prediction script that processes PDFs and outputs structured JSON
- **`outline_model.pkl`**: Trained model file (generated after running model_trainer.py)
- **`Think.txt`**: Project planning and analysis document

## Training Data

- `file01.pdf` → `file01.json`: LTC application form
- `file02.pdf` → `file02.json`: Testing syllabus with hierarchical structure  
- `file03.pdf` → `file03.json`: Research paper with complex outline
- `file04.pdf` → `file04.json`: STEM flyer with simple structure
- `file05.pdf` → `file05.json`: Event flyer

## Usage

### 1. Train the Model (one-time setup)
```bash
python model_trainer.py
```

### 2. Extract Outline from PDF
```bash
python outline_extractor.py input.pdf output.json
```

## Output Format

```json
{
  "title": "Document Title",
  "outline": [
    {"level": "H1", "text": "Chapter 1", "page": 1},
    {"level": "H2", "text": "Section 1.1", "page": 2},
    {"level": "H3", "text": "Subsection", "page": 3}
  ]
}
```

## Key Features

- **Font Size Analysis**: Larger fonts typically indicate headings
- **Font Weight Detection**: Bold text often represents structure
- **Position Analysis**: Early segments more likely to be titles
- **Text Pattern Recognition**: Numbered sections, colons, capitalization
- **Relative Sizing**: Compares font sizes within each document

## Model Performance

- **Training Samples**: 446 segments from 5 documents
- **Key Features**: Relative font size (15.4%), absolute size (10.9%), text length (9.3%)
- **Labels**: Title, H1, H2, H3, ignore (for paragraphs/H4+)
- **Confidence Threshold**: 0.4 (balance between precision and recall)

## Docker Compatibility

- CPU-only operation (no GPU required)
- Lightweight Random Forest model (<200MB)
- Fast processing (<10s for typical documents)
- Offline operation (no internet required)

## Dependencies

```bash
pip install PyMuPDF scikit-learn numpy
```

## Example Test

```bash
# Test on training data
python outline_extractor.py file02.pdf test_output.json

# Expected output: Title + 15 headings with proper hierarchy
```
