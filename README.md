# Adobe India Hackathon 2025

> [**Connecting the Dots ...**](https://d8it4huxumps7.cloudfront.net/uploads/submissions_case/6874faecd848a_Adobe_India_Hackathon_-_Challenge_Doc.pdf)

## 🔗 Repositories

* 🔹 [Round 1A Repository](https://github.com/dhruv-git-sys/Adobe-Hackathon25-GCPD-1A)
* 🔹 [Round 1B Repository](https://github.com/dhruv-git-sys/Adobe-Hackathon25-GCPD-1B)

## 👥 Team

**Team Name:** 🚀 *GCPD*  
**College:** 🎓 *The National Institute of Engineering, Mysore*

| Name                                                 | Role          |
| ---------------------------------------------------- | ------------- |
| [Ankit Kumar Shah](https://github.com/ankitkrshah30) | **👑 Leader** |
| [Dhruv Agrawal](https://github.com/dhruv-git-sys)    | ⭐ Member      |
| [Arnav Sharma](https://github.com/ArnavSharma2908/)  | ⭐ Member      |

---

## ✅ Round 1A – Structured Outline Extractor

### 🧠 Problem Statement

* **Input**: 📄 PDF file (≤ 50 pages)
* **Output**: 📝 JSON with `title`, `H1`, `H2`, `H3` headings + page numbers
* **Constraints**: Offline, Dockerized (CPU-only), ≤10s runtime, ≤200MB model size

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

### 🎯 Core Innovation: Rule-Compliant Design

Our approach is built on **5 strict rules** for universal, robust outline extraction:

1. **No Hardcoded Font Sizes**: Uses relative size analysis only.
2. **No Hardcoded Keywords**: No assumptions about text content or language.
3. **No Fixed Thresholds**: All decisions are data-adaptive.
4. **Adaptive Statistical Analysis**: Document-wide profiling and z-scores.
5. **Truly Generic Processing**: Works for any language, layout, or format.

### 🧱 Architecture

* **`pdf2segment.py`**: Extracts text segments with formatting metadata using PyMuPDF

  * Merges spans with identical styling
  * Outputs attributes like `font`, `size`, `flags`, `bbox`, `text`

* **`main_extractor.py`**:

  * **Phase 1**: Style profiling
  * **Phase 2**: Standalone detection
  * **Phase 3**: Dynamic heading assignment and output

### 🌟 Highlights

* 🚀 **Language-Independent**: Works with any script, language, or symbol set
* ⚙️ **Pure Formatting-Based**: Uses visual and positional clues, not semantics
* 🧠 **Lightweight & Fast**: ML-free, <200MB, Docker-friendly, completes in seconds
* 📁 **Batch Ready**: Automatically processes all files in a directory

---

## 📚 Libraries Used

```python
PyMuPDF>=1.20.0    # PDF processing
json               # Output formatting
sys                # Command line args
collections        # Data structures
```

---

## ⚙️ Setup & Usage

### 🐳 Docker (Recommended)

```bash
git clone https://github.com/dhruv-git-sys/Adobe-Hackathon25-GCPD
cd Adobe-Hackathon25-GCPD
# 📂 Place your PDF files inside the ./input folder (mapped to /app/input in Docker)
docker build --platform linux/amd64 -t mysolutionname:somerandomidentifier .
docker run --rm -v "$(pwd)/input":/app/input -v "$(pwd)/output":/app/output --network none mysolutionname:somerandomidentifier
```

### 💻 Local Development

```bash
git clone https://github.com/dhruv-git-sys/Adobe-Hackathon25-GCPD
cd Adobe-Hackathon25-GCPD
pip install PyMuPDF
python main_extractor.py input.pdf output.json
```

---

> Built with ❤️ by Team GCPD
