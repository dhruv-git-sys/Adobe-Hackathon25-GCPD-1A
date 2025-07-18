# Adobe India Hackathon 2025
> [**Connecting the Dots ...**](https://d8it4huxumps7.cloudfront.net/uploads/submissions_case/6874faecd848a_Adobe_India_Hackathon_-_Challenge_Doc.pdf)

## 👥 Team

**Team Name:** 🚀 _GCPD_  
**College:** 🎓 _The National Institute of Engineering, Mysore_

| Name                                                  |     Role    |
|-------------------------------------------------------|-------------|
| [Ankit Kumar Shah ](https://github.com/ankitkrshah30) |**👑 Leader** |
| [Dhruv Agrawal](https://github.com/dhruv-git-sys)     |⭐ Member     |
| [Arnav Sharma](https://github.com/ArnavSharma2908/)   |⭐ Member    |

---

## 🧠 Problem Statement

### ✅ Round 1A – Structured Outline Extractor

- Input: 📄 PDF file (≤ 50 pages)
- Output: 📝 JSON with @title@, @H1@, @H2@, @H3@ headings + page numbers
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

### 📘 Round 1B – Persona-Based Document Intelligence

- Input: 📚 3–10 related PDFs + a persona + a job-to-be-done
- Output: 📊 JSON with key sections/subsections relevant to the persona’s task
- Goal: Extract, rank, and summarize the most useful parts of each document 🎯
- Constraints: CPU-only, ≤1GB model, ≤60s processing, no internet 🚫🌐

---

### 🐳 Docker Instructions

```bash
docker build --platform linux/amd64 -t gcpd-pdf .
docker run --rm @  -v $(pwd)/input:/app/input @  -v $(pwd)/output:/app/output @  --network none @  gcpd-pdf
```

## 🛠 Proposed Solution

यहाँ हमें अपना सारा काम करना है !!!
- ### Approach 💡
- ### Libraries Used 📚
- ### Installation/Setup ⚙️

---

> Built with ❤️ by Team GCPD
