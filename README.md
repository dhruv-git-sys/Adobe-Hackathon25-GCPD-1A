\# Adobe India Hackathon 2025



\## 👥 Team



\*\*Team Name:\*\* 🚀 \_GCPD\_  

\*\*College:\*\* 🎓 \_The National Institute of Engineering, Mysore\_



| Name               | Role          |

|--------------------|---------------|

| Ankit Kumar Shah   | Team Leader   |

| Dhruv Agrawal      | Team Member   |

| Arnav Sharma       | Team Member   |



---



\## 🧠 Problem Statement



\### ✅ Round 1A – Structured Outline Extractor



\- Input: 📄 PDF file (≤ 50 pages)

\- Output: 📝 JSON with @title@, @H1@, @H2@, @H3@ headings + page numbers

\- Must run offline in Docker (CPU-only, ≤10s, ≤200MB model) 💻

\- JSON Example:

```json

{

&nbsp; "title": "Understanding AI",

&nbsp; "outline": \[

&nbsp;   { "level": "H1", "text": "Intro", "page": 1 },

&nbsp;   { "level": "H2", "text": "What is AI?", "page": 2 },

&nbsp;   { "level": "H3", "text": "History", "page": 3 }

&nbsp; ]

}

```



\### 📘 Round 1B – Persona-Based Document Intelligence



\- Input: 📚 3–10 related PDFs + a persona + a job-to-be-done

\- Output: 📊 JSON with key sections/subsections relevant to the persona’s task

\- Goal: Extract, rank, and summarize the most useful parts of each document 🎯

\- Constraints: CPU-only, ≤1GB model, ≤60s processing, no internet 🚫🌐



---



\### 🐳 Docker Instructions



```bash

docker build --platform linux/amd64 -t gcpd-pdf .

docker run --rm @  -v $(pwd)/input:/app/input @  -v $(pwd)/output:/app/output @  --network none @  gcpd-pdf

```



\## 🛠 Proposed Solution



यहाँ हमें अपना सारा काम करना है !!!

\- ### Approach 💡

\- ### Libraries Used 📚

\- ### Installation/Setup ⚙️



---



> Built with ❤️ by Team GCPD

