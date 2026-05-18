# 🦅 Tax RAG Assistant

![Python](https://img.shields.io/badge/Python-3.14-blue)
![LangChain](https://img.shields.io/badge/LangChain-Latest-green)
![Groq](https://img.shields.io/badge/Groq-Llama3.3-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-Latest-red)
![FAISS](https://img.shields.io/badge/FAISS-Vector_Store-purple)

A production-grade RAG (Retrieval-Augmented Generation) 
system that answers US federal tax questions with 
cited sources from IRS documents.

Built as part of the Enterprise AI Solutions portfolio
targeting AI/ML Engineer roles in professional services.

---

## 🎯 What it does

Ask any US federal tax question in plain English.
Get a professional answer cited directly from 
IRS documents — with the exact page number.

**Example:**
```
Question: What is the child tax credit for 2025?

Answer: The child tax credit (CTC) for 2025 is $2,200 
per qualifying child, of which $1,700 can be claimed 
as the additional child tax credit (ACTC).
[Source: i1040gi.pdf, Page 6] [Source: p17.pdf, Page 109]
```

---

## ✨ Features

- **268 pages** of IRS documents ingested and searchable
- **3,797 chunks** embedded and indexed for fast retrieval
- **Cited answers** — every response traces to source + page
- **Responsible AI** — refuses to hallucinate missing answers
- **Clean Streamlit UI** — chat interface with sources panel
- **Fast retrieval** — FAISS vector search in milliseconds
- **Free stack** — runs on Groq free tier + local embeddings

---

## 🏗️ Architecture

```
User Question (Streamlit UI)
        ↓
Embed question (HuggingFace all-MiniLM-L6-v2)
        ↓
Vector similarity search (FAISS)
        ↓
Top 5 relevant IRS document chunks retrieved
        ↓
Groq LLM (llama-3.3-70b-versatile) generates answer
        ↓
Cited answer displayed in Streamlit chat
```

---

## 🚀 Quick Start

### 1. Clone the repository
```bash
git clone https://github.com/GeoffreyOkwi/enterprise-ai-solutions.git
cd enterprise-ai-solutions/01-tax-rag-assistant
```

### 2. Create virtual environment
```bash
python -m venv .venv
.venv\Scripts\activate    # Windows
source .venv/bin/activate # Mac/Linux
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Set up environment variables
```bash
# Create .env file
cp .env.example .env

# Add your Groq API key (free at console.groq.com)
GROQ_API_KEY=your_key_here
DOCS_FOLDER=path/to/your/pdf/folder
```

### 5. Add IRS documents
```bash
# Download free IRS documents
# Place in your DOCS_FOLDER:
# - IRS Publication 17 (p17.pdf)
# - IRS Form 1040 Instructions (i1040gi.pdf)
```

### 6. Build the knowledge base
```bash
python src/embedder.py
```

### 7. Run the app
```bash
streamlit run app.py
```

Open http://localhost:8501 and start asking tax questions!

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| LLM | Groq — llama-3.3-70b-versatile |
| Embeddings | HuggingFace — all-MiniLM-L6-v2 |
| Vector Store | FAISS (local) |
| Orchestration | LangChain + LangChain-Core |
| Frontend | Streamlit |
| Document Loading | LangChain PyPDFLoader |
| Language | Python 3.14 |

---

## 📁 Project Structure

```
01-tax-rag-assistant/
├── .env.example          # Environment template
├── .gitignore            # Git ignore rules
├── app.py                # Streamlit frontend
├── requirements.txt      # Dependencies
├── README.md             # This file
├── src/
│   ├── __init__.py
│   ├── config.py         # Configuration management
│   ├── loader.py         # PDF loading and chunking
│   ├── embedder.py       # Vector embeddings + FAISS
│   └── retriever.py      # RAG chain + Groq LLM
├── data/
│   └── tax_docs/         # Place IRS PDFs here
├── faiss_index/          # Auto-generated vector index
└── screenshots/          # App screenshots
```

---

## 💬 Sample Questions

```
"What is the standard deduction for 2025?"
"How do I report freelance income?"
"What are the tax brackets for single filers?"
"What is the child tax credit?"
"How do I claim the earned income credit?"
```

---

## 🛡️ Responsible AI

This system is designed with responsible AI principles:

- **Grounded answers** — responds only from source documents
- **Honest uncertainty** — says "I could not find this" 
  rather than hallucinating
- **Full citations** — every answer traceable to source + page
- **Human escalation** — recommends tax professionals
  for complex queries

---

## 👤 Author

**Geoffrey Okwi**
AI/ML Engineer | Stanford AI/ML Alumni | Python Developer

- GitHub: [@GeoffreyOkwi](https://github.com/GeoffreyOkwi)
- Built with: LangChain, Groq, FAISS, Streamlit

---

## 📄 License

MIT License — feel free to use and adapt.

*Part of the Enterprise AI Solutions portfolio*