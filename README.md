# UN‑GenSearch

**An end‑to‑end Retrieval‑Augmented Generation (RAG) system for United Nations documents, integrating robust web crawling, hybrid BM25+FAISS retrieval, and an LLM‑powered chatbot.**

---

## 🚀 Overview

UN‑GenSearch tackles the challenges of factual accuracy and bias in large language models (LLMs) when summarizing and answering questions about United Nations proceedings. By grounding responses in authoritative UN Digital Library data, it delivers precise, unbiased, and fully‑referenced answers.

**Key components:**
1. **Data Collection & Processing**  
   • Automated web crawler for UN voting records & resolutions  
   • PDF download, OCR (pytesseract), and paragraph segmentation (TextTiling)

2. **Hybrid Retriever**  
   • Sparse (TF‑IDF / BM25) + Dense (SentenceTransformer embeddings + FAISS)  
   • Document chunking for long transcripts  
   • Weighted score fusion (α·BM25 + β·dense) with disk caching & live indexing feedback

3. **Chatbot Integration**  
   • Flask API (`/retrieve`, `/answer`, `/status`)  
   • Prompt engineering with “context‑document” template  
   • DeepSeek LLM API for final answer generation  
   • Web UI with real‑time progress bar and “thinking” logs  

---

## ✨ Features

- **High‑Coverage Corpus**  
  3,740 UN‑voting PDFs → 180,067 segmented text chunks  
- **Hybrid Search**  
  Balances exact keyword matching with deep semantic embeddings  
- **Chunk‑Aware Indexing**  
  Mitigates long‑document degradation by splitting into fixed‑length chunks  
- **Asynchronous & Cached**  
  Background index build, persistent caches (`df.pkl`, `tfidf_*.pkl`, `faiss.index`)  
- **Live UX Feedback**  
  Real‑time log streaming & progress bar in the web interface  
- **RAG Chatbot**  
  Fully‑referenced answers with transparent “📚 References”  

---

## 🏗️ Architecture

```
┌──────────┐   crawl   ┌───────────┐   index   ┌────────────┐   query    ┌───────────┐   prompt  ┌───────────┐
│  UN DL   │ ────────> │  Task 1   │ ───────> │  Task 2    │ ───────> │  Task 3   │ ──────> │   LLM     │
│(PDF/OCR) │           │ (Crawler) │          │ (Retriever)│          │ (Chatbot) │         │           │
└──────────┘           └───────────┘          └────────────┘          └───────────┘         └───────────┘
```

1. **Task 1 – Data Collection**  
   - Scrapes UN Digital Library, extracts metadata & text  
   - OCR + segmentation → structured CSV  

2. **Task 2 – Hybrid Retriever**  
   - Preprocess (NLTK), TF‑IDF & embedding indexes  
   - FAISS inner‑product search over normalized chunk embeddings  
   - Hybrid scoring & top‑K ranking  

3. **Task 3 – Chatbot Integration**  
   - Flask serves UI and API endpoints  
   - Builds RAG prompts, calls DeepSeek API, appends references  

---

## ⚙️ Prerequisites

- Python 3.8+  
- CUDA‑enabled GPU (optional, for faster embedding)  
- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) (for `pytesseract`)  
- `.env` file with:  
  ```text
  DEEPSEEK_API_KEY=<your_deepseek_api_key>
  ```

---

## 🛠️ Installation

```bash
git clone https://github.com/your-org/un-gensearch.git
cd un-gensearch
python -m venv venv
source venv/bin/activate      # Linux / macOS
venv\Scripts\activate         # Windows

pip install -r requirements.txt
```

---

## 📡 Data Collection (Task 1)

```bash
# Configure tesseract path if necessary:
export TESSDATA_PREFIX=/usr/share/tesseract-ocr/4.00

# Run crawler:
python scripts/crawl_un_docs.py \
  --output-dir data/raw_pdfs \
  --max-retries 5
```

- **Output:**  
  - `data/raw_pdfs/`: downloaded PDFs  
  - `data/corpus.csv`: combined cover metadata + paragraph segments  

---

## 🗄️ Index Building (Task 2)

```bash
# Starts background thread; serves UI immediately
python app.py
```

- **Process:**  
  1. Loads caches if available  
  2. Otherwise builds:  
     - `df.pkl`, `tfidf_vectorizer.pkl`, `tfidf_matrix.pkl`  
     - `faiss.index` with chunked embeddings  

- **Monitor:** Open browser → `http://localhost:5000/` to watch live logs & progress bar  

---

## 💬 Chatbot Service (Task 3)

Once indexing completes:

1. **Retrieve**  
   ```bash
   curl -X POST http://localhost:5000/retrieve \
     -H "Content-Type: application/json" \
     -d '{"query":"UN resolution on climate finance"}'
   ```

2. **Answer**  
   ```bash
   curl -X POST http://localhost:5000/answer \
     -H "Content-Type: application/json" \
     -d '{"query":"UN resolution on climate finance"}'
   ```

- **Web UI:** type in the textbox & click **Send**  

---

## 📈 Evaluation

- **Task 1**:  
  • 3,740 PDFs crawled (98% coverage)  
  • 180,067 text segments (166 MB)  
  • Metadata accuracy: 98% (spot‑check)  
  • OCR accuracy: ~92% (word‑level)

- **Task 2**:  
  • Precision@5 = 0.74, Recall@5 = 0.68 (50‑query benchmark)  
  • Ablations: sparse‑only (0.70/0.65), dense‑only (0.62/0.58)  
  • +31% recall gain for long docs with chunking

- **Task 3**:  
  • Human evaluation shows improved factual consistency vs. vanilla LLM  
  • End‑to‑end latency <2 s per query (excluding embedding build)  

---

## 🛣️ Future Work

- **Stronger Embeddings:** benchmark `all‑mpnet‑base‑v2`, `bge‑base`, quantized variants  
- **Learnable Fusion:** replace fixed α/β with trainable reranker  
- **User Feedback Loop:** interactive reranking & personalization  
- **Passage‑Level QA & Explainability:** extract answer spans + citation highlighting  

---

## 📜 License

This project is released under the [MIT License](LICENSE).

---

## 🤝 Acknowledgements

- Thakur et al. for the BEIR benchmark  
- Reimers & Gurevych for Sentence‑BERT  
- Lewis et al. for the RAG framework  
- Johnson et al. for FAISS  
- Hearst for TextTiling segmentation  

---