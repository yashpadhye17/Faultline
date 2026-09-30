# Engineering Intelligence & Reliability Copilot (EIRC)

**Short blurb (resume / GitHub):**  
Built a production-style RAG copilots platform on NASA NTRS technical repair/failure reports (FastAPI + FAISS + cross-encoder reranking + Groq LLMs), with four engineering features, citation-grounded answers, and a multi-model RAGAS-oriented evaluation dashboard for LLMOps experimentation.

---

## 1. What the project does

EIRC is an **Engineering Intelligence & Reliability Copilot** that answers grounded questions over a curated NASA Technical Reports Server (NTRS) corpus focused on **technical repair, failure, anomaly, mishap, and FMEA-style reports** (not open research papers).

| Feature | User job | Package |
|---|---|---|
| **Root Cause Analysis** | Investigate incidents / anomalies | `src/document_analyze` |
| **Maintenance Troubleshooting** | Diagnostic steps for technicians | `src/single_docchat` |
| **Engineering Comparison** | Compare methods / approaches with evidence | `src/document_comparison` |
| **Research Copilot** | Open questions over the repair corpus | `src/multidoc_chat` |

Optional PDF uploads can be attached as extra session context. Every answer returns **NASA citations** (`doc_id`, title, snippet, NTRS URL) plus **run metadata** (model, latency, prompt version, fetch/top-k, reranker).

---

## 2. Technical architecture

```
User query
  → Feature service (prompt + filters)
  → FAISS recall (fetch_k)
  → Cross-encoder rerank (BAAI/bge-reranker-base) → top_k
  → Context dedupe + hybrid score fusion
  → Groq chat LLM
  → Structured / markdown answer + citations
  → Optional evaluation job → metrics artifacts → Metrics UI
```

### Stack
- **API / UI:** FastAPI, Jinja2 templates, vanilla JS + CSS
- **Retrieval:** FAISS vector store, HuggingFace MiniLM embeddings (index)
- **Rerank:** local HuggingFace cross-encoder `BAAI/bge-reranker-base`
- **Generation:** Groq Chat models (configurable)
- **Chunking:** RecursiveCharacterTextSplitter with overlap (size 1000 / overlap 200)
- **Logging:** structured `structlog`-style app logger under `logger/`
- **Config:** `config/config.yaml` + `.env` (`GROQ_API_KEY`)

### Corpus pipeline
1. `scripts/fetch_ntrs.py --replace` — downloads only repair/failure-relevant STI types (TM/TP/CR/SP) with title/abstract filters  
2. `scripts/enrich_feature_tags.py` — stamps `rca` / `maintenance` / `comparison` / `research` tags  
3. `scripts/build_vectorstore.py --force` — rebuilds FAISS + build manifest  

---

## 3. Metrics system (complete process)

### 3.1 Purpose
Measure whether the RAG pipeline is **faithful, relevant, and correctly grounded**, and compare **multiple generation LLMs × scoring embeddings** under the same gold questions.

### 3.2 Gold dataset
Path: `data/evaluation/datasets/nasa_rag_v1.jsonl`  
Each sample has: `sample_id`, `feature`, `question`, `ground_truth`, `relevant_doc_ids`.

Questions are realistic feature queries tied to docs that exist in the repair corpus (e.g., NEXT PPU capacitor RCA, Mod‑1 wind-turbine FMEA, bearing early-warning maintenance).

### 3.3 Evaluation flow
1. User opens **`/metrics-dashboard`**
2. Selects:
   - one or more **Generation LLMs**
   - one or more **Scoring embeddings**
   - which **metrics** to compute
   - `feature`, `sample limit`, `temperature`, `top_k`, `fetch_k`
3. UI calls `POST /api/v1/evaluation/run`
4. Background job runs an **LLM × embedding matrix**
5. Artifacts land in `data/evaluation/runs/<experiment_id>/` and per-cell run folders
6. UI polls `GET /api/v1/evaluation/status` and renders aggregates / matrix / report

### 3.4 Important technical distinction
| Model role | What it affects |
|---|---|
| **Generation LLM** | Answer text produced for each sample |
| **Scoring embedding** | Cosine metrics (relevancy, correctness, context precision/recall) |
| **Index embedding** | FAISS retrieval (fixed until you rebuild the vectorstore) |

Changing scoring embeddings **does not** rebuild FAISS. To change retrieval embeddings, rebuild with `scripts/build_vectorstore.py` after updating `embedding_model.model_name` in config.

### 3.5 Catalog (defaults in config)
**LLMs (Groq):**
- `openai/gpt-oss-20b` (product default)
- `llama-3.1-8b-instant`
- `llama-3.3-70b-versatile`
- `gemma2-9b-it`

**Scoring embeddings (local HF):**
- `sentence-transformers/all-MiniLM-L6-v2` (also index default)
- `sentence-transformers/all-mpnet-base-v2`
- `BAAI/bge-small-en-v1.5`
- `sentence-transformers/multi-qa-MiniLM-L6-cos-v1`

### 3.6 Metric definitions

| Metric | What it measures | How EIRC computes it |
|---|---|---|
| **Faithfulness** | Answer grounded in retrieved context (anti-hallucination) | Split answer into sentences; fraction with sufficient token overlap vs full retrieved chunk text |
| **Answer relevancy** | Answer addresses the question | MiniLM/selected-embedding cosine(question, answer) |
| **Context precision** | Retrieved chunks are on-topic | Fraction of contexts with cosine(question, context) ≥ 0.25 |
| **Context recall** | Retrieved context covers gold truth | Cosine(ground_truth, joined_contexts) |
| **Answer correctness** | Answer matches gold truth | Cosine(ground_truth, answer) |

Thresholds (configurable gate) live under `evaluation.thresholds` in `config/config.yaml`.

### 3.7 Artifacts per run
- `manifest.json` — fingerprint (models, top_k/fetch_k, sample count, pass/fail)
- `predictions.jsonl` — question, answer, contexts, citations, per-sample metrics
- `metrics.json` — aggregates (+ experiment matrix when multi-model)
- `report.md` — human-readable summary

CLI equivalent:
```bash
export PYTHONPATH=.
python scripts/run_evaluation.py --feature all --limit 8
```

---

## 4. Product UX details
- Sample question chips on each feature page (click-to-fill)
- Markdown-rendered answers (headings, lists, bold) via `marked.js`
- Metrics dashboard with ops strip (doc/chunk counts, reranker, models) and experiment matrix table

---

## 5. How to run locally

```bash
cd Document_Portal
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export PYTHONPATH=.
# .env must contain GROQ_API_KEY=...
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Open:
- Product: `http://localhost:8000/`
- Metrics: `http://localhost:8000/metrics-dashboard`

---

## 6. Resume bullet (copy/paste)

- Designed and shipped **EIRC**, a FastAPI RAG platform over **NASA NTRS repair/failure reports** with four copilots (RCA, maintenance, comparison, research), FAISS + cross-encoder reranking, citation-grounded generation (Groq), and a **multi-LLM / multi-embedding evaluation dashboard** computing faithfulness, relevancy, context precision/recall, and correctness for LLMOps experimentation.

---

## 7. Attribution
Data provided by NASA Scientific and Technical Information Program  
https://ntrs.nasa.gov/api/openapi/
