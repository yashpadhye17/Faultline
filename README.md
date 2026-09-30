# Engineering Intelligence & Reliability Copilot

Production FastAPI product for NASA-grounded engineering copilots:
Root Cause Analysis, Maintenance Troubleshooting, Engineering Comparison, and Research over **technical repair / failure / anomaly reports**.

> Project Report : [`docs/PROJECT_REPORT.md`](docs/PROJECT_REPORT.md)

## Setup

```bash
cd EIRC
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export PYTHONPATH=.
```

Ensure `.env` contains:

```bash
GROQ_API_KEY=your_key
```

Default generation model: `openai/gpt-oss-20b` (Groq). Override via Metrics UI or `config/config.yaml`.

## Corpus preparation

```bash
# wipe prior PDFs and refetch repair-focused reports only
python scripts/fetch_ntrs.py --replace --limit 250 --per-query-limit 40
python scripts/enrich_feature_tags.py
python scripts/build_vectorstore.py --force
```

## Run the product

```bash
export PYTHONPATH=.
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Open:
- http://localhost:8000/
- `/rca` `/maintenance` `/comparison` `/research`
- `/healthz` `/readyz` `/metrics-dashboard`

## Docker deployment

Build a local image and run the API with your env file and a mounted FAISS index (the image does not bake PDFs or the vectorstore).

```bash
cp .env.example .env   # set GROQ_API_KEY
docker build -t eirc:local .
docker run --rm -p 8000:8000 --env-file .env \
  -v "$(pwd)/data/vectorstore:/app/data/vectorstore" \
  -v "$(pwd)/data/uploads:/app/data/uploads" \
  eirc:local
```

Optional Hugging Face cache mount (faster restarts after first model download):

```bash
docker run --rm -p 8000:8000 --env-file .env \
  -v "$(pwd)/data/vectorstore:/app/data/vectorstore" \
  -v "$(pwd)/data/uploads:/app/data/uploads" \
  -v "$(pwd)/.cache/huggingface:/app/.cache/huggingface" \
  eirc:local
```

Checks:
- Health: http://localhost:8000/healthz
- Ready (index loaded): http://localhost:8000/readyz

## Evaluation / LLMOps

Offline evaluation is available in the product UI at `/metrics-dashboard` (sidebar **Evaluations**) and via CLI / APIs.

```bash
export PYTHONPATH=.
python scripts/run_evaluation.py --feature all --limit 8
```

Artifacts: `data/evaluation/runs/<run_or_experiment_id>/`

### Metrics tracked
- **Answer relevancy** — cosine(question, answer)  
- **Context precision** — on-topic retrieved contexts  
- **Context recall** — coverage of gold ground truth  
- **Answer correctness** — cosine(ground_truth, answer)

Gate thresholds default to **0.5** for each metric (`config/config.yaml`).

Retrieval stack: hybrid BM25 + HNSW → `BAAI/bge-reranker-base` → `top_k`, overlapping chunks.

## Architecture

| Feature | Package |
|---|---|
| Root Cause Analysis | `src/document_analyze` |
| Maintenance Troubleshooting | `src/single_docchat` |
| Engineering Comparison | `src/document_comparison` |
| NASA Research Copilot | `src/multidoc_chat` |

Shared platform: `utils/`, `models/`, `prompts/`, `config/`, `logger/`, `exception/`.

Attribution: Data provided by NASA Scientific and Technical Information Program (https://ntrs.nasa.gov/api/openapi/).
