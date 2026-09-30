# Engineering Intelligence & Reliability Copilot

Production FastAPI product for NASA-grounded engineering copilots:
Root Cause Analysis, Maintenance Troubleshooting, Engineering Comparison, and Research over **technical repair / failure / anomaly reports**.

> Full write-up for GitHub / resume: [`docs/PROJECT_REPORT.md`](docs/PROJECT_REPORT.md)

## Setup

```bash
cd Document_Portal
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
- `/healthz` `/readyz` `/metrics`

## Docker

```bash
cp .env.example .env   # set GROQ_API_KEY
docker build -t eirc:local .
docker run --rm -p 8000:8000 --env-file .env \
  -v "$(pwd)/data/vectorstore:/app/data/vectorstore" \
  eirc:local
```

Mount a pre-built FAISS index at `/app/data/vectorstore` (the repo does not bake PDFs or the index into the image).

## CI / CD (GitHub Actions)

| Workflow | File | Purpose |
|---|---|---|
| CI | [`.github/workflows/ci.yaml`](.github/workflows/ci.yaml) | Compile, config check, app import, Docker build |
| Deploy definition | [`.github/workflows/defination.yaml`](.github/workflows/defination.yaml) | Reusable: ECR push + ECS task register + service update |
| AWS Deploy | [`.github/workflows/aws.yaml`](.github/workflows/aws.yaml) | Runs CI, then deploys staging (on `main`) or production (manual) |

**Secrets** (repo or environment): `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `GROQ_API_KEY`

**Environment variables** (GitHub Environments `staging` / `production`):
- `ECS_EXECUTION_ROLE_ARN`, `ECS_TASK_ROLE_ARN` (required)
- Optional overrides: `AWS_REGION`, `ECR_REPOSITORY`, `ECS_CLUSTER_STAGING`, `ECS_SERVICE_STAGING`, `ECS_CLUSTER_PRODUCTION`, `ECS_SERVICE_PRODUCTION`

Create the ECS cluster, service, VPC networking, and IAM roles once in AWS; the workflows assume they already exist.

## Evaluation / LLMOps

Offline evaluation stays in the project (CLI + APIs). It is not linked in the product UI.

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
