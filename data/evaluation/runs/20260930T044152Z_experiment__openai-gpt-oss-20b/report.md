# EIRC Evaluation Report — 20260930T044152Z_experiment__openai-gpt-oss-20b

- Created: `2026-09-30T04:42:13.079057+00:00`
- Dataset: `/Users/yash/Desktop/Document_Portal/data/evaluation/datasets/nasa_rag_v1.jsonl`
- Samples: **4**
- Generation LLM: `openai/gpt-oss-20b`
- Corpus embedding: `sentence-transformers/all-MiniLM-L6-v2`
- top_k: `6` · fetch_k: `20` · temp: `0.2`
- Gate passed: **True**

## Aggregate metrics

| Metric | Score | Threshold |
|---|---:|---:|
| answer_relevancy | 0.7409 | 0.5 |
| context_precision | 1.0000 | 0.5 |
| context_recall | 0.7158 | 0.5 |
| answer_correctness | 0.5777 | 0.5 |

## What each metric means

- **Faithfulness** — how grounded answer sentences are in retrieved context (anti-hallucination).
- **Answer relevancy** — semantic similarity of answer to the question.
- **Context precision** — fraction of retrieved chunks that are on-topic for the question.
- **Context recall** — how well retrieved context covers the gold ground-truth.
- **Answer correctness** — semantic similarity of answer to gold ground-truth.

## Scoring notes

- Retrieval uses the prebuilt FAISS + reranker pipeline.
- Cosine metrics use the same corpus embedding model as retrieval indexing.
