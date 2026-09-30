# EIRC Multi-Model Experiment — 20260930T044152Z_experiment

Compares generation LLMs on the gold evaluation set (shared BM25+HNSW retrieval + reranker).

## Configuration

```json
{
  "feature": "all",
  "limit": 4,
  "llm_models": [
    "openai/gpt-oss-20b"
  ],
  "metrics": [],
  "temperature": null,
  "top_k": null,
  "fetch_k": null,
  "reranker_enabled": null
}
```

## Matrix results

| LLM | Relevancy | Ctx precision | Ctx recall | Correctness | Passed |
|---|---:|---:|---:|---:|---|
| openai/gpt-oss-20b | 0.741 | 1.000 | 0.716 | 0.578 | True |

## Best cell per metric

- **answer_relevancy**: 0.7409 — `openai/gpt-oss-20b`
- **context_precision**: 1.0000 — `openai/gpt-oss-20b`
- **context_recall**: 0.7158 — `openai/gpt-oss-20b`
- **answer_correctness**: 0.5777 — `openai/gpt-oss-20b`
