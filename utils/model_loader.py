import os
from functools import lru_cache

from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq

from exception.exceptions import ReportsException
from logger.logger import get_app_logger
from utils.config_loader import get_config_loader

log = get_app_logger("model_loader")
load_dotenv()


class ModelLoader:
    def __init__(self, config: dict | None = None):
        self.config_loader = get_config_loader()
        self.config = config or self.config_loader.config
        self.validate_env()
        self.validate_config()
        self._embeddings_cache: dict[str, object] = {}
        self._llm_cache: dict[str, object] = {}

    def validate_env(self):
        required_env_vars = ["GROQ_API_KEY"]
        missing_vars = [var for var in required_env_vars if not os.getenv(var)]
        if missing_vars:
            raise ReportsException(
                f"Missing required environment variables: {', '.join(missing_vars)}"
            )
        log.info("Environment variables validated successfully")

    def validate_config(self):
        required_config_keys = ["embedding_model", "retriever", "llm"]
        missing_keys = [key for key in required_config_keys if key not in self.config]
        if missing_keys:
            raise ReportsException(
                f"Missing required config keys: {', '.join(missing_keys)}"
            )
        log.info("Config validated successfully")

    def load_embeddings(self, model_name: str | None = None):
        emb_cfg = self.config["embedding_model"]
        name = model_name or emb_cfg["model_name"]
        if name in self._embeddings_cache:
            return self._embeddings_cache[name]
        try:
            log.info("Loading embedding model", model=name, provider=emb_cfg.get("provider"))
            embeddings = HuggingFaceEmbeddings(
                model_name=name,
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True},
            )
            self._embeddings_cache[name] = embeddings
            return embeddings
        except Exception as e:
            log.error("Error loading embedding model", error=str(e), model=name)
            raise ReportsException(f"Failed to load embedding model: {str(e)}")

    def load_llm(
        self,
        model_name: str | None = None,
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ):
        llm_cfg = self.config["llm"]
        provider = llm_cfg["provider"]
        name = model_name or llm_cfg["model_name"]
        temp = temperature if temperature is not None else llm_cfg.get("temperature", 0.5)
        tokens = max_tokens if max_tokens is not None else llm_cfg.get("max_tokens", 2048)
        cache_key = f"{provider}:{name}:{temp}:{tokens}"
        if cache_key in self._llm_cache:
            return self._llm_cache[cache_key]
        try:
            log.info("Loading LLM model", model=name, provider=provider)
            if provider != "groq":
                raise ReportsException(f"Unsupported provider: {provider}")
            llm = ChatGroq(
                model=name,
                temperature=temp,
                max_tokens=tokens,
                groq_api_key=os.getenv("GROQ_API_KEY"),
            )
            self._llm_cache[cache_key] = llm
            log.info("LLM model loaded successfully", model=name, provider=provider)
            return llm
        except ReportsException:
            raise
        except Exception as e:
            log.error("Error loading LLM model", error=str(e), model=name)
            raise ReportsException(f"Failed to load LLM model: {str(e)}")


@lru_cache(maxsize=1)
def get_model_catalog() -> dict:
    loader = get_config_loader()
    evaluation = loader.get("evaluation", default={}) or {}
    catalog = evaluation.get("catalog") or {}
    return {
        "llms": catalog.get("llms")
        or [
            {"id": "openai/gpt-oss-20b", "label": "Groq GPT-OSS 20B"},
            {"id": "llama-3.1-8b-instant", "label": "Llama 3.1 8B Instant"},
            {"id": "llama-3.3-70b-versatile", "label": "Llama 3.3 70B"},
            {"id": "gemma2-9b-it", "label": "Gemma2 9B IT"},
        ],
        "metrics": evaluation.get("metrics")
        or [
            "answer_relevancy",
            "context_precision",
            "context_recall",
            "answer_correctness",
        ],
        "defaults": {
            "llm": loader.get("llm", "model_name"),
            "embedding": loader.get("embedding_model", "model_name"),
            "feature": "all",
            "limit": evaluation.get("sample_limit", 8),
            "top_k": loader.get("retriever", "top_k", default=6),
            "fetch_k": loader.get("retriever", "fetch_k", default=20),
            "temperature": loader.get("llm", "temperature", default=0.2),
        },
        "notes": {
            "retrieval": (
                "Retrieval uses BM25 + HNSW dense search (RRF fusion), cross-encoder reranking, "
                "and citation enforcement. Metrics and retrieval share the same corpus embedding model."
            )
        },
    }
