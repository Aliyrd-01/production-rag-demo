"""
common.py — единая точка доступа к LLM для всех скриптов.
Бэкенд: OpenCode Zen + Mistral API. Доступны модели:
  - big-pickle            (основная, сильная)
  - mistral-medium-3.5    (быстрая альтернатива, Mistral API)
  - deepseek-v4-flash-free (flash, резерв)

Пул моделей: big-pickle -> deepseek-v4-flash-free -> Mistral -> OpenRouter.
При ошибке/лимите одной модели — следующая.

ВАЖНО: load_dotenv(override=True) — иначе глобальная переменная Windows
OPENAI_API_KEY (sk-proj-...) перекрывает .env и скрипты незаметно уходят не туда.
"""

import os, time, threading
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.runnables import Runnable
from openai import APIError, RateLimitError, AuthenticationError, InternalServerError, Timeout

load_dotenv(override=True)

ZEN_BASE = os.getenv("OPENAI_API_BASE", "https://opencode.ai/zen/v1")
ZEN_KEY = os.getenv("OPENAI_API_KEY")
ZEN_MODELS = ["big-pickle", "deepseek-v4-flash-free"]

MISTRAL_KEY = os.getenv("MISTRAL_API_KEY")
MISTRAL_BASE = "https://api.mistral.ai/v1"

OR_MODEL = os.getenv("OPENROUTER_MODEL", "openrouter/free")
OR_KEY = os.getenv("OPENROUTER_API_KEY")
OR_BASE = "https://openrouter.ai/api/v1"

QUBAX_BASE = os.getenv("QUBAX_API_BASE", "https://api.qubax.ai/v1")
QUBAX_KEY = os.getenv("QUBAX_API_KEY")
QUBAX_MODELS = [m.strip() for m in
                os.getenv("QUBAX_MODELS", "qwen3-235b-a22b-2507,minimax-m2.5").split(",")
                if m.strip()]

_KEYS_WARNED = False


def _check_api_keys(need_zen: bool = False):
    global _KEYS_WARNED
    if not ZEN_KEY and not OR_KEY and not MISTRAL_KEY and not QUBAX_KEY:
        raise RuntimeError(
            "No API keys. Set QUBAX_API_KEY / OPENAI_API_KEY (Zen) "
            "and/or OPENROUTER_API_KEY, MISTRAL_API_KEY in .env"
        )
    if need_zen and not ZEN_KEY:
        raise RuntimeError("OPENAI_API_KEY required for get_judge_llm")
    if not _KEYS_WARNED:
        if not QUBAX_KEY:
            print("[!] QUBAX_API_KEY not set - Qubax models skipped")
        if not ZEN_KEY:
            print("[!] OPENAI_API_KEY not set - Zen models skipped")
        if not MISTRAL_KEY:
            print("[!] MISTRAL_API_KEY not set - Mistral fallback unavailable")
        if not OR_KEY:
            print("[!] OPENROUTER_API_KEY not set - OpenRouter fallback unavailable")
        _KEYS_WARNED = True


def _make(temperature: float, model: str, base: str, key: str) -> ChatOpenAI:
    return ChatOpenAI(
        model=model,
        openai_api_key=key,
        openai_api_base=base,
        temperature=temperature,
        timeout=60,
        max_retries=2,
    )


def _build_pool(temperature: float):
    """Список LLM: Qubax -> Zen -> Mistral -> OpenRouter. При лимите/ошибке — следующий."""
    _check_api_keys()
    pool = []
    if QUBAX_KEY:
        pool.extend(_make(temperature, m, QUBAX_BASE, QUBAX_KEY) for m in QUBAX_MODELS)
    if ZEN_KEY:
        pool.extend(_make(temperature, m, ZEN_BASE, ZEN_KEY) for m in ZEN_MODELS)
    if MISTRAL_KEY:
        pool.append(_make(temperature, "mistral-medium-3.5", MISTRAL_BASE, MISTRAL_KEY))
    if OR_KEY and os.getenv("ALLOW_OPENROUTER", "").lower() in ("1", "true", "yes"):
        pool.append(_make(temperature, OR_MODEL, OR_BASE, OR_KEY))
    if not pool:
        raise RuntimeError(
            "Пул LLM пуст: нет валидных ключей Qubax/Zen/Mistral/OpenRouter"
        )
    return pool


def _is_retryable(err: Exception) -> bool:
    """Только временные ошибки сети/лимита — переключиться на следующую модель.
    401/Auth — не retryable (ключ невалиден), иначе пул молча уходит на
    следующую модель и отвечает неправильным бэкендом."""
    if isinstance(err, AuthenticationError):
        return False
    return isinstance(err, (RateLimitError, Timeout, InternalServerError, APIError))


class _PooledLLM(Runnable):
    """Пул LLM с invoke + stream + асинхронными версиями.
    После первого успешного вызова рабочая модель запоминается
    и используется первой (без повторных фейлов пула)."""

    def __init__(self, pool):
        self._pool = pool
        self._working_idx = 0
        self._served_model = None
        self._lock = threading.Lock()

    def _capture(self, out):
        meta = getattr(out, "response_metadata", None) or {}
        served = meta.get("model_name") or meta.get("model")
        if served:
            self._served_model = served

    def _ordered(self):
        with self._lock:
            idx = self._working_idx
        pool = self._pool
        return list(range(idx, len(pool))) + list(range(0, idx))

    def _mark_working(self, idx):
        with self._lock:
            self._working_idx = idx

    def invoke(self, input, config=None, **kwargs):
        last_err = None
        for idx in self._ordered():
            llm = self._pool[idx]
            try:
                out = llm.invoke(input, config=config, **kwargs)
                self._capture(out)
                self._mark_working(idx)
                return out
            except Exception as e:
                if not _is_retryable(e):
                    raise
                last_err = e
                if isinstance(e, RateLimitError):
                    time.sleep(2)
                continue
        raise last_err or RuntimeError("all models failed")

    def stream(self, input, config=None, **kwargs):
        yielded_any = False
        for idx in self._ordered():
            llm = self._pool[idx]
            try:
                for chunk in llm.stream(input, config=config, **kwargs):
                    self._capture(chunk)
                    yielded_any = True
                    yield chunk
                self._mark_working(idx)
                return
            except Exception:
                if yielded_any:
                    raise
                continue
        raise RuntimeError("all models failed for stream")

    async def ainvoke(self, input, config=None, **kwargs):
        import asyncio
        last_err = None
        for idx in self._ordered():
            llm = self._pool[idx]
            try:
                out = await llm.ainvoke(input, config=config, **kwargs)
                self._capture(out)
                self._mark_working(idx)
                return out
            except Exception as e:
                if not _is_retryable(e):
                    raise
                last_err = e
                if isinstance(e, RateLimitError):
                    await asyncio.sleep(2)
                continue
        raise last_err or RuntimeError("all models failed")

    async def astream(self, input, config=None, **kwargs):
        yielded_any = False
        for idx in self._ordered():
            llm = self._pool[idx]
            try:
                async for chunk in llm.astream(input, config=config, **kwargs):
                    self._capture(chunk)
                    yielded_any = True
                    yield chunk
                self._mark_working(idx)
                return
            except Exception:
                if yielded_any:
                    raise
                continue
        raise RuntimeError("all models failed for astream")

    def generate(self, prompts, stop=None, callbacks=None, **kwargs):
        """RAGAS-совместимый generate: принимает list[str] промптов."""
        results = []
        for p in prompts:
            ai_msg = self.invoke(p)
            results.append(ai_msg)
        return results


def get_chat_llm(temperature: float = 0.7):
    """Основной чат-LLM: пул Zen-моделей + fallback на Mistral/OpenRouter."""
    return _PooledLLM(_build_pool(temperature))


_PROVIDERS = (
    ("api.qubax.ai", "Qubax"),
    ("opencode.ai", "OpenCode Zen"),
    ("api.mistral.ai", "Mistral"),
    ("openrouter.ai", "OpenRouter"),
)


def _provider_of(base_url: str) -> str:
    for needle, name in _PROVIDERS:
        if needle in (base_url or ""):
            return name
    return "unknown"


def describe_llm(llm) -> dict:
    """Кто реально ответил: провайдер, запрошенная и фактически отдавшая модель."""
    pool = getattr(llm, "_pool", None) or []
    if not pool:
        return {"provider": "?", "requested": "?", "served": "?", "pool": []}
    active = pool[getattr(llm, "_working_idx", 0)]
    requested = getattr(active, "model_name", "?")
    return {
        "provider": _provider_of(getattr(active, "openai_api_base", "")),
        "requested": requested,
        "served": getattr(llm, "_served_model", None) or requested,
        "pool": [getattr(m, "model_name", "?") for m in pool],
    }


def get_judge_llm():
    """LLM для оценки faithfulness — deepseek-v4-flash-free, timeout 120с."""
    _check_api_keys(need_zen=True)
    return ChatOpenAI(
        model="deepseek-v4-flash-free",
        openai_api_key=ZEN_KEY,
        openai_api_base=ZEN_BASE,
        temperature=0,
        timeout=120,
        max_retries=2,
    )


def get_struct_llm():
    return get_chat_llm(temperature=0)


def get_fallback_struct_llm():
    _check_api_keys()
    if not OR_KEY:
        raise RuntimeError("OPENROUTER_API_KEY не задан — get_fallback_struct_llm недоступен")
    return _make(0, OR_MODEL, OR_BASE, OR_KEY)


# ---------------------------------------------------------------------------
# Embeddings: обёртка поверх ollama.Client
# ---------------------------------------------------------------------------
_OLLAMA_HOST = os.getenv("OLLAMA_EMBED_HOST", "http://127.0.0.1:11434")

EMBEDDING_MODELS = {
    "nomic-embed-text":     {"dim": 768,  "size_mb": 274, "description": "Current baseline"},
    "bge-m3":               {"dim": 1024, "size_mb": 1200, "description": "Multilingual, planned primary"},
    "qwen3-embedding:0.6b": {"dim": 1024, "size_mb": 639,  "description": "Top multilingual MTEB (64.3)"},
    "mxbai-embed-large":    {"dim": 1024, "size_mb": 669,  "description": "General purpose"},
    "snowflake-arctic-embed": {"dim": 768, "size_mb": 669, "description": "Lightweight"},
    "all-minilm":           {"dim": 384,  "size_mb": 45,   "description": "Tiny baseline"},
    "qwen3-embedding:4b":   {"dim": 2560, "size_mb": 2500, "description": "Strongest multilingual MTEB (69.5)"},
}


def get_embeddings(model: str = "nomic-embed-text"):
    # Cloud-deploy: dense эмбеддинги берём из FastEmbed (Python, без Ollama).
    from langchain_core.embeddings import Embeddings

    class _FastEmbedEmb(Embeddings):
        def __init__(self):
            self._fe = get_fastembed_dense()

        def embed_documents(self, texts, batch_size=64):
            return [v.tolist() for v in self._fe.embed(texts, batch_size=batch_size)]

        def embed_query(self, text):
            return list(self._fe.embed([text]))[0].tolist()

    return _FastEmbedEmb()


# ---------------------------------------------------------------------------
# FastEmbed models (for Qdrant search)
# ---------------------------------------------------------------------------

_FASTEMBED_DENSE_MODEL = os.getenv("FASTEMBED_DENSE_MODEL", "nomic-ai/nomic-embed-text-v1.5")
_FASTEMBED_SPARSE_MODEL = os.getenv("FASTEMBED_SPARSE_MODEL", "Qdrant/bm25")
_FASTEMBED_RERANKER_MODEL = os.getenv("FASTEMBED_RERANKER_MODEL", "BAAI/bge-reranker-base")

_fast_dense = None
_fast_sparse = None
_fast_reranker = None


def get_fastembed_dense():
    global _fast_dense
    if _fast_dense is None:
        from fastembed import TextEmbedding
        _fast_dense = TextEmbedding(_FASTEMBED_DENSE_MODEL)
    return _fast_dense


def get_fastembed_sparse():
    global _fast_sparse
    if _fast_sparse is None:
        from fastembed import SparseTextEmbedding
        _fast_sparse = SparseTextEmbedding(_FASTEMBED_SPARSE_MODEL)
    return _fast_sparse


def get_fastembed_reranker():
    global _fast_reranker
    if _fast_reranker is None:
        from sentence_transformers import CrossEncoder
        _fast_reranker = CrossEncoder(_FASTEMBED_RERANKER_MODEL)
    return _fast_reranker
