# Production RAG — Hybrid Retrieval + Citations

Демо RAG-пайплайна, развёрнутое на **Streamlit Community Cloud** (бесплатно, без карты).

## Что внутри

| Слой | Реализация |
|---|---|
| Векторное хранилище | Qdrant Cloud, коллекция `rag_v2_q_test` — 310 746 пассажей MS MARCO |
| Dense-векторы | `nomic-ai/nomic-embed-text-v1.5` (768d, COSINE, INT8) через FastEmbed |
| Sparse-векторы | `Qdrant/bm25` через FastEmbed |
| Retrieval | dense + sparse → RRF fusion, либо LLM-rerank, либо CrossEncoder (`bge-reranker-base`) |
| Генерация | LLM-пул (Zen → Mistral → OpenRouter) с prompt, требующим цитаты `[N](msmarco#id)` |
| UI | Streamlit |

Датасет **не пересобирается**: dense-модель в приложении обязана совпадать с той,
по которой построен индекс. Смена модели = переиндексация 310k пассажей.

## Стратегии retrieval

- `hybrid` — dense + sparse, Reciprocal Rank Fusion (дефолт)
- `dense_only` — только векторная семантика
- `sparse_only` — только лексика (BM25)
- `hybrid_llm_rerank` — гибридный поиск, затем LLM переоценивает релевантность
- `hybrid_cross_encoder` — гибридный поиск, затем `bge-reranker-base`

## Запуск локально

```powershell
pip install -r requirements.txt
Copy-Item .env.example .env   # заполнить ключи
streamlit run app.py
```

## Секреты в облаке

`.env` нет. Ключи задаются в **App Settings → Secrets** в формате TOML:

```toml
QDRANT_URL = "https://<cluster>.europe-west3-0.gcp.cloud.qdrant.io:6333"
QDRANT_API_KEY = "..."
QDRANT_COLLECTION = "rag_v2_q_test"
FASTEMBED_DENSE_MODEL = "nomic-ai/nomic-embed-text-v1.5"
FASTEMBED_SPARSE_MODEL = "Qdrant/bm25"
FASTEMBED_RERANKER_MODEL = "BAAI/bge-reranker-base"
OPENAI_API_BASE = "https://opencode.ai/zen/v1"
OPENAI_API_KEY = "..."
OPENROUTER_API_KEY = "..."
OPENROUTER_MODEL = "openrouter/free"
ALLOW_OPENROUTER = 1
```

`app.py` переносит секреты из `st.secrets` в `os.environ` до импорта
`common.py` / `module_5_retrieval.py`, которые читают `os.getenv()`.

## Ограничения Community Cloud

- 2 CPU, **2.7 GB RAM** на приложение, 50 GB диск
- приложение засыпает после 12 ч без трафика; любой посетитель будит его одним кликом
- первый запуск скачивает модели FastEmbed (~500 MB) — занимает 1–3 минуты
- публичный репозиторий: код видно всем, секреты — только в Streamlit Secrets

## Файлы

```
app.py                        Streamlit UI
scripts/common.py             LLM-пул + FastEmbed-обёртки
scripts/module_5_retrieval.py Qdrant hybrid search, rerank
scripts/module_7_generation.py цитаты и контекст
scripts/module_12_api.py      тот же пайплайн через FastAPI (локальный вариант)
```