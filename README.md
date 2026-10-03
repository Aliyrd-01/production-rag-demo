# RAG для резюме — облачный деплой (план)

Цель: отдать работодателю живую ссылку на RAG без твоего компа.

## Что уже облачное
- **Qdrant** — уже Qdrant Cloud (QDRANT_URL в .env). Менять не надо.

## Что осталось локально (блокеры)
1. **LLM генерация** — `common.py` пул: Zen → Mistral → OpenRouter → **Ollama qwen2.5:3b (локально)**. На Render Ollama нет.
2. **Embeddings dense** — `common.py get_embeddings()` → **Ollama nomic-embed-text (768d, локально)**. На Render Ollama нет.
3. **Индекс** — Qdrant коллекция `rag_v2` построена эмбеддингами *nomic-embed-text (768d)*, которые считала Ollama. Если сменить embeddings — надо ПЕРЕСОЗДАТЬ индекс.
4. **API-сервер** — `module_12_api.py` на 127.0.0.1:8000. Вынести в облако.

> sparse-часть (Qdrant/bm25) и reranker (bge-reranker) идут черещ FastEmbed — это **Python-библиотека**, работает и в контейнере, Ollama не нужна. Только dense — заложник Ollama.

## Два пути

### Путь A (быстро, без переиндексации) — НЕ стоит
Оставить nomic-embed-text Cloud? Ollama в облаке нет → не работает.
Вывод: Путь A отсутствует, переиндексация неизбежна.

### Путь B (правильный) — переиндексация на облачные embeddings
1. Выбрать облачные embeddings (Qubax / OpenAI text-embedding-3-small / иное).
2. Пересоздать коллекцию `rag_v2_new` нужной размерности; пересканировать корпус MS MARCO.
3. В `.env` прописать новую коллекцию и облачные embeddings в common.py.
4. LLM генерацию переключить на Qubax (или оставить Zen/Mistral/OpenRouter — они уже облачные, работают из контейнера, Ollama не нужна).
5. Деплой на Render (см. ниже).

## Деплой на Render (простой, бесплатно)
1. Залить код в Git-репо (D:\AI\LangChain\final\scripts + requirements.txt + Dockerfile).
2. Render → New Web Service → указать репо; Environment загрузить из `.env.example` (настоящие ключи).
3. Render автоматически соберёт Docker-образ. Даёт URL `https://<имя>.onrender.com`.
4. Проверка: `/health` → ok; `/docs` → Swagger; `/query` → JSON-ответ.
5. HTML-демка на `/` (index.html).

Нюансы Render:
- Free tier "засыпает" после ~15 мин простоя; первый запрос после сна идёт ~30–60 сек.
- Стриминг SSE лучше держать на платном tier (или просто оставить `/query`).

## Декомпозиция: кто что делает
- cloudflared (туннель) — только для n8n/Telegram, к облачному RAG не относится.
- Render — просто хостинг FastAPI-приложения. Qdrant Cloud и MySQL уже в облаке.

## Что уже подготовлено здесь (deploy-rag/)
- requirements.txt — зависимости для Docker/Render.
- Dockerfile — образ python:3.14, копирует scripts/ + index.html, cmd module_12_api.py на 0.0.0.0:8000.
- .env.example — шаблон переменных (ключи вписать свои).

## Открытые вопросы (нужны решения пользователя)
1. Qubax ключ для генерации и/или embeddings? Или оставить Zen/Mistral/OpenRouter (уже облачные)?
2. Какие embeddings для нового индекса (размерность определит переиндексацию)?
3. Хостинг: Render (рекомендуется) или Railway / Docker VPS?
4. Готовность к переиндексации коллекции (прогрев на облаке).