# Hugging Face Spaces (Docker SDK) - deterministic build for the RAG demo.
# The vector index lives in Qdrant Cloud, so this image only runs the thin
# Streamlit app: FastEmbed (dense+sparse) + LLM pool + citations.
#
# Runs anywhere Docker runs, not just HF:
#   docker build -t production-rag .
#   docker run -p 7860:7860 --env-file .env production-rag
# -> http://localhost:7860

# Full image (not -slim): it already ships libgomp1 + gcc that onnxruntime
# (FastEmbed) needs, so the build needs no apt-get and stays reproducible.
FROM python:3.12

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 \
    TOKENIZERS_PARALLELISM=false \
    FASTEMBED_CACHE_PATH=/data/fastembed \
    STREAMLIT_SERVER_PORT=7860 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

WORKDIR /app

# Separate layer so dependency installs are cached between code changes.
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY . .

EXPOSE 7860

CMD ["streamlit", "run", "app.py", "--server.port=7860", "--server.address=0.0.0.0"]
