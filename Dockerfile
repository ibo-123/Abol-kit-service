# abol-mms-stt — Meta MMS ASR HTTP wrapper
FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/models/hf \
    PORT=8000

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install CPU torch first from the official index, then the rest.
COPY requirements.txt .
# CPU torch from PyTorch index; strip torch line so pip does not pull a CUDA wheel.
RUN pip install --upgrade pip \
    && pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu \
    && grep -vE '^torch==' requirements.txt > /tmp/requirements.notorch.txt \
    && pip install -r /tmp/requirements.notorch.txt

COPY app ./app
COPY scripts ./scripts
COPY tests ./tests
COPY pytest.ini ./pytest.ini

RUN mkdir -p /models/hf /app/outputs/vtt \
    && useradd --create-home --uid 10001 appuser \
    && chown -R appuser:appuser /app /models

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=5 \
  CMD curl -fsS "http://127.0.0.1:${PORT}/health" || exit 1

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT}"]
