# ── Dockerfile.worker4 ────────────────────────────────────────────────────
# Imagem leve — worker4 só lê .npy e escreve no Neo4j, não precisa do TF
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc \
        libpq-dev \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir poetry==1.8.3

COPY pyproject.toml poetry.lock* ./

RUN poetry config virtualenvs.create false \
    && poetry install --no-interaction --no-ansi

COPY . .

# Diretório compartilhado com o trainer via volume
RUN mkdir -p /app/embeddings

CMD ["python", "embedding_saver/embedding-saver.py"]