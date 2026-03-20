# ── Dockerfile.train ──────────────────────────────────────────────────────
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc \
        libpq-dev \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir poetry==1.8.3

COPY pyproject.toml poetry.lock* ./

# --with trainer instala o grupo [tool.poetry.group.trainer.dependencies]
RUN poetry config virtualenvs.create false \
    && poetry install --no-interaction --no-ansi --with trainer

COPY . .

RUN mkdir -p /app/embeddings

CMD ["python", "trainer/train.py"]