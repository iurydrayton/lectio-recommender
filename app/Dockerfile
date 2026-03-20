# ── Dockerfile (API) ──────────────────────────────────────────────────────
# Imagem base leve com Python 3.12
FROM python:3.12-slim

# Evita arquivos .pyc e mantém logs em tempo real
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Instala dependências do sistema necessárias para psycopg2
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc \
        libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Instala o Poetry
RUN pip install --no-cache-dir poetry==1.8.3

# Copia apenas os arquivos de dependência primeiro (cache de camadas)
COPY ../pyproject.toml ../poetry.lock* ./

# Instala dependências sem criar virtualenv (já estamos dentro do container)
RUN poetry config virtualenvs.create false \
    && poetry install --no-interaction --no-ansi

# Copia o restante do projeto
COPY . .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]