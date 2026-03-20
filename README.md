# 📚 Lectio Recommender

Sistema de recomendação de livros construído com **TensorFlow**, **Neo4j** e **FastAPI**. Aprende os gostos dos usuários a partir do histórico de compras e recomenda livros com base em similaridade de embeddings.

---

## 🧠 Como funciona

O sistema usa um **two-tower model** — duas redes neurais independentes que aprendem a representar usuários e livros como vetores (embeddings) em um mesmo espaço semântico. Usuários e livros que interagiram ficam próximos nesse espaço; os que nunca interagiram ficam distantes.

```
Torre do Usuário          Torre do Livro
   user_id ──►  Embedding ──► Dense ──► vetor [64d]
   book_id ──►  Embedding ──► Dense ──► vetor [64d]
                                  │
                             Dot Product
                                  │
                            Score [0, 1]
```

Após o treinamento, os vetores são salvos no **Neo4j** como propriedades dos nós. A recomendação em tempo real usa o índice vetorial nativo do Neo4j (HNSW) para buscar os livros mais próximos do usuário em milissegundos.

---

## 🏗️ Arquitetura

```
Cliente
  │
  ▼
API Gateway (FastAPI)
  │
  ├── PostgreSQL  ──  users, books, purchases
  │
  └── Neo4j       ──  (:User)-[:BOUGHT]->(:Book)
                       embeddings + vector index
```

**Pipeline de treinamento** (roda offline):

```
PostgreSQL ──► Workers 1/2/3 (paralelo) ──► TensorFlow Two-Tower
                                                      │
                                          user_embeddings.npy
                                          book_embeddings.npy
                                                      │
                                              Worker 4 (embedding_saver)
                                                      │
                                                   Neo4j
```

---

## 🚀 Rodando o projeto

### Pré-requisitos

- Docker e Docker Compose
- Python 3.11+
- Poetry

### 1. Clone e configure

```bash
git clone https://github.com/seu-usuario/lectio-recommender.git
cd lectio-recommender
cp .env.example .env
# edite o .env com suas senhas
```

### 2. Suba os bancos de dados

```bash
docker compose up db neo4j -d
```

### 3. Popule o banco com dados de seed

```bash
poetry install
poetry run python3 seed.py
```

### 4. Treine o modelo e salve os embeddings no Neo4j

```bash
# Treina o modelo TensorFlow e gera os embeddings
docker compose --profile training up trainer

# Salva os embeddings no Neo4j
docker compose --profile training up embedding_saver
```

### 5. Suba a API

```bash
poetry run uvicorn main:app --reload
```

Acesse a documentação interativa em: `http://localhost:8000/docs`

---

## 📡 Endpoints

| Método | Rota | Descrição |
|--------|------|-----------|
| `POST` | `/users/` | Cria um usuário |
| `GET` | `/users/{id}` | Busca um usuário |
| `POST` | `/books/` | Cria um livro |
| `POST` | `/purchases/` | Registra uma compra |
| `GET` | `/recommendations/{user_id}` | Retorna livros recomendados |

### Exemplo de resposta — recomendações

```json
{
  "user_id": 42,
  "total": 10,
  "books": [
    {
      "id": 137,
      "title": "Duna",
      "author": "Frank Herbert",
      "genre": "Ficção Científica",
      "price": 49.90,
      "rate": 4.8,
      "score": 0.9423
    }
  ]
}
```

---

## 🗂️ Estrutura do projeto

```
lectio-recommender/
├── main.py                   # FastAPI app
├── schemas.py                # Pydantic schemas
├── crud.py                   # Operações no banco
├── database.py               # Conexão PostgreSQL
├── neo4j_connection.py       # Conexão Neo4j
├── seed.py                   # Geração de dados fictícios
├── routers/
│   └── recommendations.py    # Endpoint de recomendação
├── trainer/
│   ├── train.py              # Modelo TensorFlow two-tower
│   └── Dockerfile
├── embedding_saver/
│   ├── worker4.py            # Salva embeddings no Neo4j
│   └── Dockerfile
├── embeddings/               # Gerado após o treinamento (gitignored)
├── docker-compose.yml
├── Dockerfile
├── pyproject.toml
└── .env.example
```

---

## 🛠️ Stack

| Camada | Tecnologia |
|--------|-----------|
| API | FastAPI + Uvicorn |
| ORM | SQLAlchemy 2.0 |
| Banco relacional | PostgreSQL 16 |
| Banco de grafos | Neo4j 5 |
| Machine Learning | TensorFlow 2.16 |
| Validação | Pydantic v2 |
| Containerização | Docker + Docker Compose |
| Gerenciamento de deps | Poetry |

---

## 📄 Licença

MIT