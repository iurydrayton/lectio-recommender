"""
worker4.py — lê os embeddings gerados pelo train.py e popula o Neo4j.

Dependências:
    pip install neo4j numpy sqlalchemy psycopg2-binary

Uso:
    DATABASE_URL=postgresql://... NEO4J_URL=bolt://localhost:7687 python worker4.py
    NEO4J_USER=neo4j NEO4J_PASSWORD=sua_senha python worker4.py
"""

import os
import numpy as np
from pathlib import Path
from neo4j import GraphDatabase
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session

# ---------------------------------------------------------------------------
# Configuração
# ---------------------------------------------------------------------------

DATABASE_URL   = os.getenv("DATABASE_URL",   "postgresql://postgres:senha123@localhost:5432/book_recommender")
NEO4J_URL      = os.getenv("NEO4J_URL",      "bolt://localhost:7687")
NEO4J_USER     = os.getenv("NEO4J_USER",     "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "senha123")
EMBEDDINGS_DIR = Path(os.getenv("EMBEDDINGS_DIR", "./embeddings"))

EMBEDDING_DIM  = 64
BATCH_SIZE     = 100   # nós por transação no Neo4j


# ---------------------------------------------------------------------------
# Leitura dos embeddings e dados do PostgreSQL
# ---------------------------------------------------------------------------

def load_embeddings() -> tuple[dict[int, list[float]], dict[int, list[float]]]:
    """Carrega os .npy gerados pelo train.py e converte para list[float]."""
    user_path = EMBEDDINGS_DIR / "user_embeddings.npy"
    book_path = EMBEDDINGS_DIR / "book_embeddings.npy"

    if not user_path.exists() or not book_path.exists():
        raise FileNotFoundError(
            f"Embeddings não encontrados em {EMBEDDINGS_DIR}. "
            "Execute o train.py antes do worker4.py."
        )

    # allow_pickle=True pois o .npy contém um dict Python
    user_embeddings: dict = np.load(user_path, allow_pickle=True).item()
    book_embeddings: dict = np.load(book_path, allow_pickle=True).item()

    # Converte np.ndarray → list[float] (Neo4j não aceita numpy arrays)
    user_embeddings = {uid: vec.tolist() for uid, vec in user_embeddings.items()}
    book_embeddings = {bid: vec.tolist() for bid, vec in book_embeddings.items()}

    return user_embeddings, book_embeddings


def fetch_users(db_url: str) -> list[dict]:
    """Busca dados completos dos usuários no PostgreSQL."""
    engine = create_engine(db_url, echo=False)
    with Session(engine) as session:
        rows = session.execute(text("""
            SELECT id, name, email, country, sex
            FROM users ORDER BY id
        """)).fetchall()
    engine.dispose()
    return [dict(r._mapping) for r in rows]


def fetch_books(db_url: str) -> list[dict]:
    """Busca dados completos dos livros no PostgreSQL."""
    engine = create_engine(db_url, echo=False)
    with Session(engine) as session:
        rows = session.execute(text("""
            SELECT id, title, author, genre, price, rate
            FROM books ORDER BY id
        """)).fetchall()
    engine.dispose()
    return [dict(r._mapping) for r in rows]


def fetch_purchases(db_url: str) -> list[dict]:
    """Busca todas as compras para criar arestas [:BOUGHT]."""
    engine = create_engine(db_url, echo=False)
    with Session(engine) as session:
        rows = session.execute(text("""
            SELECT user_id, book_id,
                   purchased_at::text AS purchased_at
            FROM purchases ORDER BY id
        """)).fetchall()
    engine.dispose()
    return [dict(r._mapping) for r in rows]


# ---------------------------------------------------------------------------
# Operações no Neo4j
# ---------------------------------------------------------------------------

def create_constraints(driver: GraphDatabase.driver):
    """Garante unicidade de :User(id) e :Book(id)."""
    with driver.session() as session:
        session.run("""
            CREATE CONSTRAINT user_id_unique IF NOT EXISTS
            FOR (u:User) REQUIRE u.id IS UNIQUE
        """)
        session.run("""
            CREATE CONSTRAINT book_id_unique IF NOT EXISTS
            FOR (b:Book) REQUIRE b.id IS UNIQUE
        """)
    print("   ✅ Constraints criadas")


def create_vector_index(driver: GraphDatabase.driver, dim: int):
    """Cria índice vetorial para busca ANN nos embeddings de livros."""
    with driver.session() as session:
        session.run(f"""
            CREATE VECTOR INDEX `book-embeddings` IF NOT EXISTS
            FOR (b:Book) ON (b.embedding)
            OPTIONS {{
                indexConfig: {{
                    `vector.dimensions`: {dim},
                    `vector.similarity_function`: 'cosine'
                }}
            }}
        """)
        session.run(f"""
            CREATE VECTOR INDEX `user-embeddings` IF NOT EXISTS
            FOR (u:User) ON (u.embedding)
            OPTIONS {{
                indexConfig: {{
                    `vector.dimensions`: {dim},
                    `vector.similarity_function`: 'cosine'
                }}
            }}
        """)
    print("   ✅ Índices vetoriais criados")


def upsert_users(
    driver: GraphDatabase.driver,
    users: list[dict],
    user_embeddings: dict[int, list[float]],
    batch_size: int = BATCH_SIZE,
):
    """
    Cria ou atualiza nós (:User) com embedding.
    Usa MERGE para idempotência — seguro rodar mais de uma vez.
    """
    total = 0
    for i in range(0, len(users), batch_size):
        batch = users[i : i + batch_size]
        params = [
            {
                "id":        u["id"],
                "name":      u["name"],
                "email":     u["email"],
                "country":   u["country"],
                "sex":       u["sex"],
                "embedding": user_embeddings.get(u["id"], []),
            }
            for u in batch
        ]
        with driver.session() as session:
            session.run("""
                UNWIND $rows AS row
                MERGE (u:User {id: row.id})
                SET u.name      = row.name,
                    u.email     = row.email,
                    u.country   = row.country,
                    u.sex       = row.sex,
                    u.embedding = row.embedding
            """, rows=params)
        total += len(batch)
        print(f"   👤 Usuários: {total}/{len(users)}", end="\r")
    print(f"   👤 Usuários: {total}/{len(users)} ✅")


def upsert_books(
    driver: GraphDatabase.driver,
    books: list[dict],
    book_embeddings: dict[int, list[float]],
    batch_size: int = BATCH_SIZE,
):
    """Cria ou atualiza nós (:Book) com embedding."""
    total = 0
    for i in range(0, len(books), batch_size):
        batch = books[i : i + batch_size]
        params = [
            {
                "id":        b["id"],
                "title":     b["title"],
                "author":    b["author"],
                "genre":     b["genre"] or "",
                "price":     float(b["price"] or 0),
                "rate":      float(b["rate"]  or 0),
                "embedding": book_embeddings.get(b["id"], []),
            }
            for b in batch
        ]
        with driver.session() as session:
            session.run("""
                UNWIND $rows AS row
                MERGE (b:Book {id: row.id})
                SET b.title     = row.title,
                    b.author    = row.author,
                    b.genre     = row.genre,
                    b.price     = row.price,
                    b.rate      = row.rate,
                    b.embedding = row.embedding
            """, rows=params)
        total += len(batch)
        print(f"   📚 Livros: {total}/{len(books)}", end="\r")
    print(f"   📚 Livros: {total}/{len(books)} ✅")


def upsert_purchases(
    driver: GraphDatabase.driver,
    purchases: list[dict],
    batch_size: int = BATCH_SIZE,
):
    """Cria arestas [:BOUGHT] entre (:User) e (:Book)."""
    total = 0
    for i in range(0, len(purchases), batch_size):
        batch = purchases[i : i + batch_size]
        with driver.session() as session:
            session.run("""
                UNWIND $rows AS row
                MATCH (u:User {id: row.user_id})
                MATCH (b:Book {id: row.book_id})
                MERGE (u)-[r:BOUGHT]->(b)
                SET r.purchased_at = row.purchased_at
            """, rows=batch)
        total += len(batch)
        print(f"   🛒 Compras: {total}/{len(purchases)}", end="\r")
    print(f"   🛒 Compras: {total}/{len(purchases)} ✅")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("📦 Carregando embeddings...")
    user_embeddings, book_embeddings = load_embeddings()
    print(f"   {len(user_embeddings)} usuários | {len(book_embeddings)} livros\n")

    print("🗄️  Buscando dados do PostgreSQL...")
    users     = fetch_users(DATABASE_URL)
    books     = fetch_books(DATABASE_URL)
    purchases = fetch_purchases(DATABASE_URL)
    print(f"   {len(users)} usuários | {len(books)} livros | {len(purchases)} compras\n")

    print("🔗 Conectando ao Neo4j...")
    driver = GraphDatabase.driver(NEO4J_URL, auth=(NEO4J_USER, NEO4J_PASSWORD))
    driver.verify_connectivity()
    print("   Conexão OK\n")

    print("⚙️  Criando constraints e índices...")
    create_constraints(driver)
    create_vector_index(driver, EMBEDDING_DIM)
    print()

    print("📤 Inserindo nós e arestas...")
    upsert_users(driver, users, user_embeddings)
    upsert_books(driver, books, book_embeddings)
    upsert_purchases(driver, purchases)

    driver.close()

    print("\n✅ Worker 4 concluído! Grafo populado no Neo4j.")
    print("   Acesse http://localhost:7474 para visualizar.")


if __name__ == "__main__":
    main()