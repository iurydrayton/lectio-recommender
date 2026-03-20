"""
seed.py — popula o PostgreSQL com dados fictícios para o sistema de recomendação.

Dependências:
    pip install faker sqlalchemy psycopg2-binary

Uso:
    DATABASE_URL=postgresql://user:pass@localhost:5432/dbname python seed.py
"""

import os
import random
from datetime import datetime, timezone
from faker import Faker
from sqlalchemy import create_engine, Column, Integer, String, Float, Date, DateTime, ForeignKey, text
from sqlalchemy.orm import declarative_base, Session

# ---------------------------------------------------------------------------
# Configuração
# ---------------------------------------------------------------------------

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:senha123@localhost:5432/book_recommender")

N_USERS     = 100
N_BOOKS     = 500
N_PURCHASES = 1500   # ~15 compras por usuário em média (com duplicatas removidas)

fake = Faker("pt_BR")
Faker.seed(42)
random.seed(42)

# ---------------------------------------------------------------------------
# Models (espelho do seu schemas.py / models.py)
# ---------------------------------------------------------------------------

Base = declarative_base()


class User(Base):
    __tablename__ = "users"

    id         = Column(Integer, primary_key=True, autoincrement=True)
    email      = Column(String, unique=True, nullable=False)
    name       = Column(String, nullable=False)
    birth_date = Column(Date, nullable=False)
    sex        = Column(String, nullable=False)
    country    = Column(String, nullable=False)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))


class Book(Base):
    __tablename__ = "books"

    id               = Column(Integer, primary_key=True, autoincrement=True)
    title            = Column(String, nullable=False)
    author           = Column(String, nullable=False)
    isbn             = Column(String, nullable=True)
    genre            = Column(String, nullable=True)
    rate             = Column(Float, nullable=True)
    price            = Column(Float, nullable=True)
    country          = Column(String, nullable=True)
    publication_date = Column(Integer, nullable=True)
    created_at       = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))


class Purchase(Base):
    __tablename__ = "purchases"

    id           = Column(Integer, primary_key=True, autoincrement=True)
    user_id      = Column(Integer, ForeignKey("users.id"), nullable=False)
    book_id      = Column(Integer, ForeignKey("books.id"), nullable=False)
    purchased_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))


# ---------------------------------------------------------------------------
# Dados de apoio
# ---------------------------------------------------------------------------

GENRES = [
    "Ficção Científica", "Fantasia", "Romance", "Thriller", "Terror",
    "Biografia", "História", "Autoajuda", "Filosofia", "Tecnologia",
    "Economia", "Poesia", "Infantil", "HQ", "Culinária",
]

COUNTRIES = ["BR", "PT", "US", "UK", "FR", "DE", "JP"]

SEXES = ["M", "F"]


def make_user(index: int) -> dict:
    sex = random.choice(SEXES)
    return dict(
        email      = fake.unique.email(),
        name       = fake.name(),
        birth_date = fake.date_of_birth(minimum_age=16, maximum_age=80),
        sex        = sex,
        country    = random.choice(COUNTRIES),
        created_at = fake.date_time_between(start_date="-2y", end_date="now", tzinfo=timezone.utc),
    )


def make_book(index: int) -> dict:
    pub_year = random.randint(1950, 2026)
    return dict(
        title            = fake.catch_phrase().title(),
        author           = fake.name(),
        isbn             = fake.isbn13(separator="-"),
        genre            = random.choice(GENRES),
        rate             = round(random.uniform(1.0, 5.0), 1),
        price            = round(random.uniform(9.90, 149.90), 2),
        country          = random.choice(COUNTRIES),
        publication_date = pub_year,
        created_at       = fake.date_time_between(start_date="-2y", end_date="now", tzinfo=timezone.utc),
    )


# ---------------------------------------------------------------------------
# Seed
# ---------------------------------------------------------------------------

def run_seed():
    engine = create_engine(DATABASE_URL, echo=False)

    # Cria as tabelas se não existirem
    Base.metadata.create_all(engine)

    with Session(engine) as session:
        # Limpa dados antigos (ordem respeitando FK)
        print("🗑️  Limpando dados antigos...")
        session.execute(text("TRUNCATE purchases, books, users RESTART IDENTITY CASCADE"))
        session.commit()

        # ----- Usuários -----
        print(f"👤 Inserindo {N_USERS} usuários...")
        users = [User(**make_user(i)) for i in range(N_USERS)]
        session.add_all(users)
        session.flush()   # garante que os IDs sejam gerados
        user_ids = [u.id for u in users]

        # ----- Livros -----
        print(f"📚 Inserindo {N_BOOKS} livros...")
        books = [Book(**make_book(i)) for i in range(N_BOOKS)]
        session.add_all(books)
        session.flush()
        book_ids = [b.id for b in books]

        # ----- Compras -----
        print(f"🛒 Inserindo até {N_PURCHASES} compras únicas...")
        pairs_seen: set[tuple[int, int]] = set()
        purchases = []

        attempts = 0
        while len(purchases) < N_PURCHASES and attempts < N_PURCHASES * 5:
            attempts += 1
            uid = random.choice(user_ids)
            bid = random.choice(book_ids)
            if (uid, bid) in pairs_seen:
                continue
            pairs_seen.add((uid, bid))
            purchases.append(Purchase(
                user_id      = uid,
                book_id      = bid,
                purchased_at = fake.date_time_between(start_date="-2y", end_date="now", tzinfo=timezone.utc),
            ))

        session.add_all(purchases)
        session.commit()

    print(f"\n✅ Seed concluído!")
    print(f"   Usuários : {N_USERS}")
    print(f"   Livros   : {N_BOOKS}")
    print(f"   Compras  : {len(purchases)}")


if __name__ == "__main__":
    run_seed()