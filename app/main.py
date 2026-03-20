# main.py
from fastapi import FastAPI, Depends, HTTPException
from sqlalchemy.orm import Session
from contextlib import asynccontextmanager

import app.crud
from app.schemas import UserCreate, User, BookCreate, Book, PurchaseCreate, Purchase
from app.database import SessionLocal, engine, get_db, Base
from app.neo4j_connection import init_driver, close_driver
from app.recommendations import router as recommendations_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_driver()
    yield
    close_driver()


app = FastAPI(
    title="Lectio Recommender",
    description="API de recomendação de livros com TensorFlow + Neo4j",
    version="0.1.0",
    lifespan=lifespan,
)

Base.metadata.create_all(bind=engine)

# Routers
app.include_router(recommendations_router)


# ── Users ──────────────────────────────────────────────────────────────────

@app.post("/users/", response_model=User)
def create_user(user: UserCreate, db: Session = Depends(get_db)):
    db_user = crud.get_user_by_email(db, email=user.email)
    if db_user:
        raise HTTPException(status_code=400, detail="Email já cadastrado")
    return crud.create_user(db=db, user=user)


@app.get("/users/{user_id}", response_model=User)
def read_user(user_id: int, db: Session = Depends(get_db)):
    db_user = crud.get_user(db, user_id=user_id)
    if db_user is None:
        raise HTTPException(status_code=404, detail="Usuário não encontrado")
    return db_user


# ── Books ──────────────────────────────────────────────────────────────────

@app.post("/books/", response_model=Book)
def create_book(book: BookCreate, db: Session = Depends(get_db)):
    return crud.create_book(db=db, book=book)


# ── Purchases ──────────────────────────────────────────────────────────────

@app.post("/purchases/", response_model=Purchase)
def create_purchase(purchase: PurchaseCreate, db: Session = Depends(get_db)):
    return crud.create_purchase(db=db, purchase=purchase)