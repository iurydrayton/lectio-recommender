"""
routers/recommendations.py — endpoint de recomendação de livros por embedding.
"""

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from sqlalchemy import text
from neo4j import Driver
from pydantic import BaseModel
from typing import Optional
from datetime import datetime

from app.database import get_db
from app.neo4j_connection import get_driver

router = APIRouter(prefix="/recommendations", tags=["recommendations"])

# ---------------------------------------------------------------------------
# Schemas de response
# ---------------------------------------------------------------------------

class RecommendedBook(BaseModel):
    id:               int
    title:            str
    author:           str
    genre:            Optional[str]      = None
    price:            Optional[float]    = None
    rate:             Optional[float]    = None
    publication_date: Optional[datetime] = None
    score:            float

    class Config:
        from_attributes = True


class RecommendationResponse(BaseModel):
    user_id: int
    total:   int
    books:   list[RecommendedBook]


# ---------------------------------------------------------------------------
# Queries Cypher
# ---------------------------------------------------------------------------

CYPHER_USER_EXISTS = """
MATCH (u:User {id: $user_id})
RETURN u.id AS id
"""

CYPHER_RECOMMEND = """
MATCH (u:User {id: $user_id})
CALL db.index.vector.queryNodes('book-embeddings', $candidates, u.embedding)
YIELD node AS b, score
WHERE NOT (u)-[:BOUGHT]->(b)
RETURN b.id AS book_id, score
ORDER BY score DESC
LIMIT $limit
"""

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def user_exists_in_graph(driver: Driver, user_id: int) -> bool:
    with driver.session() as session:
        result = session.run(CYPHER_USER_EXISTS, user_id=user_id)
        return result.single() is not None


def get_recommendations_from_graph(
    driver: Driver,
    user_id: int,
    limit: int,
    candidates: int,
) -> list[dict]:
    with driver.session() as session:
        result = session.run(
            CYPHER_RECOMMEND,
            user_id=user_id,
            limit=limit,
            candidates=candidates,
        )
        return [{"book_id": r["book_id"], "score": r["score"]} for r in result]


def fetch_books_by_ids(db: Session, book_ids: list[int]) -> dict[int, dict]:
    if not book_ids:
        return {}
    rows = db.execute(
        text("""
            SELECT id, title, author, genre, price, rate, publication_date
            FROM books
            WHERE id = ANY(:ids)
        """),
        {"ids": book_ids},
    ).fetchall()
    return {r.id: dict(r._mapping) for r in rows}


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------

@router.get("/{user_id}", response_model=RecommendationResponse)
def recommend_books(
    user_id:    int,
    limit:      int = Query(default=10, ge=1,  le=50,  description="Livros a retornar"),
    candidates: int = Query(default=50, ge=10, le=200, description="Candidatos antes de filtrar comprados"),
    driver:     Driver  = Depends(get_driver),
    db:         Session = Depends(get_db),
):
    """
    Retorna livros recomendados para o usuário com base em similaridade de embedding.
    Filtra automaticamente os livros que o usuário já comprou.
    """
    if not user_exists_in_graph(driver, user_id):
        raise HTTPException(
            status_code=404,
            detail=f"Usuário {user_id} não encontrado no grafo. Execute o pipeline de treinamento primeiro.",
        )

    recommendations = get_recommendations_from_graph(driver, user_id, limit, candidates)

    if not recommendations:
        return RecommendationResponse(user_id=user_id, total=0, books=[])

    book_ids    = [r["book_id"] for r in recommendations]
    books_by_id = fetch_books_by_ids(db, book_ids)

    books = [
        RecommendedBook(**books_by_id[r["book_id"]], score=round(r["score"], 4))
        for r in recommendations
        if r["book_id"] in books_by_id
    ]

    return RecommendationResponse(user_id=user_id, total=len(books), books=books)