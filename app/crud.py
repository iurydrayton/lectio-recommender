# crud.py
from sqlalchemy.orm import Session
import app.models
import app.schemas


def get_user(db: Session, user_id: int):
    return db.query(app.models.User).filter(app.models.User.id == user_id).first()


def get_user_by_email(db: Session, email: str):
    return db.query(app.models.User).filter(app.models.User.email == email).first()


def create_user(db: Session, user: app.schemas.UserCreate):
    db_user = app.models.User(email=user.email, name=user.name)
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user


def create_book(db: Session, book: app.schemas.BookCreate):
    db_book = app.models.Book(**book.dict())
    db.add(db_book)
    db.commit()
    db.refresh(db_book)
    return db_book


def create_purchase(db: Session, purchase: app.schemas.PurchaseCreate):
    db_purchase = app.models.Purchase(**purchase.dict())
    db.add(db_purchase)
    db.commit()
    db.refresh(db_purchase)
    return db_purchase