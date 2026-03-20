# schemas.py
from pydantic import BaseModel
from datetime import date, datetime
from typing import Optional


class UserBase(BaseModel):
    email: str
    name: str
    birth_date: date
    sex: str
    country: str

class UserCreate(UserBase):
    pass


class User(UserBase):
    id: int
    created_at: datetime
    
    class Config:
        from_attributes = True  # permite converter de SQLAlchemy


class BookBase(BaseModel):
    title: str
    author: str
    isbn: Optional[str] = None
    genre: Optional[str] = None
    rate: Optional[float] = None
    price: Optional[float] = None
    country: Optional[str] = None
    publication_date: Optional[int] = None


class BookCreate(BookBase):
    pass


class Book(BookBase):
    id: int
    created_at: datetime

    class Config:
        from_attributes = True


class PurchaseCreate(BaseModel):
    user_id: int
    book_id: int


class Purchase(PurchaseCreate):
    id: int
    purchased_at: datetime

    class Config:
        from_attributes = True