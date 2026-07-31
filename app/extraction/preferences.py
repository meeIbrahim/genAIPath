from __future__ import annotations

from pydantic import BaseModel, Field

from app.extraction.interests import extract_interests
from app.extraction.location import extract_city
from app.extraction.price import extract_price


class QueryPreferences(BaseModel):
    city: str | None = None
    budget: float | None = None
    interests: list[str] = Field(default_factory=list)


def extract_preferences(query: str) -> QueryPreferences:
    return QueryPreferences(
        city=extract_city(query),
        budget=extract_price(query),
        interests=extract_interests(query),
    )
