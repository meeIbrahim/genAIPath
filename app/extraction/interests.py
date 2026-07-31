from __future__ import annotations

_INTEREST_KEYWORDS: dict[str, str] = {
    "hiking": "outdoors",
    "trek": "outdoors",
    "trekking": "outdoors",
    "camping": "outdoors",
    "food": "food",
    "restaurant": "food",
    "cuisine": "food",
    "museum": "culture",
    "history": "culture",
    "art": "culture",
    "shopping": "shopping",
    "nightlife": "nightlife",
}


def extract_interests(text: str) -> list[str]:
    lowered = text.lower()
    matched: list[str] = []
    for keyword, category in _INTEREST_KEYWORDS.items():
        if keyword in lowered and category not in matched:
            matched.append(category)
    return matched
