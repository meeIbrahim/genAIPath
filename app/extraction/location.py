from __future__ import annotations

import spacy

_nlp = spacy.load("en_core_web_sm")


def extract_city(text: str) -> str | None:
    doc = _nlp(text)
    for ent in doc.ents:
        if ent.label_ == "GPE":
            return ent.text.lower()
    return None
