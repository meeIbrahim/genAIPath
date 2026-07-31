from __future__ import annotations

import re

_PRICE_RE = re.compile(
    r"\$\s?(?P<dollar>\d[\d,]*\.?\d*)(?P<dollar_k>[kK])?"
    r"|(?:under|less than|budget of|budget is|around|for)\s+\$?\s?(?P<qualified>\d[\d,]*\.?\d*)(?P<qualified_k>[kK])?"
    r"|(?P<bare_k>\d[\d,]*\.?\d*)\s?[kK]\b",
    re.IGNORECASE,
)


def extract_price(text: str) -> float | None:
    match = _PRICE_RE.search(text)
    if not match:
        return None

    groups = match.groupdict()
    if groups["dollar"] is not None:
        raw, is_k = groups["dollar"], groups["dollar_k"]
    elif groups["qualified"] is not None:
        raw, is_k = groups["qualified"], groups["qualified_k"]
    else:
        raw, is_k = groups["bare_k"], "k"

    value = float(raw.replace(",", ""))
    return value * 1000 if is_k else value
