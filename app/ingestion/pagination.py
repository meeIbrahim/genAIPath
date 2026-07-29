from __future__ import annotations

import re
from dataclasses import dataclass
from urllib.parse import urljoin, urlparse

from bs4 import BeautifulSoup

_NEXT_TEXT_RE = re.compile(r"^(next|›|»|more)$", re.IGNORECASE)
_NUMBERED_RE = re.compile(r"(\?page=|&page=|/page/)(\d+)")


@dataclass(frozen=True)
class PaginationPlan:
    mode: str  # "single" | "chain" | "template"
    next_url: str | None = None
    template: str | None = None
    start_page_number: int = 1


def _same_path_prefix(current_url: str, candidate_url: str) -> bool:
    current = urlparse(current_url)
    candidate = urlparse(candidate_url)
    if current.netloc != candidate.netloc:
        return False
    return candidate.path == current.path or candidate.path.startswith(current.path.rstrip("/") + "/")


def detect_next_url(html: str, current_url: str) -> str | None:
    soup = BeautifulSoup(html, "lxml")

    link_next = soup.find("link", rel="next")
    if link_next and link_next.get("href"):
        return urljoin(current_url, link_next["href"])

    for anchor in soup.find_all("a", href=True):
        rel = anchor.get("rel") or []
        text = anchor.get_text(strip=True)
        is_next_rel = "next" in rel
        is_next_text = bool(text) and bool(_NEXT_TEXT_RE.match(text))
        if not (is_next_rel or is_next_text):
            continue
        candidate = urljoin(current_url, anchor["href"])
        if _same_path_prefix(current_url, candidate):
            return candidate
    return None


def detect_numbered_template(html: str, current_url: str) -> str | None:
    soup = BeautifulSoup(html, "lxml")
    current_domain = urlparse(current_url).netloc
    candidates: dict[str, set[int]] = {}

    for anchor in soup.find_all("a", href=True):
        href = urljoin(current_url, anchor["href"])
        if urlparse(href).netloc != current_domain:
            continue
        match = _NUMBERED_RE.search(href)
        if not match:
            continue
        number = int(match.group(2))
        template = href[: match.start(2)] + "{n}" + href[match.end(2):]
        candidates.setdefault(template, set()).add(number)

    clusters = {template: nums for template, nums in candidates.items() if len(nums) >= 2}
    if not clusters:
        return None
    return max(clusters, key=lambda template: len(clusters[template]))


def render_template(template: str, page_number: int) -> str:
    return template.replace("{n}", str(page_number))


def detect_pagination(html: str, current_url: str) -> PaginationPlan:
    next_url = detect_next_url(html, current_url)
    if next_url:
        return PaginationPlan(mode="chain", next_url=next_url)

    template = detect_numbered_template(html, current_url)
    if template:
        current_match = _NUMBERED_RE.search(current_url)
        start_page_number = int(current_match.group(2)) + 1 if current_match else 2
        return PaginationPlan(mode="template", template=template, start_page_number=start_page_number)

    return PaginationPlan(mode="single")
