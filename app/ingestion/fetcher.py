from __future__ import annotations

import httpx

from app.config import Settings


class FetchError(Exception):
    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(reason)


async def fetch_page(client: httpx.AsyncClient, url: str, settings: Settings) -> str:
    try:
        response = await client.get(
            url,
            timeout=settings.fetch_timeout_seconds,
            headers={"User-Agent": settings.user_agent},
            follow_redirects=True,
        )
    except httpx.TimeoutException as exc:
        raise FetchError(f"timeout fetching {url}") from exc
    except httpx.RequestError as exc:
        raise FetchError(f"request failed for {url}: {exc}") from exc

    if response.status_code != 200:
        raise FetchError(f"non-200 status {response.status_code} for {url}")
    return response.text
