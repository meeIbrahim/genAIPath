import httpx
import pytest

from app.config import Settings
from app.ingestion.fetcher import FetchError, fetch_page

SETTINGS = Settings()


def _client(handler) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


async def test_fetch_page_returns_body_on_200():
    def handler(request):
        assert request.headers["user-agent"] == SETTINGS.user_agent
        return httpx.Response(200, text="<html>ok</html>")

    async with _client(handler) as client:
        body = await fetch_page(client, "https://example.com", SETTINGS)
    assert body == "<html>ok</html>"


async def test_fetch_page_raises_on_non_200():
    def handler(request):
        return httpx.Response(404, text="not found")

    async with _client(handler) as client:
        with pytest.raises(FetchError, match="404"):
            await fetch_page(client, "https://example.com", SETTINGS)


async def test_fetch_page_raises_on_timeout():
    def handler(request):
        raise httpx.TimeoutException("timed out")

    async with _client(handler) as client:
        with pytest.raises(FetchError, match="timeout"):
            await fetch_page(client, "https://example.com", SETTINGS)
