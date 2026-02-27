from __future__ import annotations

import os
import time

import httpx

from combat.adapters.base import CrawlerAdapter, CrawlResult


class FirecrawlAdapter(CrawlerAdapter):
    """Adapter for self-hosted Firecrawl (open-source).

    Firecrawl requires a full stack (API + Redis + Postgres) via their
    docker-compose.  This adapter connects to an existing instance or
    skips gracefully.

    To run Firecrawl locally:
        git clone https://github.com/firecrawl/firecrawl
        cd firecrawl && docker compose up -d
    Then set FIRECRAWL_URL=http://localhost:3002
    """

    name = "Firecrawl"

    def __init__(self) -> None:
        self.base_url = os.environ.get("FIRECRAWL_URL", "http://localhost:3002")
        self.api_key = os.environ.get("FIRECRAWL_API_KEY", "fc-combat-test")
        self._client: httpx.AsyncClient | None = None

    async def setup(self) -> None:
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=60,
            headers={"Authorization": f"Bearer {self.api_key}"},
        )
        try:
            resp = await self._client.get("/", timeout=5)
            if resp.status_code >= 500:
                raise ConnectionError(f"Firecrawl returned {resp.status_code}")
        except (httpx.ConnectError, httpx.ConnectTimeout) as exc:
            await self._client.aclose()
            self._client = None
            raise RuntimeError(
                f"No Firecrawl at {self.base_url}. "
                "Run their docker-compose stack first — see adapter docstring."
            ) from exc

    async def teardown(self) -> None:
        if self._client:
            await self._client.aclose()

    async def crawl_one(self, url: str, *, javascript: bool = True, timeout: int = 30) -> CrawlResult:
        assert self._client is not None, "Call setup() first"
        payload = {
            "url": url,
            "formats": ["markdown", "html"],
        }
        t0 = time.perf_counter()
        try:
            resp = await self._client.post("/v1/scrape", json=payload, timeout=timeout + 10)
            elapsed = (time.perf_counter() - t0) * 1000
            data = resp.json()
            content = data.get("data", {})
            cr = CrawlResult(
                url=url,
                markdown=content.get("markdown", ""),
                html=content.get("html", ""),
                elapsed_ms=elapsed,
                success=data.get("success", resp.is_success),
                error=data.get("error"),
            )
            cr.compute_quality_metrics()
            return cr
        except Exception as exc:
            elapsed = (time.perf_counter() - t0) * 1000
            return CrawlResult(url=url, elapsed_ms=elapsed, error=str(exc))
