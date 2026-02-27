from __future__ import annotations

import time

from combat.adapters.base import CrawlerAdapter, CrawlResult


class Crawl4AIAdapter(CrawlerAdapter):
    """Adapter for Crawl4AI (pip install crawl4ai)."""

    name = "Crawl4AI"

    def __init__(self) -> None:
        self._crawler = None

    async def setup(self) -> None:
        from crawl4ai import AsyncWebCrawler

        self._crawler = AsyncWebCrawler()
        await self._crawler.__aenter__()

    async def teardown(self) -> None:
        if self._crawler:
            await self._crawler.__aexit__(None, None, None)

    async def crawl_one(self, url: str, *, javascript: bool = True, timeout: int = 30) -> CrawlResult:
        assert self._crawler is not None, "Call setup() first"
        t0 = time.perf_counter()
        try:
            result = await self._crawler.arun(url=url)
            elapsed = (time.perf_counter() - t0) * 1000
            cr = CrawlResult(
                url=url,
                markdown=getattr(result, "markdown", "") or "",
                html=getattr(result, "html", "") or "",
                elapsed_ms=elapsed,
                success=bool(getattr(result, "success", True)),
                error=getattr(result, "error_message", None),
            )
            cr.compute_quality_metrics()
            return cr
        except Exception as exc:
            elapsed = (time.perf_counter() - t0) * 1000
            return CrawlResult(url=url, elapsed_ms=elapsed, error=str(exc))
