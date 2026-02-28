from __future__ import annotations

import os
import time

import httpx

from combat.adapters.base import CrawlerAdapter, CrawlResult


class GrubAdapter(CrawlerAdapter):
    """Adapter for Grub via its POST /api/markdown endpoint."""

    name = "Grub"

    def __init__(self) -> None:
        self.base_url = os.environ.get("GRUB_COMBAT_URL", "http://localhost:6792")
        self.customer_id = os.environ.get("GRUB_COMBAT_CUSTOMER_ID", "combat-bench")
        self._client: httpx.AsyncClient | None = None

    async def setup(self) -> None:
        self._client = httpx.AsyncClient(base_url=self.base_url, timeout=60)
        # Verify Grub is reachable
        resp = await self._client.get("/health")
        resp.raise_for_status()

    async def teardown(self) -> None:
        if self._client:
            await self._client.aclose()

    async def crawl_batch(self, urls: list[str], *, concurrency: int = 3, timeout: int = 45) -> list[CrawlResult]:
        """Use Grub's native /api/batch endpoint for fair batch comparison."""
        assert self._client is not None, "Call setup() first"
        import time as _time
        payload = {
            "urls": urls,
            "customer_id": self.customer_id,
            "concurrent": min(concurrency, 3),
            "options": {
                "javascript": True,
                "timeout": 30,
                "screenshot": False,
                "wait_after_load_ms": 0,
                "include_html": True,
            },
        }
        t0 = _time.perf_counter()
        try:
            resp = await self._client.post("/api/batch", json=payload, timeout=timeout * len(urls))
            elapsed = (_time.perf_counter() - t0) * 1000
            data = resp.json()
            results = []
            for item in data.get("results", []):
                cr = CrawlResult(
                    url=item.get("url", ""),
                    markdown=item.get("markdown") or "",
                    html=item.get("html") or "",
                    elapsed_ms=elapsed / max(len(urls), 1),
                    success=item.get("success", False),
                    error=item.get("error"),
                )
                cr.compute_quality_metrics()
                results.append(cr)
            # Fill missing URLs with errors
            result_urls = {r.url for r in results}
            for u in urls:
                if u not in result_urls:
                    results.append(CrawlResult(url=u, elapsed_ms=elapsed, error="missing from batch response"))
            return results
        except Exception as exc:
            elapsed = (_time.perf_counter() - t0) * 1000
            return [CrawlResult(url=u, elapsed_ms=elapsed, error=str(exc)) for u in urls]

    async def crawl_one(self, url: str, *, javascript: bool = True, timeout: int = 30) -> CrawlResult:
        assert self._client is not None, "Call setup() first"
        payload = {
            "url": url,
            "customer_id": self.customer_id,
            "options": {
                "javascript": javascript,
                "timeout": timeout,
                "screenshot": False,
                "wait_after_load_ms": 0,  # no dead wait — fair race
                "include_html": True,
            },
        }
        t0 = time.perf_counter()
        try:
            resp = await self._client.post("/api/crawl", json=payload)
            elapsed = (time.perf_counter() - t0) * 1000
            data = resp.json()

            # Extract server-side phase timings
            server_timings = data.get("timings_ms", {})

            result = CrawlResult(
                url=url,
                markdown=data.get("markdown") or "",
                html=data.get("html") or "",
                elapsed_ms=elapsed,
                success=data.get("success", False),
                error=data.get("error"),
                timings={
                    "client_total_ms": round(elapsed, 1),
                    **{k: v for k, v in server_timings.items()},
                },
            )
            result.compute_quality_metrics()
            return result
        except Exception as exc:
            elapsed = (time.perf_counter() - t0) * 1000
            return CrawlResult(url=url, elapsed_ms=elapsed, error=str(exc))
