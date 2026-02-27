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
