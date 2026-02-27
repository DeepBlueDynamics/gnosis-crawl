from __future__ import annotations

import time

from combat.adapters.base import CrawlerAdapter, CrawlResult


class PlaywrightRawAdapter(CrawlerAdapter):
    """Baseline adapter using raw Playwright — no extraction pipeline.

    Shows what Grub adds on top of bare browser automation.
    """

    name = "PW Raw"

    def __init__(self) -> None:
        self._pw = None
        self._browser = None

    async def setup(self) -> None:
        from playwright.async_api import async_playwright

        self._pw = await async_playwright().start()
        self._browser = await self._pw.chromium.launch(headless=True)

    async def teardown(self) -> None:
        if self._browser:
            await self._browser.close()
        if self._pw:
            await self._pw.stop()

    async def crawl_one(self, url: str, *, javascript: bool = True, timeout: int = 30) -> CrawlResult:
        assert self._browser is not None, "Call setup() first"
        from markdownify import markdownify as md

        page = None
        t0 = time.perf_counter()
        try:
            page = await self._browser.new_page()
            t_nav_start = time.perf_counter()
            await page.goto(url, wait_until="domcontentloaded", timeout=timeout * 1000)
            t_nav_end = time.perf_counter()

            html = await page.content()
            t_content = time.perf_counter()

            markdown = md(html) if html else ""
            t_md = time.perf_counter()

            elapsed = (t_md - t0) * 1000
            cr = CrawlResult(
                url=url,
                markdown=markdown,
                html=html,
                elapsed_ms=elapsed,
                success=bool(html),
                timings={
                    "navigation_ms": round((t_nav_end - t_nav_start) * 1000, 1),
                    "content_ms": round((t_content - t_nav_end) * 1000, 1),
                    "markdown_ms": round((t_md - t_content) * 1000, 1),
                    "total_ms": round(elapsed, 1),
                },
            )
            cr.compute_quality_metrics()
            return cr
        except Exception as exc:
            elapsed = (time.perf_counter() - t0) * 1000
            return CrawlResult(url=url, elapsed_ms=elapsed, error=str(exc))
        finally:
            if page:
                await page.close()
