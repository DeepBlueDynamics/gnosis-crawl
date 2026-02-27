from __future__ import annotations

import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class CrawlResult:
    url: str
    markdown: str = ""
    html: str = ""
    elapsed_ms: float = 0.0
    success: bool = False
    error: Optional[str] = None
    word_count: int = 0
    char_count: int = 0
    has_headings: bool = False
    has_links: bool = False
    content_ratio: float = 0.0
    # Per-phase timing breakdown (adapter-specific, may be empty)
    timings: dict = field(default_factory=dict)

    def compute_quality_metrics(self) -> None:
        """Derive quality metrics from markdown/html content."""
        self.word_count = len(self.markdown.split()) if self.markdown else 0
        self.char_count = len(self.markdown) if self.markdown else 0
        self.has_headings = bool(re.search(r"^#{1,6}\s", self.markdown, re.MULTILINE)) if self.markdown else False
        self.has_links = bool(re.search(r"\[.+?\]\(.+?\)", self.markdown)) if self.markdown else False
        if self.html and self.markdown:
            # Strip tags/scripts/styles to get visible text length.
            # Raw HTML bytes is an unfair denominator because some adapters
            # return the full page while others return pre-filtered content.
            visible = re.sub(r'<script[^>]*>.*?</script>', '', self.html, flags=re.DOTALL | re.IGNORECASE)
            visible = re.sub(r'<style[^>]*>.*?</style>', '', visible, flags=re.DOTALL | re.IGNORECASE)
            visible = re.sub(r'<[^>]+>', '', visible)
            visible_len = len(visible.strip())
            self.content_ratio = len(self.markdown) / visible_len if visible_len else 0.0
        else:
            self.content_ratio = 0.0


class CrawlerAdapter(ABC):
    """Base class all crawler adapters must implement."""

    name: str = "base"

    async def setup(self) -> None:
        """Initialise resources (browser, client, etc.)."""

    async def teardown(self) -> None:
        """Release resources."""

    @abstractmethod
    async def crawl_one(self, url: str, *, javascript: bool = True, timeout: int = 30) -> CrawlResult:
        """Crawl a single URL and return a CrawlResult."""
        ...

    async def crawl_batch(self, urls: list[str], *, concurrency: int = 3, timeout: int = 45) -> list[CrawlResult]:
        """Crawl multiple URLs with bounded concurrency.

        Default implementation uses asyncio.Semaphore around crawl_one.
        Each individual crawl is hard-capped at *timeout* seconds so one
        stalled page can't block the whole batch.
        """
        import asyncio

        sem = asyncio.Semaphore(concurrency)

        async def _bounded(u: str) -> CrawlResult:
            async with sem:
                try:
                    return await asyncio.wait_for(self.crawl_one(u), timeout=timeout)
                except asyncio.TimeoutError:
                    return CrawlResult(url=u, elapsed_ms=timeout * 1000, error="timeout")

        return await asyncio.gather(*[_bounded(u) for u in urls])
