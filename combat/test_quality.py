"""Content quality comparison — who extracts the richest markdown?"""
from __future__ import annotations

import asyncio

import pytest

from combat.adapters.base import CrawlResult
from combat.conftest import QUALITY_URLS

CRAWL_TIMEOUT = 45


@pytest.mark.combat
@pytest.mark.asyncio(loop_scope="session")
@pytest.mark.parametrize("url", QUALITY_URLS, ids=[u.split("//")[1].split("/")[0] for u in QUALITY_URLS])
async def test_content_quality(adapters, combat_results, url):
    """Compare markdown richness across adapters for a single URL.

    Metrics collected per adapter:
      - word_count, char_count
      - has_headings, has_links
      - content_ratio  (markdown / html length)
    """
    url_key = url.split("//")[1].rstrip("/")
    row: dict = {}

    for adapter in adapters:
        try:
            result = await asyncio.wait_for(adapter.crawl_one(url), timeout=CRAWL_TIMEOUT)
        except asyncio.TimeoutError:
            result = CrawlResult(url=url, elapsed_ms=CRAWL_TIMEOUT * 1000, error="timeout")
        row[adapter.name] = {
            "word_count": result.word_count,
            "char_count": result.char_count,
            "has_headings": result.has_headings,
            "has_links": result.has_links,
            "content_ratio": round(result.content_ratio, 3),
            "success": result.success,
            "error": result.error,
        }

    combat_results["quality"][url_key] = row

    parts = [f"{name}: {d['word_count']}w" for name, d in row.items()]
    print(f"  {url_key}  ->  {' | '.join(parts)}")
