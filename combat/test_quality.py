"""Content quality comparison - who extracts the richest markdown?

Reuses crawl results from the speed test to avoid double-hitting URLs
(Wikipedia and others rate-limit rapid sequential requests).
"""
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

    If the speed test already crawled this URL, reuse that data to
    extract quality metrics (avoids rate-limiting on repeated hits).
    Otherwise crawl fresh.
    """
    url_key = url.split("//")[1].rstrip("/")
    row: dict = {}

    # Check if speed test already has results for this URL
    speed_row = combat_results.get("speed", {}).get(url_key)

    crawl_count = 0
    for adapter in adapters:
        # Try to reuse the speed test result if it was successful
        speed_data = speed_row.get(adapter.name, {}) if speed_row else {}
        if speed_data.get("success") and speed_data.get("word_count", 0) > 0:
            # Reuse - the speed test already collected word_count
            row[adapter.name] = {
                "word_count": speed_data.get("word_count", 0),
                "char_count": speed_data.get("char_count", 0),
                "has_headings": speed_data.get("has_headings", False),
                "has_links": speed_data.get("has_links", False),
                "content_ratio": speed_data.get("content_ratio", 0.0),
                "success": True,
                "error": None,
            }
        else:
            # No speed data or it failed - crawl fresh
            if crawl_count > 0:
                await asyncio.sleep(10)
            crawl_count += 1
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
