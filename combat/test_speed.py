"""Single-URL speed races — how fast can each crawler fetch + convert one page?"""
from __future__ import annotations

import asyncio

import pytest

from combat.adapters.base import CrawlResult
from combat.conftest import SPEED_URLS

CRAWL_TIMEOUT = 45  # seconds — hard cap per adapter per URL
INTER_ADAPTER_DELAY = 10  # seconds between adapter runs to avoid rate-limiting


@pytest.mark.combat
@pytest.mark.asyncio(loop_scope="session")
@pytest.mark.parametrize("url", SPEED_URLS, ids=[u.split("//")[1].split("/")[0] for u in SPEED_URLS])
async def test_single_url_speed(adapters, combat_results, url):
    """Race every available adapter on a single URL.

    No assertions — this is data-collection only.  Results are stored in
    the shared ``combat_results`` dict and written to ``results.json``
    after the session.

    Grub always runs first (baseline).  Remaining adapters run in shuffled
    order with a delay between each to avoid rate-limiting.
    """
    url_key = url.split("//")[1].rstrip("/")
    row: dict = {}

    for i, adapter in enumerate(adapters):
        if i > 0:
            await asyncio.sleep(INTER_ADAPTER_DELAY)
        try:
            result = await asyncio.wait_for(adapter.crawl_one(url), timeout=CRAWL_TIMEOUT)
        except asyncio.TimeoutError:
            result = CrawlResult(url=url, elapsed_ms=CRAWL_TIMEOUT * 1000, error="timeout")
        row[adapter.name] = {
            "elapsed_ms": round(result.elapsed_ms, 1),
            "success": result.success,
            "word_count": result.word_count,
            "char_count": result.char_count,
            "has_headings": result.has_headings,
            "has_links": result.has_links,
            "content_ratio": round(result.content_ratio, 3),
            "error": result.error,
            "timings": result.timings,
        }

    combat_results["speed"][url_key] = row

    # Summary line
    parts = [f"{name}: {d['elapsed_ms']:.0f}ms" for name, d in row.items()]
    print(f"  {url_key}  ->  {' | '.join(parts)}")

    # Phase breakdown for adapters that report it
    for name, d in row.items():
        t = d.get("timings")
        if t:
            phases = "  ".join(f"{k}={v}" for k, v in t.items() if k != "client_total_ms" and k != "total_ms")
            if phases:
                print(f"    {name}: {phases}")
