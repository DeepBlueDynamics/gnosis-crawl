from __future__ import annotations

import json
import pathlib
import random

import pytest
import pytest_asyncio

# ---------------------------------------------------------------------------
# URL lists
# ---------------------------------------------------------------------------

SPEED_URLS = [
    "https://example.com",
    "https://news.ycombinator.com",
    "https://en.wikipedia.org/wiki/Web_crawling",
    "https://httpbin.org/html",
    "https://quotes.toscrape.com",
]

QUALITY_URLS = SPEED_URLS  # same set, different metrics

BATCH_URLS = [f"https://quotes.toscrape.com/page/{i}/" for i in range(1, 51)]

# ---------------------------------------------------------------------------
# Shared results collector
# ---------------------------------------------------------------------------

_combat_results: dict = {"speed": {}, "quality": {}, "batch": {}}

RESULTS_PATH = pathlib.Path(__file__).parent / "results.json"


def pytest_configure(config):
    config.addinivalue_line("markers", "combat: combat suite benchmarks")


def pytest_sessionfinish(session, exitstatus):
    """Dump collected results to JSON after the run."""
    if _combat_results["speed"] or _combat_results["quality"] or _combat_results["batch"]:
        RESULTS_PATH.write_text(json.dumps(_combat_results, indent=2, default=str))


@pytest.fixture(scope="session")
def combat_results():
    """Mutable dict shared across all combat tests for report generation."""
    return _combat_results


# ---------------------------------------------------------------------------
# Adapter fixtures
# ---------------------------------------------------------------------------

@pytest_asyncio.fixture(loop_scope="session", scope="session")
async def adapters():
    """Initialise all available adapters, skip unavailable ones."""
    from combat.adapters.grub import GrubAdapter
    from combat.adapters.crawl4ai import Crawl4AIAdapter
    from combat.adapters.firecrawl import FirecrawlAdapter
    from combat.adapters.scrapy_adapter import ScrapyAdapter

    candidates = [
        GrubAdapter(),
        Crawl4AIAdapter(),
        FirecrawlAdapter(),
        ScrapyAdapter(),
    ]
    available: list = []
    for adapter in candidates:
        try:
            await adapter.setup()
            available.append(adapter)
            print(f"  [combat] {adapter.name}: ready")
        except Exception as exc:
            print(f"  [combat] {adapter.name}: skipped ({exc})")

    if not available:
        pytest.skip("No crawl adapters available")

    # Grub always runs first (baseline), shuffle the rest to avoid
    # order-dependent rate-limiting advantages.
    grub = [a for a in available if a.name == "Grub"]
    others = [a for a in available if a.name != "Grub"]
    random.shuffle(others)
    available = grub + others

    yield available

    for adapter in available:
        try:
            await adapter.teardown()
        except Exception:
            pass


@pytest.fixture(scope="session")
def batch_urls():
    return list(BATCH_URLS)
