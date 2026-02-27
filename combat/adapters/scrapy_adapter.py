from __future__ import annotations

import asyncio
import json
import sys
import time
import textwrap

from combat.adapters.base import CrawlerAdapter, CrawlResult

# Scrapy's Twisted reactor can only start once per process.  We shell out
# to a tiny helper script in a subprocess so every crawl gets a fresh reactor.

_HELPER_SCRIPT = textwrap.dedent("""\
    import json, sys, scrapy
    from scrapy.crawler import CrawlerProcess

    url, timeout = sys.argv[1], int(sys.argv[2])
    collected = {}

    class S(scrapy.Spider):
        name = "c"
        start_urls = [url]
        custom_settings = {
            "DOWNLOAD_TIMEOUT": timeout,
            "LOG_ENABLED": False,
            "ROBOTSTXT_OBEY": False,
            "REQUEST_FINGERPRINTER_IMPLEMENTATION": "2.7",
        }
        def parse(self, response):
            collected["html"] = response.text
            collected["status"] = response.status

    p = CrawlerProcess({"LOG_ENABLED": False, "REQUEST_FINGERPRINTER_IMPLEMENTATION": "2.7"})
    p.crawl(S)
    p.start()
    json.dump(collected, sys.stdout)
""")


class ScrapyAdapter(CrawlerAdapter):
    """Adapter for Scrapy — HTTP-only (no JS rendering).

    Demonstrates the content gap when JavaScript rendering is absent.
    Each crawl runs in a subprocess to avoid Twisted reactor issues.
    """

    name = "Scrapy"

    async def setup(self) -> None:
        import scrapy  # noqa: F401 — verify importable
        import markdownify  # noqa: F401

    async def teardown(self) -> None:
        pass

    async def crawl_one(self, url: str, *, javascript: bool = True, timeout: int = 30) -> CrawlResult:
        from markdownify import markdownify as md

        t0 = time.perf_counter()
        try:
            proc = await asyncio.create_subprocess_exec(
                sys.executable, "-c", _HELPER_SCRIPT, url, str(timeout),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout + 10)
            elapsed = (time.perf_counter() - t0) * 1000

            if proc.returncode != 0:
                return CrawlResult(url=url, elapsed_ms=elapsed, error=stderr.decode(errors="replace")[:500])

            collected = json.loads(stdout.decode())
            html = collected.get("html", "")
            markdown = md(html) if html else ""
            cr = CrawlResult(
                url=url,
                markdown=markdown,
                html=html,
                elapsed_ms=elapsed,
                success=bool(html),
            )
            cr.compute_quality_metrics()
            return cr
        except Exception as exc:
            elapsed = (time.perf_counter() - t0) * 1000
            return CrawlResult(url=url, elapsed_ms=elapsed, error=str(exc))
