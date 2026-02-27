"""Smoke test: run the actual batch test flow to find the hang."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import asyncio
import time
import os

os.environ["PYTHONIOENCODING"] = "utf-8"

URL = "https://example.com"
BATCH_URLS = [f"https://quotes.toscrape.com/page/{i}/" for i in range(1, 4)]


async def main():
    from combat.adapters.grub import GrubAdapter
    from combat.adapters.crawl4ai import Crawl4AIAdapter
    from combat.adapters.scrapy_adapter import ScrapyAdapter

    adapters = []
    for name, cls in [("Grub", GrubAdapter), ("Crawl4AI", Crawl4AIAdapter), ("Scrapy", ScrapyAdapter)]:
        a = cls()
        try:
            await asyncio.wait_for(a.setup(), timeout=30)
            adapters.append(a)
            print(f"[setup] {name}: ready", flush=True)
        except Exception as e:
            print(f"[setup] {name}: skip ({e})", flush=True)

    print(f"\n=== SINGLE URL TEST ===", flush=True)
    for a in adapters:
        print(f"  [{a.name}] starting crawl_one...", flush=True)
        t0 = time.perf_counter()
        try:
            r = await asyncio.wait_for(a.crawl_one(URL), timeout=30)
            print(f"  [{a.name}] done {(time.perf_counter()-t0)*1000:.0f}ms success={r.success}", flush=True)
        except asyncio.TimeoutError:
            print(f"  [{a.name}] TIMEOUT after 30s", flush=True)
        except Exception as e:
            print(f"  [{a.name}] ERROR: {e}", flush=True)

    print(f"\n=== BATCH TEST (3 URLs) ===", flush=True)
    for a in adapters:
        print(f"  [{a.name}] starting crawl_batch...", flush=True)
        t0 = time.perf_counter()
        try:
            results = await asyncio.wait_for(a.crawl_batch(BATCH_URLS, concurrency=3), timeout=90)
            ok = sum(1 for r in results if r.success)
            print(f"  [{a.name}] done {(time.perf_counter()-t0)*1000:.0f}ms {ok}/{len(results)} ok", flush=True)
        except asyncio.TimeoutError:
            print(f"  [{a.name}] TIMEOUT after 90s", flush=True)
        except Exception as e:
            print(f"  [{a.name}] ERROR: {e}", flush=True)

    print(f"\n=== TEARDOWN ===", flush=True)
    for a in adapters:
        print(f"  [{a.name}] teardown...", flush=True)
        try:
            await asyncio.wait_for(a.teardown(), timeout=15)
            print(f"  [{a.name}] done", flush=True)
        except asyncio.TimeoutError:
            print(f"  [{a.name}] TIMEOUT on teardown", flush=True)
        except Exception as e:
            print(f"  [{a.name}] ERROR: {e}", flush=True)

    print("\n--- finished ---", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
