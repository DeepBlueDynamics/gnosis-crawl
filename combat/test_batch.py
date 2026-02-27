"""Batch throughput benchmarks — how well does each crawler handle concurrency?"""
from __future__ import annotations

import time

import pytest


@pytest.mark.combat
@pytest.mark.asyncio(loop_scope="session")
@pytest.mark.parametrize("batch_size", [10, 25, 50])
async def test_batch_throughput(adapters, combat_results, batch_urls, batch_size):
    """Race adapters on a batch of URLs with concurrency=3.

    Metrics:
      - total_ms: wall-clock time for the full batch
      - per_url_ms: total_ms / batch_size
      - success_rate: fraction that succeeded
      - total_words: sum of word counts across successful results
    """
    urls = batch_urls[:batch_size]
    row: dict = {}

    for adapter in adapters:
        t0 = time.perf_counter()
        results = await adapter.crawl_batch(urls, concurrency=3)
        total_ms = (time.perf_counter() - t0) * 1000

        successes = [r for r in results if r.success]
        row[adapter.name] = {
            "total_ms": round(total_ms, 1),
            "per_url_ms": round(total_ms / batch_size, 1),
            "success_rate": round(len(successes) / len(results), 2) if results else 0,
            "total_words": sum(r.word_count for r in successes),
        }

    combat_results["batch"][str(batch_size)] = row

    parts = [f"{name}: {d['total_ms']:.0f}ms ({d['success_rate']:.0%})" for name, d in row.items()]
    print(f"  batch={batch_size}  ->  {' | '.join(parts)}")
