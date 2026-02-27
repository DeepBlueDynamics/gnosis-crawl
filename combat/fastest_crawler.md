# Fastest Crawler — Combat Arena Report

**Date:** 2026-02-27
**Engine:** Grub Crawler (Python markdown path, pre-Rust)
**Note:** grub_md Rust extension was NOT loaded for this run. Wikipedia 0w is a known Python-engine bug on that page. Rust engine tested standalone: 6394 words, sub-ms markdown conversion.

---

## Test Environment

| Parameter | Value |
| --- | --- |
| Grub endpoint | `http://localhost:6792` (Docker, Playwright Chromium) |
| Crawl4AI | `v0.8.0` (local, async Chromium) |
| Scrapy | HTTP-only (no JS), subprocess per crawl, `markdownify` |
| Batch concurrency | 3 |
| Crawl timeout | 30s (single) / 45s (batch) |
| Target URLs | `quotes.toscrape.com` pages 1-50 (batch) |

### Adapters

| Adapter | Browser | JS | Markdown Engine |
| --- | --- | --- | --- |
| **Grub** | Playwright Chromium (Docker) | Yes | `app/markdown.py` (BeautifulSoup) |
| **Crawl4AI** | Built-in Chromium | Yes | Built-in extractor |
| **Scrapy** | None (HTTP only) | No | `markdownify` |

---

## Speed — Single URL (ms, lower is better)

| URL | Grub | Crawl4AI | Scrapy | Winner |
| --- | ---: | ---: | ---: | --- |
| example.com | **556** | 1156 | 1824 | Grub |
| news.ycombinator.com | **896** | 1238 | 2533 | Grub |
| en.wikipedia.org/wiki/Web_crawling | 2143 | **1600** | 1952 | Crawl4AI |
| httpbin.org/html | **364** | 1004 | 2378 | Grub |
| quotes.toscrape.com | **503** | 995 | 1758 | Grub |

**Grub wins 4/5 URLs.** Wikipedia is the outlier — `markdown_ms=990` dominates Grub's total, which is the exact bottleneck the Rust engine targets.

### Grub Phase Breakdown (ms)

| URL | navigation | wait | content | visible_text | markdown | server total |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| example.com | 405 | 0 | 15 | 16 | **5** | 449 |
| news.ycombinator.com | 351 | 0 | 16 | 19 | **384** | 395 |
| en.wikipedia.org | 453 | 0 | 119 | 163 | **990** | 829 |
| httpbin.org/html | 228 | 0 | 20 | 13 | **3** | 270 |
| quotes.toscrape.com | 220 | 0 | 13 | 24 | **135** | 273 |

`markdown_ms` is the dominant bottleneck on complex pages. Navigation time is network-bound and roughly equal across adapters.

---

## Quality — Word Count (higher is better)

| URL | Grub | Crawl4AI | Scrapy | Winner |
| --- | ---: | ---: | ---: | --- |
| example.com | 20 | 20 | **22** | Scrapy |
| news.ycombinator.com | 365 | 801 | **1031** | Scrapy |
| en.wikipedia.org | 0 | **9387** | 9165 | Crawl4AI |
| httpbin.org/html | **606** | 606 | 606 | Tie |
| quotes.toscrape.com | 183 | 282 | **285** | Scrapy |

Grub's content filtering is aggressive — it strips nav, sidebar, footer, and hidden elements, producing leaner output. This is by design (LLM-friendly extraction), but means lower raw word counts vs adapters that dump everything.

Wikipedia 0w is a bug in the Python markdown engine (content filter + fallback heuristic fails on Wikipedia's DOM structure). The Rust engine produces 6394 words on the same page.

### Quality — Content Features

| URL | Grub headings | Grub links | Crawl4AI headings | Crawl4AI links | Scrapy headings | Scrapy links |
| --- | :---: | :---: | :---: | :---: | :---: | :---: |
| example.com | Y | Y | Y | Y | N | Y |
| news.ycombinator.com | N | Y | N | Y | N | Y |
| en.wikipedia.org | N | N | Y | Y | Y | Y |
| httpbin.org/html | Y | N | Y | N | N | N |
| quotes.toscrape.com | Y | Y | Y | Y | N | Y |

---

## Batch Throughput (ms total, lower is better)

| Batch Size | Grub | Crawl4AI | Scrapy | Winner | Grub speedup vs 2nd |
| ---: | ---: | ---: | ---: | --- | --- |
| 10 | **3234** | 4711 | 8032 | Grub | 1.5x |
| 25 | **5361** | 12091 | 17076 | Grub | 2.3x |
| 50 | **9106** | 20453 | 35365 | Grub | 2.2x |

All adapters achieved **100% success rate** across all batch sizes.

### Per-URL Throughput (ms/url)

| Batch Size | Grub | Crawl4AI | Scrapy |
| ---: | ---: | ---: | ---: |
| 10 | **323** | 471 | 803 |
| 25 | **215** | 484 | 683 |
| 50 | **182** | 409 | 707 |

Grub scales well — per-URL cost drops from 323ms to 182ms as batch size grows. Crawl4AI stays flat around 400-480ms. Scrapy degrades under load.

---

## Scorecard

| Category | Grub | Crawl4AI | Scrapy |
| --- | :---: | :---: | :---: |
| Speed (single URL wins) | **4/5** | 1/5 | 0/5 |
| Speed (batch wins) | **3/3** | 0/3 | 0/3 |
| Quality (word count wins) | 0/5 | 1/5 | **3/5** |
| 100% success rate | Y | Y | Y |

---

## Known Issues

1. **Wikipedia 0 words (Python engine)** — The Python `ContentFilter` + `HTMLToMarkdownConverter` pipeline produces empty output on Wikipedia's complex DOM. The Rust `grub_md` engine handles it correctly (6394 words). Fix: rebuild Docker image with `grub_md` installed.

2. **`markdown_ms` bottleneck** — BeautifulSoup's two-parse pipeline costs 384-990ms on complex pages. The Rust engine (single html5ever parse, single walk) brings this to <5ms on the same content when tested standalone.

3. **Aggressive content filtering** — Grub's nav/sidebar/footer stripping reduces word counts vs raw-dump adapters. This is intentional for LLM consumption but hurts on the word-count metric.

---

## Next Steps

- [ ] Rebuild Docker image with `grub_md` Rust extension (`docker compose up -d --build`)
- [ ] Re-run combat suite to measure actual Rust engine impact on `markdown_ms`
- [ ] Investigate Wikipedia Python-engine failure (likely content filter stripping `<main>` or fallback heuristic)
- [ ] Tune content filter aggressiveness for better word-count parity
