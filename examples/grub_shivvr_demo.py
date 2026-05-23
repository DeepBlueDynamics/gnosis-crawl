"""Grub + Shivvr — crawl pages, vector-search them in ~10 lines.

What this does:
  1. Uses grub (https://grub.nuts.services) to crawl URLs → markdown
  2. Pushes each markdown blob into a shivvr temp collection (ephemeral
     vector store, no setup required, no DB to manage)
  3. Runs a semantic search over the collection
  4. Cleans up the temp collection

Why temp instead of a persistent session: temp stores are zero-setup —
you pick a name, start ingesting, no auth needed for ingest/organize-search
when shivvr isn't configured with an OpenAI key (which is the default
deployment). Delete cleans up. Use a persistent session
(POST /sessions/:id/ingest) when you want the vectors to outlive the run.

Setup:
  export NUTS_AUTH_TOKEN=ahp_...   # get one at https://auth.nuts.services
  pip install requests
  python examples/grub_shivvr_demo.py
"""

import os
import requests

GRUB     = "https://grub.nuts.services"
SHIVVR   = "https://shivvr.nuts.services"
H        = {"Authorization": f"Bearer {os.environ['NUTS_AUTH_TOKEN']}"}
COLL     = "grub-hn-demo"

URLS = [
    "https://horace.io/brrr_intro.html",
    "https://www.reenigne.org/blog/80386-microcode-disassembled/",
    "https://www.dpolakovic.space/blogs/zork-part2",
]

# Crawl each URL through grub and ingest its markdown into a shivvr temp collection.
for url in URLS:
    md = requests.post(f"{GRUB}/api/markdown", json={"url": url}, headers=H, timeout=60).json()["markdown"]
    requests.post(f"{SHIVVR}/temp/{COLL}/ingest", json={"text": md, "source": url}, timeout=30)

# Semantic search — free, runs on local GTR-T5-base embeddings inside shivvr.
hits = requests.get(f"{SHIVVR}/temp/{COLL}/search",
                    params={"q": "how do gpu kernels stay fast", "n": 3},
                    timeout=15).json()
for h in hits["results"]:
    print(f"{h['score']:.3f}  {h['source']}\n  -> {h['text'][:120]}...\n")

# Clean up the temp collection (auth required for DELETE).
requests.delete(f"{SHIVVR}/temp/{COLL}", headers=H, timeout=15)
