"""Warm-up navigation: Google search -> click -> target URL.

Navigates to Google, searches for the target, and clicks through if a
matching link is found. This establishes a natural referrer chain and
picks up Google cookies, making subsequent navigation to review sites
appear more organic to Cloudflare bot detection.
"""

import asyncio
import logging
import random
import re
import urllib.parse
from typing import Optional
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

# Platform domains for warm-up queries
PLATFORM_DOMAINS = {
    "trustpilot": "trustpilot.com",
    "g2": "g2.com",
    "capterra": "capterra.com",
    "trustradius": "trustradius.com",
}


def build_warmup_query(competitor_name: str, platform: str) -> str:
    """Build a natural-looking search query for warm-up navigation."""
    domain = PLATFORM_DOMAINS.get(platform, "")
    if domain:
        return f'"{competitor_name}" reviews site:{domain}'
    return f'"{competitor_name}" reviews'


def build_warmup_query_for_url(url: str) -> str:
    """Derive a natural-looking Google search query from any URL."""
    parsed = urlparse(url)
    domain = parsed.netloc.replace("www.", "")
    path_parts = [p for p in parsed.path.split("/") if p and not p.isdigit()]
    if path_parts:
        terms = " ".join(path_parts[-2:]).replace("-", " ").replace("_", " ")
        return f"{terms} {domain}"
    return domain


async def warmup_via_homepage(
    page,
    target_url: str,
    timeout_ms: int = 15000,
    settle_ms: float = 3.0,
) -> bool:
    """Navigate to the target site's homepage first to establish a session.

    Visiting the homepage lets the site's bot-detection JS (Akamai, Cloudflare, etc.)
    run and set its cookies. Subsequent navigation to the product page within the same
    context sees a valid session instead of a cold bot fingerprint.

    Returns True always (even on failure) so the caller proceeds with the direct goto
    using whatever session state was established.
    """
    try:
        parsed = urlparse(target_url)
        homepage = f"{parsed.scheme}://{parsed.netloc}/"
        if homepage == target_url.rstrip("/") + "/":
            return False  # Already targeting the homepage — skip

        logger.info(f"Warmup: visiting {homepage} to establish session before product page")
        await page.goto(homepage, timeout=timeout_ms, wait_until="domcontentloaded")
        await asyncio.sleep(random.uniform(settle_ms * 0.8, settle_ms * 1.2))
        return True
    except Exception as e:
        logger.debug(f"Warmup homepage visit failed: {e}")
        return False


async def warmup_via_google(
    page,
    target_url: str,
    search_query: str,
    timeout_ms: int = 12000,
) -> bool:
    """Navigate to Google, search for target, click through if found.

    Returns True if warm-up succeeded (clicked through to target domain), False if
    skipped/failed. Falls back gracefully -- caller should proceed
    with direct navigation on failure.
    """
    try:
        encoded_query = urllib.parse.quote(search_query)
        google_url = f"https://www.google.com/search?q={encoded_query}"

        await page.goto(google_url, timeout=timeout_ms, wait_until="domcontentloaded")
        await asyncio.sleep(random.uniform(1.0, 2.5))

        # Extract domain from target URL for matching
        domain_match = re.search(r"//([^/]+)", target_url)
        if not domain_match:
            return False
        domain = domain_match.group(1).replace("www.", "")

        # Google wraps result hrefs as /url?q=https://domain/... — match both forms
        links = await page.query_selector_all(f'a[href*="{domain}"]')

        if links:
            await links[0].click()
            try:
                await page.wait_for_load_state("domcontentloaded", timeout=timeout_ms)
            except Exception:
                pass
            # Verify we actually landed on the target domain, not a Google redirect.
            # Check hostname only — domain can appear in query strings of Google redirects.
            current = page.url
            current_host = urlparse(current).netloc.replace("www.", "")
            if domain in current_host:
                return True
            logger.debug(f"Warmup click landed on {current!r}, not target domain — will navigate directly with Google context")
            return False

        return False
    except Exception as e:
        logger.debug(f"Warm-up navigation failed: {e}")
        return False
