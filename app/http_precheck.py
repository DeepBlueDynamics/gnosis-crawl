"""Lightweight HTTP pre-check with TLS fingerprint impersonation via curl_cffi."""

import logging
from dataclasses import dataclass, field
from typing import Optional

from app.config import settings

logger = logging.getLogger(__name__)

try:
    from curl_cffi.requests import AsyncSession
    _HAS_CURL_CFFI = True
except ImportError:
    _HAS_CURL_CFFI = False

# Markers that indicate the page needs a real browser
_BROWSER_NEEDED_MARKERS = [
    'cf-browser-verification',
    'cf-challenge-running',
    'challenge-platform',
    '_cf_chl',
    'managed-challenge',
    '<noscript>',
    'enable javascript',
    'browser check',
    'ddos-guard',
    'datadome',
    '_incapsula_resource',
    'incapsula_resource',
    'visid_incap_',
]


@dataclass
class PrecheckResult:
    """Result of an HTTP pre-check."""
    url: str = ""
    success: bool = False
    needs_browser: bool = True  # default to True (safe fallback)
    status_code: Optional[int] = None
    content: str = ""
    content_length: int = 0
    headers: dict = field(default_factory=dict)
    error: Optional[str] = None
    usable_content: Optional[str] = None  # markdown-converted content when browser not needed


def _check_needs_browser(status_code: Optional[int], content: str, content_length: int) -> bool:
    """Heuristic: does this response need a full browser to get real content?"""
    # HTTP errors that typically indicate challenge pages
    if status_code in (403, 503):
        return True

    # Very short responses are often challenge/redirect pages
    if content_length < 1024:
        return True

    # Check for known browser-verification markers
    content_lower = content[:5000].lower()  # only scan first 5KB
    for marker in _BROWSER_NEEDED_MARKERS:
        if marker in content_lower:
            return True

    return False


async def http_precheck(url: str, timeout: Optional[int] = None, impersonate: Optional[str] = None) -> PrecheckResult:
    """Perform a lightweight HTTP fetch with TLS fingerprint impersonation.

    Returns a PrecheckResult indicating whether the content is usable or
    if a full browser crawl is needed.
    """
    result = PrecheckResult(url=url)

    if not _HAS_CURL_CFFI:
        result.error = "curl_cffi not installed"
        return result

    if not settings.http_precheck_enabled:
        result.error = "precheck disabled"
        return result

    effective_timeout = timeout or settings.http_precheck_timeout
    effective_impersonate = impersonate or settings.http_precheck_impersonate

    try:
        async with AsyncSession(impersonate=effective_impersonate) as session:
            response = await session.get(
                url,
                timeout=effective_timeout,
                allow_redirects=True,
                headers={
                    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                    'Accept-Language': 'en-US,en;q=0.9',
                    'Referer': 'https://www.google.com/',
                },
            )

        result.status_code = response.status_code
        result.content = response.text or ""
        result.content_length = len(result.content)
        result.headers = dict(response.headers) if response.headers else {}
        result.needs_browser = _check_needs_browser(
            result.status_code, result.content, result.content_length
        )
        # When browser isn't needed, provide the content directly
        if not result.needs_browser and result.content_length > 1024:
            result.usable_content = result.content
        result.success = True

    except Exception as exc:
        logger.warning("HTTP precheck failed for %s: %s", url, exc)
        result.error = str(exc)
        result.needs_browser = True  # fail-safe: fall back to browser

    return result
