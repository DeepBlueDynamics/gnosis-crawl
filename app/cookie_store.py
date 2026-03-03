"""Per-domain cookie persistence for Cloudflare clearance reuse."""

import time
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# Cookie names worth persisting (Cloudflare anti-bot tokens)
_CF_COOKIE_NAMES = {"__cf_bm", "cf_clearance", "__cflb"}


@dataclass
class StoredCookie:
    name: str
    value: str
    domain: str
    path: str = "/"
    stored_at: float = field(default_factory=time.time)
    ttl_seconds: float = 1500  # 25 minutes

    @property
    def is_expired(self) -> bool:
        return (time.time() - self.stored_at) > self.ttl_seconds


class CookieStore:
    def __init__(self):
        self._store: Dict[str, List[StoredCookie]] = {}
        # CapSolver UA cache: domain -> (user_agent, stored_at)
        # cf_clearance is bound to the UA that CapSolver used to solve the challenge.
        # Subsequent contexts for the same domain MUST use this UA.
        self._capsolver_ua: Dict[str, tuple[str, float]] = {}

    def _key(self, domain: str, proxy_server: Optional[str] = None) -> str:
        return f"{domain}|{proxy_server or 'direct'}"

    async def save_from_context(self, context, domain: str, proxy_server: Optional[str] = None):
        """Extract Cloudflare cookies from browser context and store them."""
        cookies = await context.cookies()
        key = self._key(domain, proxy_server)
        self._store[key] = [
            StoredCookie(
                name=c["name"],
                value=c["value"],
                domain=c.get("domain", domain),
                path=c.get("path", "/"),
            )
            for c in cookies
            if c.get("name") in _CF_COOKIE_NAMES
        ]

    async def load_into_context(self, context, domain: str, proxy_server: Optional[str] = None) -> int:
        """Load stored cookies into a fresh browser context. Returns count loaded."""
        key = self._key(domain, proxy_server)
        stored = self._store.get(key, [])
        valid = [c for c in stored if not c.is_expired]
        if not valid:
            return 0
        playwright_cookies = [
            {
                "name": c.name,
                "value": c.value,
                "domain": c.domain,
                "path": c.path,
                "httpOnly": True,
                "secure": True,
            }
            for c in valid
        ]
        await context.add_cookies(playwright_cookies)
        return len(valid)

    def save_capsolver_ua(self, domain: str, user_agent: str):
        """Cache the user agent CapSolver used to solve a challenge for this domain.

        cf_clearance cookies are bound to the UA that solved the challenge.
        Subsequent contexts for the same domain should use this UA to reuse the cookie.
        """
        self._capsolver_ua[domain] = (user_agent, time.time())
        logger.info(f"Cached CapSolver UA for {domain}: {user_agent[:60]}...")

    def get_capsolver_ua(self, domain: str, max_age_seconds: float = 1500) -> Optional[str]:
        """Get the cached CapSolver UA for a domain, if still fresh."""
        entry = self._capsolver_ua.get(domain)
        if not entry:
            return None
        ua, stored_at = entry
        if (time.time() - stored_at) > max_age_seconds:
            del self._capsolver_ua[domain]
            return None
        return ua

    def clear_expired(self):
        for key in list(self._store):
            self._store[key] = [c for c in self._store[key] if not c.is_expired]
            if not self._store[key]:
                del self._store[key]
        # Also clear expired CapSolver UAs
        now = time.time()
        for domain in list(self._capsolver_ua):
            _, stored_at = self._capsolver_ua[domain]
            if (now - stored_at) > 1500:
                del self._capsolver_ua[domain]


_global_store: Optional[CookieStore] = None


def get_cookie_store() -> CookieStore:
    global _global_store
    if _global_store is None:
        _global_store = CookieStore()
    return _global_store
