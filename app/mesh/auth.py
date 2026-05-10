"""HMAC token signing and verification for inter-node mesh auth.

Every mesh HTTP call carries a mesh_token. The token is an HMAC-SHA256
signature over a timestamp (and optionally a nonce), so nodes sharing the
same MESH_SECRET can verify each other without a central auth service.

Replay protection
-----------------
Without nonces, an intercepted mesh_token can be replayed by anyone on the
network within the TTL window. For trusted private subnets this is fine and
``MESH_REQUIRE_NONCE`` defaults to false, keeping the token format compact.

For zero-trust deployments where mesh traffic transits public networks, set
``MESH_REQUIRE_NONCE=true``. The verifier then:

    1. requires every incoming token to carry a unique nonce,
    2. caches each accepted nonce until its TTL elapses, and
    3. rejects any token whose nonce has already been seen.

Combined with a tighter TTL (10s when nonce mode is on), this limits the
replay window to single-use.
"""

from __future__ import annotations

import hashlib
import hmac
import secrets
import time
import threading
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# Tokens are valid for 60 seconds in legacy (private network) mode to absorb
# clock skew. With nonce-mode on, we tighten to 10 seconds — clocks should be
# NTP-aligned in any modern deployment, and a tighter window further limits
# the replay envelope.
TOKEN_TTL_S = 60
TOKEN_TTL_NONCE_S = 10

# How many nonces to keep before evicting the oldest. 8192 covers ~136 calls
# per second sustained over the TTL window, far above realistic mesh chatter.
_NONCE_CACHE_MAX = 8192


class NonceCache:
    """Thread-safe TTL/LRU nonce cache.

    Backed by a plain dict keyed on nonce → expiration_ms. We sweep expired
    entries on every check_and_insert call (cheap; amortized O(1)) and cap
    the size to evict the oldest when it grows beyond _NONCE_CACHE_MAX.
    """

    def __init__(self, max_size: int = _NONCE_CACHE_MAX):
        self._max_size = max_size
        self._entries: dict[str, int] = {}
        self._lock = threading.Lock()

    def check_and_insert(self, nonce: str, ttl_ms: int) -> bool:
        """Return True if the nonce is fresh; False if already seen.

        On a fresh nonce, inserts it with expiration = now + ttl_ms.
        """
        now = int(time.time() * 1000)
        expires = now + ttl_ms

        with self._lock:
            # Sweep expired entries — bounded by current size, fast in practice.
            if self._entries:
                stale = [k for k, exp in self._entries.items() if exp <= now]
                for k in stale:
                    del self._entries[k]

            if nonce in self._entries:
                return False

            # Cap the cache by evicting the oldest entry (lowest expiration).
            if len(self._entries) >= self._max_size:
                oldest = min(self._entries, key=self._entries.get)
                del self._entries[oldest]

            self._entries[nonce] = expires
            return True

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)


# Module-level singleton — one cache per process.
_nonce_cache = NonceCache()


def _new_nonce() -> str:
    """Generate a 128-bit random nonce as urlsafe-base64 (no padding)."""
    return secrets.token_urlsafe(16)


def _signing_message(ts_ms: int, nonce: Optional[str]) -> bytes:
    """Build the canonical message to sign.

    Legacy format (no nonce): ``"{ts}"``
    Nonce format:             ``"{ts}.{nonce}"``
    """
    if nonce:
        return f"{ts_ms}.{nonce}".encode()
    return str(ts_ms).encode()


def sign_mesh_token(
    secret: str,
    timestamp_ms: int | None = None,
    *,
    require_nonce: bool = False,
) -> str:
    """Create an HMAC-SHA256 mesh token.

    Format (legacy):    ``{timestamp_ms}.{hex_signature}``
    Format (nonce mode): ``{timestamp_ms}.{nonce}.{hex_signature}``

    The verifier auto-detects which format it received based on dot count
    in the token, so legacy senders interoperate with non-strict verifiers.
    Pass ``require_nonce=True`` to always emit a nonce — recommended when
    the receiver enforces ``MESH_REQUIRE_NONCE``.
    """
    ts = timestamp_ms or int(time.time() * 1000)
    nonce = _new_nonce() if require_nonce else None
    sig = hmac.new(secret.encode(), _signing_message(ts, nonce), hashlib.sha256).hexdigest()
    if nonce:
        return f"{ts}.{nonce}.{sig}"
    return f"{ts}.{sig}"


def verify_mesh_token(
    token: str,
    secret: str,
    *,
    require_nonce: bool = False,
) -> bool:
    """Verify an HMAC-SHA256 mesh token.

    When ``require_nonce`` is False (default, legacy/private-network mode):
      - accepts both 2-segment and 3-segment tokens
      - checks signature and TTL only

    When ``require_nonce`` is True (zero-trust / public-network mode):
      - rejects 2-segment tokens (no replay protection)
      - checks signature and tightened TTL
      - checks nonce against the in-process cache; rejects on duplicate
    """
    try:
        parts = token.split(".")
        if len(parts) == 2:
            if require_nonce:
                logger.debug("Mesh token rejected: nonce required but missing")
                return False
            ts_str, sig = parts
            nonce = None
            ttl_s = TOKEN_TTL_S
        elif len(parts) == 3:
            ts_str, nonce, sig = parts
            ttl_s = TOKEN_TTL_NONCE_S if require_nonce else TOKEN_TTL_S
        else:
            return False

        ts = int(ts_str)
        now = int(time.time() * 1000)
        if abs(now - ts) > ttl_s * 1000:
            logger.debug("Mesh token expired: age=%dms ttl=%ds", abs(now - ts), ttl_s)
            return False

        expected = hmac.new(secret.encode(), _signing_message(ts, nonce), hashlib.sha256).hexdigest()
        if not hmac.compare_digest(sig, expected):
            return False

        # Replay check — only when nonce mode is active and a nonce was sent.
        if require_nonce and nonce is not None:
            ttl_ms = ttl_s * 1000
            if not _nonce_cache.check_and_insert(nonce, ttl_ms):
                logger.warning("Mesh token replay detected: nonce=%s", nonce[:8])
                return False

        return True
    except Exception:
        logger.debug("Mesh token verification failed", exc_info=True)
        return False
