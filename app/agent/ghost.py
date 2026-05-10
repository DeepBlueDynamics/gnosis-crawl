"""Ghost Protocol: diagnostic mode for anti-bot blocked pages.

When a crawl returns blocked or thin content, Ghost Protocol:

1. Takes a screenshot of exactly what the browser rendered
2. Sends the screenshot to a vision LLM
3. LLM diagnoses what's blocking (Cloudflare, CAPTCHA, login wall, etc.)
4. Returns a structured diagnosis with a recommended next action

The caller (engine.py) acts on the diagnosis — retry with warmup, tell the
agent to inject cookies, or mark the URL as unsolvable.

Ghost Protocol is diagnostic, not extractive. The screenshot shows the BLOCK
PAGE (challenge, wall, error), not the target content. The point is to
identify the block type so the system can route around it.

Requires AGENT_GHOST_ENABLED=true.
Auto-triggers on detected blocks when AGENT_GHOST_AUTO_TRIGGER=true.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Block detection (used before triggering Ghost Protocol)
# ---------------------------------------------------------------------------

class BlockSignal(str, Enum):
    """Categorized anti-bot block signals."""
    CLOUDFLARE = "cloudflare_challenge"
    CAPTCHA = "captcha"
    SESSION_VERIFY = "session_verification"
    ACCESS_DENIED = "access_denied"
    BOT_CHALLENGE = "bot_challenge"
    EMPTY_SHELL = "empty_spa_shell"
    HTTP_403 = "http_403"
    HTTP_429 = "http_429"
    HTTP_503 = "http_503"


@dataclass
class BlockDetection:
    """Result of block signal analysis."""
    blocked: bool = False
    signal: Optional[BlockSignal] = None
    reason: str = ""
    captcha_detected: bool = False
    confidence: float = 0.0


_BLOCK_PATTERNS: List[tuple] = [
    ("cloudflare", BlockSignal.CLOUDFLARE, 0.95),
    ("verify your session", BlockSignal.SESSION_VERIFY, 0.9),
    ("captcha", BlockSignal.CAPTCHA, 0.95),
    ("recaptcha", BlockSignal.CAPTCHA, 0.95),
    ("hcaptcha", BlockSignal.CAPTCHA, 0.95),
    ("access denied", BlockSignal.ACCESS_DENIED, 0.8),
    ("just a moment", BlockSignal.BOT_CHALLENGE, 0.85),
    ("are you human", BlockSignal.BOT_CHALLENGE, 0.9),
    ("attention required", BlockSignal.BOT_CHALLENGE, 0.85),
    ("checking your browser", BlockSignal.BOT_CHALLENGE, 0.9),
    ("please wait while we verify", BlockSignal.BOT_CHALLENGE, 0.9),
    ("enable javascript and cookies", BlockSignal.BOT_CHALLENGE, 0.8),
]

_EMPTY_SHELL_CHAR_THRESHOLD = 200
_EMPTY_SHELL_WORD_THRESHOLD = 30


def detect_block(
    *,
    html: str = "",
    markdown: str = "",
    status_code: Optional[int] = None,
    body_char_count: int = 0,
    body_word_count: int = 0,
    content_quality: str = "",
) -> BlockDetection:
    """Analyze crawl output for anti-bot block signals."""
    combined = f"{html or ''}\n{markdown or ''}".lower()

    for phrase, signal, confidence in _BLOCK_PATTERNS:
        if phrase in combined:
            return BlockDetection(
                blocked=True,
                signal=signal,
                reason=f"Detected '{phrase}' in page content",
                captcha_detected=signal == BlockSignal.CAPTCHA,
                confidence=confidence,
            )

    if status_code == 403:
        return BlockDetection(blocked=True, signal=BlockSignal.HTTP_403, reason="HTTP 403 Forbidden", confidence=0.7)
    if status_code == 429:
        return BlockDetection(blocked=True, signal=BlockSignal.HTTP_429, reason="HTTP 429 Too Many Requests", confidence=0.8)
    if status_code == 503:
        return BlockDetection(blocked=True, signal=BlockSignal.HTTP_503, reason="HTTP 503 (common anti-bot response)", confidence=0.75)

    if (
        body_char_count < _EMPTY_SHELL_CHAR_THRESHOLD
        and body_word_count < _EMPTY_SHELL_WORD_THRESHOLD
        and html
        and len(html) > 500
    ):
        return BlockDetection(
            blocked=True,
            signal=BlockSignal.EMPTY_SHELL,
            reason="Empty SPA shell: HTML present but minimal text content",
            confidence=0.6,
        )

    if content_quality == "blocked":
        return BlockDetection(
            blocked=True,
            signal=BlockSignal.BOT_CHALLENGE,
            reason="Crawler classified content_quality as 'blocked'",
            confidence=0.85,
        )

    return BlockDetection(blocked=False)


def should_trigger_ghost(
    detection: BlockDetection,
    *,
    ghost_enabled: bool = False,
    auto_trigger: bool = True,
) -> bool:
    """Determine whether to activate Ghost Protocol for this detection."""
    if not ghost_enabled:
        return False
    if not detection.blocked:
        return False
    if not auto_trigger:
        return False
    if detection.signal == BlockSignal.ACCESS_DENIED and detection.confidence < 0.85:
        return False
    return True


# ---------------------------------------------------------------------------
# Screenshot capture
# ---------------------------------------------------------------------------

@dataclass
class GhostCapture:
    """Result of a Ghost Protocol screenshot capture."""
    success: bool = False
    image_bytes: bytes = b""
    content_type: str = "image/png"
    url: str = ""
    capture_ms: int = 0
    error: Optional[str] = None


async def capture_screenshot(
    url: str,
    *,
    max_width: int = 1280,
    timeout: int = 30,
    javascript: bool = True,
    proxy=None,
) -> GhostCapture:
    """Take a screenshot of the URL as the browser currently renders it.

    The screenshot shows whatever the browser got — including block pages,
    CAPTCHAs, cookie walls, etc. That's the diagnostic data we need.
    """
    start = time.monotonic()
    try:
        from app.browser import get_browser_engine
        browser = await get_browser_engine()

        html, page_info, screenshot_data = await browser.crawl_with_context(
            url=url,
            javascript_enabled=javascript,
            timeout=timeout * 1000,
            take_screenshot=True,
            wait_until="domcontentloaded",
            wait_after_load_ms=1500,
            proxy=proxy,
        )

        capture_ms = int((time.monotonic() - start) * 1000)

        if screenshot_data is None:
            return GhostCapture(success=False, url=url, capture_ms=capture_ms, error="Screenshot returned None")

        image_bytes = screenshot_data[0] if isinstance(screenshot_data, list) else screenshot_data
        return GhostCapture(success=True, image_bytes=image_bytes, url=url, capture_ms=capture_ms)

    except Exception as exc:
        capture_ms = int((time.monotonic() - start) * 1000)
        logger.error("Ghost screenshot capture failed for %s: %s", url, exc, exc_info=True)
        return GhostCapture(success=False, url=url, capture_ms=capture_ms, error=str(exc))


# ---------------------------------------------------------------------------
# Vision diagnosis
# ---------------------------------------------------------------------------

GHOST_DIAGNOSIS_PROMPT = """You are a web crawler diagnostic agent. The crawler attempted to fetch a URL but returned empty or blocked content. You are looking at a screenshot of exactly what the browser rendered.

Diagnose what is blocking the crawler and recommend what should be tried next.

Respond in EXACTLY this format (no extra text before or after):

BLOCK_TYPE: <one of: CLOUDFLARE | AKAMAI | CAPTCHA | LOGIN_WALL | COOKIE_CONSENT | JS_GATE | RATE_LIMITED | ACCESS_DENIED | ERROR_PAGE | EMPTY | UNKNOWN>
DESCRIPTION: <one sentence — exactly what you see on screen>
ACTION: <one of: WARMUP | ACCEPT_COOKIES | INJECT_COOKIES | RETRY_JS | LOGIN_REQUIRED | UNSOLVABLE>
ACTION_REASON: <one sentence — why this action would help>

BLOCK_TYPE meanings:
- CLOUDFLARE: Cloudflare "Just a moment" / "Checking your browser" challenge
- AKAMAI: Akamai Bot Manager / "Access Denied" screen
- CAPTCHA: Visible CAPTCHA (reCAPTCHA, hCAPTCHA, image puzzle)
- LOGIN_WALL: Login or signup page blocking access to content
- COOKIE_CONSENT: Cookie consent/GDPR popup blocking the page
- JS_GATE: Blank page that requires JavaScript to render content
- RATE_LIMITED: 429 or explicit rate limit message
- ACCESS_DENIED: Generic 403/access denied without specific anti-bot branding
- ERROR_PAGE: 404, 500, or other server error page
- EMPTY: Page rendered but shows actual content (crawl failure was transient or the content was sparse)
- UNKNOWN: Cannot identify the block type

ACTION meanings:
- WARMUP: Visit the site's homepage first (or use Google click-through) to establish a real session cookie before retrying
- ACCEPT_COOKIES: Dismiss cookie consent popup — retry with JavaScript enabled
- INJECT_COOKIES: User must provide real browser session cookies (anti-bot or login cookie required)
- RETRY_JS: Enable JavaScript and retry — page needs JS to render
- LOGIN_REQUIRED: Page requires authenticated user session — cannot crawl without credentials
- UNSOLVABLE: Cannot bypass programmatically (sophisticated CAPTCHA, IP ban, paywalled behind login)"""


@dataclass
class GhostDiagnosis:
    """Structured diagnosis from Ghost Protocol."""
    success: bool = False
    url: str = ""
    block_type: str = "UNKNOWN"
    description: str = ""
    action: str = "UNSOLVABLE"
    action_reason: str = ""
    capture_ms: int = 0
    diagnosis_ms: int = 0
    total_ms: int = 0
    provider: str = ""
    error: Optional[str] = None


def _parse_diagnosis(text: str) -> Tuple[str, str, str, str]:
    """Parse the 4-field structured response from the vision LLM."""
    block_type = "UNKNOWN"
    description = ""
    action = "UNSOLVABLE"
    action_reason = ""

    for line in text.strip().splitlines():
        line = line.strip()
        if line.upper().startswith("BLOCK_TYPE:"):
            block_type = line.split(":", 1)[1].strip().upper()
        elif line.upper().startswith("DESCRIPTION:"):
            description = line.split(":", 1)[1].strip()
        elif line.upper().startswith("ACTION:") and not line.upper().startswith("ACTION_REASON:"):
            action = line.split(":", 1)[1].strip().upper()
        elif line.upper().startswith("ACTION_REASON:"):
            action_reason = line.split(":", 1)[1].strip()

    # Normalize to valid enums
    valid_block_types = {"CLOUDFLARE", "AKAMAI", "CAPTCHA", "LOGIN_WALL", "COOKIE_CONSENT", "JS_GATE", "RATE_LIMITED", "ACCESS_DENIED", "ERROR_PAGE", "EMPTY", "UNKNOWN"}
    valid_actions = {"WARMUP", "ACCEPT_COOKIES", "INJECT_COOKIES", "RETRY_JS", "LOGIN_REQUIRED", "UNSOLVABLE"}
    if block_type not in valid_block_types:
        block_type = "UNKNOWN"
    if action not in valid_actions:
        action = "UNSOLVABLE"

    return block_type, description, action, action_reason


async def diagnose_via_vision(
    capture: GhostCapture,
    *,
    provider: Optional[Any] = None,
) -> GhostDiagnosis:
    """Send a screenshot to a vision LLM and get a structured block diagnosis."""
    if not capture.success or not capture.image_bytes:
        return GhostDiagnosis(
            success=False,
            url=capture.url,
            error=capture.error or "No screenshot data",
        )

    if provider is None:
        return GhostDiagnosis(success=False, url=capture.url, error="No vision provider configured")

    start = time.monotonic()
    try:
        raw_text = await provider.vision(
            capture.image_bytes,
            GHOST_DIAGNOSIS_PROMPT,
            detail="low",  # we want block-type classification, not fine text
        )
        diagnosis_ms = int((time.monotonic() - start) * 1000)

        block_type, description, action, action_reason = _parse_diagnosis(raw_text)
        provider_name = provider.__class__.__name__

        logger.info(
            "Ghost diagnosis for %s: block_type=%s action=%s (%dms)",
            capture.url, block_type, action, diagnosis_ms,
        )

        return GhostDiagnosis(
            success=True,
            url=capture.url,
            block_type=block_type,
            description=description,
            action=action,
            action_reason=action_reason,
            capture_ms=capture.capture_ms,
            diagnosis_ms=diagnosis_ms,
            total_ms=capture.capture_ms + diagnosis_ms,
            provider=provider_name,
        )

    except NotImplementedError:
        diagnosis_ms = int((time.monotonic() - start) * 1000)
        return GhostDiagnosis(
            success=False,
            url=capture.url,
            error=f"Provider does not support vision",
            diagnosis_ms=diagnosis_ms,
        )
    except Exception as exc:
        diagnosis_ms = int((time.monotonic() - start) * 1000)
        logger.error("Ghost vision diagnosis failed: %s", exc, exc_info=True)
        return GhostDiagnosis(
            success=False,
            url=capture.url,
            error=str(exc),
            diagnosis_ms=diagnosis_ms,
        )


# ---------------------------------------------------------------------------
# Full Ghost Protocol pipeline
# ---------------------------------------------------------------------------

async def run_ghost_protocol(
    url: str,
    *,
    provider: Optional[Any] = None,
    max_width: int = 1280,
    timeout: int = 30,
    block_detection: Optional[BlockDetection] = None,
    proxy=None,
) -> GhostDiagnosis:
    """Execute the Ghost Protocol pipeline: screenshot → diagnose → recommend.

    Takes a screenshot of what the browser actually rendered (which is typically
    a block page, challenge, or error), sends it to a vision LLM, and returns
    a structured diagnosis with a recommended next action.

    The caller is responsible for acting on the diagnosis (retry with warmup,
    inject cookies, inform the agent, etc.).
    """
    pipeline_start = time.monotonic()
    logger.info("Ghost Protocol activated for %s", url)

    capture = await capture_screenshot(url, max_width=max_width, timeout=timeout, proxy=proxy)

    if not capture.success:
        total_ms = int((time.monotonic() - pipeline_start) * 1000)
        return GhostDiagnosis(
            success=False,
            url=url,
            capture_ms=capture.capture_ms,
            total_ms=total_ms,
            error=f"Screenshot failed: {capture.error}",
        )

    diagnosis = await diagnose_via_vision(capture, provider=provider)
    diagnosis.total_ms = int((time.monotonic() - pipeline_start) * 1000)
    return diagnosis


# ---------------------------------------------------------------------------
# Vision provider factory
# ---------------------------------------------------------------------------

def create_ghost_provider():
    """Create a vision-capable provider for Ghost Protocol."""
    from app.config import settings
    from app.agent.providers.base import create_provider, _pick_key, _pick_model, _pick_base_url

    provider_name = settings.agent_ghost_vision_provider or settings.agent_provider
    return create_provider(
        provider_name,
        api_key=_pick_key(settings, provider_name),
        model=_pick_model(settings, provider_name),
        base_url=_pick_base_url(settings, provider_name),
    )
