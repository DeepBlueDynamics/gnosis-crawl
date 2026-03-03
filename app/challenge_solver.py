"""
Cloudflare Challenge Solver for gnosis-crawl.

Detects and waits for Cloudflare challenges to auto-resolve on Playwright pages.
Falls back to CapSolver API for visible Turnstile challenges that don't auto-resolve.
"""

import asyncio
import logging
import os
from dataclasses import dataclass
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)


class ChallengeType(str, Enum):
    """Types of Cloudflare challenges."""
    TURNSTILE = "turnstile"
    JS_CHALLENGE = "js_challenge"
    BROWSER_CHECK = "browser_check"
    MANAGED = "managed_challenge"
    NONE = "none"


@dataclass
class ChallengeDetection:
    """Result of challenge detection."""
    detected: bool = False
    challenge_type: ChallengeType = ChallengeType.NONE
    confidence: float = 0.0
    selector_matched: str = ""


@dataclass
class ChallengeResult:
    """Result of challenge resolution attempt."""
    resolved: bool = False
    challenge_type: ChallengeType = ChallengeType.NONE
    method: str = "none"  # "auto_resolve", "capsolver", "none"
    wait_time_ms: int = 0
    error: Optional[str] = None


# Selectors that indicate a Cloudflare challenge is present
CHALLENGE_SELECTORS = [
    ('#challenge-running', ChallengeType.JS_CHALLENGE),
    ('#challenge-stage', ChallengeType.JS_CHALLENGE),
    ('.cf-browser-verification', ChallengeType.BROWSER_CHECK),
    ('iframe[src*="challenges.cloudflare.com"]', ChallengeType.TURNSTILE),
    ('#turnstile-wrapper', ChallengeType.TURNSTILE),
    ('#cf-challenge-running', ChallengeType.MANAGED),
    ('.cf-turnstile', ChallengeType.TURNSTILE),
]

# Selectors that indicate the challenge has been resolved
RESOLVED_SELECTORS = [
    '#challenge-success',
    '#challenge-stage[style*="display: none"]',
]

# Title patterns that indicate a challenge page
CHALLENGE_TITLE_PATTERNS = [
    # English
    'just a moment',
    'attention required',
    'checking your browser',
    'please wait',
    'one more step',
    'verify you are human',
    # Portuguese
    'um momento',
    'verificação de segurança',
    # Spanish
    'un momento',
    'verificación de seguridad',
    # French
    'un instant',
    'vérification de sécurité',
    # German
    'einen moment',
    'sicherheitsüberprüfung',
]


async def detect_challenge(page) -> ChallengeDetection:
    """
    Detect if a Cloudflare challenge is present on the page.

    Detection priority: DOM selectors > content heuristic > title.
    Title match alone only indicates *a* challenge is present; DOM selectors
    determine the *type* (Turnstile, Managed, JS).  This matters because
    only TURNSTILE and MANAGED types are eligible for the CapSolver fallback.

    Args:
        page: Playwright page object

    Returns:
        ChallengeDetection with type and confidence
    """
    # Step 1: Title check — fast signal that *some* challenge is present.
    # Don't return yet; use it as a flag so DOM selectors can refine the type.
    title_matched_pattern = None
    try:
        title = await page.title()
        title_lower = title.lower() if title else ""
        for pattern in CHALLENGE_TITLE_PATTERNS:
            if pattern in title_lower:
                title_matched_pattern = pattern
                break
    except Exception:
        pass

    # Step 2: DOM selectors — the most accurate type classification.
    # Turnstile iframes / widgets override the generic JS_CHALLENGE type.
    for selector, challenge_type in CHALLENGE_SELECTORS:
        try:
            element = await page.query_selector(selector)
            if element:
                visible = await element.is_visible()
                confidence = 0.95 if visible else 0.7
                return ChallengeDetection(
                    detected=True,
                    challenge_type=challenge_type,
                    confidence=confidence,
                    selector_matched=selector,
                )
        except Exception:
            continue

    # Step 3: Content-based heuristic — catches custom Cloudflare interstitials
    # and Managed Challenges whose DOM selectors are embedded in heavy JS.
    # Cloudflare challenge pages can be 30-50K of JS scaffolding, so the
    # threshold must be high enough to include them.
    try:
        content = await page.content()
        if content and len(content) < 50000:
            content_lower = content.lower()
            cf_signals = [
                "cloudflare", "cf-browser-verification", "ray id",
                "challenge-platform", "turnstile", "cf_chl_opt",
                "performance & security by",
            ]
            matched_signals = [s for s in cf_signals if s in content_lower]
            if len(matched_signals) >= 2:
                logger.info(f"Challenge detected via content heuristic: {matched_signals}")
                return ChallengeDetection(
                    detected=True,
                    challenge_type=ChallengeType.MANAGED,
                    confidence=0.8,
                    selector_matched=f"content_heuristic:{','.join(matched_signals[:3])}",
                )
    except Exception:
        pass

    # Step 4: Title-only fallback — if DOM selectors and content heuristic
    # found nothing, but the title matched, classify as MANAGED (not JS_CHALLENGE)
    # because Cloudflare uses the same "Just a moment..." title for all challenge
    # types.  Classifying as MANAGED ensures CapSolver is eligible as a fallback.
    if title_matched_pattern:
        return ChallengeDetection(
            detected=True,
            challenge_type=ChallengeType.MANAGED,
            confidence=0.9,
            selector_matched=f"title:{title_matched_pattern}",
        )

    return ChallengeDetection(detected=False)


async def wait_for_challenge_resolution(
    page,
    timeout_ms: int = 15000,
    poll_interval_ms: int = 500,
    site_url: str = None,
) -> ChallengeResult:
    """
    Wait for a Cloudflare challenge to auto-resolve.

    Many Turnstile challenges are invisible and auto-resolve within seconds.
    This function polls the page to detect when the challenge is gone.

    For Managed Challenges, Cloudflare's JS verifies the browser and shows
    "Verification successful. Waiting for <site> to respond".  At that point
    cf_clearance is set, but the page may NOT auto-navigate.  If we detect
    this state, we navigate to the URL again (with cf_clearance cookie).

    Note: detect_challenge uses a content heuristic that matches Cloudflare
    keywords in HTML — these keywords remain even after verification succeeds.
    So we must also check #challenge-success selector and body text.
    """
    detection = await detect_challenge(page)
    if not detection.detected:
        return ChallengeResult(resolved=True, method="none", wait_time_ms=0)

    start_ms = int(asyncio.get_event_loop().time() * 1000)
    elapsed = 0
    verification_seen = False

    while elapsed < timeout_ms:
        await asyncio.sleep(poll_interval_ms / 1000)
        elapsed = int(asyncio.get_event_loop().time() * 1000) - start_ms

        # Check for resolved indicators FIRST — these are most reliable
        for sel in RESOLVED_SELECTORS:
            try:
                el = await page.query_selector(sel)
                if el:
                    logger.info(f"Challenge resolved via selector {sel} in {elapsed}ms")
                    return ChallengeResult(
                        resolved=True,
                        challenge_type=detection.challenge_type,
                        method="auto_resolve",
                        wait_time_ms=elapsed,
                    )
            except Exception:
                continue

        # Check if challenge page navigated away entirely
        current = await detect_challenge(page)
        if not current.detected:
            return ChallengeResult(
                resolved=True,
                challenge_type=detection.challenge_type,
                method="auto_resolve",
                wait_time_ms=elapsed,
            )

        # Check for "Verification successful" — Cloudflare verified the browser
        # but the page hasn't navigated yet.  The cf_clearance cookie should be
        # set at this point.  Navigate to URL with the cookie.
        if not verification_seen:
            try:
                body_text = await page.inner_text("body")
                if "verification successful" in body_text.lower():
                    verification_seen = True
                    logger.info("Cloudflare verification successful — waiting for redirect")
                    # Give Cloudflare 5 seconds to redirect naturally
                    await asyncio.sleep(5)
                    elapsed = int(asyncio.get_event_loop().time() * 1000) - start_ms
                    # Check if it navigated
                    post_wait = await detect_challenge(page)
                    if not post_wait.detected:
                        return ChallengeResult(
                            resolved=True,
                            challenge_type=detection.challenge_type,
                            method="auto_resolve",
                            wait_time_ms=elapsed,
                        )
                    # Still on challenge page — navigate fresh with cf_clearance
                    _nav_url = site_url or page.url
                    logger.info(f"Redirect didn't happen — navigating to {_nav_url} with cf_clearance")
                    try:
                        await page.goto(_nav_url, timeout=20000, wait_until="domcontentloaded")
                        await asyncio.sleep(2)
                        elapsed = int(asyncio.get_event_loop().time() * 1000) - start_ms
                        # Check the NEW page title — if it's no longer a challenge title
                        try:
                            new_title = await page.title()
                            if new_title and not any(
                                p in new_title.lower()
                                for p in CHALLENGE_TITLE_PATTERNS
                            ):
                                logger.info(f"Challenge resolved via goto after verification in {elapsed}ms (title: {new_title})")
                                return ChallengeResult(
                                    resolved=True,
                                    challenge_type=detection.challenge_type,
                                    method="auto_resolve",
                                    wait_time_ms=elapsed,
                                )
                        except Exception:
                            pass
                        post_nav = await detect_challenge(page)
                        if not post_nav.detected:
                            logger.info(f"Challenge resolved via goto after verification in {elapsed}ms")
                            return ChallengeResult(
                                resolved=True,
                                challenge_type=detection.challenge_type,
                                method="auto_resolve",
                                wait_time_ms=elapsed,
                            )
                    except Exception as nav_err:
                        logger.debug(f"Navigation after verification failed: {nav_err}")
            except Exception:
                pass

    # Timeout — challenge didn't auto-resolve
    return ChallengeResult(
        resolved=False,
        challenge_type=detection.challenge_type,
        method="none",
        wait_time_ms=elapsed,
        error=f"Challenge auto-resolve timeout after {timeout_ms}ms",
    )


async def solve_turnstile_capsolver(
    page,
    site_url: str,
    api_key: Optional[str] = None,
    timeout_ms: int = 30000,
) -> ChallengeResult:
    """
    Attempt to solve a visible Turnstile challenge via CapSolver API.

    Only called when auto-resolve fails and a visible Turnstile is detected.
    Requires CAPSOLVER_API_KEY environment variable.
    """
    key = api_key or os.environ.get("CAPSOLVER_API_KEY")
    if not key:
        logger.warning("CAPSOLVER_API_KEY not configured — CapSolver fallback disabled")
        return ChallengeResult(
            resolved=False,
            challenge_type=ChallengeType.TURNSTILE,
            method="none",
            error="CAPSOLVER_API_KEY not configured",
        )

    start_ms = int(asyncio.get_event_loop().time() * 1000)

    try:
        # Extract sitekey from Turnstile iframe/widget
        sitekey = await _extract_turnstile_sitekey(page)
        if not sitekey:
            return ChallengeResult(
                resolved=False,
                challenge_type=ChallengeType.TURNSTILE,
                method="none",
                error="Could not extract Turnstile sitekey",
            )

        # Call CapSolver API
        token = await _call_capsolver(key, site_url, sitekey, timeout_ms)
        if not token:
            elapsed = int(asyncio.get_event_loop().time() * 1000) - start_ms
            return ChallengeResult(
                resolved=False,
                challenge_type=ChallengeType.TURNSTILE,
                method="capsolver",
                wait_time_ms=elapsed,
                error="CapSolver failed to return token",
            )

        # Inject token into the page
        await _inject_turnstile_token(page, token)

        # Wait briefly for page to process the token
        await asyncio.sleep(2)

        # Verify resolution
        current = await detect_challenge(page)
        elapsed = int(asyncio.get_event_loop().time() * 1000) - start_ms

        if not current.detected:
            return ChallengeResult(
                resolved=True,
                challenge_type=ChallengeType.TURNSTILE,
                method="capsolver",
                wait_time_ms=elapsed,
            )
        else:
            return ChallengeResult(
                resolved=False,
                challenge_type=ChallengeType.TURNSTILE,
                method="capsolver",
                wait_time_ms=elapsed,
                error="Token injected but challenge still present",
            )

    except Exception as e:
        elapsed = int(asyncio.get_event_loop().time() * 1000) - start_ms
        return ChallengeResult(
            resolved=False,
            challenge_type=ChallengeType.TURNSTILE,
            method="capsolver",
            wait_time_ms=elapsed,
            error=str(e),
        )


async def solve_managed_challenge_capsolver(
    page,
    site_url: str,
    proxy_config: Optional[dict] = None,
    api_key: Optional[str] = None,
    timeout_ms: int = 30000,
) -> ChallengeResult:
    """
    Solve a Cloudflare Managed Challenge via CapSolver's AntiCloudflareTask.

    Unlike AntiTurnstileTaskProxyLess, this task type:
    - Does NOT need a sitekey
    - DOES need a proxy (same IP as the browser)
    - Is designed specifically for full-page Cloudflare Managed Challenges

    Returns ChallengeResult with cookies to inject on success.
    """
    key = api_key or os.environ.get("CAPSOLVER_API_KEY")
    if not key:
        logger.warning("CAPSOLVER_API_KEY not configured — managed challenge solver disabled")
        return ChallengeResult(
            resolved=False,
            challenge_type=ChallengeType.MANAGED,
            method="none",
            error="CAPSOLVER_API_KEY not configured",
        )

    proxy_str = _format_proxy_for_capsolver(proxy_config)
    if not proxy_str:
        logger.warning("No proxy config for AntiCloudflareTask — proxy required")
        return ChallengeResult(
            resolved=False,
            challenge_type=ChallengeType.MANAGED,
            method="none",
            error="AntiCloudflareTask requires proxy config",
        )

    start_ms = int(asyncio.get_event_loop().time() * 1000)

    try:
        token_or_cookies = await _call_capsolver_managed(
            key, site_url, proxy_str, timeout_ms
        )
        if not token_or_cookies:
            elapsed = int(asyncio.get_event_loop().time() * 1000) - start_ms
            return ChallengeResult(
                resolved=False,
                challenge_type=ChallengeType.MANAGED,
                method="capsolver_managed",
                wait_time_ms=elapsed,
                error="CapSolver AntiCloudflareTask failed",
            )

        # Inject cookies into the browser context
        cookies_dict = token_or_cookies.get("cookies", {})
        capsolver_ua = token_or_cookies.get("userAgent")

        if cookies_dict and hasattr(page, "context"):
            from urllib.parse import urlparse
            parsed = urlparse(site_url)
            domain = parsed.netloc
            if domain.startswith("www."):
                domain = domain[4:]
            cookie_list = [
                {
                    "name": name,
                    "value": value,
                    "domain": f".{domain}",
                    "path": "/",
                    "secure": True,
                    "httpOnly": True,
                    "sameSite": "None",
                }
                for name, value in cookies_dict.items()
            ]
            try:
                await page.context.add_cookies(cookie_list)
                logger.info(f"Injected {len(cookie_list)} cookies from AntiCloudflareTask: {list(cookies_dict.keys())}")
            except Exception as e:
                logger.debug(f"Cookie injection failed: {e}")

        # The cf_clearance cookie from CapSolver is bound to the user agent
        # that CapSolver used.  To make Cloudflare accept it, we must navigate
        # with the same user agent.  Create a fresh page in a new context
        # with the matching UA and cookies, then swap it in.
        if capsolver_ua and cookies_dict and hasattr(page, "context"):
            logger.info(f"CapSolver UA: {capsolver_ua[:60]}... — creating matched context")
            try:
                browser = page.context.browser
                if browser:
                    new_ctx = await browser.new_context(user_agent=capsolver_ua)
                    # Set cookies on the new context
                    from urllib.parse import urlparse
                    parsed = urlparse(site_url)
                    domain = parsed.netloc
                    if domain.startswith("www."):
                        domain = domain[4:]
                    cookie_list_new = [
                        {
                            "name": name,
                            "value": value,
                            "domain": f".{domain}",
                            "path": "/",
                            "secure": True,
                            "httpOnly": True,
                            "sameSite": "None",
                        }
                        for name, value in cookies_dict.items()
                    ]
                    await new_ctx.add_cookies(cookie_list_new)
                    new_page = await new_ctx.new_page()
                    try:
                        resp = await new_page.goto(site_url, timeout=20000, wait_until="domcontentloaded")
                        await asyncio.sleep(2)
                        # Check if we got real content
                        new_title = await new_page.title()
                        post_nav = await detect_challenge(new_page)
                        if not post_nav.detected:
                            elapsed = int(asyncio.get_event_loop().time() * 1000) - start_ms
                            logger.info(f"Challenge resolved via CapSolver UA-matched context in {elapsed}ms (title: {new_title})")
                            # Cache the CapSolver UA so subsequent crawls to this domain
                            # can create contexts with the same UA and reuse cf_clearance
                            try:
                                from app.cookie_store import get_cookie_store
                                get_cookie_store().save_capsolver_ua(domain, capsolver_ua)
                            except Exception:
                                pass
                            result = ChallengeResult(
                                resolved=True,
                                challenge_type=ChallengeType.MANAGED,
                                method="capsolver_managed",
                                wait_time_ms=elapsed,
                            )
                            result._new_page = new_page
                            result._new_context = new_ctx
                            return result
                        else:
                            logger.info(f"UA-matched context still has challenge (title: {new_title})")
                            await new_page.close()
                            await new_ctx.close()
                    except Exception as nav_err:
                        logger.debug(f"UA-matched navigation failed: {nav_err}")
                        try:
                            await new_page.close()
                            await new_ctx.close()
                        except Exception:
                            pass
            except Exception as ctx_err:
                logger.debug(f"Failed to create UA-matched context: {ctx_err}")

        # Fallback: Navigate the current page fresh with cookies
        try:
            await page.goto(site_url, timeout=20000, wait_until="domcontentloaded")
            await asyncio.sleep(2)
        except Exception as e:
            logger.debug(f"Navigation after cookie injection failed: {e}")

        # Check if challenge is resolved
        current = await detect_challenge(page)
        elapsed = int(asyncio.get_event_loop().time() * 1000) - start_ms

        if not current.detected:
            return ChallengeResult(
                resolved=True,
                challenge_type=ChallengeType.MANAGED,
                method="capsolver_managed",
                wait_time_ms=elapsed,
            )
        else:
            return ChallengeResult(
                resolved=False,
                challenge_type=ChallengeType.MANAGED,
                method="capsolver_managed",
                wait_time_ms=elapsed,
                error="Cookies injected but challenge still present",
            )

    except Exception as e:
        elapsed = int(asyncio.get_event_loop().time() * 1000) - start_ms
        return ChallengeResult(
            resolved=False,
            challenge_type=ChallengeType.MANAGED,
            method="capsolver_managed",
            wait_time_ms=elapsed,
            error=str(e),
        )


async def resolve_challenge(
    page,
    site_url: str,
    auto_wait_ms: int = 15000,
    capsolver_timeout_ms: int = 30000,
    proxy_config: Optional[dict] = None,
) -> ChallengeResult:
    """
    Full challenge resolution pipeline:
    1. Detect if challenge is present
    2. Wait for auto-resolve (invisible Turnstile, JS challenges)
    3. Try clicking interactive Turnstile checkbox
    4. Try AntiCloudflareTask (managed challenges, needs proxy)
    5. Try AntiTurnstileTaskProxyLess (standalone Turnstile, needs sitekey)
    """
    detection = await detect_challenge(page)
    if not detection.detected:
        return ChallengeResult(resolved=True, method="none", wait_time_ms=0)

    logger.info(f"Challenge detected: {detection.challenge_type} (confidence: {detection.confidence}, selector: {detection.selector_matched})")

    # Step 1: Try auto-resolve (handles invisible Turnstile, simple JS challenges)
    auto_result = await wait_for_challenge_resolution(page, timeout_ms=auto_wait_ms, site_url=site_url)
    if auto_result.resolved:
        logger.info(f"Challenge auto-resolved in {auto_result.wait_time_ms}ms")
        return auto_result

    # Step 2: Re-detect the challenge type after auto-resolve wait.
    current_detection = await detect_challenge(page)
    effective_type = current_detection.challenge_type if current_detection.detected else detection.challenge_type
    logger.info(f"Auto-resolve failed. Re-detected challenge type: {effective_type} (initial: {detection.challenge_type})")

    if effective_type not in (ChallengeType.TURNSTILE, ChallengeType.MANAGED):
        logger.warning(f"Challenge type {effective_type} not eligible for further resolution")
        return ChallengeResult(
            resolved=False,
            challenge_type=effective_type,
            method="none",
            wait_time_ms=auto_result.wait_time_ms,
            error=auto_result.error or "Challenge not resolved",
        )

    # Step 3: Try clicking the interactive Turnstile checkbox.
    # This is fast and free — works when cType is 'interactive'.
    logger.info("Attempting Turnstile checkbox click")
    click_ok = await _click_turnstile_checkbox(page)
    if click_ok:
        # Give the page a moment to process the click
        await asyncio.sleep(3)
        post_click = await detect_challenge(page)
        if not post_click.detected:
            total_ms = auto_result.wait_time_ms + 3000
            logger.info(f"Challenge resolved via Turnstile checkbox click in {total_ms}ms")
            return ChallengeResult(
                resolved=True,
                challenge_type=effective_type,
                method="click",
                wait_time_ms=total_ms,
            )
        logger.info("Click succeeded but challenge still present — trying CapSolver")

    # Step 4: Try AntiCloudflareTask for managed challenges (proxy-based, no sitekey)
    if effective_type == ChallengeType.MANAGED and proxy_config:
        logger.info("Attempting CapSolver AntiCloudflareTask for managed challenge")
        managed_result = await solve_managed_challenge_capsolver(
            page, site_url, proxy_config=proxy_config,
            timeout_ms=capsolver_timeout_ms,
        )
        if managed_result.resolved:
            total_ms = auto_result.wait_time_ms + managed_result.wait_time_ms
            result = ChallengeResult(
                resolved=True,
                challenge_type=ChallengeType.MANAGED,
                method="capsolver_managed",
                wait_time_ms=total_ms,
            )
            # Forward the UA-matched page/context if CapSolver created one
            if hasattr(managed_result, '_new_page'):
                result._new_page = managed_result._new_page
                result._new_context = managed_result._new_context
            return result
        logger.warning(f"AntiCloudflareTask failed: {managed_result.error}")

    # Step 5: Fall back to AntiTurnstileTaskProxyLess (needs sitekey)
    logger.info(f"Attempting CapSolver AntiTurnstileTaskProxyLess for {effective_type} challenge")
    capsolver_result = await solve_turnstile_capsolver(
        page, site_url, timeout_ms=capsolver_timeout_ms
    )
    if capsolver_result.resolved:
        total_ms = auto_result.wait_time_ms + capsolver_result.wait_time_ms
        return ChallengeResult(
            resolved=True,
            challenge_type=ChallengeType.TURNSTILE,
            method="capsolver",
            wait_time_ms=total_ms,
        )
    logger.warning(f"AntiTurnstileTaskProxyLess failed: {capsolver_result.error}")

    # All attempts failed
    total_ms = auto_result.wait_time_ms
    return ChallengeResult(
        resolved=False,
        challenge_type=effective_type,
        method="none",
        wait_time_ms=total_ms,
        error=auto_result.error or "Challenge not resolved",
    )


# --- Internal helpers ---

async def _extract_turnstile_sitekey(page) -> Optional[str]:
    """Extract the Turnstile sitekey from the page.

    Checks multiple sources in order:
    1. DOM elements with data-sitekey attribute (.cf-turnstile, etc.)
    2. Turnstile iframe src parameter
    3. JavaScript window.turnstile widget instances (render=explicit mode)
    4. Turnstile script URL path (Cloudflare Managed Challenge embeds the
       sitekey in the script URL as /turnstile/v0/g/{sitekey}/api.js)
    5. HTML content regex as last resort
    """
    # Method 1: DOM attributes
    dom_selectors = [
        '.cf-turnstile[data-sitekey]',
        'div[data-turnstile-sitekey]',
        'iframe[src*="challenges.cloudflare.com"]',
    ]
    for sel in dom_selectors:
        try:
            el = await page.query_selector(sel)
            if el:
                sitekey = await el.get_attribute('data-sitekey') or await el.get_attribute('data-turnstile-sitekey')
                if sitekey:
                    logger.info(f"Turnstile sitekey from DOM attribute: {sitekey}")
                    return sitekey
                # Try extracting from iframe src parameter
                src = await el.get_attribute('src')
                if src and 'sitekey=' in src:
                    key = src.split('sitekey=')[1].split('&')[0]
                    logger.info(f"Turnstile sitekey from iframe src param: {key}")
                    return key
        except Exception:
            continue

    # Method 2: JavaScript — query rendered Turnstile widget instances
    try:
        js_sitekey = await page.evaluate("""() => {
            // Check window.turnstile for rendered widgets (render=explicit mode)
            if (window.turnstile && typeof window.turnstile.getResponse === 'function') {
                // Turnstile v0 stores widgets internally
                const containers = document.querySelectorAll('[data-sitekey]');
                for (const c of containers) {
                    const key = c.getAttribute('data-sitekey');
                    if (key) return key;
                }
            }
            // Check for sitekey in any cf_chl_opt object
            if (window._cf_chl_opt && window._cf_chl_opt.cK) {
                return window._cf_chl_opt.cK;
            }
            return null;
        }""")
        if js_sitekey:
            logger.info(f"Turnstile sitekey from JS context: {js_sitekey}")
            return js_sitekey
    except Exception:
        pass

    # Method 3: Extract from Turnstile script URL path
    # Cloudflare Managed Challenges embed the sitekey in the script src:
    # https://challenges.cloudflare.com/turnstile/v0/g/{sitekey}/api.js
    try:
        script_sitekey = await page.evaluate("""() => {
            const scripts = document.querySelectorAll('script[src*="challenges.cloudflare.com/turnstile"]');
            for (const s of scripts) {
                const src = s.getAttribute('src') || '';
                const match = src.match(/\\/turnstile\\/v0\\/(?:g|i)\\/([0-9a-fA-Fx-]+)\\/api\\.js/);
                if (match) return match[1];
            }
            return null;
        }""")
        if script_sitekey:
            logger.info(f"Turnstile sitekey from script URL path: {script_sitekey}")
            return script_sitekey
    except Exception:
        pass

    # Method 4: HTML regex fallback
    try:
        import re
        html = await page.content()
        # Pattern: /turnstile/v0/g/{sitekey}/api.js
        match = re.search(r'/turnstile/v0/(?:g|i)/([0-9a-fA-Fx-]+)/api\.js', html)
        if match:
            key = match.group(1)
            logger.info(f"Turnstile sitekey from HTML regex: {key}")
            return key
        # Pattern: data-sitekey="..." anywhere in HTML
        match = re.search(r'data-sitekey=["\']([^"\']+)', html)
        if match:
            key = match.group(1)
            logger.info(f"Turnstile sitekey from HTML data-sitekey: {key}")
            return key
    except Exception:
        pass

    return None


async def _call_capsolver(
    api_key: str,
    site_url: str,
    sitekey: str,
    timeout_ms: int,
) -> Optional[str]:
    """Call CapSolver API to solve Turnstile. Returns token or None."""
    import aiohttp

    create_url = "https://api.capsolver.com/createTask"
    result_url = "https://api.capsolver.com/getTaskResult"

    payload = {
        "clientKey": api_key,
        "task": {
            "type": "AntiTurnstileTaskProxyLess",
            "websiteURL": site_url,
            "websiteKey": sitekey,
        },
    }

    try:
        async with aiohttp.ClientSession() as session:
            # Create task
            async with session.post(create_url, json=payload, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                data = await resp.json()
                if data.get("errorId", 1) != 0:
                    logger.warning(f"CapSolver create error: {data.get('errorDescription')}")
                    return None
                task_id = data.get("taskId")
                if not task_id:
                    return None

            # Poll for result
            poll_payload = {"clientKey": api_key, "taskId": task_id}
            elapsed = 0
            while elapsed < timeout_ms:
                await asyncio.sleep(3)
                elapsed += 3000
                async with session.post(result_url, json=poll_payload, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                    data = await resp.json()
                    status = data.get("status")
                    if status == "ready":
                        return data.get("solution", {}).get("token")
                    if status == "failed":
                        logger.warning(f"CapSolver task failed: {data.get('errorDescription')}")
                        return None

            logger.warning(f"CapSolver timeout after {timeout_ms}ms")
            return None

    except Exception as e:
        logger.warning(f"CapSolver error: {e}")
        return None


async def _inject_turnstile_token(page, token: str):
    """Inject the solved Turnstile token and trigger Cloudflare's callback."""
    await page.evaluate(f"""() => {{
        // Set token in known Turnstile response inputs
        const inputs = document.querySelectorAll('input[name="cf-turnstile-response"]');
        inputs.forEach(input => {{ input.value = '{token}'; }});

        const hiddenInputs = document.querySelectorAll('[name*="turnstile"]');
        hiddenInputs.forEach(input => {{ input.value = '{token}'; }});

        // Trigger Cloudflare's callback to process the token
        // Method 1: Call turnstile callback if available on the widget
        const widgets = document.querySelectorAll('.cf-turnstile, [data-turnstile-sitekey]');
        for (const w of widgets) {{
            const callbackName = w.getAttribute('data-callback');
            if (callbackName && typeof window[callbackName] === 'function') {{
                window[callbackName]('{token}');
            }}
        }}

        // Method 2: Submit the challenge form if present
        const forms = document.querySelectorAll('form[action*="challenge"]');
        if (forms.length > 0) {{
            forms[0].submit();
        }}

        // Method 3: Dispatch input event to trigger any listeners
        inputs.forEach(input => {{
            input.dispatchEvent(new Event('input', {{ bubbles: true }}));
            input.dispatchEvent(new Event('change', {{ bubbles: true }}));
        }});
    }}""")


async def _click_turnstile_checkbox(page) -> bool:
    """
    Try to click the interactive Turnstile widget checkbox.

    Cloudflare Managed Challenges with cType='interactive' render a Turnstile
    widget in a cross-domain iframe from challenges.cloudflare.com.  The widget
    contains a checkbox that must be clicked to proceed.

    Returns True if a click was successfully dispatched, False otherwise.
    """
    try:
        # Find the Turnstile iframe
        turnstile_frame = page.frame_locator(
            'iframe[src*="challenges.cloudflare.com"]'
        )

        # Try 1: Look for the standard checkbox input
        checkbox = turnstile_frame.locator('input[type="checkbox"]')
        if await checkbox.count() > 0:
            logger.info("Found Turnstile checkbox — clicking")
            await checkbox.first.click()
            return True

        # Try 2: Look for a clickable label/div (Turnstile uses various elements)
        for sel in [
            'label',
            '#challenge-stage',
            '[role="checkbox"]',
            '.ctp-checkbox-container',
            'div.widget',
        ]:
            el = turnstile_frame.locator(sel)
            if await el.count() > 0:
                logger.info(f"Found Turnstile clickable element '{sel}' — clicking")
                await el.first.click()
                return True

        # Try 3: Click the body of the iframe as last resort
        body = turnstile_frame.locator('body')
        if await body.count() > 0:
            logger.info("Clicking Turnstile iframe body as fallback")
            await body.first.click()
            return True

        logger.debug("No clickable element found in Turnstile iframe")
        return False

    except Exception as e:
        logger.debug(f"Turnstile checkbox click failed: {e}")
        return False


def _format_proxy_for_capsolver(proxy_config: Optional[dict]) -> Optional[str]:
    """
    Convert a Playwright proxy config dict to CapSolver proxy string format.

    Playwright: {"server": "http://host:port", "username": "u", "password": "p"}
    CapSolver:  "host:port:username:password"
    """
    if not proxy_config or not proxy_config.get("server"):
        return None

    username = proxy_config.get("username")
    password = proxy_config.get("password")
    if not username or not password:
        return None

    server = proxy_config["server"]
    # Strip scheme (http:// or https://)
    if "://" in server:
        server = server.split("://", 1)[1]

    return f"{server}:{username}:{password}"


async def _call_capsolver_managed(
    api_key: str,
    site_url: str,
    proxy_str: str,
    timeout_ms: int,
) -> Optional[dict]:
    """
    Call CapSolver API with AntiCloudflareTask for Managed Challenges.

    Returns solution dict with cookies on success, or None on failure.
    """
    import aiohttp

    create_url = "https://api.capsolver.com/createTask"
    result_url = "https://api.capsolver.com/getTaskResult"

    payload = {
        "clientKey": api_key,
        "task": {
            "type": "AntiCloudflareTask",
            "websiteURL": site_url,
            "proxy": proxy_str,
        },
    }

    try:
        async with aiohttp.ClientSession() as session:
            # Create task
            async with session.post(
                create_url, json=payload,
                timeout=aiohttp.ClientTimeout(total=10),
            ) as resp:
                data = await resp.json()
                if data.get("errorId", 1) != 0:
                    logger.warning(
                        f"CapSolver AntiCloudflareTask create error: "
                        f"{data.get('errorDescription')}"
                    )
                    return None
                task_id = data.get("taskId")
                if not task_id:
                    return None

            # Poll for result
            poll_payload = {"clientKey": api_key, "taskId": task_id}
            elapsed = 0
            while elapsed < timeout_ms:
                await asyncio.sleep(3)
                elapsed += 3000
                async with session.post(
                    result_url, json=poll_payload,
                    timeout=aiohttp.ClientTimeout(total=10),
                ) as resp:
                    data = await resp.json()
                    status = data.get("status")
                    if status == "ready":
                        solution = data.get("solution", {})
                        logger.info(
                            f"AntiCloudflareTask solved — "
                            f"cookies: {list(solution.get('cookies', {}).keys())}"
                        )
                        return solution
                    if status == "failed":
                        logger.warning(
                            f"AntiCloudflareTask failed: "
                            f"{data.get('errorDescription')}"
                        )
                        return None

            logger.warning(f"AntiCloudflareTask timeout after {timeout_ms}ms")
            return None

    except Exception as e:
        logger.warning(f"AntiCloudflareTask error: {e}")
        return None
