"""Stealth module: playwright-stealth patches, request interception, JS fingerprint patches."""

import logging

from app.config import settings

logger = logging.getLogger(__name__)

# Domains to block (analytics & advertising telemetry only).
#
# DO NOT add anti-bot vendors (Datadome, PerimeterX, Imperva/Incapsula, Kasada,
# Cloudflare, Akamai, queue-it). Those WAFs operate guilty-until-proven-innocent:
# if their telemetry script can't load and submit a fingerprint, the WAF defaults
# to a hard 403 on every subsequent request. We *want* their JS to run so the
# spoofed-stealth fingerprint gets evaluated and a clearance cookie is issued.
BLOCKED_DOMAINS = [
    "google-analytics.com",
    "googletagmanager.com",
    "facebook.net",
    "connect.facebook.net",
    "doubleclick.net",
    "hotjar.com",
    "clarity.ms",
    "sentry.io",
    "bugsnag.com",
    "segment.io",
    "mixpanel.com",
    "amplitude.com",
    "intercom.io",
    "drift.com",
]


async def apply_stealth(context) -> None:
    """Apply playwright-stealth patches to a browser context."""
    if not settings.stealth_enabled:
        return
    if settings.browser_engine == "camoufox":
        logger.debug("Camoufox engine: stealth is built-in, skipping playwright-stealth")
        return
    try:
        from playwright_stealth import Stealth
        stealth = Stealth()
        await stealth.apply_stealth_async(context)
        logger.debug("Applied playwright-stealth patches")
    except ImportError:
        logger.warning("playwright-stealth not installed, skipping stealth patches")
    except Exception as exc:
        logger.warning("Failed to apply stealth patches: %s", exc)


_CHROMIUM_JS_PATCHES = """
// Fix Notification.permission (headless returns 'denied', a detection signal)
try {
    Object.defineProperty(Notification, 'permission', {
        get: () => 'default',
        configurable: true
    });
} catch(e) {}

// Remove Playwright global markers
const pwGlobals = Object.getOwnPropertyNames(window).filter(
    k => k.startsWith('__playwright') || k === '__pwInitScripts'
);
for (const key of pwGlobals) {
    try { delete window[key]; } catch(e) {}
}

// WebGL renderer spoofing (hide SwiftShader/headless indicators)
try {
    const getParam = WebGLRenderingContext.prototype.getParameter;
    WebGLRenderingContext.prototype.getParameter = function(param) {
        if (param === 37445) return 'Google Inc. (Intel)';
        if (param === 37446) return 'ANGLE (Intel, Intel(R) UHD Graphics 630, OpenGL 4.1)';
        return getParam.call(this, param);
    };
} catch(e) {}

// AudioContext fingerprint noise — STATIC per session.
// Real hardware produces a stable audio fingerprint; using Math.random() on
// every call makes the fingerprint drift between samples taken milliseconds
// apart, which Datadome/Cloudflare Turnstile flag as active spoofing.
// We generate the noise vector once and reuse it for the lifetime of the page.
try {
    const origGetFloatFreqData = AnalyserNode.prototype.getFloatFrequencyData;
    let _noiseVec = null;
    AnalyserNode.prototype.getFloatFrequencyData = function(array) {
        origGetFloatFreqData.call(this, array);
        if (_noiseVec === null || _noiseVec.length !== array.length) {
            _noiseVec = new Float32Array(array.length);
            for (let i = 0; i < array.length; i++) {
                _noiseVec[i] = (Math.random() - 0.5) * 0.001;
            }
        }
        for (let i = 0; i < array.length; i++) {
            array[i] += _noiseVec[i];
        }
    };
} catch(e) {}
"""


async def apply_chromium_js_patches(page) -> None:
    """Inject JS patches to hide Chromium/Playwright detection signals.

    Skipped for Camoufox which handles stealth at C++ level.
    """
    if settings.browser_engine == "camoufox":
        return
    try:
        await page.add_init_script(_CHROMIUM_JS_PATCHES)
        logger.debug("Applied Chromium JS stealth patches")
    except Exception as exc:
        logger.warning("Failed to apply JS stealth patches: %s", exc)


async def setup_request_interception(context) -> None:
    """Register request interception to block tracking/analytics domains.

    For Camoufox (Firefox-based) with proxy: uses per-domain route patterns
    that only call ``route.abort()``.  A catch-all ``context.route("**/*", ...)``
    would require ``route.continue_()`` for non-blocked requests, which fails
    on Firefox to re-route through the proxy.  Domain-specific routes avoid
    this — unmatched requests flow through the proxy normally.
    """
    if not settings.block_tracking_domains:
        return

    if settings.browser_engine == "camoufox":
        # Per-domain routes: only abort(), never continue_()
        for domain in BLOCKED_DOMAINS:
            await context.route(
                f"**/*{domain}*",
                lambda route: route.abort(),
            )
        logger.debug("Camoufox: blocking %d tracking domains via per-domain routes", len(BLOCKED_DOMAINS))
        return

    async def _route_handler(route):
        url = route.request.url.lower()
        for domain in BLOCKED_DOMAINS:
            if domain in url:
                logger.debug("Blocked request to %s", domain)
                await route.abort()
                return
        await route.continue_()

    await context.route("**/*", _route_handler)
    logger.debug("Request interception enabled (%d blocked domains)", len(BLOCKED_DOMAINS))
