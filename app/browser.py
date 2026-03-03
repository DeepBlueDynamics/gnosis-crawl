"""
Browser automation engine for Grub Crawler
Ported from gnosis-wraith with enhanced stability and anti-detection
"""
import os
import random
import asyncio
import logging
from typing import Optional, Dict, Any
try:
    from patchright.async_api import async_playwright as async_patchright
    _HAS_PATCHRIGHT = True
except ImportError:
    _HAS_PATCHRIGHT = False
try:
    from browserforge.headers import HeaderGenerator
    _HAS_BROWSERFORGE = True
except ImportError:
    _HAS_BROWSERFORGE = False
from playwright.async_api import async_playwright, Browser, BrowserContext, Page
from PIL import Image, ImageDraw, ImageFont
from io import BytesIO

from app.config import settings
from app.stealth import apply_chromium_js_patches

logger = logging.getLogger(__name__)

# Comprehensive Chromium launch args for stealth and stability.
# Shared by BrowserEngine and BrowserPool.
CHROMIUM_STEALTH_ARGS = [
    # GPU / sandbox / stability
    '--disable-gpu',
    '--no-sandbox',
    '--disable-dev-shm-usage',
    '--disable-accelerated-2d-canvas',
    '--disable-accelerated-video-decode',
    '--disable-features=site-per-process',
    '--disable-extensions',
    '--disable-background-networking',
    '--disable-default-apps',
    '--disable-sync',
    '--disable-translate',
    '--hide-scrollbars',
    '--metrics-recording-only',
    '--mute-audio',
    '--no-first-run',
    # Automation detection
    '--disable-blink-features=AutomationControlled',
    '--disable-features=VizDisplayCompositor',
    # WebRTC leak protection
    '--webrtc-ip-handling-policy=disable_non_proxied_udp',
    '--force-webrtc-ip-handling-policy',
    # Canvas fingerprint noise
    '--fingerprinting-canvas-image-data-noise',
    # Background / throttling
    '--disable-background-timer-throttling',
    '--disable-backgrounding-occluded-windows',
    '--disable-renderer-backgrounding',
    # Crash / hang reporting (detection signals)
    '--disable-breakpad',
    '--disable-hang-monitor',
    '--disable-domain-reliability',
    '--disable-ipc-flooding-protection',
    '--disable-component-extensions-with-background-pages',
    # Consistent rendering
    '--force-color-profile=srgb',
    '--font-render-hinting=none',
    # Network service
    '--enable-features=NetworkService,NetworkServiceInProcess',
    # Pointer / hover consistency (looks like real desktop)
    '--blink-settings=primaryHoverType=2,availableHoverTypes=2,primaryPointerType=4,availablePointerTypes=4',
]


_VISIBLE_TEXT_JS = r"""
({ maxChars }) => {
  const HIDDEN_CLASSES = new Set([
    'sr-only', 'sr_only', 'srOnly',
    'visually-hidden', 'visually_hidden',
    'screen-reader-only', 'screen_reader_only',
    'a11y-only', 'a11y_only'
  ]);

  function hasHiddenClass(el) {
    if (!el || !el.classList) return false;
    for (const cls of el.classList) {
      if (HIDDEN_CLASSES.has(cls)) return true;
    }
    return false;
  }

  function isAriaHidden(el) {
    if (!el) return false;
    if (el.getAttribute && el.getAttribute('aria-hidden') === 'true') return true;
    if (el.hasAttribute && el.hasAttribute('hidden')) return true;
    return false;
  }

  function looksLikeSrOnly(style, rect) {
    if (!style || !rect) return false;
    const w = rect.width || 0;
    const h = rect.height || 0;
    const area = w * h;
    if (area > 16) return false;
    // Common sr-only recipe: clipped + overflow hidden + tiny box.
    const clip = (style.clip || '').toString();
    const clipPath = (style.clipPath || '').toString();
    const overflow = (style.overflow || '').toString();
    const whiteSpace = (style.whiteSpace || '').toString();
    const position = (style.position || '').toString();
    const offscreen = rect.right < -100 || rect.bottom < -100 || rect.left > 100000 || rect.top > 100000;
    const clipped = (clip && clip !== 'auto') || (clipPath && clipPath !== 'none');
    const tiny = w <= 2 && h <= 2;
    const absish = position === 'absolute' || position === 'fixed';
    const recipe = clipped && (overflow === 'hidden' || whiteSpace === 'nowrap') && (tiny || offscreen) && absish;
    return recipe;
  }

  function isVisibleTextNode(node) {
    if (!node) return false;
    const parent = node.parentElement;
    if (!parent) return false;

    // Quick attribute checks up the tree
    if (parent.closest && parent.closest('[aria-hidden="true"], [hidden]')) return false;
    if (hasHiddenClass(parent)) return false;

    const style = window.getComputedStyle(parent);
    if (!style) return false;
    if (style.display === 'none') return false;
    if (style.visibility === 'hidden') return false;
    if (style.opacity === '0') return false;
    if (style.fontSize === '0px') return false;

    // Bounding box heuristics to drop 1px clipped sr-only content.
    let rect;
    try {
      rect = parent.getBoundingClientRect();
    } catch (_) {
      rect = null;
    }
    if (rect) {
      if ((rect.width || 0) === 0 || (rect.height || 0) === 0) return false;
      if (looksLikeSrOnly(style, rect)) return false;
    }

    // final attribute checks
    if (isAriaHidden(parent)) return false;
    return true;
  }

  const walker = document.createTreeWalker(
    document.body || document.documentElement,
    NodeFilter.SHOW_TEXT,
    {
      acceptNode: (node) => {
        const text = (node && node.nodeValue) ? node.nodeValue : '';
        if (!text || !text.trim()) return NodeFilter.FILTER_REJECT;
        if (!isVisibleTextNode(node)) return NodeFilter.FILTER_REJECT;
        return NodeFilter.FILTER_ACCEPT;
      }
    }
  );

  let out = '';
  let node;
  while ((node = walker.nextNode())) {
    const chunk = (node.nodeValue || '').replace(/\s+/g, ' ').trim();
    if (!chunk) continue;
    if (out.length + chunk.length + 1 > maxChars) break;
    out += (out ? ' ' : '') + chunk;
  }

  const wordCount = out ? (out.match(/\b\w+\b/g) || []).length : 0;
  return {
    text: out,
    char_count: out.length,
    word_count: wordCount
  };
}
"""


def split_image_by_height(image_bytesio: BytesIO, viewport_width: int, output_format: str = 'PNG') -> list[BytesIO]:
    """
    Splits an image stored in a BytesIO object by height into segments.
    Uses reasonable proportions based on 8.5x11 aspect ratio (~2.5x viewport height).

    Args:
        image_bytesio (BytesIO): BytesIO object containing the image.
        viewport_width (int): Width of the viewport used for capture.
        output_format (str): Output format for segmented images (e.g., 'PNG', 'JPEG').

    Returns:
        List of BytesIO objects containing segmented images.
    """
    try:
        # Calculate reasonable max height based on 8.5x11 proportions
        # 8.5x11 aspect ratio: height = width * (11/8.5) = width * 1.294
        segment_height = int(viewport_width * (11/8.5))
        
        # Open the image from BytesIO
        image_bytesio.seek(0)
        image = Image.open(image_bytesio)
        
        width, height = image.size
        
        # If image is smaller than segment height, return as-is
        if height <= segment_height:
            image_bytesio.seek(0)
            return [image_bytesio]
        
        # Calculate number of segments needed
        num_segments = (height + segment_height - 1) // segment_height
        segments = []
        
        logger.info(f"Splitting {width}x{height} image into {num_segments} segments of {segment_height}px each")
        
        for i in range(num_segments):
            # Calculate segment boundaries
            top = i * segment_height
            bottom = min((i + 1) * segment_height, height)
            
            # Crop the segment
            segment = image.crop((0, top, width, bottom))
            
            # Save segment to BytesIO
            segment_bytesio = BytesIO()
            segment.save(segment_bytesio, format=output_format)
            segment_bytesio.seek(0)
            segments.append(segment_bytesio)
            
            logger.debug(f"Created segment {i + 1}/{num_segments}: {width}x{bottom - top}px")
        
        return segments
        
    except Exception as e:
        logger.error(f"Failed to split image: {e}")
        # Return original image as fallback
        image_bytesio.seek(0)
        return [image_bytesio]


class BrowserEngine:
    """Advanced browser automation with anti-detection and stability features."""
    
    def __init__(self):
        self.playwright = None
        self.browser: Optional[Browser] = None
        self.context: Optional[BrowserContext] = None
        self.page: Optional[Page] = None
        self._browser_lock = asyncio.Lock()
        self._camoufox_cm = None
        self._context_semaphore = asyncio.Semaphore(settings.max_concurrent_crawls)
        # Serialize context creation for Camoufox + residential proxy:
        # concurrent proxy auth handshakes cause HTTP 407 rejections.
        self._context_create_lock = asyncio.Lock()
        # Per-domain locks: only one crawl per domain at a time.
        # Hitting the same domain from multiple proxy IPs simultaneously
        # is an anti-bot red flag and triggers Cloudflare rate limiting.
        self._domain_locks: Dict[str, asyncio.Lock] = {}
        # Consecutive failure tracking for proxy burnout detection.
        # After N consecutive timeouts, restart browser with a fresh proxy session.
        self._consecutive_failures = 0
        self._max_failures_before_restart = settings.proxy_restart_after_failures
        # Store the proxy config for CapSolver AntiCloudflareTask (needs proxy details).
        self._proxy_config: Optional[dict] = None
        
    async def start_browser(self, javascript_enabled: bool = True) -> None:
        """Start browser with enhanced configuration and anti-detection."""
        async with self._browser_lock:
            if self.browser and not self.browser.is_connected():
                logger.info("Browser disconnected, recreating")
                await self.close()
            
            if self.browser:
                logger.debug("Browser already running")
                return
                
            try:
                if settings.browser_engine == "camoufox":
                    logger.info("Starting Camoufox browser engine")
                    from camoufox.async_api import AsyncCamoufox
                    import uuid as _uuid

                    session_id = _uuid.uuid4().hex[:12]
                    proxy = settings.get_sticky_proxy_config(session_id=session_id)
                    if proxy:
                        logger.info(f"Camoufox proxy session: {session_id} ({settings.proxy_session_duration_minutes} min sticky)")
                        self._proxy_config = proxy
                    else:
                        logger.info("No proxy configured for Camoufox")
                    headless_mode = "virtual" if settings.browser_headless else False

                    self._camoufox_cm = AsyncCamoufox(
                        headless=headless_mode,
                        proxy=proxy,
                        geoip=True,
                        os=["windows", "macos", "linux"],
                    )
                    self.browser = await self._camoufox_cm.__aenter__()
                    self.context = await self.browser.new_context()
                    self.page = await self.context.new_page()

                    logger.info("Camoufox browser started successfully")
                    await self._check_exit_ip()
                else:
                    if _HAS_PATCHRIGHT:
                        logger.info("Starting Patchright (CDP-patched) browser")
                        self.playwright = await async_patchright().start()
                    else:
                        logger.info("Starting Playwright browser (patchright not available)")
                        self.playwright = await async_playwright().start()

                    # Use shared stealth args constant
                    browser_args = list(CHROMIUM_STEALTH_ARGS)

                    # Add headless configuration
                    headless_mode = settings.browser_headless
                    if headless_mode:
                        browser_args.extend([
                            '--headless=new',
                            '--disable-notifications',
                            '--disable-infobars'
                        ])

                    logger.info(f"Launching browser (headless={headless_mode}, js_enabled={javascript_enabled})")
                    self.browser = await self.playwright.chromium.launch(
                        headless=headless_mode,
                        args=browser_args
                    )

                    # Create context with randomized fingerprint
                    viewport = self._get_random_viewport()
                    user_agent = self._get_random_user_agent()
                    timezone_id = self._get_random_timezone()
                    locale = self._get_random_locale()

                    logger.info(f"Browser context: viewport={viewport}, timezone={timezone_id}")

                    self.context = await self.browser.new_context(
                        viewport=viewport,
                        user_agent=user_agent,
                        locale=locale,
                        timezone_id=timezone_id,
                        has_touch=random.choice([True, False]),
                        java_script_enabled=javascript_enabled,
                        ignore_https_errors=True
                    )

                    # Create page with enhanced headers
                    self.page = await self.context.new_page()
                    await apply_chromium_js_patches(self.page)
                    await self._set_realistic_headers()

                    logger.info("Browser started successfully")
                    await self._check_exit_ip()

            except Exception as e:
                logger.error(f"Failed to start browser: {e}", exc_info=True)
                await self.close()
                raise
    
    async def create_isolated_context(self, javascript_enabled: bool = True, proxy=None, domain: str = None, proxy_server: str = None) -> tuple[BrowserContext, Page]:
        """Create a new isolated browser context and page for concurrent operations."""
        if not self.browser:
            logger.info("Browser not started, initializing")
            await self.start_browser(javascript_enabled=javascript_enabled)

        from app.stealth import apply_stealth, setup_request_interception, apply_chromium_js_patches

        if settings.browser_engine == "camoufox":
            # Camoufox auto-generates realistic fingerprints per context.
            # Do NOT set proxy per-context — the browser-level proxy already
            # routes all traffic.  Setting it again causes 407 auth races when
            # multiple contexts start simultaneously.
            ctx_kwargs = {}

            # If we have a cached CapSolver UA for this domain, use it.
            # cf_clearance cookies are bound to the UA that CapSolver used to
            # solve the challenge.  Reusing the same UA lets us skip CapSolver
            # on subsequent crawls to the same domain.
            if domain:
                from app.cookie_store import get_cookie_store
                cached_ua = get_cookie_store().get_capsolver_ua(domain)
                if cached_ua:
                    ctx_kwargs["user_agent"] = cached_ua
                    logger.info(f"Using cached CapSolver UA for {domain}")

            context = await self.browser.new_context(**ctx_kwargs)
            page = await context.new_page()
            # Stealth + request interception are built-in at C++ level
            await setup_request_interception(context)

            if domain:
                from app.cookie_store import get_cookie_store
                loaded = await get_cookie_store().load_into_context(context, domain, proxy_server)
                if loaded:
                    logger.info(f"[review-crawl] Loaded {loaded} cookies from store for {domain}")

            return context, page

        # Chromium path: manual fingerprinting + playwright-stealth
        context_options = {
            'viewport': {
                'width': random.randint(1200, 1920),
                'height': random.randint(800, 1080)
            },
            'user_agent': self._get_random_user_agent(),
            'locale': random.choice(['en-US', 'en-GB', 'en-CA']),
            'timezone_id': random.choice([
                'America/New_York', 'America/Los_Angeles', 'Europe/London',
                'Europe/Paris', 'Asia/Tokyo', 'Australia/Sydney'
            ]),
            'permissions': [],
            'java_script_enabled': javascript_enabled
        }

        if proxy is not None:
            context_options['proxy'] = proxy

        context = await self.browser.new_context(**context_options)
        page = await context.new_page()

        # Apply stealth patches, JS patches, and request interception
        await apply_stealth(context)
        await apply_chromium_js_patches(page)
        await setup_request_interception(context)

        if domain:
            from app.cookie_store import get_cookie_store
            loaded = await get_cookie_store().load_into_context(context, domain, proxy_server)
            if loaded:
                logger.info(f"[review-crawl] Loaded {loaded} cookies from store for {domain}")

        return context, page

    def _get_domain_lock(self, url: str) -> asyncio.Lock:
        """Get or create a per-domain lock. Only one crawl per domain at a time."""
        from urllib.parse import urlparse
        netloc = urlparse(url).netloc.lower()
        # Strip www. so www.example.com and example.com share a lock
        if netloc.startswith("www."):
            netloc = netloc[4:]
        if netloc not in self._domain_locks:
            self._domain_locks[netloc] = asyncio.Lock()
        return self._domain_locks[netloc]

    async def crawl_with_context(
        self,
        url: str,
        javascript_enabled: bool = True,
        timeout: int = 30000,
        take_screenshot: bool = False,
        javascript_payload: Optional[str] = None,
        wait_until: str = "domcontentloaded",
        wait_for_selector: Optional[str] = None,
        wait_after_load_ms: int = 1000,
        proxy=None,
        client_timeout_seconds: Optional[int] = None,
        domain: str = None,
        proxy_server: str = None,
        challenge_wait_ms: int = None,
        simulate_human: bool = False,
        scroll_count: int = 3,
        platform: str = None,
    ) -> tuple[str, dict, bytes]:
        """Crawl URL using an isolated context for concurrent operations.

        Concurrency controls (stealthiest-first):
        1. Per-domain lock — only one crawl per domain at a time.
           Hitting the same domain from multiple proxy IPs simultaneously
           is an anti-bot fingerprint and triggers Cloudflare rate limiting.
        2. Global semaphore — caps total concurrent browser contexts.
        3. Context creation lock — serializes proxy auth handshakes.
        """
        import time as _time

        domain_lock = self._get_domain_lock(url)
        if domain_lock.locked():
            logger.info(f"Domain lock active, queuing request for {url}")

        if self._context_semaphore.locked():
            logger.info(f"Browser context limit reached ({settings.max_concurrent_crawls}), queuing request for {url}")

        # Compute client deadline if X-Client-Timeout was provided
        deadline = (_time.monotonic() + client_timeout_seconds) if client_timeout_seconds else None

        async with domain_lock:
            async with self._context_semaphore:
                # Serialize context creation to avoid concurrent proxy auth
                # handshakes — residential proxies reject simultaneous 407 handshakes.
                async with self._context_create_lock:
                    context, page = await self.create_isolated_context(javascript_enabled, proxy=proxy, domain=domain, proxy_server=proxy_server)

                try:
                    original_context = context  # Track for cleanup if CapSolver swaps context
                    max_retries = 2
                    for attempt in range(max_retries):
                        # Check if client has likely given up (5s safety margin)
                        if deadline and _time.monotonic() > deadline - 5:
                            logger.warning(f"Client timeout budget exhausted for {url}, stopping retries at attempt {attempt + 1}")
                            break

                        try:
                            logger.info(f"Navigating to {url} (attempt {attempt + 1}/{max_retries})")

                            crawl_started_at = asyncio.get_running_loop().time()
                            wait_strategy = wait_until if wait_until in {"domcontentloaded", "networkidle", "selector"} else "domcontentloaded"
                            goto_wait_until = "domcontentloaded" if wait_strategy == "selector" else wait_strategy

                            navigation_started_at = asyncio.get_running_loop().time()
                            response = await page.goto(url, timeout=timeout, wait_until=goto_wait_until)
                            navigation_ms = int((asyncio.get_running_loop().time() - navigation_started_at) * 1000)

                            wait_started_at = asyncio.get_running_loop().time()
                            if wait_strategy == "selector" and wait_for_selector:
                                await page.wait_for_selector(wait_for_selector, timeout=timeout)
                            if wait_after_load_ms > 0:
                                await asyncio.sleep(wait_after_load_ms / 1000.0)
                            wait_ms = int((asyncio.get_running_loop().time() - wait_started_at) * 1000)

                            # Challenge detection + resolution (integrated from challenge_solver)
                            challenge_result = None
                            _challenge_wait = challenge_wait_ms or settings.challenge_auto_wait_ms
                            try:
                                from app.challenge_solver import resolve_challenge
                                challenge_result = await resolve_challenge(
                                    page, site_url=url,
                                    auto_wait_ms=_challenge_wait,
                                    capsolver_timeout_ms=min(settings.challenge_capsolver_timeout_ms, _challenge_wait),
                                    proxy_config=self._proxy_config,
                                )
                                if challenge_result.resolved and challenge_result.method != "none":
                                    logger.info(f"Challenge resolved via {challenge_result.method} in {challenge_result.wait_time_ms}ms")
                                    # If CapSolver returned a new page with matched UA, use it
                                    if hasattr(challenge_result, '_new_page') and challenge_result._new_page:
                                        logger.info("Switching to CapSolver UA-matched page")
                                        try:
                                            if not page.is_closed():
                                                await page.close()
                                        except Exception:
                                            pass
                                        page = challenge_result._new_page
                                        context = challenge_result._new_context
                                    else:
                                        try:
                                            await page.wait_for_load_state("domcontentloaded", timeout=10000)
                                        except Exception:
                                            await asyncio.sleep(3)
                                    wait_ms += challenge_result.wait_time_ms
                            except Exception as e:
                                logger.debug(f"Challenge resolution skipped: {e}")

                            logger.info(f"Successfully navigated to {url}")
                            self._consecutive_failures = 0

                            # Human behavior simulation before content extraction
                            if simulate_human:
                                try:
                                    from app.human_behavior import human_scroll, simulate_mouse_movement
                                    from app.behavior_profile import BehaviorProfile
                                    profile = BehaviorProfile.random()
                                    await simulate_mouse_movement(page, profile=profile)
                                    await human_scroll(page, scroll_count=scroll_count, platform=platform, profile=profile)
                                except Exception as hb_err:
                                    logger.debug(f"Human behavior simulation failed (non-fatal): {hb_err}")

                            # Save cookies after successful navigation
                            if domain:
                                from app.cookie_store import get_cookie_store
                                await get_cookie_store().save_from_context(context, domain, proxy_server)

                            # Execute user-provided JavaScript payload before capturing HTML
                            if javascript_enabled and javascript_payload:
                                try:
                                    logger.info("Executing custom JavaScript payload")
                                    await page.evaluate(
                                        """
                                        async payload => {
                                            const executor = new Function("return (async () => { " + payload + "\\n})();");
                                            try {
                                                return await executor();
                                            } catch (error) {
                                                console.error("Injected JavaScript payload failed", error);
                                                throw error;
                                            }
                                        }
                                        """,
                                        javascript_payload
                                    )
                                    # Give DOM a brief moment to settle after script execution
                                    await asyncio.sleep(0.5)
                                    wait_ms += 500
                                except Exception as e:
                                    logger.warning(f"JavaScript payload execution failed: {e}")

                            # Get page content
                            content_started_at = asyncio.get_running_loop().time()
                            content = await page.content()
                            content_ms = int((asyncio.get_running_loop().time() - content_started_at) * 1000)
                            logger.debug(f"Retrieved content ({len(content)} characters)")

                            # Best-effort visible text capture for hidden-text prompt injection detection.
                            visible_started_at = asyncio.get_running_loop().time()
                            visible_payload = None
                            try:
                                visible_payload = await page.evaluate(_VISIBLE_TEXT_JS, {"maxChars": 20000})
                            except Exception as e:
                                logger.debug(f"Visible text capture failed: {e}")
                            visible_ms = int((asyncio.get_running_loop().time() - visible_started_at) * 1000)

                            # Get page info
                            page_info = {
                                "title": await page.title(),
                                "url": page.url,
                                "status_code": response.status if response else None,
                                "content_type": "text/html",
                                "content_length": len(content),
                                "render_mode": "js_rendered" if javascript_enabled else "html_only",
                                "wait_strategy": wait_strategy,
                                "challenge_detected": bool(challenge_result and challenge_result.challenge_type.value != "none"),
                                "challenge_resolved": bool(challenge_result and challenge_result.resolved),
                                "challenge_method": challenge_result.method if challenge_result else None,
                                "challenge_wait_ms": challenge_result.wait_time_ms if challenge_result else 0,
                                "timings_ms": {
                                    "navigation_ms": navigation_ms,
                                    "wait_ms": wait_ms,
                                    "content_ms": content_ms,
                                    "visible_text_ms": visible_ms,
                                    "total_ms": int((asyncio.get_running_loop().time() - crawl_started_at) * 1000)
                                }
                            }

                            if isinstance(visible_payload, dict):
                                # Private/internal field; caller must NOT expose the raw visible text.
                                page_info["_visible_text"] = visible_payload.get("text") or ""
                                page_info["visible_char_count"] = int(visible_payload.get("char_count") or 0)
                                page_info["visible_word_count"] = int(visible_payload.get("word_count") or 0)

                            # Take screenshot if requested
                            screenshot_data = None
                            if take_screenshot:
                                try:
                                    # Capture full page screenshot
                                    raw_screenshot = await page.screenshot(full_page=True)
                                    logger.debug(f"Screenshot captured ({len(raw_screenshot)} bytes)")

                                    # Split image if it's very long (using viewport-proportional height)
                                    screenshot_bytesio = BytesIO(raw_screenshot)

                                    # Get viewport width from context
                                    viewport = page.viewport_size
                                    viewport_width = viewport['width'] if viewport else 1429

                                    screenshot_segments = split_image_by_height(screenshot_bytesio, viewport_width)

                                    if len(screenshot_segments) > 1:
                                        logger.info(f"Screenshot split into {len(screenshot_segments)} segments")
                                        # Return all segments as a list
                                        screenshot_data = [segment.getvalue() for segment in screenshot_segments]
                                    else:
                                        # Single image - return as bytes
                                        screenshot_data = raw_screenshot

                                except Exception as e:
                                    logger.warning(f"Failed to take screenshot: {e}")
                                    screenshot_data = None

                            # Track proxy health on success
                            try:
                                from app.proxy_pool import get_proxy_pool
                                from urllib.parse import urlparse as _urlparse
                                _pool = get_proxy_pool()
                                _domain = _urlparse(url).netloc.lower()
                                if _domain.startswith("www."):
                                    _domain = _domain[4:]
                                _pool.mark_success(_domain)
                            except Exception:
                                pass

                            return content, page_info, screenshot_data

                        except Exception as e:
                            is_timeout = "Timeout" in type(e).__name__ or "timeout" in str(e).lower()
                            is_proxy_error = "NS_ERROR_PROXY" in str(e)

                            # Proxy errors (BAD_GATEWAY, CONNECTION_REFUSED) mean the
                            # proxy session is dead.  Restart immediately — no point
                            # retrying with the same burned proxy.
                            if is_proxy_error:
                                logger.warning(f"Proxy error detected: {e}")
                                self._consecutive_failures += 1

                                # Track proxy health on failure
                                try:
                                    from app.proxy_pool import get_proxy_pool
                                    from urllib.parse import urlparse as _urlparse
                                    _pool = get_proxy_pool()
                                    _domain = _urlparse(url).netloc.lower()
                                    if _domain.startswith("www."):
                                        _domain = _domain[4:]
                                    _pool.mark_failed(_domain)
                                except Exception:
                                    pass

                                await self._restart_with_fresh_proxy()
                                raise  # Let caller know — context is stale, cannot retry here

                            # On navigation timeout, the page may have loaded a Cloudflare challenge.
                            # Try to detect + solve it before giving up.
                            if is_timeout:
                                logger.info(f"Navigation timed out on attempt {attempt + 1}, checking for Cloudflare challenge on partial page")
                                try:
                                    from app.challenge_solver import resolve_challenge
                                    _challenge_wait = challenge_wait_ms or settings.challenge_auto_wait_ms
                                    challenge_result = await resolve_challenge(
                                        page, site_url=url,
                                        auto_wait_ms=_challenge_wait,
                                        proxy_config=self._proxy_config,
                                    )
                                    if challenge_result.resolved and challenge_result.method != "none":
                                        logger.info(f"Challenge resolved after timeout via {challenge_result.method} in {challenge_result.wait_time_ms}ms — extracting content")
                                        # Challenge solved — page should now have real content. Wait for DOM to settle.
                                        try:
                                            await page.wait_for_load_state("domcontentloaded", timeout=15000)
                                        except Exception:
                                            await asyncio.sleep(3)

                                        # Extract content from the now-loaded page (same as happy path)
                                        navigation_ms = int((asyncio.get_running_loop().time() - navigation_started_at) * 1000)
                                        content = await page.content()
                                        page_info = {
                                            "title": await page.title(),
                                            "url": page.url,
                                            "status_code": None,
                                            "content_type": "text/html",
                                            "content_length": len(content),
                                            "render_mode": "js_rendered" if javascript_enabled else "html_only",
                                            "wait_strategy": wait_strategy,
                                            "challenge_detected": True,
                                            "challenge_resolved": True,
                                            "challenge_method": challenge_result.method,
                                            "challenge_wait_ms": challenge_result.wait_time_ms,
                                            "timings_ms": {
                                                "navigation_ms": navigation_ms,
                                                "wait_ms": challenge_result.wait_time_ms,
                                                "content_ms": 0,
                                                "visible_text_ms": 0,
                                                "total_ms": int((asyncio.get_running_loop().time() - crawl_started_at) * 1000)
                                            }
                                        }

                                        # Save cookies (esp. cf_clearance) after challenge resolution
                                        if domain:
                                            try:
                                                from app.cookie_store import get_cookie_store
                                                await get_cookie_store().save_from_context(context, domain, proxy_server)
                                            except Exception:
                                                pass

                                        screenshot_data = None
                                        if take_screenshot:
                                            try:
                                                screenshot_data = await page.screenshot(full_page=True)
                                            except Exception:
                                                pass

                                        return content, page_info, screenshot_data
                                except Exception as challenge_err:
                                    logger.debug(f"Post-timeout challenge resolution failed: {challenge_err}")

                            # Track proxy health on failure
                            try:
                                from app.proxy_pool import get_proxy_pool
                                from urllib.parse import urlparse as _urlparse
                                _pool = get_proxy_pool()
                                _domain = _urlparse(url).netloc.lower()
                                if _domain.startswith("www."):
                                    _domain = _domain[4:]
                                _pool.mark_failed(_domain)
                            except Exception:
                                pass

                            # Track consecutive failures for proxy burnout detection
                            if is_timeout:
                                self._consecutive_failures += 1
                                if self._consecutive_failures >= self._max_failures_before_restart:
                                    await self._restart_with_fresh_proxy()

                            logger.warning(f"Navigation error on attempt {attempt + 1}: {e}")
                            if attempt == max_retries - 1:
                                logger.error(f"Failed to navigate to {url} after {max_retries} attempts")
                                raise

                            # Wait before retry
                            await asyncio.sleep(1)

                    # If we exit the for loop without returning (e.g. deadline break),
                    # raise so caller doesn't get None
                    raise TimeoutError(f"Navigation to {url} exhausted all retries or deadline")

                finally:
                    # Always cleanup the isolated context(s).
                    # If CapSolver created a UA-matched context, close both
                    # the original and the new one.
                    for ctx in {original_context, context}:
                        try:
                            await ctx.close()
                        except Exception as e:
                            logger.warning(f"Error closing context: {e}")

    async def navigate(self, url: str, javascript_enabled: bool = True, timeout: int = 30000) -> None:
        """Navigate to URL with enhanced error handling and retry logic."""
        if not self.page:
            logger.info("Browser not started, initializing")
            await self.start_browser(javascript_enabled=javascript_enabled)
        
        max_retries = 2
        retry_count = 0
        
        while retry_count <= max_retries:
            try:
                logger.info(f"Navigating to {url} (attempt {retry_count + 1}/{max_retries + 1})")
                
                # Navigate with DOM content loaded strategy
                await self.page.goto(url, timeout=timeout, wait_until='domcontentloaded')
                
                # Wait for render stability
                logger.info("Waiting for render stability")
                await self._wait_for_render_stability(javascript_enabled)
                
                if javascript_enabled:
                    await asyncio.sleep(3)  # Additional JS execution time
                
                logger.info(f"Successfully navigated to {url}")
                return
                
            except Exception as e:
                retry_count += 1
                error_message = str(e)
                logger.warning(f"Navigation error on attempt {retry_count}: {error_message}")
                
                # Handle page crashes
                if "Page crashed" in error_message or "Target closed" in error_message:
                    logger.info("Page crash detected, recreating page")
                    try:
                        if self.page and not self.page.is_closed():
                            await self.page.close()
                        self.page = await self.context.new_page()
                        await self._set_realistic_headers()
                        logger.info("Created new page after crash")
                    except Exception as recovery_error:
                        logger.error(f"Error during page recreation: {recovery_error}")
                
                if retry_count > max_retries:
                    logger.error(f"Failed to navigate to {url} after {max_retries + 1} attempts")
                    raise
                
                await asyncio.sleep(2)  # Wait before retry
    
    async def get_content(self) -> str:
        """Get page HTML content."""
        if not self.page:
            raise Exception("Browser not started or page not available")
        
        try:
            content = await self.page.content()
            logger.debug(f"Retrieved content ({len(content)} characters)")
            return content
        except Exception as e:
            logger.error(f"Error getting page content: {e}")
            raise
    
    async def get_page_info(self) -> Dict[str, Any]:
        """Get comprehensive page information."""
        if not self.page:
            raise Exception("Browser not started or page not available")
        
        try:
            # Get basic page info
            title = await self.page.title()
            url = self.page.url
            
            # Get page metrics
            metrics = await self.page.evaluate("""() => {
                return {
                    elements: document.querySelectorAll('*').length,
                    links: document.querySelectorAll('a').length,
                    images: document.querySelectorAll('img').length,
                    scripts: document.querySelectorAll('script').length,
                    readyState: document.readyState,
                    contentLength: document.documentElement.innerHTML.length
                };
            }""")
            
            return {
                "title": title,
                "url": url,
                "metrics": metrics
            }
            
        except Exception as e:
            logger.error(f"Error getting page info: {e}")
            return {"title": "", "url": "", "metrics": {}}
    
    async def take_screenshot(self, path: str, mode: str = "full") -> bool:
        """Take screenshot with error handling and fallback."""
        if not self.page:
            raise Exception("Browser not started or page not available")
        
        try:
            if mode == "top":
                await self.page.screenshot(path=path, full_page=False)
                logger.info(f"Top viewport screenshot saved to {path}")
            elif mode == "full":
                await self.page.screenshot(path=path, full_page=True)
                logger.info(f"Full page screenshot saved to {path}")
            else:
                logger.info(f"Screenshot skipped (mode: {mode})")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error taking screenshot: {e}")
            
            # Create error image as fallback
            try:
                self._create_error_image(path, str(e))
                logger.info(f"Created error image at {path}")
                return False
            except Exception as img_error:
                logger.error(f"Error creating error image: {img_error}")
                raise e
    
    async def execute_javascript(self, script: str) -> Any:
        """Execute JavaScript in the page context."""
        if not self.page:
            raise Exception("Browser not started or page not available")
        
        try:
            result = await self.page.evaluate(script)
            logger.debug(f"Executed JavaScript: {script[:100]}...")
            return result
        except Exception as e:
            logger.error(f"Error executing JavaScript: {e}")
            raise
    
    async def close(self) -> None:
        """Close browser and release resources."""
        async with self._browser_lock:
            try:
                if self.page and not self.page.is_closed():
                    await self.page.close()
                    self.page = None

                if self.context:
                    await self.context.close()
                    self.context = None

                if self.browser:
                    await self.browser.close()
                    self.browser = None

                if self._camoufox_cm:
                    await self._camoufox_cm.__aexit__(None, None, None)
                    self._camoufox_cm = None

                if self.playwright:
                    await self.playwright.stop()
                    self.playwright = None

                logger.info("Browser closed successfully")

            except Exception as e:
                logger.error(f"Error closing browser: {e}")
    
    async def _check_exit_ip(self):
        """Check and log the proxy exit IP address.

        Opens a disposable page, hits an IP-check service, logs the result,
        and always closes the page — even on failure.  Non-fatal: if the
        check fails the browser continues normally.
        """
        if not self.browser:
            return
        _ip_page = None
        try:
            _ctx = await self.browser.new_context()
            _ip_page = await _ctx.new_page()
            resp = await _ip_page.goto(
                "https://httpbin.org/ip", timeout=10000, wait_until="domcontentloaded"
            )
            if resp and resp.ok:
                body = await _ip_page.inner_text("body")
                try:
                    import json as _json
                    ip_addr = _json.loads(body).get("origin", body.strip())
                except Exception:
                    ip_addr = body.strip().replace("\n", " ")
                logger.info(f"Proxy exit IP: {ip_addr}")
            else:
                logger.warning(f"IP check returned HTTP {resp.status if resp else 'no response'}")
        except Exception as e:
            logger.warning(f"IP check failed (non-fatal): {e}")
        finally:
            try:
                if _ip_page and not _ip_page.is_closed():
                    await _ip_page.close()
                if _ctx:
                    await _ctx.close()
            except Exception:
                pass

    async def _restart_with_fresh_proxy(self):
        """Close browser and start fresh one with new proxy session."""
        logger.warning("Restarting browser with fresh proxy session due to consecutive failures")
        await self.close()
        await self.start_browser()
        self._consecutive_failures = 0
        # IP check already runs inside start_browser(), no need to call again

    async def _wait_for_render_stability(self, javascript_enabled: bool = False) -> bool:
        """Wait for page render stability by monitoring DOM changes."""
        if not javascript_enabled:
            await asyncio.sleep(1)
            return True
        
        try:
            await asyncio.sleep(1)  # Initial delay
            
            previous_count = -1
            stable_checks = 0
            max_checks = 10
            check_interval = 0.5
            
            for i in range(max_checks):
                current_count = await self.page.evaluate("""() => {
                    return document.querySelectorAll('*').length;
                }""")
                
                logger.debug(f"DOM stability check {i+1}/{max_checks}: {current_count} elements")
                
                if current_count == previous_count and current_count > 10:
                    stable_checks += 1
                    if stable_checks >= 2:
                        logger.info(f"DOM appears stable after {i+1} checks")
                        break
                else:
                    stable_checks = 0
                
                previous_count = current_count
                await asyncio.sleep(check_interval)
            
            # Additional wait for animations
            await asyncio.sleep(2)
            
            # Verify content exists
            content_check = await self.page.evaluate("""() => {
                const mainContent = document.querySelector('main') || 
                                    document.querySelector('#content') || 
                                    document.querySelector('.content') ||
                                    document.querySelector('article') ||
                                    document.querySelector('body');
                                    
                return mainContent ? mainContent.querySelectorAll('*').length : 0;
            }""")
            
            logger.info(f"Content elements found: {content_check}")
            
            if content_check < 5:
                logger.warning("Few content elements detected, waiting longer")
                await asyncio.sleep(3)
            
            return True
            
        except Exception as e:
            logger.warning(f"Error during render stability check: {e}")
            await asyncio.sleep(3)
            return False
    
    async def _set_realistic_headers(self) -> None:
        """Set realistic browser headers using browserforge when available."""
        is_camoufox = settings.browser_engine == "camoufox"
        # Sec-CH-UA* headers are Chrome Client Hints -- Firefox/Camoufox never sends them
        chrome_only_prefixes = ('Sec-CH-UA',)

        if _HAS_BROWSERFORGE:
            try:
                browser_type = 'firefox' if is_camoufox else 'chrome'
                gen = HeaderGenerator(browser=browser_type)
                headers = gen.generate()
                request_headers = {}
                for key in ['Accept', 'Accept-Language', 'Accept-Encoding',
                            'Sec-CH-UA', 'Sec-CH-UA-Mobile', 'Sec-CH-UA-Platform',
                            'Sec-Fetch-Dest', 'Sec-Fetch-Mode', 'Sec-Fetch-Site',
                            'Sec-Fetch-User', 'Upgrade-Insecure-Requests']:
                    if is_camoufox and key.startswith(chrome_only_prefixes):
                        continue
                    for hkey in [key, key.lower()]:
                        if hkey in headers:
                            request_headers[key] = headers[hkey]
                            break
                request_headers.setdefault('Referer', 'https://www.google.com/')
                request_headers.setdefault('Connection', 'keep-alive')
                await self.page.set_extra_http_headers(request_headers)
                return
            except Exception:
                pass  # fall through to static headers

        static_headers = {
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Sec-Fetch-Dest': 'document',
            'Sec-Fetch-Mode': 'navigate',
            'Sec-Fetch-Site': 'none',
            'Sec-Fetch-User': '?1',
            'Referer': 'https://www.google.com/',
        }
        if is_camoufox:
            static_headers = {k: v for k, v in static_headers.items()
                              if not k.startswith(chrome_only_prefixes)}
        await self.page.set_extra_http_headers(static_headers)
    
    def _get_random_user_agent(self) -> str:
        """Return a random, realistic user agent string."""
        if _HAS_BROWSERFORGE:
            try:
                gen = HeaderGenerator(browser='chrome')
                headers = gen.generate()
                ua = headers.get('User-Agent') or headers.get('user-agent')
                if ua:
                    return ua
            except Exception:
                pass  # fall through to manual

        chrome_versions = [
            '133.0.6917.92', '133.0.6917.127', '134.0.6944.59',
            '134.0.6944.85', '135.0.6972.61', '135.0.6972.108',
        ]

        os_versions = [
            ('Windows NT 10.0; Win64; x64', 'Windows 10/11'),
            ('Macintosh; Intel Mac OS X 10_15_7', 'macOS Sonoma'),
            ('Macintosh; Intel Mac OS X 14_5', 'macOS Sequoia'),
            ('X11; Linux x86_64', 'Linux'),
        ]

        chrome_version = random.choice(chrome_versions)
        os_info, _ = random.choice(os_versions)

        return f'Mozilla/5.0 ({os_info}) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/{chrome_version} Safari/537.36'
    
    def _get_random_viewport(self) -> Dict[str, int]:
        """Return a random viewport size based on common screen resolutions."""
        common_resolutions = [
            {'width': 1366, 'height': 768},
            {'width': 1920, 'height': 1080},
            {'width': 1536, 'height': 864},
            {'width': 1440, 'height': 900},
            {'width': 1280, 'height': 800},
            {'width': 1680, 'height': 1050},
        ]
        
        selected = random.choice(common_resolutions)
        jitter = random.randint(-10, 10)
        
        return {
            'width': max(800, selected['width'] + jitter),
            'height': max(600, selected['height'] + jitter)
        }
    
    def _get_random_timezone(self) -> str:
        """Return a random timezone."""
        timezones = [
            'America/New_York', 'America/Chicago', 'America/Los_Angeles',
            'Europe/London', 'Europe/Paris', 'Asia/Tokyo'
        ]
        return random.choice(timezones)
    
    def _get_random_locale(self) -> str:
        """Return a random locale."""
        locales = ['en-US', 'en-GB', 'en-CA', 'fr-FR', 'de-DE']
        return random.choice(locales)
    
    def _create_error_image(self, path: str, error_message: str) -> None:
        """Create an error image when screenshot fails."""
        img = Image.new('RGB', (1280, 800), color=(240, 240, 240))
        draw = ImageDraw.Draw(img)
        
        try:
            font = ImageFont.truetype("Arial", 20)
        except:
            font = ImageFont.load_default()
        
        draw.text((50, 50), "Error Capturing Page", fill=(255, 0, 0), font=font)
        draw.text((50, 100), f"Error: {error_message}", fill=(0, 0, 0), font=font)
        
        if self.page:
            draw.text((50, 150), f"URL: {self.page.url}", fill=(0, 0, 0), font=font)
        
        img.save(path)


# Global browser instance
_browser_engine = None

async def get_browser_engine() -> BrowserEngine:
    """Get or create the global browser engine instance."""
    global _browser_engine
    if _browser_engine is None:
        _browser_engine = BrowserEngine()
    return _browser_engine

async def cleanup_browser():
    """Cleanup the global browser instance."""
    global _browser_engine
    if _browser_engine:
        await _browser_engine.close()
        _browser_engine = None
