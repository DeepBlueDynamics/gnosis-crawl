"""
Core crawler orchestration for Grub Crawler
Combines browser automation with markdown generation
"""
import asyncio
import hashlib
import logging
import time
import mimetypes
import re
from typing import Dict, Any, Optional, List
from urllib.parse import urlparse, urljoin, unquote
from pathlib import Path

import httpx

from app.browser import get_browser_engine, cleanup_browser
from app.markdown import MarkdownGenerator, ContentFilter
from app.storage import CrawlStorageService
from app.config import settings
from app import __version__
from app.policy.injection import analyze_hidden_prompt_injection

logger = logging.getLogger(__name__)


class CrawlResult:
    """Result of a single URL crawl operation."""
    
    def __init__(self, url: str, success: bool = False):
        self.url = url
        self.success = success
        self.timestamp = time.time()
        
        # Content
        self.html = ""
        self.markdown = ""
        self.markdown_plain = ""
        self.content = ""
        self.title = ""
        self.status_code = None
        self.final_url = ""

        # Metadata
        self.page_info = {}
        self.screenshot_path = ""
        self.error_message = ""
        self.crawl_options = {}
        self.render_mode = "html_only"
        self.wait_strategy = "domcontentloaded"
        self.timings_ms: Dict[str, int] = {}
        self.blocked = False
        self.block_reason = ""
        self.captcha_detected = False
        self.http_error_family = ""
        self.body_char_count = 0
        self.body_word_count = 0
        self.content_quality = "empty"
        self.extractor_version = f"grub-crawl/{__version__}"
        self.normalized_url = ""
        self.content_hash = ""
        self.policy_flags: List[str] = []
        self.quarantined: bool = False
        self.quarantine_reason: str = ""
        self.visible_char_count: int = 0
        self.visible_word_count: int = 0
        self.visible_similarity: Optional[float] = None

        # Processing info
        self.processing_time = 0.0
        self.browser_time = 0.0
        self.markdown_time = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        return {
            "url": self.url,
            "success": self.success,
            "timestamp": self.timestamp,
            "html": self.html,
            "markdown": self.markdown,
            "markdown_plain": self.markdown_plain,
            "content": self.content,
            "title": self.title,
            "status_code": self.status_code,
            "final_url": self.final_url,
            "page_info": self.page_info,
            "screenshot_path": self.screenshot_path,
            "error_message": self.error_message,
            "crawl_options": self.crawl_options,
            "render_mode": self.render_mode,
            "wait_strategy": self.wait_strategy,
            "timings_ms": self.timings_ms,
            "blocked": self.blocked,
            "block_reason": self.block_reason,
            "captcha_detected": self.captcha_detected,
            "http_error_family": self.http_error_family,
            "body_char_count": self.body_char_count,
            "body_word_count": self.body_word_count,
            "content_quality": self.content_quality,
            "extractor_version": self.extractor_version,
            "normalized_url": self.normalized_url,
            "content_hash": self.content_hash,
            "policy_flags": self.policy_flags,
            "quarantined": self.quarantined,
            "quarantine_reason": self.quarantine_reason,
            "visible_char_count": self.visible_char_count,
            "visible_word_count": self.visible_word_count,
            "visible_similarity": self.visible_similarity,
            "processing_time": self.processing_time,
            "browser_time": self.browser_time,
            "markdown_time": self.markdown_time
        }


class CrawlerEngine:
    """Main crawler engine that orchestrates browser and markdown generation."""
    
    def __init__(self, user_email: Optional[str] = None):
        self.user_email = user_email
        self.storage = CrawlStorageService(user_email)
        self.markdown_generator = MarkdownGenerator(ContentFilter())
        self._crawl_lock = asyncio.Lock()
    
    async def crawl_url(
        self,
        url: str,
        javascript: bool = True,
        screenshot: bool = False,
        screenshot_mode: str = "full",
        timeout: int = None,
        session_id: Optional[str] = None,
        javascript_payload: Optional[str] = None,
        dedupe_tables: bool = True,
        wait_until: str = "domcontentloaded",
        wait_for_selector: Optional[str] = None,
        wait_after_load_ms: int = 1000,
        retry_with_js_if_thin: bool = False,
        proxy=None,
        client_timeout_seconds: Optional[int] = None,
        domain: str = None,
        proxy_server: str = None
    ) -> CrawlResult:
        """
        Crawl a single URL and return comprehensive results.
        
        Args:
            url: URL to crawl
            javascript: Enable JavaScript execution
            screenshot: Take screenshot
            screenshot_mode: Screenshot mode ("full", "top", "off")
            timeout: Browser timeout in milliseconds
            session_id: Session ID for storage organization
            
        Returns:
            CrawlResult with all crawl data
        """
        start_time = time.time()
        result = CrawlResult(url)
        result.crawl_options = {
            "javascript": javascript,
            "screenshot": screenshot,
            "screenshot_mode": screenshot_mode,
            "timeout": timeout or settings.browser_timeout,
            "dedupe_tables": dedupe_tables,
            "wait_until": wait_until,
            "wait_for_selector": wait_for_selector,
            "wait_after_load_ms": wait_after_load_ms,
            "retry_with_js_if_thin": retry_with_js_if_thin
        }

        try:
            # Validate URL
            parsed_url = urlparse(url)
            if not parsed_url.scheme or not parsed_url.netloc:
                raise ValueError(f"Invalid URL: {url}")

            # Auto-derive domain for cookie store / CapSolver UA caching
            if not domain:
                domain = parsed_url.netloc.lower()
                if domain.startswith("www."):
                    domain = domain[4:]

            # Get browser engine and crawl with isolated context
            browser = await get_browser_engine()

            # Browser operations with isolated context (no lock needed)
            browser_start = time.time()
            timeout_ms = (timeout * 1000) if timeout else settings.browser_timeout
            take_screenshot = screenshot and screenshot_mode != "off"

            async def run_capture(javascript_enabled: bool):
                return await browser.crawl_with_context(
                    url,
                    javascript_enabled=javascript_enabled,
                    timeout=timeout_ms,
                    take_screenshot=take_screenshot,
                    javascript_payload=javascript_payload,
                    wait_until=wait_until,
                    wait_for_selector=wait_for_selector,
                    wait_after_load_ms=wait_after_load_ms,
                    proxy=proxy,
                    client_timeout_seconds=client_timeout_seconds,
                    domain=domain,
                    proxy_server=proxy_server
                )

            result.html, result.page_info, screenshot_data = await run_capture(javascript)
            self._populate_result_metadata(result)
            self._populate_content_fields(result, url, dedupe_tables=dedupe_tables)

            # Retry with JavaScript when initial response is too thin.
            if retry_with_js_if_thin and (not javascript) and result.content_quality in {"empty", "minimal"}:
                result.page_info["retried_with_js"] = True
                retry_html, retry_page_info, retry_screenshot_data = await run_capture(True)

                retry_result = CrawlResult(url)
                retry_result.html = retry_html
                retry_result.page_info = retry_page_info or {}
                self._populate_result_metadata(retry_result)
                self._populate_content_fields(retry_result, url, dedupe_tables=dedupe_tables)

                if retry_result.content_quality == "sufficient" or (
                    retry_result.body_word_count > result.body_word_count
                ):
                    result.html = retry_result.html
                    result.page_info = retry_result.page_info
                    result.markdown = retry_result.markdown
                    result.markdown_plain = retry_result.markdown_plain
                    result.content = retry_result.content
                    result.title = retry_result.title
                    result.status_code = retry_result.status_code
                    result.final_url = retry_result.final_url
                    result.render_mode = retry_result.render_mode
                    result.wait_strategy = retry_result.wait_strategy
                    result.timings_ms = retry_result.timings_ms
                    result.blocked = retry_result.blocked
                    result.block_reason = retry_result.block_reason
                    result.captcha_detected = retry_result.captcha_detected
                    result.http_error_family = retry_result.http_error_family
                    result.body_char_count = retry_result.body_char_count
                    result.body_word_count = retry_result.body_word_count
                    result.content_quality = retry_result.content_quality
                    result.normalized_url = retry_result.normalized_url
                    result.content_hash = retry_result.content_hash
                    screenshot_data = retry_screenshot_data

            result.browser_time = time.time() - browser_start
            result.markdown_time = result.timings_ms.get("markdown_ms", 0) / 1000.0

            # Handle screenshot deterministically when requested
            if take_screenshot:
                if screenshot_data and session_id:
                    screenshot_paths = await self._save_screenshot_data(screenshot_data, url, session_id)
                    result.screenshot_path = screenshot_paths
                elif screenshot_data and not session_id:
                    result.screenshot_path = "inline_screenshot"
                else:
                    result.screenshot_path = ""

            # Save results to storage
            if session_id:
                await self._save_crawl_result(result, session_id)
            
            result.success = True
            result.processing_time = time.time() - start_time
            
            logger.info(f"Successfully crawled {url} in {result.processing_time:.2f}s")
            
        except Exception as e:
            result.error_message = str(e)
            result.processing_time = time.time() - start_time
            logger.error(f"Failed to crawl {url}: {e}", exc_info=True)
        
        return result

    def _populate_result_metadata(self, result: CrawlResult) -> None:
        page_info = result.page_info or {}
        result.title = page_info.get("title", "")
        result.final_url = page_info.get("url", result.url)
        result.status_code = page_info.get("status_code")
        result.render_mode = page_info.get("render_mode", "html_only")
        result.wait_strategy = page_info.get("wait_strategy", "domcontentloaded")
        result.timings_ms = page_info.get("timings_ms", {}) or {}
        result.http_error_family = self._http_error_family(result.status_code)
        result.normalized_url = self._normalize_url(result.final_url or result.url)

    def _populate_content_fields(self, result: CrawlResult, source_url: str, dedupe_tables: bool = True) -> None:
        markdown_start = time.time()
        markdown_result = self.markdown_generator.generate_markdown(
            result.html,
            source_url,
            dedupe_tables=dedupe_tables
        )
        result.markdown = markdown_result.clean_markdown
        result.markdown_plain = markdown_result.markdown_plain or markdown_result.clean_markdown
        result.content = result.markdown_plain or result.markdown
        result.timings_ms["markdown_ms"] = int((time.time() - markdown_start) * 1000)

        # Hidden-text prompt injection detection (best-effort).
        visible_text = ""
        page_info = result.page_info or {}
        if isinstance(page_info, dict):
            visible_text = page_info.pop("_visible_text", "") or ""
            # Keep only metrics, never the raw visible text.
            result.visible_char_count = int(page_info.get("visible_char_count") or 0)
            result.visible_word_count = int(page_info.get("visible_word_count") or 0)

        result.body_char_count = len((result.content or "").strip())
        result.body_word_count = len(re.findall(r"\b\w+\b", result.content or ""))
        result.blocked, result.block_reason, result.captcha_detected = self._detect_block_signals(
            result.html,
            result.markdown,
            result.status_code
        )
        result.content_quality = self._classify_content_quality(
            body_char_count=result.body_char_count,
            body_word_count=result.body_word_count,
            blocked=result.blocked,
            status_code=result.status_code,
            content=result.content
        )

        analysis = analyze_hidden_prompt_injection(
            extracted_text=result.content or "",
            visible_text=visible_text,
        )
        if analysis.flags:
            result.policy_flags.extend(analysis.flags)
        result.visible_similarity = analysis.visible_similarity

        if analysis.quarantined:
            result.quarantined = True
            result.quarantine_reason = analysis.quarantine_reason or "quarantined"
            # Never allow quarantined content to be treated as sufficient.
            result.content_quality = "minimal"

            # Fail-closed: do not return potentially malicious hidden instructions.
            result.markdown = ""
            result.markdown_plain = ""
            result.content = ""
            result.body_char_count = 0
            result.body_word_count = 0

        result.content_hash = hashlib.sha256((result.content or "").encode("utf-8")).hexdigest() if result.content else ""

    def _http_error_family(self, status_code: Optional[int]) -> str:
        if status_code is None:
            return ""
        try:
            family = int(status_code) // 100
        except Exception:
            return ""
        if family in {4, 5}:
            return f"{family}xx"
        return ""

    def _detect_block_signals(self, html: str, markdown: str, status_code: Optional[int]) -> tuple[bool, str, bool]:
        combined = f"{html or ''}\n{markdown or ''}".lower()
        html_len = len(html or "")
        markdown_len = len(markdown or "")
        # Real challenge pages are small (<3K HTML). If we have 5K+ HTML or 2K+
        # markdown, block phrases are from boilerplate scripts/headers, not a challenge.
        has_substantial_content = html_len > 5000 or markdown_len > 2000

        # Cloudflare challenge: lots of HTML (JS scripts) but almost no markdown.
        # Challenge pages inject ~8K bytes of JS, triggering has_substantial_content,
        # but produce <100 chars of actual markdown content.
        html_heavy_no_markdown = html_len > 3000 and markdown_len < 100

        patterns = [
            ("cloudflare", "cloudflare_challenge"),
            ("verify your session", "session_verification"),
            ("captcha", "captcha"),
            ("access denied", "access_denied"),
            ("just a moment", "bot_challenge"),
            ("are you human", "bot_challenge"),
            ("attention required", "bot_challenge"),
        ]

        for phrase, reason in patterns:
            if phrase in combined:
                # Don't flag as blocked when we have substantial content — real challenge
                # pages are small (<5K HTML). If we have 10K+ chars, the block phrase is
                # likely from boilerplate scripts/headers, not an actual challenge.
                # Exception: large HTML with tiny markdown = challenge page with JS bloat.
                if has_substantial_content and not html_heavy_no_markdown:
                    logger.debug(
                        f"Block phrase '{phrase}' found but page has substantial content "
                        f"(html={html_len}, md={markdown_len}), not flagging as blocked"
                    )
                    continue
                return True, reason, "captcha" in phrase or "captcha" in combined

        if status_code in {401, 403, 429, 503}:
            # 403 with substantial content = soft-block (page served despite status code)
            if has_substantial_content and status_code == 403:
                logger.debug(
                    f"HTTP 403 but page has substantial content "
                    f"(html={html_len}, md={markdown_len}), not flagging as blocked"
                )
                return False, "", False
            return True, f"http_{status_code}", False

        return False, "", False

    def _classify_content_quality(
        self,
        body_char_count: int,
        body_word_count: int,
        blocked: bool,
        status_code: Optional[int] = None,
        content: Optional[str] = None
    ) -> str:
        # Anti-bot/CAPTCHA/challenge pages are never usable content.
        if blocked:
            return "blocked"

        # Catch Cloudflare challenge pages served with HTTP 403/503 + thin body
        if status_code in (403, 503) and body_char_count < 200:
            return "blocked"

        # HTTP errors should never be "sufficient" for downstream summarization,
        # UNLESS we have substantial content (soft-block: page served despite status code).
        if status_code is not None:
            try:
                code = int(status_code)
            except Exception:
                code = None
            if code is not None:
                if code >= 500:
                    return "blocked"
                if code >= 400:
                    # 4xx with substantial content = soft-block, don't downgrade
                    if body_char_count > 2000:
                        pass  # fall through to normal quality assessment
                    else:
                        return "minimal"

        normalized = (content or "").lower()
        error_page_signatures = [
            "error code: 404",
            "you've arrived at an empty lot",
            "page not found",
            "not found",
            "doesn't look like there's anything at this address",
            "access denied",
        ]
        if any(sig in normalized for sig in error_page_signatures):
            return "minimal"

        # Cloudflare challenge pages served as HTTP 200
        cf_challenge_signatures = [
            "just a moment",
            "performance & security by cloudflare",
            "ray id:",
            "cf-browser-verification",
            "enable javascript and cookies to continue",
        ]
        cf_matches = sum(1 for sig in cf_challenge_signatures if sig in normalized)
        if cf_matches >= 2 and body_char_count < 500:
            return "blocked"

        # Thin pages should not be treated as sufficient; this catches quiz/header-only pages.
        if body_char_count < 80 or body_word_count < 15:
            return "empty"
        if body_char_count < 600 or body_word_count < 120:
            return "minimal"

        return "sufficient"

    def _normalize_url(self, url: str) -> str:
        parsed = urlparse(url)
        if not parsed.scheme and not parsed.netloc:
            return url
        path = parsed.path.rstrip("/") or "/"
        return parsed._replace(
            scheme=parsed.scheme.lower(),
            netloc=parsed.netloc.lower(),
            path=path,
            query="",
            fragment=""
        ).geturl()
    
    async def crawl_for_markdown_only(
        self,
        url: str,
        javascript: bool = True,
        timeout: int = None,
        javascript_payload: Optional[str] = None,
        dedupe_tables: bool = True,
        wait_until: str = "domcontentloaded",
        wait_for_selector: Optional[str] = None,
        wait_after_load_ms: int = 1000,
        retry_with_js_if_thin: bool = False,
        proxy=None
    ) -> str:
        """
        Crawl URL and return only the markdown content.
        
        Args:
            url: URL to crawl
            javascript: Enable JavaScript execution
            timeout: Browser timeout in milliseconds
            
        Returns:
            Markdown content as string
        """
        try:
            result = await self.crawl_url(
                url=url,
                javascript=javascript,
                screenshot=False,
                timeout=timeout,
                javascript_payload=javascript_payload,
                dedupe_tables=dedupe_tables,
                wait_until=wait_until,
                wait_for_selector=wait_for_selector,
                wait_after_load_ms=wait_after_load_ms,
                retry_with_js_if_thin=retry_with_js_if_thin,
                proxy=proxy
            )
            
            if result.success:
                return result.markdown
            else:
                return f"Error crawling {url}: {result.error_message}"
                
        except Exception as e:
            logger.error(f"Error in markdown-only crawl: {e}")
            return f"Error: {str(e)}"

    async def crawl_raw_html(
        self,
        url: str,
        javascript: bool = True,
        timeout: int = None,
        javascript_payload: Optional[str] = None,
        proxy=None
    ) -> Dict[str, Any]:
        """
        Crawl URL and return only raw HTML content.
        """
        start_time = time.time()
        result: Dict[str, Any] = {
            "success": False,
            "html": "",
            "error": None,
            "page_info": {},
            "processing_time": 0.0
        }
        try:
            parsed_url = urlparse(url)
            if not parsed_url.scheme or not parsed_url.netloc:
                raise ValueError(f"Invalid URL: {url}")

            browser = await get_browser_engine()
            timeout_ms = (timeout * 1000) if timeout else settings.browser_timeout

            html, page_info, _ = await browser.crawl_with_context(
                url,
                javascript_enabled=javascript,
                timeout=timeout_ms,
                take_screenshot=False,
                javascript_payload=javascript_payload,
                proxy=proxy
            )

            result["success"] = True
            result["html"] = html
            result["page_info"] = page_info
        except Exception as e:
            logger.error(f"Failed to fetch raw HTML for {url}: {e}", exc_info=True)
            result["error"] = str(e)
        finally:
            result["processing_time"] = time.time() - start_time

        return result

    async def fetch_binary(
        self,
        url: str,
        use_browser: bool = False,
        javascript: bool = True,
        timeout: int = None,
        session_id: Optional[str] = None,
        filename: Optional[str] = None
    ) -> Dict[str, Any]:
        """Fetch a binary payload (e.g., PDF) from a URL."""
        result: Dict[str, Any] = {
            "success": False,
            "url": url,
            "status_code": None,
            "content": b"",
            "content_type": "application/octet-stream",
            "content_disposition": "",
            "filename": None,
            "saved_path": None,
            "error": None
        }
        try:
            if use_browser:
                browser = await get_browser_engine()
                context, page = await browser.create_isolated_context(javascript_enabled=javascript)
                try:
                    timeout_ms = (timeout * 1000) if timeout else settings.browser_timeout
                    response = await page.goto(url, timeout=timeout_ms, wait_until="domcontentloaded")
                    if response is None:
                        raise RuntimeError("No response received from browser navigation")
                    body = await response.body()
                    headers = response.headers
                    status = response.status
                finally:
                    try:
                        await context.close()
                    except Exception as close_error:
                        logger.warning(f"Error closing browser context: {close_error}")
            else:
                request_timeout = timeout or settings.crawl_timeout
                async with httpx.AsyncClient(follow_redirects=True, timeout=request_timeout) as client:
                    resp = await client.get(url)
                    status = resp.status_code
                    headers = resp.headers
                    body = resp.content

            content_type = headers.get("content-type", "application/octet-stream")
            content_disposition = headers.get("content-disposition", "")

            final_name = filename or self._derive_filename(url, content_type, content_disposition)
            if final_name and "." not in final_name:
                guessed_ext = mimetypes.guess_extension(content_type.split(";")[0].strip()) or ""
                final_name = f"{final_name}{guessed_ext}" if guessed_ext else final_name
            final_name = self._sanitize_filename(final_name) if final_name else "download"

            result.update({
                "success": True,
                "status_code": status,
                "content": body,
                "content_type": content_type,
                "content_disposition": content_disposition,
                "filename": final_name
            })

            if session_id:
                storage_path = f"downloads/{final_name}"
                await self.storage.save_file(body, storage_path, session_id)
                result["saved_path"] = storage_path

        except Exception as e:
            logger.error(f"Failed to fetch binary from {url}: {e}", exc_info=True)
            result["error"] = str(e)

        return result

    def _derive_filename(self, url: str, content_type: str, content_disposition: str) -> str:
        filename = self._extract_filename_from_disposition(content_disposition)
        if not filename:
            path = urlparse(url).path
            if path:
                candidate = Path(path).name
                if candidate:
                    filename = candidate
        if not filename:
            ext = mimetypes.guess_extension(content_type.split(";")[0].strip()) or ""
            filename = f"download{ext}"
        return filename

    def _extract_filename_from_disposition(self, content_disposition: str) -> Optional[str]:
        if not content_disposition:
            return None
        match = re.search(r"filename\\*=([^']*)''([^;]+)", content_disposition, flags=re.IGNORECASE)
        if match:
            return unquote(match.group(2))
        match = re.search(r'filename=\"?([^\";]+)\"?', content_disposition, flags=re.IGNORECASE)
        if match:
            return match.group(1)
        return None

    def _sanitize_filename(self, filename: str) -> str:
        candidate = Path(filename).name
        candidate = candidate.encode("ascii", "ignore").decode("ascii")
        if not candidate:
            return "download"
        safe = "".join(ch if ch.isalnum() or ch in " ._-()" else "_" for ch in candidate)
        return safe or "download"
    
    async def batch_crawl(
        self,
        urls: List[str],
        javascript: bool = True,
        screenshot: bool = False,
        max_concurrent: int = 3,
        session_id: Optional[str] = None,
        javascript_payload: Optional[str] = None,
        dedupe_tables: bool = True,
        wait_until: str = "domcontentloaded",
        wait_for_selector: Optional[str] = None,
        wait_after_load_ms: int = 1000,
        retry_with_js_if_thin: bool = False,
        proxy=None
    ) -> Dict[str, Any]:
        """
        Crawl multiple URLs concurrently.
        
        Args:
            urls: List of URLs to crawl
            javascript: Enable JavaScript execution
            screenshot: Take screenshots
            max_concurrent: Maximum concurrent crawls
            session_id: Session ID for storage organization
            
        Returns:
            Dictionary with batch results
        """
        if not urls:
            return {"urls": [], "results": [], "summary": {"total": 0, "success": 0, "failed": 0}}
        
        # Limit concurrent crawls
        semaphore = asyncio.Semaphore(min(max_concurrent, settings.max_concurrent_crawls))
        
        async def crawl_with_semaphore(url: str) -> CrawlResult:
            async with semaphore:
                return await self.crawl_url(
                    url=url,
                    javascript=javascript,
                    screenshot=screenshot,
                    session_id=session_id,
                    javascript_payload=javascript_payload,
                    dedupe_tables=dedupe_tables,
                    wait_until=wait_until,
                    wait_for_selector=wait_for_selector,
                    wait_after_load_ms=wait_after_load_ms,
                    retry_with_js_if_thin=retry_with_js_if_thin,
                    proxy=proxy
                )
        
        # Execute crawls concurrently
        start_time = time.time()
        results = await asyncio.gather(
            *[crawl_with_semaphore(url) for url in urls],
            return_exceptions=True
        )
        
        # Process results into a single stable schema
        all_results: List[Dict[str, Any]] = []
        
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                all_results.append({
                    "url": urls[i],
                    "final_url": "",
                    "success": False,
                    "status_code": None,
                    "markdown": "",
                    "markdown_plain": "",
                    "content": "",
                    "error": str(result),
                    "content_quality": "empty"
                })
            elif isinstance(result, CrawlResult):
                payload = result.to_dict()
                payload["error"] = result.error_message if not result.success else ""
                all_results.append(payload)
        
        total_time = time.time() - start_time
        success_count = len([r for r in all_results if r.get("success")])
        failed_results = [r for r in all_results if not r.get("success")]
        
        batch_result = {
            "urls": urls,
            "results": all_results,
            "failed": failed_results,
            "summary": {
                "total": len(urls),
                "success": success_count,
                "failed": len(failed_results),
                "processing_time": total_time
            },
            "session_id": session_id
        }
        
        # Save batch results
        if session_id:
            await self._save_batch_result(batch_result, session_id)
        
        logger.info(f"Batch crawl completed: {success_count}/{len(urls)} successful in {total_time:.2f}s")
        
        return batch_result
    
    def _extract_page_info_from_html(self, html: str) -> Dict[str, Any]:
        """Extract basic page information from HTML content."""
        try:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(html, 'html.parser')
            
            # Extract title
            title_tag = soup.find('title')
            title = title_tag.text.strip() if title_tag else ""
            
            # Extract meta description
            meta_desc = soup.find('meta', attrs={'name': 'description'})
            description = meta_desc.get('content', '') if meta_desc else ""
            
            # Extract basic info
            return {
                "title": title,
                "description": description,
                "url": "",  # Will be set by caller
                "status_code": 200,  # Assume success if we got HTML
                "content_type": "text/html",
                "content_length": len(html)
            }
        except Exception as e:
            logger.warning(f"Error extracting page info from HTML: {e}")
            return {
                "title": "",
                "description": "",
                "url": "",
                "status_code": 200,
                "content_type": "text/html",
                "content_length": len(html)
            }
    
    async def _save_screenshot_data(self, screenshot_data, url: str, session_id: str):
        """Save screenshot data to storage and return the path(s)."""
        try:
            # Generate base info from URL
            from urllib.parse import urlparse
            parsed = urlparse(url)
            domain = parsed.netloc.replace(":", "_").replace("/", "_")
            timestamp = int(time.time())
            
            logger.info(f"Saving screenshot data for {url}, session_id: {session_id}")
            logger.info(f"Screenshot data type: {type(screenshot_data)}, size: {len(screenshot_data) if isinstance(screenshot_data, (list, bytes)) else 'unknown'}")
            
            # Handle multiple screenshot segments
            if isinstance(screenshot_data, list):
                # Multiple segments from long page
                saved_paths = []
                for i, segment_data in enumerate(screenshot_data):
                    filename = f"screenshot_{domain}_{timestamp}_segment_{i+1}.png"
                    logger.info(f"Saving screenshot segment {i+1}: {filename}, size: {len(segment_data)} bytes")
                    await self.storage.save_file(segment_data, filename, session_id)
                    saved_paths.append(filename)
                    logger.info(f"Successfully saved screenshot segment {i+1}/{len(screenshot_data)}: {filename}")
                
                logger.info(f"Saved {len(saved_paths)} screenshot segments for {url}")
                return saved_paths
            
            else:
                # Single screenshot
                filename = f"screenshot_{domain}_{timestamp}.png"
                logger.info(f"Saving single screenshot: {filename}, size: {len(screenshot_data)} bytes")
                await self.storage.save_file(screenshot_data, filename, session_id)
                logger.info(f"Successfully saved screenshot: {filename}")
                return filename
            
        except Exception as e:
            logger.error(f"Failed to save screenshot: {e}", exc_info=True)
            return None
    
    async def _save_screenshot(
        self,
        browser,
        url: str,
        mode: str,
        session_id: Optional[str]
    ) -> str:
        """Save screenshot and return the path."""
        try:
            # Generate screenshot filename
            parsed_url = urlparse(url)
            domain = parsed_url.netloc.replace(".", "_")
            timestamp = int(time.time())
            filename = f"screenshot_{domain}_{timestamp}.png"
            
            # Get storage path
            if session_id:
                screenshot_dir = self.storage.get_session_path(session_id) / "screenshots"
                screenshot_dir.mkdir(exist_ok=True)
                screenshot_path = screenshot_dir / filename
            else:
                screenshot_path = Path(f"./screenshots/{filename}")
                screenshot_path.parent.mkdir(exist_ok=True)
            
            # Take screenshot
            success = await browser.take_screenshot(str(screenshot_path), mode)
            
            if success:
                logger.info(f"Screenshot saved: {screenshot_path}")
                return str(screenshot_path)
            else:
                return ""
                
        except Exception as e:
            logger.error(f"Error saving screenshot: {e}")
            return ""
    
    async def _save_crawl_result(self, result: CrawlResult, session_id: str) -> str:
        """Save crawl result to storage and return filename."""
        try:
            # Generate filename
            parsed_url = urlparse(result.url)
            domain = parsed_url.netloc.replace(".", "_")
            timestamp = int(time.time())
            filename = f"crawl_{domain}_{timestamp}.json"
            
            # Save to storage
            await self.storage.save_json(
                data=result.to_dict(),
                filename=filename,
                session_id=session_id
            )
            
            logger.debug(f"Saved crawl result: {filename}")
            return filename
        except Exception as e:
            logger.error(f"Error saving crawl result: {e}")
            return ""
    
    async def _save_batch_result(self, batch_result: Dict[str, Any], session_id: str) -> None:
        """Save batch crawl result to storage."""
        try:
            timestamp = int(time.time())
            filename = f"batch_crawl_{timestamp}.json"
            
            await self.storage.save_json(
                data=batch_result,
                filename=filename,
                session_id=session_id
            )
            
            logger.debug(f"Saved batch result: {filename}")
            
        except Exception as e:
            logger.error(f"Error saving batch result: {e}")
    
    async def cleanup(self) -> None:
        """Cleanup crawler resources."""
        try:
            await cleanup_browser()
            logger.info("Crawler cleanup completed")
        except Exception as e:
            logger.error(f"Error during crawler cleanup: {e}")


# Global crawler instance cache
_crawler_instances = {}

async def get_crawler_engine(user_email: Optional[str] = None) -> CrawlerEngine:
    """Get or create a crawler engine instance for a user."""
    key = user_email or "default"
    
    if key not in _crawler_instances:
        _crawler_instances[key] = CrawlerEngine(user_email)
    
    return _crawler_instances[key]

async def cleanup_all_crawlers():
    """Cleanup all crawler instances."""
    for crawler in _crawler_instances.values():
        await crawler.cleanup()
    _crawler_instances.clear()
    logger.info("All crawlers cleaned up")
