"""
Tests for _detect_block_signals and _classify_content_quality in CrawlerEngine.

Verifies that block phrases (e.g. "just a moment", "cloudflare") in HTML/markdown
do NOT trigger a false-positive block when the page has real content.

Capterra pages with ~9K HTML commonly include Cloudflare challenge boilerplate
in headers/scripts even when successfully loaded. The substantial-content guard
must prevent false positives.

Also tests content quality classification for Cloudflare fake successes:
- HTTP 403/503 + thin body = "blocked"
- HTTP 200 + Cloudflare challenge signatures + thin body = "blocked"

Created: 2026-02-27
Updated: 2026-02-28 — added Cloudflare fake success detection tests
"""

import pytest
from unittest.mock import MagicMock
from app.crawler import CrawlerEngine


@pytest.fixture
def handler():
    """Create a minimal CrawlerEngine for unit-testing _detect_block_signals."""
    return CrawlerEngine.__new__(CrawlerEngine)


class TestDetectBlockSignals:
    """Block detection with substantial-content guard."""

    def test_small_challenge_page_is_blocked(self, handler):
        """A small page with 'just a moment' is a real challenge — should be blocked."""
        html = "<html><head><title>Just a moment...</title></head><body>Please wait</body></html>"
        blocked, reason, captcha = handler._detect_block_signals(html, "", None)
        assert blocked is True
        assert reason == "bot_challenge"

    def test_small_cloudflare_page_is_blocked(self, handler):
        """A small page with 'cloudflare' is a real challenge."""
        html = "<html><body>Checking your browser... Cloudflare</body></html>"
        blocked, reason, captcha = handler._detect_block_signals(html, "", None)
        assert blocked is True
        assert reason == "cloudflare_challenge"

    def test_large_html_with_challenge_phrase_not_blocked(self, handler):
        """A page with >10K HTML containing 'cloudflare' in scripts is NOT blocked."""
        # Simulate a real review page with Cloudflare script references in boilerplate
        real_content = "Great product review. " * 500  # ~10K chars
        html = f"<html><head><script src='cloudflare-cdn.js'></script></head><body>{real_content}</body></html>"
        assert len(html) > 10000

        # In real flow, markdown is always populated by the markdown generator.
        # A page with 10K real HTML produces substantial markdown.
        markdown = "Great product review. " * 200
        assert len(markdown) > 2000

        blocked, reason, captcha = handler._detect_block_signals(html, markdown, None)
        assert blocked is False

    def test_moderate_html_with_rich_markdown_not_blocked(self, handler):
        """9K HTML + 3K markdown with 'just a moment' in header is NOT blocked.

        This is the Capterra false positive scenario: the page loaded fine (~9K HTML),
        markdown extraction produced real content (>2K), but the HTML header
        contains 'just a moment' from Cloudflare's initial challenge.
        """
        # Capterra-like HTML: under 10K but has real content
        html = "<html><head><title>Just a moment</title></head><body>" + "Review text. " * 600 + "</body></html>"
        assert len(html) < 10000

        # Markdown extraction produced real review content
        markdown = "# Dovetail Reviews\n\n" + "Great product for user research. " * 80
        assert len(markdown) > 2000

        blocked, reason, captcha = handler._detect_block_signals(html, markdown, None)
        assert blocked is False

    def test_small_html_with_short_markdown_is_blocked(self, handler):
        """Small HTML (<5K) + tiny markdown with 'just a moment' IS blocked.

        If HTML is under 5K AND markdown is under threshold, the page
        is likely a real challenge/block page.
        """
        html = "<html><head><title>Just a moment</title></head><body>" + "x" * 3000 + "</body></html>"
        assert len(html) < 5000

        # Very short markdown = page didn't render properly
        markdown = "Loading..."
        assert len(markdown) < 500

        blocked, reason, captcha = handler._detect_block_signals(html, markdown, None)
        assert blocked is True

    def test_403_with_substantial_content_not_blocked(self, handler):
        """HTTP 403 with large content is a soft-block — should not be flagged."""
        html = "x" * 15000
        blocked, reason, captcha = handler._detect_block_signals(html, "", 403)
        assert blocked is False

    def test_403_without_content_is_blocked(self, handler):
        """HTTP 403 with no content is a hard block."""
        blocked, reason, captcha = handler._detect_block_signals("", "", 403)
        assert blocked is True
        assert reason == "http_403"

    def test_captcha_detection(self, handler):
        """Pages with 'captcha' should set captcha_detected=True."""
        html = "<html><body>Please solve this captcha to continue</body></html>"
        blocked, reason, captcha = handler._detect_block_signals(html, "", None)
        assert blocked is True
        assert captcha is True

    def test_html_heavy_no_markdown_is_blocked(self, handler):
        """Cloudflare challenge: 8K HTML (JS bloat) but <100 chars markdown = blocked.

        Cloudflare injects ~8KB of JavaScript in challenge pages, which triggers the
        has_substantial_content guard (html_len > 5000). But the markdown output is
        tiny (<100 chars), which is the html_heavy_no_markdown signal.
        """
        # Simulate a Cloudflare challenge page with lots of JS
        js_bloat = "<script>" + "var x=1;" * 800 + "</script>"
        html = f"<html><head>{js_bloat}</head><body>Just a moment...</body></html>"
        assert len(html) > 5000  # triggers has_substantial_content

        # Markdown extraction produced almost nothing
        markdown = "Just a moment..."
        assert len(markdown) < 100

        blocked, reason, captcha = handler._detect_block_signals(html, markdown, None)
        assert blocked is True
        assert reason == "bot_challenge"

    def test_html_heavy_with_real_markdown_not_blocked(self, handler):
        """Large HTML with rich markdown (>100 chars) is NOT a Cloudflare fake success."""
        html = "<html><head><script src='cloudflare-cdn.js'></script></head><body>" + "Real content. " * 600 + "</body></html>"
        assert len(html) > 5000

        markdown = "# Reviews\n\n" + "Great product review. " * 20
        assert len(markdown) > 100

        blocked, reason, captcha = handler._detect_block_signals(html, markdown, None)
        assert blocked is False

    def test_403_html_heavy_no_markdown_with_phrase_is_blocked(self, handler):
        """HTTP 403 + large HTML (JS bloat) + tiny markdown + block phrase = blocked.

        Even though HTML > 5K (has_substantial_content=True), the html_heavy_no_markdown
        override detects that the page is a Cloudflare challenge (lots of JS, no markdown).
        """
        js_bloat = "<script>" + "var x=1;" * 800 + "</script>"
        html = f"<html><head>{js_bloat}</head><body>Checking your browser cloudflare</body></html>"
        assert len(html) > 5000

        markdown = ""
        blocked, reason, captcha = handler._detect_block_signals(html, markdown, None)
        assert blocked is True
        assert reason == "cloudflare_challenge"

    def test_clean_page_not_blocked(self, handler):
        """A normal page with no challenge phrases is not blocked."""
        html = "<html><body>Welcome to our product reviews</body></html>"
        blocked, reason, captcha = handler._detect_block_signals(html, "", 200)
        assert blocked is False


class TestClassifyContentQuality:
    """Content quality classification with Cloudflare fake success detection."""

    def test_blocked_stays_blocked(self, handler):
        quality = handler._classify_content_quality(100, 20, blocked=True, status_code=200)
        assert quality == "blocked"

    def test_403_thin_body_is_blocked(self, handler):
        """HTTP 403 with <200 chars body = Cloudflare challenge."""
        quality = handler._classify_content_quality(50, 8, blocked=False, status_code=403)
        assert quality == "blocked"

    def test_503_thin_body_is_blocked(self, handler):
        """HTTP 503 with <200 chars body = Cloudflare challenge."""
        quality = handler._classify_content_quality(100, 15, blocked=False, status_code=503)
        assert quality == "blocked"

    def test_403_substantial_body_is_not_blocked(self, handler):
        """HTTP 403 with >2000 chars = soft-block, not challenge."""
        quality = handler._classify_content_quality(
            3000, 500, blocked=False, status_code=403,
            content="This is a real page with lots of content. " * 100
        )
        assert quality == "sufficient"

    def test_cf_challenge_signatures_http200_blocked(self, handler):
        """HTTP 200 + 2+ Cloudflare challenge signatures + <500 chars = blocked."""
        content = "Just a moment... Performance & security by Cloudflare. Ray ID: abc123"
        quality = handler._classify_content_quality(
            len(content), 10, blocked=False, status_code=200, content=content
        )
        assert quality == "blocked"

    def test_cf_single_signature_not_blocked(self, handler):
        """Only 1 Cloudflare signature is not enough to block."""
        content = "Just a moment... Loading your dashboard."
        quality = handler._classify_content_quality(
            len(content), 8, blocked=False, status_code=200, content=content
        )
        # Should not be "blocked" — only 1 CF signature
        assert quality != "blocked"

    def test_cf_signatures_large_body_not_blocked(self, handler):
        """Cloudflare signatures in a large body (>500 chars) = real content, not challenge."""
        content = ("Just a moment. Performance & security by Cloudflare. " +
                   "This is real content. " * 50)
        assert len(content) > 500
        quality = handler._classify_content_quality(
            len(content), 200, blocked=False, status_code=200, content=content
        )
        assert quality == "sufficient"

    def test_sufficient_quality(self, handler):
        quality = handler._classify_content_quality(
            5000, 800, blocked=False, status_code=200,
            content="Lots of real review content here. " * 200
        )
        assert quality == "sufficient"

    def test_empty_quality(self, handler):
        quality = handler._classify_content_quality(10, 2, blocked=False, status_code=200)
        assert quality == "empty"

    def test_minimal_quality(self, handler):
        quality = handler._classify_content_quality(200, 40, blocked=False, status_code=200, content="Some text")
        assert quality == "minimal"

    def test_error_page_signature_is_minimal(self, handler):
        content = "Page not found - the resource you requested does not exist"
        quality = handler._classify_content_quality(
            len(content), 10, blocked=False, status_code=200, content=content
        )
        assert quality == "minimal"
