"""Unit tests for app.agent.ghost — Ghost Protocol block detection and logic."""

import pytest

from app.agent.ghost import (
    BlockDetection,
    BlockSignal,
    GhostCapture,
    GhostExtraction,
    GhostResult,
    GHOST_EXTRACTION_PROMPT,
    detect_block,
    extract_via_vision,
    should_trigger_ghost,
)


class TestGhostRouteImports:
    def test_agent_ghost_uses_proxy_module_for_proxy_resolution(self):
        """Regression: /api/agent/ghost must not import resolve_proxy from stealth."""
        import pathlib

        route_path = pathlib.Path(__file__).parent.parent / "app" / "agent_routes.py"
        source = route_path.read_text()

        assert "from app.proxy import resolve_proxy" in source
        assert "from app.stealth import resolve_proxy" not in source


class TestBlockSignal:
    def test_all_signals(self):
        assert BlockSignal.CLOUDFLARE == "cloudflare_challenge"
        assert BlockSignal.CAPTCHA == "captcha"
        assert BlockSignal.SESSION_VERIFY == "session_verification"
        assert BlockSignal.ACCESS_DENIED == "access_denied"
        assert BlockSignal.BOT_CHALLENGE == "bot_challenge"
        assert BlockSignal.EMPTY_SHELL == "empty_spa_shell"
        assert BlockSignal.HTTP_403 == "http_403"
        assert BlockSignal.HTTP_429 == "http_429"
        assert BlockSignal.HTTP_503 == "http_503"


class TestDetectBlock:
    def test_cloudflare_detected(self):
        d = detect_block(html="<div>Checking your browser... Cloudflare</div>")
        assert d.blocked is True
        assert d.signal == BlockSignal.CLOUDFLARE
        assert d.confidence >= 0.9

    def test_captcha_detected(self):
        d = detect_block(html="<div>Please solve the CAPTCHA below</div>")
        assert d.blocked is True
        assert d.signal == BlockSignal.CAPTCHA
        assert d.captcha_detected is True

    def test_recaptcha_detected(self):
        d = detect_block(html='<div class="g-recaptcha" data-sitekey="abc"></div>')
        assert d.blocked is True
        assert d.signal == BlockSignal.CAPTCHA

    def test_hcaptcha_detected(self):
        d = detect_block(html='<iframe src="https://hcaptcha.com/challenge"></iframe>')
        assert d.blocked is True
        assert d.signal == BlockSignal.CAPTCHA

    def test_bot_challenge_just_a_moment(self):
        d = detect_block(html="<title>Just a moment...</title>")
        assert d.blocked is True
        assert d.signal == BlockSignal.BOT_CHALLENGE

    def test_bot_challenge_are_you_human(self):
        d = detect_block(html="<h1>Are you human?</h1>")
        assert d.blocked is True
        assert d.signal == BlockSignal.BOT_CHALLENGE

    def test_checking_your_browser(self):
        d = detect_block(html="<p>Checking your browser before accessing</p>")
        assert d.blocked is True
        assert d.signal == BlockSignal.BOT_CHALLENGE

    def test_access_denied(self):
        d = detect_block(html="<h1>Access Denied</h1>")
        assert d.blocked is True
        assert d.signal == BlockSignal.ACCESS_DENIED

    def test_http_403(self):
        d = detect_block(html="", status_code=403)
        assert d.blocked is True
        assert d.signal == BlockSignal.HTTP_403

    def test_http_429(self):
        d = detect_block(html="", status_code=429)
        assert d.blocked is True
        assert d.signal == BlockSignal.HTTP_429

    def test_http_503(self):
        d = detect_block(html="", status_code=503)
        assert d.blocked is True
        assert d.signal == BlockSignal.HTTP_503

    def test_empty_spa_shell(self):
        # Big HTML but almost no text content
        html = '<html><head><script src="app.js"></script></head><body><div id="root"></div></body></html>' * 10
        d = detect_block(
            html=html,
            body_char_count=50,
            body_word_count=5,
        )
        assert d.blocked is True
        assert d.signal == BlockSignal.EMPTY_SHELL

    def test_content_quality_blocked(self):
        d = detect_block(html="<html></html>", content_quality="blocked")
        assert d.blocked is True

    def test_normal_page_not_blocked(self):
        d = detect_block(
            html="<html><body><h1>Hello World</h1><p>This is content.</p></body></html>",
            body_char_count=500,
            body_word_count=80,
        )
        assert d.blocked is False

    def test_http_200_not_blocked(self):
        d = detect_block(html="<html><body>OK</body></html>", status_code=200)
        assert d.blocked is False

    def test_markdown_also_checked(self):
        d = detect_block(html="", markdown="# Cloudflare challenge page")
        assert d.blocked is True
        assert d.signal == BlockSignal.CLOUDFLARE

    def test_case_insensitive(self):
        d = detect_block(html="<div>CLOUDFLARE RAY ID</div>")
        assert d.blocked is True


class TestShouldTriggerGhost:
    def test_trigger_on_cloudflare(self):
        d = BlockDetection(blocked=True, signal=BlockSignal.CLOUDFLARE, confidence=0.95)
        assert should_trigger_ghost(d, ghost_enabled=True, auto_trigger=True) is True

    def test_no_trigger_when_disabled(self):
        d = BlockDetection(blocked=True, signal=BlockSignal.CLOUDFLARE, confidence=0.95)
        assert should_trigger_ghost(d, ghost_enabled=False, auto_trigger=True) is False

    def test_no_trigger_when_auto_off(self):
        d = BlockDetection(blocked=True, signal=BlockSignal.CLOUDFLARE, confidence=0.95)
        assert should_trigger_ghost(d, ghost_enabled=True, auto_trigger=False) is False

    def test_no_trigger_when_not_blocked(self):
        d = BlockDetection(blocked=False)
        assert should_trigger_ghost(d, ghost_enabled=True, auto_trigger=True) is False

    def test_no_trigger_on_low_confidence_access_denied(self):
        d = BlockDetection(blocked=True, signal=BlockSignal.ACCESS_DENIED, confidence=0.7)
        assert should_trigger_ghost(d, ghost_enabled=True, auto_trigger=True) is False

    def test_trigger_on_captcha(self):
        d = BlockDetection(blocked=True, signal=BlockSignal.CAPTCHA, confidence=0.95)
        assert should_trigger_ghost(d, ghost_enabled=True, auto_trigger=True) is True

    def test_trigger_on_empty_shell(self):
        d = BlockDetection(blocked=True, signal=BlockSignal.EMPTY_SHELL, confidence=0.6)
        assert should_trigger_ghost(d, ghost_enabled=True, auto_trigger=True) is True


class TestGhostDataclasses:
    def test_ghost_capture_defaults(self):
        c = GhostCapture()
        assert c.success is False
        assert c.image_bytes == b""
        assert c.content_type == "image/png"

    def test_ghost_extraction_defaults(self):
        e = GhostExtraction()
        assert e.success is False
        assert e.render_mode == "ghost"
        assert e.blocked_content is False

    def test_ghost_result_defaults(self):
        r = GhostResult()
        assert r.success is False
        assert r.render_mode == "ghost"

    def test_extraction_prompt_exists(self):
        assert len(GHOST_EXTRACTION_PROMPT) > 100
        assert "screenshot" in GHOST_EXTRACTION_PROMPT.lower()

    def test_extraction_prompt_has_page_type_classification(self):
        """Updated prompt should include PAGE_TYPE classification instructions."""
        assert "PAGE_TYPE: BLOCKED" in GHOST_EXTRACTION_PROMPT
        assert "PAGE_TYPE: CONTENT" in GHOST_EXTRACTION_PROMPT
        assert "PAGE_TYPE: ERROR" in GHOST_EXTRACTION_PROMPT
        assert "PAGE_TYPE: EMPTY" in GHOST_EXTRACTION_PROMPT

    def test_extraction_prompt_mentions_cloudflare(self):
        """Updated prompt should mention Cloudflare-specific signals."""
        prompt_lower = GHOST_EXTRACTION_PROMPT.lower()
        assert "cloudflare" in prompt_lower or "just a moment" in prompt_lower
        assert "checking your browser" in prompt_lower or "captcha" in prompt_lower


class _MockVisionProvider:
    """Mock vision provider that returns a pre-set response."""

    def __init__(self, response: str):
        self._response = response

    async def vision(self, image_bytes, prompt, detail="auto"):
        return self._response


class TestExtractViaVisionBlocked:
    """Tests for blocked content detection in extract_via_vision."""

    @pytest.fixture
    def capture(self):
        return GhostCapture(success=True, image_bytes=b"fake-png-data", url="https://example.com")

    @pytest.mark.asyncio
    async def test_page_type_blocked_in_first_line(self, capture):
        """PAGE_TYPE: BLOCKED classification in first line -> blocked_content: True."""
        provider = _MockVisionProvider("PAGE_TYPE: BLOCKED\nCloudflare challenge page with verify button.")
        result = await extract_via_vision(capture, provider=provider)
        assert result.success is True
        assert result.blocked_content is True

    @pytest.mark.asyncio
    async def test_page_type_blocked_no_space(self, capture):
        """PAGE_TYPE:BLOCKED (no space) -> blocked_content: True."""
        provider = _MockVisionProvider("PAGE_TYPE:BLOCKED\nSecurity check required.")
        result = await extract_via_vision(capture, provider=provider)
        assert result.blocked_content is True

    @pytest.mark.asyncio
    async def test_page_type_error(self, capture):
        """PAGE_TYPE: ERROR -> blocked_content: True."""
        provider = _MockVisionProvider("PAGE_TYPE: ERROR\n404 Page not found.")
        result = await extract_via_vision(capture, provider=provider)
        assert result.blocked_content is True

    @pytest.mark.asyncio
    async def test_page_type_empty(self, capture):
        """PAGE_TYPE: EMPTY -> blocked_content: True."""
        provider = _MockVisionProvider("PAGE_TYPE: EMPTY\nBlank page with only a header.")
        result = await extract_via_vision(capture, provider=provider)
        assert result.blocked_content is True

    @pytest.mark.asyncio
    async def test_page_type_content_not_blocked(self, capture):
        """PAGE_TYPE: CONTENT -> blocked_content: False (actual content)."""
        provider = _MockVisionProvider("PAGE_TYPE: CONTENT\n# Product Reviews\nGreat product, 5 stars.")
        result = await extract_via_vision(capture, provider=provider)
        assert result.success is True
        assert result.blocked_content is False

    @pytest.mark.asyncio
    async def test_cloudflare_just_a_moment(self, capture):
        """'just a moment' in content -> blocked_content: True (fallback pattern)."""
        provider = _MockVisionProvider("The page shows: Just a moment... Please wait.")
        result = await extract_via_vision(capture, provider=provider)
        assert result.blocked_content is True

    @pytest.mark.asyncio
    async def test_checking_your_browser(self, capture):
        """'checking your browser' -> blocked_content: True."""
        provider = _MockVisionProvider("Checking your browser before accessing the website.")
        result = await extract_via_vision(capture, provider=provider)
        assert result.blocked_content is True

    @pytest.mark.asyncio
    async def test_cloudflare_ray_id(self, capture):
        """'ray id' in content -> blocked_content: True."""
        provider = _MockVisionProvider("Performance by Cloudflare. Ray ID: 8abc123def456.")
        result = await extract_via_vision(capture, provider=provider)
        assert result.blocked_content is True

    @pytest.mark.asyncio
    async def test_security_check(self, capture):
        """'security check' -> blocked_content: True."""
        provider = _MockVisionProvider("Please complete the security check to access the page.")
        result = await extract_via_vision(capture, provider=provider)
        assert result.blocked_content is True

    @pytest.mark.asyncio
    async def test_captcha_detection(self, capture):
        """'captcha' in content -> blocked_content: True."""
        provider = _MockVisionProvider("Please solve the CAPTCHA to verify you are not a robot.")
        result = await extract_via_vision(capture, provider=provider)
        assert result.blocked_content is True

    @pytest.mark.asyncio
    async def test_recaptcha_detection(self, capture):
        """'recaptcha' -> blocked_content: True."""
        provider = _MockVisionProvider("A reCAPTCHA verification box is displayed.")
        result = await extract_via_vision(capture, provider=provider)
        assert result.blocked_content is True

    @pytest.mark.asyncio
    async def test_hcaptcha_detection(self, capture):
        """'hcaptcha' -> blocked_content: True."""
        provider = _MockVisionProvider("An hCaptcha challenge is visible on the page.")
        result = await extract_via_vision(capture, provider=provider)
        assert result.blocked_content is True

    @pytest.mark.asyncio
    async def test_attention_required(self, capture):
        """'attention required' -> blocked_content: True."""
        provider = _MockVisionProvider("Attention Required! Your request has been flagged.")
        result = await extract_via_vision(capture, provider=provider)
        assert result.blocked_content is True

    @pytest.mark.asyncio
    async def test_normal_review_content_not_blocked(self, capture):
        """Normal review content should NOT be flagged as blocked."""
        review_content = (
            "## Product Reviews for Acme Analytics\n\n"
            "### Review by John D. - 5/5 Stars\n"
            "Great product for security teams. Easy to set up.\n\n"
            "### Review by Jane S. - 4/5 Stars\n"
            "Solid analytics platform with good reporting.\n"
        )
        provider = _MockVisionProvider(review_content)
        result = await extract_via_vision(capture, provider=provider)
        assert result.success is True
        assert result.blocked_content is False

    @pytest.mark.asyncio
    async def test_normal_content_with_page_type_content(self, capture):
        """PAGE_TYPE: CONTENT followed by normal content -> not blocked."""
        provider = _MockVisionProvider(
            "PAGE_TYPE: CONTENT\n# Acme Reviews\nReview by Alice: 5 stars. Excellent product."
        )
        result = await extract_via_vision(capture, provider=provider)
        assert result.blocked_content is False

    @pytest.mark.asyncio
    async def test_failed_capture_returns_error(self):
        """Failed capture -> extraction failure without blocked_content."""
        failed_capture = GhostCapture(success=False, error="Screenshot failed")
        result = await extract_via_vision(failed_capture, provider=_MockVisionProvider("anything"))
        assert result.success is False
        assert result.error is not None
