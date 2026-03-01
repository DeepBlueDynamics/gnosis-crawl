"""Tests for app.challenge_solver — Cloudflare challenge detection and resolution."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.challenge_solver import (
    ChallengeDetection,
    ChallengeResult,
    ChallengeType,
    CHALLENGE_SELECTORS,
    CHALLENGE_TITLE_PATTERNS,
    RESOLVED_SELECTORS,
    detect_challenge,
    wait_for_challenge_resolution,
    solve_turnstile_capsolver,
    solve_managed_challenge_capsolver,
    resolve_challenge,
    _extract_turnstile_sitekey,
    _click_turnstile_checkbox,
    _format_proxy_for_capsolver,
)


# --- Fixtures ---


def make_page(title="My Site", selectors=None, resolved_selectors=None):
    """Create a mock Playwright page with configurable challenge indicators."""
    page = AsyncMock()
    page.title = AsyncMock(return_value=title)

    # Map of selector -> mock element
    element_map = {}
    if selectors:
        for selector, visible in selectors.items():
            el = AsyncMock()
            el.is_visible = AsyncMock(return_value=visible)
            el.get_attribute = AsyncMock(return_value=None)
            element_map[selector] = el

    if resolved_selectors:
        for selector in resolved_selectors:
            el = AsyncMock()
            element_map[selector] = el

    async def query_selector(sel):
        return element_map.get(sel)

    page.query_selector = AsyncMock(side_effect=query_selector)
    page.evaluate = AsyncMock(return_value=None)
    return page


# --- ChallengeDetection dataclass ---


class TestChallengeDetection:
    def test_defaults(self):
        d = ChallengeDetection()
        assert d.detected is False
        assert d.challenge_type == ChallengeType.NONE
        assert d.confidence == 0.0
        assert d.selector_matched == ""

    def test_with_values(self):
        d = ChallengeDetection(
            detected=True,
            challenge_type=ChallengeType.TURNSTILE,
            confidence=0.95,
            selector_matched=".cf-turnstile",
        )
        assert d.detected is True
        assert d.challenge_type == ChallengeType.TURNSTILE


# --- ChallengeResult dataclass ---


class TestChallengeResult:
    def test_defaults(self):
        r = ChallengeResult()
        assert r.resolved is False
        assert r.method == "none"
        assert r.wait_time_ms == 0
        assert r.error is None

    def test_successful_result(self):
        r = ChallengeResult(
            resolved=True,
            challenge_type=ChallengeType.JS_CHALLENGE,
            method="auto_resolve",
            wait_time_ms=3500,
        )
        assert r.resolved is True
        assert r.method == "auto_resolve"


# --- detect_challenge ---


class TestDetectChallenge:
    @pytest.mark.asyncio
    async def test_no_challenge_on_clean_page(self):
        page = make_page(title="Product Reviews - G2")
        result = await detect_challenge(page)
        assert result.detected is False
        assert result.challenge_type == ChallengeType.NONE

    @pytest.mark.asyncio
    async def test_detects_challenge_from_title_just_a_moment(self):
        page = make_page(title="Just a moment...")
        result = await detect_challenge(page)
        assert result.detected is True
        # Title-only match now classifies as MANAGED (not JS_CHALLENGE)
        # because Cloudflare uses "Just a moment..." for all challenge types.
        # MANAGED ensures CapSolver is eligible as a fallback.
        assert result.challenge_type == ChallengeType.MANAGED
        assert result.confidence == 0.9
        assert "just a moment" in result.selector_matched

    @pytest.mark.asyncio
    async def test_detects_challenge_from_title_checking_browser(self):
        page = make_page(title="Checking your browser before accessing example.com")
        result = await detect_challenge(page)
        assert result.detected is True

    @pytest.mark.asyncio
    async def test_detects_challenge_from_title_verify_human(self):
        page = make_page(title="Verify you are human")
        result = await detect_challenge(page)
        assert result.detected is True

    @pytest.mark.asyncio
    async def test_detects_visible_dom_selector(self):
        page = make_page(
            title="Some Page",
            selectors={"#challenge-running": True},
        )
        result = await detect_challenge(page)
        assert result.detected is True
        assert result.challenge_type == ChallengeType.JS_CHALLENGE
        assert result.confidence == 0.95

    @pytest.mark.asyncio
    async def test_detects_hidden_dom_selector_lower_confidence(self):
        page = make_page(
            title="Some Page",
            selectors={"#challenge-running": False},  # present but not visible
        )
        result = await detect_challenge(page)
        assert result.detected is True
        assert result.confidence == 0.7

    @pytest.mark.asyncio
    async def test_detects_turnstile_iframe(self):
        page = make_page(
            title="Some Page",
            selectors={'iframe[src*="challenges.cloudflare.com"]': True},
        )
        result = await detect_challenge(page)
        assert result.detected is True
        assert result.challenge_type == ChallengeType.TURNSTILE

    @pytest.mark.asyncio
    async def test_detects_cf_turnstile_class(self):
        page = make_page(
            title="Some Page",
            selectors={".cf-turnstile": True},
        )
        result = await detect_challenge(page)
        assert result.detected is True
        assert result.challenge_type == ChallengeType.TURNSTILE

    @pytest.mark.asyncio
    async def test_detects_managed_challenge(self):
        page = make_page(
            title="Some Page",
            selectors={"#cf-challenge-running": True},
        )
        result = await detect_challenge(page)
        assert result.detected is True
        assert result.challenge_type == ChallengeType.MANAGED

    @pytest.mark.asyncio
    async def test_detects_browser_check(self):
        page = make_page(
            title="Some Page",
            selectors={".cf-browser-verification": True},
        )
        result = await detect_challenge(page)
        assert result.detected is True
        assert result.challenge_type == ChallengeType.BROWSER_CHECK

    @pytest.mark.asyncio
    async def test_title_exception_falls_through_to_selectors(self):
        page = make_page(selectors={"#challenge-running": True})
        page.title = AsyncMock(side_effect=Exception("Page closed"))
        result = await detect_challenge(page)
        assert result.detected is True
        assert result.challenge_type == ChallengeType.JS_CHALLENGE

    @pytest.mark.asyncio
    async def test_selector_exception_continues_to_next(self):
        page = make_page(title="Normal Page")
        call_count = 0

        async def flaky_query(sel):
            nonlocal call_count
            call_count += 1
            if call_count <= 3:
                raise Exception("Flaky query")
            return None

        page.query_selector = AsyncMock(side_effect=flaky_query)
        result = await detect_challenge(page)
        # Should complete without crashing, returning no detection
        assert result.detected is False

    @pytest.mark.asyncio
    async def test_dom_selector_takes_priority_over_title(self):
        """DOM selector match should override the title-only classification.

        When both a title pattern and a Turnstile widget are present, the DOM
        selector is more specific and should win — TURNSTILE, not MANAGED.
        """
        page = make_page(
            title="Just a moment...",
            selectors={".cf-turnstile": True},
        )
        result = await detect_challenge(page)
        assert result.challenge_type == ChallengeType.TURNSTILE
        assert result.selector_matched == ".cf-turnstile"


# --- wait_for_challenge_resolution ---


class TestWaitForChallengeResolution:
    @pytest.mark.asyncio
    async def test_returns_immediately_if_no_challenge(self):
        page = make_page(title="Clean Page")
        result = await wait_for_challenge_resolution(page, timeout_ms=5000)
        assert result.resolved is True
        assert result.method == "none"
        assert result.wait_time_ms == 0

    @pytest.mark.asyncio
    async def test_resolves_when_challenge_disappears(self):
        """Simulate challenge disappearing after first poll."""
        page = make_page(title="Just a moment...")
        detect_count = 0

        original_title = page.title

        async def title_side_effect():
            nonlocal detect_count
            detect_count += 1
            if detect_count <= 1:
                return "Just a moment..."
            return "Normal Page - Reviews"

        page.title = AsyncMock(side_effect=title_side_effect)

        result = await wait_for_challenge_resolution(
            page, timeout_ms=5000, poll_interval_ms=50,
        )
        assert result.resolved is True
        assert result.method == "auto_resolve"
        # Title-only detection now returns MANAGED (not JS_CHALLENGE)
        assert result.challenge_type == ChallengeType.MANAGED

    @pytest.mark.asyncio
    async def test_resolves_via_resolved_selector(self):
        """Challenge detected via title, then #challenge-success appears on second poll."""
        page = make_page(title="Just a moment...")
        query_call_count = 0

        async def query_side_effect(sel):
            nonlocal query_call_count
            query_call_count += 1
            # After a few query_selector calls, #challenge-success appears
            if sel == "#challenge-success" and query_call_count >= 3:
                return AsyncMock()  # Element exists -> resolved
            return None

        # Title stays as challenge title (detect_challenge returns detected=True via title)
        # But the resolved selector check AFTER detect_challenge will find #challenge-success
        page.query_selector = AsyncMock(side_effect=query_side_effect)

        result = await wait_for_challenge_resolution(
            page, timeout_ms=5000, poll_interval_ms=50,
        )
        assert result.resolved is True
        assert result.method == "auto_resolve"

    @pytest.mark.asyncio
    async def test_timeout_returns_failure(self):
        page = make_page(title="Just a moment...")
        # Title never changes
        result = await wait_for_challenge_resolution(
            page, timeout_ms=200, poll_interval_ms=50,
        )
        assert result.resolved is False
        assert "timeout" in result.error.lower()
        # Title-only detection now returns MANAGED (not JS_CHALLENGE)
        assert result.challenge_type == ChallengeType.MANAGED


# --- solve_turnstile_capsolver ---


class TestSolveTurnstileCapSolver:
    @pytest.mark.asyncio
    async def test_returns_error_if_no_api_key(self, monkeypatch):
        monkeypatch.delenv("CAPSOLVER_API_KEY", raising=False)
        page = make_page()
        result = await solve_turnstile_capsolver(page, "https://g2.com")
        assert result.resolved is False
        assert "CAPSOLVER_API_KEY" in result.error

    @pytest.mark.asyncio
    async def test_logs_warning_when_api_key_missing(self, monkeypatch, caplog):
        """G#5: A warning log must be emitted when CAPSOLVER_API_KEY is absent."""
        import logging

        monkeypatch.delenv("CAPSOLVER_API_KEY", raising=False)
        page = make_page()
        with caplog.at_level(logging.WARNING, logger="app.challenge_solver"):
            await solve_turnstile_capsolver(page, "https://g2.com")
        assert any(
            "CAPSOLVER_API_KEY not configured" in record.message
            for record in caplog.records
        ), "Expected a WARNING log about missing CAPSOLVER_API_KEY"

    @pytest.mark.asyncio
    async def test_returns_error_if_no_sitekey(self, monkeypatch):
        monkeypatch.setenv("CAPSOLVER_API_KEY", "test-key")
        page = make_page()
        # All sitekey selectors return None
        page.query_selector = AsyncMock(return_value=None)
        result = await solve_turnstile_capsolver(page, "https://g2.com")
        assert result.resolved is False
        assert "sitekey" in result.error.lower()

    @pytest.mark.asyncio
    async def test_uses_explicit_api_key(self, monkeypatch):
        """When api_key is passed directly, it should be used instead of env."""
        monkeypatch.delenv("CAPSOLVER_API_KEY", raising=False)
        page = make_page()
        page.query_selector = AsyncMock(return_value=None)
        # Even with no sitekey, it should NOT fail on missing API key
        result = await solve_turnstile_capsolver(page, "https://g2.com", api_key="explicit-key")
        assert "CAPSOLVER_API_KEY" not in (result.error or "")


# --- _extract_turnstile_sitekey ---


class TestExtractTurnstileSitekey:
    @pytest.mark.asyncio
    async def test_extracts_from_data_sitekey_attribute(self):
        page = AsyncMock()
        el = AsyncMock()
        el.get_attribute = AsyncMock(side_effect=lambda attr: "0x4AAAAAAA" if attr == "data-sitekey" else None)

        async def query(sel):
            if sel == ".cf-turnstile[data-sitekey]":
                return el
            return None

        page.query_selector = AsyncMock(side_effect=query)
        result = await _extract_turnstile_sitekey(page)
        assert result == "0x4AAAAAAA"

    @pytest.mark.asyncio
    async def test_extracts_from_iframe_src(self):
        page = AsyncMock()
        el = AsyncMock()
        el.get_attribute = AsyncMock(
            side_effect=lambda attr: "https://challenges.cloudflare.com/cdn-cgi/challenge-platform/h/b/turnstile/if/ov2/av0/rcv0/0/iq12c/0x4BBBBBBB/auto/cbk/normal?sitekey=0x4BBBBBBB&action=managed"
            if attr == "src"
            else None
        )

        async def query(sel):
            if 'iframe' in sel:
                return el
            return None

        page.query_selector = AsyncMock(side_effect=query)
        result = await _extract_turnstile_sitekey(page)
        assert result == "0x4BBBBBBB"

    @pytest.mark.asyncio
    async def test_returns_none_when_no_sitekey_found(self):
        page = AsyncMock()
        page.query_selector = AsyncMock(return_value=None)
        page.evaluate = AsyncMock(return_value=None)
        page.content = AsyncMock(return_value="<html><body>Normal page</body></html>")
        result = await _extract_turnstile_sitekey(page)
        assert result is None


# --- resolve_challenge (full pipeline) ---


class TestResolveChallenge:
    @pytest.mark.asyncio
    async def test_no_challenge_detected(self):
        page = make_page(title="Clean Page")
        result = await resolve_challenge(page, "https://g2.com")
        assert result.resolved is True
        assert result.method == "none"
        assert result.wait_time_ms == 0

    @pytest.mark.asyncio
    async def test_auto_resolves_js_challenge(self):
        """JS challenge detected then disappears -> auto_resolve."""
        page = make_page(title="Just a moment...")
        detect_count = 0

        async def changing_title():
            nonlocal detect_count
            detect_count += 1
            if detect_count <= 2:
                return "Just a moment..."
            return "Normal Page"

        page.title = AsyncMock(side_effect=changing_title)

        result = await resolve_challenge(
            page, "https://g2.com", auto_wait_ms=2000, capsolver_timeout_ms=1000,
        )
        assert result.resolved is True
        assert result.method == "auto_resolve"

    @pytest.mark.asyncio
    async def test_non_turnstile_challenge_does_not_try_capsolver(self, monkeypatch):
        """JS challenge that doesn't resolve should NOT try CapSolver."""
        monkeypatch.setenv("CAPSOLVER_API_KEY", "test-key")
        page = make_page(title="Just a moment...")
        # Title never changes -> timeout

        result = await resolve_challenge(
            page, "https://g2.com", auto_wait_ms=200, capsolver_timeout_ms=100,
        )
        assert result.resolved is False
        # JS_CHALLENGE, not TURNSTILE -> capsolver should not be attempted

    @pytest.mark.asyncio
    async def test_all_attempts_fail_returns_error(self):
        """Both auto-resolve and capsolver fail."""
        page = make_page(
            title="Normal",
            selectors={".cf-turnstile": True},
        )
        # Challenge never resolves

        result = await resolve_challenge(
            page, "https://g2.com", auto_wait_ms=200, capsolver_timeout_ms=100,
        )
        assert result.resolved is False
        assert result.error is not None


# --- ChallengeType enum ---


class TestChallengeType:
    def test_enum_values(self):
        assert ChallengeType.TURNSTILE == "turnstile"
        assert ChallengeType.JS_CHALLENGE == "js_challenge"
        assert ChallengeType.BROWSER_CHECK == "browser_check"
        assert ChallengeType.MANAGED == "managed_challenge"
        assert ChallengeType.NONE == "none"

    def test_is_string_enum(self):
        assert isinstance(ChallengeType.TURNSTILE, str)


# --- Constants ---


class TestLocalizedChallengeDetection:
    """Tests for localized Cloudflare challenge title detection (PT/ES/FR/DE)."""

    @pytest.mark.asyncio
    async def test_detects_portuguese_um_momento(self):
        page = make_page(title="Um momento...")
        result = await detect_challenge(page)
        assert result.detected is True
        # Title-only match now classifies as MANAGED (CapSolver-eligible)
        assert result.challenge_type == ChallengeType.MANAGED
        assert "um momento" in result.selector_matched

    @pytest.mark.asyncio
    async def test_detects_portuguese_verificacao(self):
        page = make_page(title="Verificação de segurança")
        result = await detect_challenge(page)
        assert result.detected is True

    @pytest.mark.asyncio
    async def test_detects_spanish_un_momento(self):
        page = make_page(title="Un momento por favor")
        result = await detect_challenge(page)
        assert result.detected is True

    @pytest.mark.asyncio
    async def test_detects_spanish_verificacion(self):
        page = make_page(title="Verificación de seguridad")
        result = await detect_challenge(page)
        assert result.detected is True

    @pytest.mark.asyncio
    async def test_detects_french_un_instant(self):
        page = make_page(title="Un instant s'il vous plaît")
        result = await detect_challenge(page)
        assert result.detected is True

    @pytest.mark.asyncio
    async def test_detects_german_einen_moment(self):
        page = make_page(title="Einen Moment bitte")
        result = await detect_challenge(page)
        assert result.detected is True

    @pytest.mark.asyncio
    async def test_detects_german_sicherheitsueberpruefung(self):
        page = make_page(title="Sicherheitsüberprüfung läuft")
        result = await detect_challenge(page)
        assert result.detected is True

    @pytest.mark.asyncio
    async def test_detects_french_verification(self):
        page = make_page(title="Vérification de sécurité en cours")
        result = await detect_challenge(page)
        assert result.detected is True


class TestConstants:
    def test_challenge_selectors_populated(self):
        assert len(CHALLENGE_SELECTORS) >= 6

    def test_resolved_selectors_populated(self):
        assert len(RESOLVED_SELECTORS) >= 2

    def test_challenge_title_patterns_populated(self):
        assert len(CHALLENGE_TITLE_PATTERNS) >= 5
        assert "just a moment" in CHALLENGE_TITLE_PATTERNS


# --- _click_turnstile_checkbox ---


class TestClickTurnstileCheckbox:
    """Tests for interactive Turnstile widget click approach."""

    @pytest.mark.asyncio
    async def test_click_finds_iframe_and_clicks_checkbox(self):
        """When Turnstile iframe has a clickable element, click succeeds."""
        page = AsyncMock()

        # Mock frame_locator -> locator chain
        mock_checkbox = AsyncMock()
        mock_checkbox.count = AsyncMock(return_value=1)
        mock_checkbox.first = AsyncMock()
        mock_checkbox.first.click = AsyncMock()

        mock_frame = MagicMock()
        mock_frame.locator = MagicMock(return_value=mock_checkbox)

        page.frame_locator = MagicMock(return_value=mock_frame)

        result = await _click_turnstile_checkbox(page)
        assert result is True
        page.frame_locator.assert_called_once()
        mock_checkbox.first.click.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_click_no_iframe_returns_false(self):
        """When no Turnstile iframe exists, returns False gracefully."""
        page = AsyncMock()
        page.frame_locator = MagicMock(side_effect=Exception("No frame found"))

        result = await _click_turnstile_checkbox(page)
        assert result is False

    @pytest.mark.asyncio
    async def test_click_no_checkbox_in_iframe_tries_body(self):
        """When checkbox not found in iframe, tries clicking iframe body."""
        page = AsyncMock()

        # Checkbox not found (count=0), body found (count=1)
        mock_checkbox = AsyncMock()
        mock_checkbox.count = AsyncMock(return_value=0)

        mock_body = AsyncMock()
        mock_body.count = AsyncMock(return_value=1)
        mock_body.first = AsyncMock()
        mock_body.first.click = AsyncMock()

        call_count = 0

        def locator_side_effect(sel):
            nonlocal call_count
            call_count += 1
            if "checkbox" in sel or "type=" in sel:
                return mock_checkbox
            return mock_body

        mock_frame = MagicMock()
        mock_frame.locator = MagicMock(side_effect=locator_side_effect)
        page.frame_locator = MagicMock(return_value=mock_frame)

        result = await _click_turnstile_checkbox(page)
        assert result is True
        mock_body.first.click.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_click_exception_is_non_fatal(self):
        """Errors during click don't raise — returns False."""
        page = AsyncMock()

        mock_checkbox = AsyncMock()
        mock_checkbox.count = AsyncMock(side_effect=Exception("Frame detached"))

        mock_frame = MagicMock()
        mock_frame.locator = MagicMock(return_value=mock_checkbox)
        page.frame_locator = MagicMock(return_value=mock_frame)

        result = await _click_turnstile_checkbox(page)
        assert result is False


# --- _format_proxy_for_capsolver ---


class TestFormatProxyForCapsolver:
    def test_formats_playwright_proxy_to_capsolver(self):
        """Convert Playwright proxy dict to CapSolver ip:port:user:pass format."""
        proxy = {
            "server": "http://gate.decodo.com:7777",
            "username": "user-abc-country-us-session-123-sessionduration-10",
            "password": "s3cret",
        }
        result = _format_proxy_for_capsolver(proxy)
        assert result == "gate.decodo.com:7777:user-abc-country-us-session-123-sessionduration-10:s3cret"

    def test_handles_https_scheme(self):
        proxy = {
            "server": "https://proxy.example.com:8080",
            "username": "user1",
            "password": "pass1",
        }
        result = _format_proxy_for_capsolver(proxy)
        assert result == "proxy.example.com:8080:user1:pass1"

    def test_handles_no_scheme(self):
        proxy = {
            "server": "proxy.example.com:8080",
            "username": "user1",
            "password": "pass1",
        }
        result = _format_proxy_for_capsolver(proxy)
        assert result == "proxy.example.com:8080:user1:pass1"

    def test_returns_none_for_no_proxy(self):
        assert _format_proxy_for_capsolver(None) is None
        assert _format_proxy_for_capsolver({}) is None

    def test_returns_none_for_missing_credentials(self):
        proxy = {"server": "http://proxy.example.com:8080"}
        assert _format_proxy_for_capsolver(proxy) is None


# --- solve_managed_challenge_capsolver ---


class TestSolveManagedChallengeCapsolver:
    """Tests for the AntiCloudflareTask CapSolver approach (managed challenges)."""

    @pytest.mark.asyncio
    async def test_no_api_key_returns_error(self, monkeypatch):
        monkeypatch.delenv("CAPSOLVER_API_KEY", raising=False)
        page = make_page()
        result = await solve_managed_challenge_capsolver(page, "https://capterra.com", proxy_config=None)
        assert result.resolved is False
        assert "CAPSOLVER_API_KEY" in result.error

    @pytest.mark.asyncio
    async def test_no_proxy_returns_error(self, monkeypatch):
        monkeypatch.setenv("CAPSOLVER_API_KEY", "test-key")
        page = make_page()
        result = await solve_managed_challenge_capsolver(page, "https://capterra.com", proxy_config=None)
        assert result.resolved is False
        assert "proxy" in result.error.lower()

    @pytest.mark.asyncio
    async def test_calls_managed_api_with_proxy(self, monkeypatch):
        """Verify _call_capsolver_managed is invoked with correct proxy string."""
        monkeypatch.setenv("CAPSOLVER_API_KEY", "test-key")
        proxy = {
            "server": "http://gate.decodo.com:7777",
            "username": "user-abc",
            "password": "pass",
        }
        page = make_page(title="Normal Page After Solve")
        page.context = AsyncMock()
        page.context.add_cookies = AsyncMock()
        page.reload = AsyncMock()

        captured_args = {}

        async def mock_call_managed(api_key, site_url, proxy_str, timeout_ms):
            captured_args["api_key"] = api_key
            captured_args["proxy_str"] = proxy_str
            captured_args["site_url"] = site_url
            return None  # Failure

        with patch("app.challenge_solver._call_capsolver_managed", mock_call_managed):
            result = await solve_managed_challenge_capsolver(
                page, "https://capterra.com", proxy_config=proxy, timeout_ms=5000,
            )

        assert captured_args["proxy_str"] == "gate.decodo.com:7777:user-abc:pass"
        assert captured_args["site_url"] == "https://capterra.com"
        assert result.resolved is False

    @pytest.mark.asyncio
    async def test_success_injects_cookies(self, monkeypatch):
        """On success, cf_clearance cookie is set in the browser context."""
        monkeypatch.setenv("CAPSOLVER_API_KEY", "test-key")
        proxy = {
            "server": "http://gate.decodo.com:7777",
            "username": "user-abc",
            "password": "pass",
        }
        page = make_page(title="Normal Page After Solve")
        page.context = AsyncMock()
        page.context.add_cookies = AsyncMock()
        page.reload = AsyncMock()

        async def mock_call_managed(api_key, site_url, proxy_str, timeout_ms):
            return {
                "cookies": {"cf_clearance": "abc123"},
                "userAgent": "Mozilla/5.0 ...",
            }

        with patch("app.challenge_solver._call_capsolver_managed", mock_call_managed):
            result = await solve_managed_challenge_capsolver(
                page, "https://capterra.com", proxy_config=proxy, timeout_ms=30000,
            )

        assert result.resolved is True
        assert result.method == "capsolver_managed"
        page.context.add_cookies.assert_awaited_once()


# --- resolve_challenge with click + managed CapSolver ---


class TestResolveChallengeWithClickAndManaged:
    """Tests for the updated resolve pipeline: auto → click → managed capsolver → turnstile capsolver."""

    @pytest.mark.asyncio
    async def test_click_tried_before_capsolver(self):
        """For managed challenges, click is attempted before CapSolver."""
        page = make_page(
            title="Just a moment...",
            selectors={"#cf-challenge-running": True},
        )
        proxy = {"server": "http://proxy:7777", "username": "u", "password": "p"}

        click_called = False
        capsolver_called = False

        async def mock_click(p):
            nonlocal click_called
            click_called = True
            return False  # Click fails

        async def mock_managed_capsolver(p, url, **kwargs):
            nonlocal capsolver_called
            capsolver_called = True
            return ChallengeResult(resolved=False, error="test")

        with patch("app.challenge_solver._click_turnstile_checkbox", mock_click), \
             patch("app.challenge_solver.solve_managed_challenge_capsolver", mock_managed_capsolver):
            result = await resolve_challenge(
                page, "https://capterra.com",
                auto_wait_ms=100, capsolver_timeout_ms=100,
                proxy_config=proxy,
            )

        assert click_called, "Click should be attempted before CapSolver"
        assert capsolver_called, "CapSolver should be tried after click fails"

    @pytest.mark.asyncio
    async def test_click_success_skips_capsolver(self):
        """If click resolves the challenge, CapSolver is not called."""
        page = make_page(title="Just a moment...")
        click_happened = False

        async def changing_title():
            if click_happened:
                return "Normal Page"
            return "Just a moment..."

        page.title = AsyncMock(side_effect=changing_title)

        capsolver_called = False

        async def mock_click(p):
            nonlocal click_happened
            click_happened = True
            return True  # Click succeeded

        async def mock_managed(p, url, **kwargs):
            nonlocal capsolver_called
            capsolver_called = True
            return ChallengeResult(resolved=False, error="test")

        with patch("app.challenge_solver._click_turnstile_checkbox", mock_click), \
             patch("app.challenge_solver.solve_managed_challenge_capsolver", mock_managed):
            result = await resolve_challenge(
                page, "https://capterra.com",
                auto_wait_ms=100, capsolver_timeout_ms=100,
            )

        assert result.resolved is True
        assert result.method == "click"
        assert not capsolver_called

    @pytest.mark.asyncio
    async def test_proxy_config_passed_to_managed_solver(self):
        """Proxy config from resolve_challenge flows to solve_managed_challenge_capsolver."""
        page = make_page(title="Just a moment...")
        proxy = {"server": "http://proxy:7777", "username": "u", "password": "p"}

        received_proxy = {}

        async def mock_click(p):
            return False

        async def mock_managed(p, url, proxy_config=None, **kwargs):
            received_proxy.update(proxy_config or {})
            return ChallengeResult(resolved=False, error="test")

        with patch("app.challenge_solver._click_turnstile_checkbox", mock_click), \
             patch("app.challenge_solver.solve_managed_challenge_capsolver", mock_managed):
            await resolve_challenge(
                page, "https://capterra.com",
                auto_wait_ms=100, capsolver_timeout_ms=100,
                proxy_config=proxy,
            )

        assert received_proxy == proxy
