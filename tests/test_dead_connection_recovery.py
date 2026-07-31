"""
Regression tests for the dead-browser-connection bug seen in production.

Symptom: a long-lived Cloud Run instance's Playwright driver subprocess dies
(its IPC pipe closes), and every subsequent crawl fails forever with:

    RuntimeError: Browser.new_context: unable to perform operation on
    <WriteUnixTransport closed=True reading=False 0x...>; the handler is closed

create_isolated_context() only called start_browser() when self.browser was
None, so a browser object that exists but is disconnected was never replaced.
Fixed by checking is_connected() before reuse, and by self-healing (restart +
one retry) if new_context() itself raises a dead-connection error.

Created: 2026-07-31
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.browser import BrowserEngine, _is_dead_connection_error


DEAD_TRANSPORT_ERROR = RuntimeError(
    "Browser.new_context: unable to perform operation on "
    "<WriteUnixTransport closed=True reading=False 0x7f9d416b0ba0>; "
    "the handler is closed"
)


class TestIsDeadConnectionError:
    def test_matches_production_error_text(self):
        assert _is_dead_connection_error(DEAD_TRANSPORT_ERROR)

    def test_does_not_match_unrelated_error(self):
        assert not _is_dead_connection_error(TimeoutError("Navigation timeout of 30000ms exceeded"))


class TestCreateIsolatedContextSelfHeals:
    """create_isolated_context() must detect and recover from a dead browser."""

    def _mock_context_and_page(self):
        context = AsyncMock()
        context.new_page = AsyncMock(return_value=AsyncMock())
        return context

    @pytest.mark.asyncio
    async def test_restarts_when_browser_disconnected(self, monkeypatch):
        """A truthy but disconnected self.browser must trigger start_browser()."""
        import app.stealth as stealth_mod

        engine = BrowserEngine()
        dead_browser = MagicMock()
        dead_browser.is_connected.return_value = False
        engine.browser = dead_browser

        fresh_browser = MagicMock()
        fresh_browser.is_connected.return_value = True
        fresh_context = self._mock_context_and_page()
        fresh_browser.new_context = AsyncMock(return_value=fresh_context)

        async def fake_start_browser(javascript_enabled=True):
            engine.browser = fresh_browser

        engine.start_browser = AsyncMock(side_effect=fake_start_browser)
        monkeypatch.setattr(stealth_mod, "apply_stealth", AsyncMock())
        monkeypatch.setattr(stealth_mod, "setup_request_interception", AsyncMock())
        monkeypatch.setattr(stealth_mod, "apply_chromium_js_patches", AsyncMock())

        context, page = await engine.create_isolated_context()

        engine.start_browser.assert_awaited_once()
        assert context is fresh_context

    @pytest.mark.asyncio
    async def test_new_context_dead_transport_triggers_restart_and_retry(self, monkeypatch):
        """new_context() raising the dead-transport error must self-heal, not propagate."""
        import app.stealth as stealth_mod

        engine = BrowserEngine()
        browser = MagicMock()
        browser.is_connected.return_value = True
        good_context = self._mock_context_and_page()
        browser.new_context = AsyncMock(side_effect=[DEAD_TRANSPORT_ERROR, good_context])
        engine.browser = browser

        engine.close = AsyncMock()

        async def fake_start_browser(javascript_enabled=True):
            pass  # engine.browser stays the same mock; only new_context's side_effect changes

        engine.start_browser = AsyncMock(side_effect=fake_start_browser)
        monkeypatch.setattr(stealth_mod, "apply_stealth", AsyncMock())
        monkeypatch.setattr(stealth_mod, "setup_request_interception", AsyncMock())
        monkeypatch.setattr(stealth_mod, "apply_chromium_js_patches", AsyncMock())

        context, page = await engine.create_isolated_context()

        engine.close.assert_awaited_once()
        engine.start_browser.assert_awaited_once()
        assert browser.new_context.await_count == 2
        assert context is good_context

    @pytest.mark.asyncio
    async def test_unrelated_new_context_error_still_propagates(self, monkeypatch):
        """A non-dead-connection error from new_context() must not be swallowed."""
        from app import browser as browser_mod

        engine = BrowserEngine()
        browser = MagicMock()
        browser.is_connected.return_value = True
        browser.new_context = AsyncMock(side_effect=ValueError("some other failure"))
        engine.browser = browser
        engine.close = AsyncMock()
        engine.start_browser = AsyncMock()

        with pytest.raises(ValueError):
            await engine.create_isolated_context()

        engine.close.assert_not_awaited()


class TestStartBrowserDoesNotDeadlock:
    """Regression test for a self-deadlock that wedged production on 2026-07-31.

    start_browser() holds self._browser_lock (asyncio.Lock, not reentrant)
    for its whole body. Both places it needs to tear down a dead/partial
    browser -- the "disconnected, recreating" branch and the launch-failure
    except clause -- used to call self.close(), which itself does
    `async with self._browser_lock:`. Re-entering a lock the same task
    already holds blocks forever: the task can never complete, so it never
    releases the lock, so every subsequent request that needs the browser
    (any domain, not just the one that failed) piles up behind it and the
    whole service goes unresponsive until the instance is killed.

    Real trigger: Camoufox segfaulting during its first launch on a fresh
    Cloud Run instance (SIGSEGV inside context.new_page()) took the launch
    exception path straight into this deadlock.
    """

    @pytest.mark.asyncio
    async def test_disconnected_browser_recreate_does_not_deadlock(self, monkeypatch):
        """start_browser() must complete (not hang) when self.browser exists but is dead."""
        import app.browser as browser_mod

        mock_settings = MagicMock()
        mock_settings.browser_engine = "camoufox"
        mock_settings.browser_headless = True
        mock_settings.max_concurrent_crawls = 4
        mock_settings.proxy_restart_after_failures = 3
        mock_settings.get_sticky_proxy_config.return_value = None
        mock_settings.proxy_session_duration_minutes = 5
        monkeypatch.setattr(browser_mod, "settings", mock_settings)

        engine = BrowserEngine()
        dead_browser = MagicMock()
        dead_browser.is_connected.return_value = False
        dead_browser.close = AsyncMock()
        engine.browser = dead_browser

        fresh_browser = MagicMock()
        fresh_browser.is_connected.return_value = True
        fresh_context = AsyncMock()
        fresh_page = AsyncMock()
        fresh_context.new_page = AsyncMock(return_value=fresh_page)
        fresh_browser.new_context = AsyncMock(return_value=fresh_context)

        mock_cm = AsyncMock()
        mock_cm.__aenter__ = AsyncMock(return_value=fresh_browser)
        monkeypatch.setattr(engine, "_check_exit_ip", AsyncMock())
        monkeypatch.setattr(browser_mod, "AsyncCamoufox", MagicMock(return_value=mock_cm), raising=False)
        monkeypatch.setitem(
            __import__("sys").modules, "camoufox.async_api",
            MagicMock(AsyncCamoufox=MagicMock(return_value=mock_cm)),
        )

        # Must finish promptly, not hang forever re-acquiring self._browser_lock.
        await asyncio.wait_for(engine.start_browser(), timeout=2.0)

        dead_browser.close.assert_awaited_once()
        assert engine.browser is fresh_browser
        assert not engine._browser_lock.locked()

    @pytest.mark.asyncio
    async def test_launch_failure_cleanup_does_not_deadlock(self, monkeypatch):
        """A crash mid-launch (e.g. Camoufox segfault) must not wedge the lock forever."""
        import app.browser as browser_mod

        mock_settings = MagicMock()
        mock_settings.browser_engine = "camoufox"
        mock_settings.browser_headless = True
        mock_settings.max_concurrent_crawls = 4
        mock_settings.proxy_restart_after_failures = 3
        mock_settings.get_sticky_proxy_config.return_value = None
        mock_settings.proxy_session_duration_minutes = 5
        monkeypatch.setattr(browser_mod, "settings", mock_settings)

        engine = BrowserEngine()

        crashed_browser = MagicMock()
        crashed_browser.close = AsyncMock()
        crashed_context = AsyncMock()
        crashed_context.new_page = AsyncMock(
            side_effect=RuntimeError("BrowserContext.new_page: Target page, context or browser has been closed")
        )
        crashed_browser.new_context = AsyncMock(return_value=crashed_context)

        mock_cm = AsyncMock()
        mock_cm.__aenter__ = AsyncMock(return_value=crashed_browser)
        monkeypatch.setitem(
            __import__("sys").modules, "camoufox.async_api",
            MagicMock(AsyncCamoufox=MagicMock(return_value=mock_cm)),
        )

        with pytest.raises(RuntimeError):
            await asyncio.wait_for(engine.start_browser(), timeout=2.0)

        assert not engine._browser_lock.locked()

        # The lock must be usable again immediately -- proves it wasn't
        # left stuck, not just that this one call happened to finish.
        async with engine._browser_lock:
            pass
