"""Tests for Camoufox browser engine integration."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock


class TestConfigBrowserEngine:
    """Config correctly exposes browser_engine setting."""

    def test_default_is_camoufox(self):
        from app.config import Settings
        s = Settings(_env_file=None)
        assert s.browser_engine == "camoufox"

    def test_camoufox_from_env(self):
        from app.config import Settings
        s = Settings(_env_file=None, browser_engine="camoufox")
        assert s.browser_engine == "camoufox"


@pytest.mark.asyncio
class TestStartBrowserCamoufox:
    """start_browser() uses AsyncCamoufox when engine=camoufox."""

    async def test_start_browser_uses_camoufox_when_configured(self):
        """start_browser() imports and calls AsyncCamoufox when engine=camoufox."""
        from app.browser import BrowserEngine

        mock_browser = AsyncMock()
        mock_context = AsyncMock()
        mock_page = AsyncMock()
        mock_browser.new_context = AsyncMock(return_value=mock_context)
        mock_context.new_page = AsyncMock(return_value=mock_page)

        mock_cm = AsyncMock()
        mock_cm.__aenter__ = AsyncMock(return_value=mock_browser)
        mock_cm.__aexit__ = AsyncMock(return_value=False)

        mock_async_camoufox = MagicMock(return_value=mock_cm)

        with patch("app.browser.settings") as mock_settings:
            mock_settings.browser_engine = "camoufox"
            mock_settings.browser_headless = True
            mock_settings.max_concurrent_crawls = 5
            mock_settings.get_proxy_config.return_value = None
            mock_settings.get_sticky_proxy_config.return_value = None

            with patch.dict(
                "sys.modules",
                {"camoufox": MagicMock(), "camoufox.async_api": MagicMock(AsyncCamoufox=mock_async_camoufox)},
            ):
                engine = BrowserEngine()
                await engine.start_browser()

                mock_async_camoufox.assert_called_once()
                call_kwargs = mock_async_camoufox.call_args[1]
                assert call_kwargs["headless"] == "virtual"
                assert call_kwargs["geoip"] is True
                assert engine.browser is mock_browser
                assert engine.context is mock_context
                assert engine.page is mock_page

                await engine.close()

    async def test_start_browser_uses_chromium_by_default(self):
        """start_browser() uses playwright.chromium.launch() when engine=chromium."""
        from app.browser import BrowserEngine

        mock_playwright = AsyncMock()
        mock_browser = AsyncMock()
        mock_browser.is_connected.return_value = True
        mock_context = AsyncMock()
        mock_page = AsyncMock()

        mock_playwright.chromium.launch = AsyncMock(return_value=mock_browser)
        mock_browser.new_context = AsyncMock(return_value=mock_context)
        mock_context.new_page = AsyncMock(return_value=mock_page)

        with patch("app.browser.settings") as mock_settings:
            mock_settings.browser_engine = "chromium"
            mock_settings.browser_headless = True
            mock_settings.max_concurrent_crawls = 5

            with patch("app.browser.async_playwright") as mock_pw_ctx:
                mock_pw_ctx.return_value = AsyncMock()
                mock_pw_ctx.return_value.start = AsyncMock(return_value=mock_playwright)

                engine = BrowserEngine()
                await engine.start_browser()

                mock_playwright.chromium.launch.assert_called_once()
                assert engine.browser is mock_browser

                await engine.close()

    async def test_start_browser_camoufox_passes_proxy(self):
        """start_browser() passes proxy config to AsyncCamoufox."""
        from app.browser import BrowserEngine

        mock_browser = AsyncMock()
        mock_context = AsyncMock()
        mock_page = AsyncMock()
        mock_browser.new_context = AsyncMock(return_value=mock_context)
        mock_context.new_page = AsyncMock(return_value=mock_page)

        mock_cm = AsyncMock()
        mock_cm.__aenter__ = AsyncMock(return_value=mock_browser)
        mock_cm.__aexit__ = AsyncMock(return_value=False)

        mock_async_camoufox = MagicMock(return_value=mock_cm)
        proxy_config = {"server": "http://proxy:8080", "username": "user", "password": "pass"}

        with patch("app.browser.settings") as mock_settings:
            mock_settings.browser_engine = "camoufox"
            mock_settings.browser_headless = True
            mock_settings.max_concurrent_crawls = 5
            mock_settings.get_proxy_config.return_value = proxy_config
            mock_settings.get_sticky_proxy_config.return_value = proxy_config

            with patch.dict(
                "sys.modules",
                {"camoufox": MagicMock(), "camoufox.async_api": MagicMock(AsyncCamoufox=mock_async_camoufox)},
            ):
                engine = BrowserEngine()
                await engine.start_browser()

                call_kwargs = mock_async_camoufox.call_args[1]
                assert call_kwargs["proxy"] == proxy_config

                await engine.close()


@pytest.mark.asyncio
class TestCreateIsolatedContextCamoufox:
    """create_isolated_context() skips manual fingerprinting for camoufox."""

    async def test_skips_stealth_for_camoufox(self):
        """create_isolated_context() does NOT call apply_stealth() for camoufox."""
        from app.browser import BrowserEngine

        mock_browser = AsyncMock()
        mock_browser.is_connected.return_value = True
        mock_context = AsyncMock()
        mock_page = AsyncMock()
        mock_browser.new_context = AsyncMock(return_value=mock_context)
        mock_context.new_page = AsyncMock(return_value=mock_page)

        engine = BrowserEngine()
        engine.browser = mock_browser

        with patch("app.browser.settings") as mock_settings:
            mock_settings.browser_engine = "camoufox"

            with patch("app.stealth.apply_stealth", new_callable=AsyncMock) as mock_apply:
                with patch("app.stealth.setup_request_interception", new_callable=AsyncMock) as mock_intercept:
                    ctx, pg = await engine.create_isolated_context()

                    mock_apply.assert_not_called()

    async def test_keeps_request_interception_for_camoufox(self):
        """create_isolated_context() still calls setup_request_interception() for camoufox."""
        from app.browser import BrowserEngine

        mock_browser = AsyncMock()
        mock_browser.is_connected.return_value = True
        mock_context = AsyncMock()
        mock_page = AsyncMock()
        mock_browser.new_context = AsyncMock(return_value=mock_context)
        mock_context.new_page = AsyncMock(return_value=mock_page)

        engine = BrowserEngine()
        engine.browser = mock_browser

        with patch("app.browser.settings") as mock_settings:
            mock_settings.browser_engine = "camoufox"

            with patch("app.stealth.apply_stealth", new_callable=AsyncMock):
                with patch("app.stealth.setup_request_interception", new_callable=AsyncMock) as mock_intercept:
                    ctx, pg = await engine.create_isolated_context()

                    mock_intercept.assert_called_once_with(mock_context)

    async def test_skips_manual_fingerprinting_for_camoufox(self):
        """create_isolated_context() doesn't set UA/viewport/timezone for camoufox."""
        from app.browser import BrowserEngine

        mock_browser = AsyncMock()
        mock_browser.is_connected.return_value = True
        mock_context = AsyncMock()
        mock_page = AsyncMock()
        mock_browser.new_context = AsyncMock(return_value=mock_context)
        mock_context.new_page = AsyncMock(return_value=mock_page)

        engine = BrowserEngine()
        engine.browser = mock_browser

        with patch("app.browser.settings") as mock_settings:
            mock_settings.browser_engine = "camoufox"

            with patch("app.stealth.apply_stealth", new_callable=AsyncMock):
                with patch("app.stealth.setup_request_interception", new_callable=AsyncMock):
                    ctx, pg = await engine.create_isolated_context()

                    # Verify new_context was called without manual fingerprint args
                    call_kwargs = mock_browser.new_context.call_args[1]
                    assert "user_agent" not in call_kwargs
                    assert "viewport" not in call_kwargs
                    assert "timezone_id" not in call_kwargs
                    assert "locale" not in call_kwargs

    async def test_skips_per_context_proxy_for_camoufox(self):
        """create_isolated_context() does NOT pass proxy to camoufox context.

        Camoufox uses browser-level proxy only.  Setting proxy per-context
        causes 407 auth races when multiple contexts start simultaneously.
        """
        from app.browser import BrowserEngine

        mock_browser = AsyncMock()
        mock_browser.is_connected.return_value = True
        mock_context = AsyncMock()
        mock_page = AsyncMock()
        mock_browser.new_context = AsyncMock(return_value=mock_context)
        mock_context.new_page = AsyncMock(return_value=mock_page)

        engine = BrowserEngine()
        engine.browser = mock_browser
        proxy = {"server": "http://proxy:9090"}

        with patch("app.browser.settings") as mock_settings:
            mock_settings.browser_engine = "camoufox"

            with patch("app.stealth.apply_stealth", new_callable=AsyncMock):
                with patch("app.stealth.setup_request_interception", new_callable=AsyncMock):
                    ctx, pg = await engine.create_isolated_context(proxy=proxy)

                    call_kwargs = mock_browser.new_context.call_args[1]
                    assert "proxy" not in call_kwargs


@pytest.mark.asyncio
class TestApplyStealthCamoufox:
    """apply_stealth() is a no-op when engine is camoufox."""

    async def test_noop_for_camoufox(self):
        """apply_stealth() returns immediately when engine=camoufox."""
        mock_context = MagicMock()

        with patch("app.stealth.settings") as mock_settings:
            mock_settings.stealth_enabled = True
            mock_settings.browser_engine = "camoufox"

            from app.stealth import apply_stealth
            # Should not raise, should not call any stealth methods
            await apply_stealth(mock_context)

    async def test_still_applies_for_chromium(self):
        """apply_stealth() still works for chromium engine."""
        mock_context = MagicMock()
        mock_apply = AsyncMock()
        mock_stealth_instance = MagicMock()
        mock_stealth_instance.apply_stealth_async = mock_apply
        mock_stealth_cls = MagicMock(return_value=mock_stealth_instance)

        with patch("app.stealth.settings") as mock_settings:
            mock_settings.stealth_enabled = True
            mock_settings.browser_engine = "chromium"
            with patch.dict("sys.modules", {"playwright_stealth": MagicMock(Stealth=mock_stealth_cls)}):
                from app.stealth import apply_stealth
                await apply_stealth(mock_context)
                mock_stealth_cls.assert_called_once()
                mock_apply.assert_called_once_with(mock_context)


@pytest.mark.asyncio
class TestCloseCamoufox:
    """close() properly cleans up Camoufox context manager."""

    async def test_calls_camoufox_aexit(self):
        """close() calls __aexit__ on Camoufox context manager."""
        from app.browser import BrowserEngine

        mock_cm = AsyncMock()
        mock_cm.__aexit__ = AsyncMock(return_value=False)

        engine = BrowserEngine()
        engine._camoufox_cm = mock_cm
        engine.browser = AsyncMock()
        engine.page = AsyncMock()
        engine.page.is_closed.return_value = False
        engine.context = AsyncMock()

        await engine.close()

        mock_cm.__aexit__.assert_called_once_with(None, None, None)
        assert engine._camoufox_cm is None

    async def test_close_without_camoufox_cm(self):
        """close() works fine when _camoufox_cm is not set (chromium path)."""
        from app.browser import BrowserEngine

        engine = BrowserEngine()
        engine.browser = AsyncMock()
        engine.page = AsyncMock()
        engine.page.is_closed.return_value = False
        engine.context = AsyncMock()

        # Should not raise
        await engine.close()
