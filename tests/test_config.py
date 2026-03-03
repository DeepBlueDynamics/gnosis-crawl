"""Unit tests for app.config — Settings and RunConfig builder."""

import pytest

from app.config import Settings, settings
from app.agent.types import RunConfig


class TestSettings:
    """Test Settings defaults using _env_file=None to isolate from local .env."""

    def test_defaults(self):
        s = Settings(_env_file=None)
        assert s.host == "0.0.0.0"
        assert s.port == 6792
        assert s.debug is False
        assert s.agent_enabled is False
        assert s.agent_provider == "openai"
        assert s.agent_ghost_enabled is False
        assert s.browser_stream_enabled is False

    def test_agent_defaults(self):
        s = Settings(_env_file=None)
        assert s.agent_max_steps == 12
        assert s.agent_max_wall_time_ms == 90_000
        assert s.agent_max_failures == 3
        assert s.agent_block_private_ranges is True
        assert s.agent_redact_secrets is True

    def test_ghost_defaults(self):
        s = Settings(_env_file=None)
        assert s.agent_ghost_enabled is False
        assert s.agent_ghost_auto_trigger is True
        assert s.agent_ghost_vision_provider == ""
        assert s.agent_ghost_max_image_width == 1280
        assert s.agent_ghost_max_retries == 1

    def test_challenge_resolution_defaults(self):
        s = Settings(_env_file=None)
        assert s.challenge_auto_wait_ms == 15000
        assert s.challenge_capsolver_timeout_ms == 30000

    def test_proxy_rotation_defaults(self):
        s = Settings(_env_file=None)
        assert s.proxy_session_duration_minutes == 10
        assert s.proxy_restart_after_failures == 3

    def test_stream_defaults(self):
        s = Settings(_env_file=None)
        assert s.browser_pool_size == 1
        assert s.browser_stream_enabled is False
        assert s.browser_stream_quality == 25
        assert s.browser_stream_max_width == 854
        assert s.browser_stream_max_lease_seconds == 300

    def test_provider_defaults(self):
        s = Settings(_env_file=None)
        assert s.openai_model == "gpt-4.1-mini"
        assert s.anthropic_model == "claude-3-5-sonnet-latest"
        assert s.ollama_model == "llama3.1:8b-instruct"
        assert s.ollama_base_url == "http://localhost:11434"


class TestAgentAllowedTools:
    def test_empty_string(self):
        s = Settings(_env_file=None, agent_allowed_tools="")
        assert s.get_agent_allowed_tools() == []

    def test_single_tool(self):
        s = Settings(_env_file=None, agent_allowed_tools="crawl")
        assert s.get_agent_allowed_tools() == ["crawl"]

    def test_multiple_tools(self):
        s = Settings(_env_file=None, agent_allowed_tools="crawl, markdown, batch")
        assert s.get_agent_allowed_tools() == ["crawl", "markdown", "batch"]

    def test_strips_whitespace(self):
        s = Settings(_env_file=None, agent_allowed_tools="  crawl , markdown  ")
        assert s.get_agent_allowed_tools() == ["crawl", "markdown"]


class TestAgentAllowedDomains:
    def test_empty_string(self):
        s = Settings(_env_file=None, agent_allowed_domains="")
        assert s.get_agent_allowed_domains() == []

    def test_multiple_domains(self):
        s = Settings(_env_file=None, agent_allowed_domains="example.com, test.com")
        assert s.get_agent_allowed_domains() == ["example.com", "test.com"]


class TestBuildRunConfig:
    def test_returns_run_config(self):
        s = Settings(_env_file=None)
        cfg = s.build_run_config()
        assert isinstance(cfg, RunConfig)

    def test_maps_values(self):
        s = Settings(
            _env_file=None,
            agent_max_steps=20,
            agent_max_wall_time_ms=60_000,
            agent_max_failures=5,
            agent_allowed_tools="crawl,markdown",
            agent_allowed_domains="example.com",
            agent_block_private_ranges=False,
            agent_redact_secrets=False,
        )
        cfg = s.build_run_config()
        assert cfg.max_steps == 20
        assert cfg.max_wall_time_ms == 60_000
        assert cfg.max_failures == 5
        assert cfg.allowed_tools == ["crawl", "markdown"]
        assert cfg.allowed_domains == ["example.com"]
        assert cfg.block_private_ranges is False
        assert cfg.redact_secrets is False


class TestStickyProxyConfig:
    """Tests for get_sticky_proxy_config() — Decodo sticky session proxy."""

    def test_returns_none_when_no_proxy_server(self):
        s = Settings(_env_file=None, proxy_server=None)
        assert s.get_sticky_proxy_config() is None

    def test_returns_sticky_username_format(self):
        s = Settings(
            _env_file=None,
            proxy_server="http://gate.decodo.com:7000",
            proxy_username="spwod13p0r",
            proxy_password="secret123",
        )
        config = s.get_sticky_proxy_config(session_id="abc123", duration_minutes=30)
        assert config is not None
        assert config["server"] == "http://gate.decodo.com:7000"
        assert config["username"] == "user-spwod13p0r-country-us-session-abc123-sessionduration-30"
        assert config["password"] == "secret123"

    def test_generates_session_id_when_not_provided(self):
        s = Settings(
            _env_file=None,
            proxy_server="http://gate.decodo.com:7000",
            proxy_username="spwod13p0r",
            proxy_password="secret123",
        )
        config = s.get_sticky_proxy_config()
        assert config is not None
        # Should contain a generated session ID (12-char hex)
        assert "session-" in config["username"]
        # Default duration is now proxy_session_duration_minutes (10)
        assert "sessionduration-10" in config["username"]

    def test_custom_duration(self):
        s = Settings(
            _env_file=None,
            proxy_server="http://gate.decodo.com:7000",
            proxy_username="spwod13p0r",
            proxy_password="secret123",
        )
        config = s.get_sticky_proxy_config(session_id="test", duration_minutes=60)
        assert "sessionduration-60" in config["username"]

    def test_no_username_still_returns_config(self):
        s = Settings(
            _env_file=None,
            proxy_server="http://gate.decodo.com:7000",
            proxy_username=None,
            proxy_password="secret123",
        )
        config = s.get_sticky_proxy_config()
        assert config is not None
        assert config["server"] == "http://gate.decodo.com:7000"
        assert "username" not in config or config.get("username") is None

    def test_does_not_include_bypass(self):
        """Sticky proxy config should not include bypass (Camoufox doesn't support it)."""
        s = Settings(
            _env_file=None,
            proxy_server="http://gate.decodo.com:7000",
            proxy_username="spwod13p0r",
            proxy_password="secret123",
            proxy_bypass="localhost",
        )
        config = s.get_sticky_proxy_config(session_id="test")
        assert "bypass" not in config


class TestGlobalSettings:
    def test_singleton_exists(self):
        assert settings is not None
        assert isinstance(settings, Settings)
