"""
Configuration management for Grub Crawler
Environment-based configuration following gnosis-ocr pattern
"""
import os
import uuid
from typing import Optional
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings from environment variables"""

    # Server Configuration
    host: str = "0.0.0.0"
    port: int = 6792
    debug: bool = False

    # Storage Configuration
    storage_path: str = "./storage"
    running_in_cloud: bool = False
    gcs_bucket_name: Optional[str] = None

    # Authentication
    disable_auth: bool = False
    gnosis_auth_url: str = "http://gnosis-auth:5000"

    # Crawling Configuration
    max_concurrent_crawls: int = 5
    crawl_timeout: int = 30
    enable_javascript: bool = True
    enable_screenshots: bool = False

    # Browser Configuration
    browser_engine: str = "camoufox"  # "camoufox" or "chromium"
    browser_headless: bool = True
    browser_timeout: int = 30000

    # Cloud Configuration
    google_cloud_project: Optional[str] = None
    cloud_tasks_queue: str = "crawl-jobs"
    cloud_tasks_location: str = "us-central1"
    worker_service_url: Optional[str] = None

    # Agent Configuration (Mode B - internal micro-agent loop)
    agent_enabled: bool = False
    agent_max_steps: int = 12
    agent_max_wall_time_ms: int = 90_000
    agent_max_failures: int = 3
    agent_allowed_tools: str = ""  # comma-separated allowlist; empty = all registered
    agent_allowed_domains: str = ""  # comma-separated domain allowlist; empty = all
    agent_block_private_ranges: bool = True
    agent_redact_secrets: bool = True
    agent_persist_raw_html: bool = False

    # Ghost Protocol Configuration (vision-based anti-bot fallback)
    agent_ghost_enabled: bool = False
    agent_ghost_auto_trigger: bool = True
    agent_ghost_vision_provider: str = ""  # inherits from agent_provider if empty
    agent_ghost_max_image_width: int = 1280
    agent_ghost_max_retries: int = 1

    # Live Browser Stream Configuration
    browser_pool_size: int = 1
    browser_stream_enabled: bool = False
    browser_stream_quality: int = 25  # JPEG quality (1-100), lower = faster
    browser_stream_max_width: int = 854
    browser_stream_max_lease_seconds: int = 300  # max time a slot can be leased

    # LLM Provider Configuration
    agent_provider: str = "openai"  # openai | anthropic | ollama
    openai_api_key: Optional[str] = None
    openai_model: str = "gpt-4.1-mini"
    anthropic_api_key: Optional[str] = None
    anthropic_model: str = "claude-3-5-sonnet-latest"
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.1:8b-instruct"

    # Proxy Configuration
    proxy_server: Optional[str] = None
    proxy_username: Optional[str] = None
    proxy_password: Optional[str] = None
    proxy_bypass: str = ""

    # Challenge Resolution Configuration
    challenge_auto_wait_ms: int = 30000       # default auto-wait for challenge resolution (30s)
    challenge_capsolver_timeout_ms: int = 30000  # CapSolver API timeout

    # Proxy Rotation Configuration
    proxy_session_duration_minutes: int = 10  # shorter sticky sessions (was 30)
    proxy_restart_after_failures: int = 3     # restart browser after N consecutive timeouts

    # Stealth Configuration
    stealth_enabled: bool = True
    block_tracking_domains: bool = True

    # TLS Impersonation
    tls_impersonate_chromium: bool = False

    # HTTP Pre-check Configuration
    http_precheck_enabled: bool = False
    http_precheck_timeout: int = 15
    http_precheck_impersonate: str = "chrome135"

    # Mesh Coordinator Configuration
    mesh_enabled: bool = False
    mesh_peers: str = ""  # comma-separated seed peer URLs
    mesh_node_name: str = ""  # human-readable; defaults to hostname
    mesh_secret: str = ""  # shared HMAC secret for inter-node auth
    mesh_heartbeat_interval_s: float = 15.0
    mesh_peer_timeout_s: float = 45.0
    mesh_peer_remove_s: float = 120.0
    mesh_prefer_local: bool = True  # bias toward local execution
    mesh_remote_timeout_ms: int = 35_000
    mesh_advertise_url: str = ""  # URL peers use to reach this node

    class Config:
        env_file = ".env"
        case_sensitive = False
        extra = "ignore"

        # Map environment variable names
        fields = {
            'running_in_cloud': {'env': 'RUNNING_IN_CLOUD'},
            'gcs_bucket_name': {'env': 'GCS_BUCKET_NAME'},
            'gnosis_auth_url': {'env': 'GNOSIS_AUTH_URL'},
            'max_concurrent_crawls': {'env': 'MAX_CONCURRENT_CRAWLS'},
            'crawl_timeout': {'env': 'CRAWL_TIMEOUT'},
            'enable_javascript': {'env': 'ENABLE_JAVASCRIPT'},
            'enable_screenshots': {'env': 'ENABLE_SCREENSHOTS'},
            'google_cloud_project': {'env': 'GOOGLE_CLOUD_PROJECT'},
            'cloud_tasks_queue': {'env': 'CLOUD_TASKS_QUEUE'},
            'cloud_tasks_location': {'env': 'CLOUD_TASKS_LOCATION'},
            'worker_service_url': {'env': 'WORKER_SERVICE_URL'},
        }

    def is_cloud_environment(self) -> bool:
        """Check if running in cloud environment"""
        return os.environ.get('RUNNING_IN_CLOUD', '').lower() == 'true'

    def get_agent_allowed_tools(self) -> list[str]:
        """Parse comma-separated tool allowlist."""
        if not self.agent_allowed_tools:
            return []
        return [t.strip() for t in self.agent_allowed_tools.split(",") if t.strip()]

    def get_agent_allowed_domains(self) -> list[str]:
        """Parse comma-separated domain allowlist."""
        if not self.agent_allowed_domains:
            return []
        return [d.strip() for d in self.agent_allowed_domains.split(",") if d.strip()]

    def get_mesh_peers(self) -> list[str]:
        """Parse comma-separated mesh seed peer URLs."""
        if not self.mesh_peers:
            return []
        return [p.strip() for p in self.mesh_peers.split(",") if p.strip()]

    def get_proxy_config(self) -> Optional[dict]:
        """Return Playwright-compatible proxy dict or None."""
        if not self.proxy_server:
            return None
        config = {"server": self.proxy_server}
        if self.proxy_username:
            config["username"] = self.proxy_username
        if self.proxy_password:
            config["password"] = self.proxy_password
        if self.proxy_bypass:
            config["bypass"] = self.proxy_bypass
        return config

    def get_sticky_proxy_config(self, session_id: str = None, duration_minutes: int = None) -> Optional[dict]:
        """Return proxy config with Decodo sticky session.

        Sticky sessions keep the same exit IP for the session duration,
        which is required for Cloudflare challenge resolution (the challenge
        page and verification callback must come from the same IP).

        Username format: user-{USERNAME}-country-us-session-{ID}-sessionduration-{MINUTES}
        """
        if not self.proxy_server:
            return None

        config = {"server": self.proxy_server}

        if self.proxy_username:
            sid = session_id or uuid.uuid4().hex[:12]
            dur = duration_minutes if duration_minutes is not None else self.proxy_session_duration_minutes
            config["username"] = f"user-{self.proxy_username}-country-us-session-{sid}-sessionduration-{dur}"

        if self.proxy_password:
            config["password"] = self.proxy_password

        return config

    def build_run_config(self):
        """Build a RunConfig from current settings. Imported lazily to avoid circular deps."""
        from app.agent.types import RunConfig
        return RunConfig(
            max_steps=self.agent_max_steps,
            max_wall_time_ms=self.agent_max_wall_time_ms,
            max_failures=self.agent_max_failures,
            allowed_tools=self.get_agent_allowed_tools(),
            allowed_domains=self.get_agent_allowed_domains(),
            block_private_ranges=self.agent_block_private_ranges,
            redact_secrets=self.agent_redact_secrets,
            persist_raw_html=self.agent_persist_raw_html,
        )


# Global settings instance
settings = Settings()
