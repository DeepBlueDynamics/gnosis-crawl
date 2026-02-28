"""
Pydantic models for Grub Crawler API requests and responses
"""
from typing import List, Optional, Dict, Any, Literal
from pydantic import BaseModel, HttpUrl, Field
from pydantic import root_validator
from datetime import datetime


# Request Models

class ProxyConfig(BaseModel):
    """Proxy configuration for per-request proxy override."""
    server: str
    username: Optional[str] = None
    password: Optional[str] = None
    bypass: Optional[str] = None


class CrawlOptions(BaseModel):
    """Options for crawling configuration"""
    javascript: bool = True
    screenshot: bool = False
    screenshot_mode: str = "full"
    full_content: bool = True
    dedupe_tables: bool = True
    timeout: int = Field(default=30, ge=5, le=300)
    javascript_payload: Optional[str] = None
    wait_until: Literal["domcontentloaded", "networkidle", "selector"] = "domcontentloaded"
    wait_for_selector: Optional[str] = None
    wait_after_load_ms: int = Field(default=1000, ge=0, le=60000)
    retry_with_js_if_thin: bool = False
    proxy: Optional[ProxyConfig] = None


class CrawlRequest(BaseModel):
    """Single URL crawl request"""
    url: HttpUrl
    options: CrawlOptions = CrawlOptions()
    session_id: Optional[str] = None
    customer_id: Optional[str] = None
    javascript_enabled: Optional[bool] = None
    javascript_payload: Optional[str] = None


class MarkdownRequest(BaseModel):
    """Markdown-only crawl request"""
    url: Optional[HttpUrl] = None
    urls: Optional[List[HttpUrl]] = Field(None, min_items=1, max_items=50)
    options: CrawlOptions = CrawlOptions()
    session_id: Optional[str] = None
    customer_id: Optional[str] = None
    javascript_enabled: Optional[bool] = None
    javascript_payload: Optional[str] = None

    @root_validator(pre=True)
    def require_url(cls, values):
        if not values.get("url") and not values.get("urls"):
            raise ValueError("url or urls required")
        return values


class RawHtmlRequest(BaseModel):
    """Raw HTML crawl request"""
    url: HttpUrl
    options: CrawlOptions = CrawlOptions()
    session_id: Optional[str] = None
    customer_id: Optional[str] = None
    javascript_enabled: Optional[bool] = None
    javascript_payload: Optional[str] = None


class BatchRequest(BaseModel):
    """Batch crawl request"""
    urls: List[HttpUrl] = Field(..., min_items=1, max_items=50)
    options: CrawlOptions = CrawlOptions()
    concurrent: int = Field(default=3, ge=1, le=10)
    session_id: Optional[str] = None
    customer_id: Optional[str] = None
    javascript_enabled: Optional[bool] = None
    javascript_payload: Optional[str] = None


# Response Models
class CrawlResult(BaseModel):
    """Single crawl result"""
    success: bool
    url: str
    html: Optional[str] = None
    markdown: Optional[str] = None
    markdown_plain: Optional[str] = None
    content: Optional[str] = None
    final_url: Optional[str] = None
    status_code: Optional[int] = None
    blocked: bool = False
    block_reason: Optional[str] = None
    captcha_detected: bool = False
    http_error_family: Optional[str] = None
    render_mode: Optional[str] = None
    wait_strategy: Optional[str] = None
    timings_ms: Dict[str, int] = {}
    body_char_count: int = 0
    body_word_count: int = 0
    content_quality: Optional[str] = None
    extractor_version: Optional[str] = None
    normalized_url: Optional[str] = None
    content_hash: Optional[str] = None
    quarantined: bool = False
    quarantine_reason: Optional[str] = None
    policy_flags: List[str] = []
    visible_char_count: int = 0
    visible_word_count: int = 0
    visible_similarity: Optional[float] = None
    screenshot_url: Optional[Any] = None  # str or list[str] for multi-segment
    metadata: Dict[str, Any] = {}
    crawled_at: datetime
    error: Optional[str] = None


class MarkdownResult(BaseModel):
    """Markdown-only result"""
    success: bool
    url: str
    markdown: Optional[str] = None
    markdown_plain: Optional[str] = None
    content: Optional[str] = None
    final_url: Optional[str] = None
    status_code: Optional[int] = None
    blocked: bool = False
    block_reason: Optional[str] = None
    captcha_detected: bool = False
    http_error_family: Optional[str] = None
    render_mode: Optional[str] = None
    wait_strategy: Optional[str] = None
    timings_ms: Dict[str, int] = {}
    body_char_count: int = 0
    body_word_count: int = 0
    content_quality: Optional[str] = None
    extractor_version: Optional[str] = None
    normalized_url: Optional[str] = None
    content_hash: Optional[str] = None
    quarantined: bool = False
    quarantine_reason: Optional[str] = None
    policy_flags: List[str] = []
    visible_char_count: int = 0
    visible_word_count: int = 0
    visible_similarity: Optional[float] = None
    metadata: Dict[str, Any] = {}
    crawled_at: datetime
    error: Optional[str] = None


class BatchItemResult(BaseModel):
    url: str
    success: bool
    final_url: Optional[str] = None
    status_code: Optional[int] = None
    markdown: Optional[str] = None
    markdown_plain: Optional[str] = None
    content: Optional[str] = None
    error: Optional[str] = None
    blocked: bool = False
    block_reason: Optional[str] = None
    captcha_detected: bool = False
    http_error_family: Optional[str] = None
    render_mode: Optional[str] = None
    wait_strategy: Optional[str] = None
    timings_ms: Dict[str, int] = {}
    body_char_count: int = 0
    body_word_count: int = 0
    content_quality: Optional[str] = None
    extractor_version: Optional[str] = None
    normalized_url: Optional[str] = None
    content_hash: Optional[str] = None
    screenshot_url: Optional[Any] = None  # str or list[str] for multi-segment


class RawHtmlResult(BaseModel):
    """Raw HTML result"""
    success: bool
    url: str
    html: Optional[str] = None
    metadata: Dict[str, Any] = {}
    crawled_at: datetime
    error: Optional[str] = None


class BatchResult(BaseModel):
    """Batch crawl result summary"""
    success: bool
    job_id: str
    total_urls: int
    message: str = "Batch job created successfully"
    results: List[BatchItemResult] = []
    summary: Dict[str, Any] = {}


class JobStatus(BaseModel):
    """Job status response"""
    job_id: str
    status: str  # pending, running, completed, failed
    progress: float = Field(ge=0.0, le=1.0)
    total_urls: int
    completed_urls: int
    results: List[CrawlResult] = []
    created_at: datetime
    updated_at: datetime
    error: Optional[str] = None


class JobListResponse(BaseModel):
    """List of user jobs"""
    jobs: List[Dict[str, Any]]
    total: int


# Health Response
class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    service: str
    version: str
    cloud_mode: bool


class CacheSearchRequest(BaseModel):
    query: str
    domain: Optional[str] = None
    url_prefix: Optional[str] = None
    min_similarity: float = Field(default=0.4, ge=0.0, le=1.0)
    max_results: int = Field(default=20, ge=1, le=200)
    quality_in: Optional[List[str]] = None
    since_ts: Optional[str] = None


class CacheUpsertRequest(BaseModel):
    url: str
    markdown: str = ""
    markdown_plain: Optional[str] = None
    content: Optional[str] = None
    quality: str = "sufficient"
    status_code: Optional[int] = None
    extractor_version: Optional[str] = None
    normalized_url: Optional[str] = None
    content_hash: Optional[str] = None
    metadata: Dict[str, Any] = {}


class CachePruneRequest(BaseModel):
    domain: Optional[str] = None
    ttl_hours: Optional[int] = Field(default=None, ge=1, le=24 * 365)
    dry_run: bool = False


# Agent Models (Mode B)

class AgentRunRequest(BaseModel):
    """Submit a task to the internal agent loop."""
    task: str = Field(..., min_length=1, max_length=4000)
    session_id: Optional[str] = None
    customer_id: Optional[str] = None
    max_steps: int = Field(default=12, ge=1, le=50)
    max_wall_time_ms: int = Field(default=90_000, ge=5000, le=300_000)
    allowed_domains: Optional[List[str]] = None
    allowed_tools: Optional[List[str]] = None


class AgentTraceEntry(BaseModel):
    """Single trace event in the agent run."""
    event: Optional[str] = None
    step_id: Optional[int] = None
    tool_name: Optional[str] = None
    duration_ms: Optional[int] = None
    status: Optional[str] = None
    error_code: Optional[str] = None
    timestamp_ms: Optional[int] = None


class AgentRunResponse(BaseModel):
    """Response from an agent run."""
    success: bool
    run_id: str
    stop_reason: str
    response: Optional[str] = None
    steps: int = 0
    wall_time_ms: int = 0
    trace: List[AgentTraceEntry] = []
    artifacts: List[Dict[str, Any]] = []
    error: Optional[str] = None


class AgentStatusResponse(BaseModel):
    """Status of a running or completed agent run."""
    run_id: str
    found: bool
    success: Optional[bool] = None
    stop_reason: Optional[str] = None
    response: Optional[str] = None
    steps: Optional[int] = None
    wall_time_ms: Optional[int] = None
    error: Optional[str] = None


# Ghost Protocol Models

class GhostExtractRequest(BaseModel):
    """Request for Ghost Protocol vision extraction."""
    url: str = Field(..., min_length=1)
    timeout: int = Field(default=30, ge=5, le=120)
    prompt: Optional[str] = None
    proxy: Optional[ProxyConfig] = None


class GhostExtractResponse(BaseModel):
    """Response from Ghost Protocol extraction."""
    success: bool
    url: str
    content: Optional[str] = None
    render_mode: str = "ghost"
    block_signal: Optional[str] = None
    block_reason: Optional[str] = None
    capture_ms: int = 0
    extraction_ms: int = 0
    total_ms: int = 0
    provider: Optional[str] = None
    blocked_content: bool = False
    error: Optional[str] = None
