# Grub Crawler — Developer Reference

> **Service:** `grubcrawler` on Cloud Run  
> **Domains:** `grub.nuts.services`, `grubcrawler.dev`  
> **Auth:** nuts-auth bearer tokens (`ahp_...` format)  
> **Port:** 6792 (local), 443 (prod)

---

## Table of Contents

1. [Architecture](#architecture)
2. [Running Locally](#running-locally)
3. [Authentication](#authentication)
4. [Core Crawl Endpoints](#core-crawl-endpoints)
5. [Agent Endpoints](#agent-endpoints)
6. [Job Queue Endpoints](#job-queue-endpoints)
7. [Cache Endpoints](#cache-endpoints)
8. [Session & File Endpoints](#session--file-endpoints)
9. [Utility Endpoints](#utility-endpoints)
10. [Request Models](#request-models)
11. [Response Models](#response-models)
12. [Environment Variables](#environment-variables)
13. [Deployment](#deployment)
14. [Error Handling](#error-handling)

---

## Architecture

```
┌─────────────────────────────────────────────────┐
│                   FastAPI App                   │
│                                                 │
│  /api/crawl ──► CrawlerEngine                  │
│  /api/agent ──► AgentEngine (Mode B)           │
│  /api/jobs  ──► JobManager ──► Cloud Tasks     │
│  /api/cache ──► RemoteCacheStore               │
│  /{tool}    ──► ToolRegistry (AHP protocol)    │
│                                                 │
│  Middleware stack (outermost → innermost):      │
│    CORS → Auth → ContentType → 404Enricher     │
└─────────────────────────────────────────────────┘
```

**Key subsystems:**

| Module | Purpose |
|--------|---------|
| `app/crawler.py` | Playwright-based crawl engine |
| `app/browser.py` | Browser context management, cookie injection |
| `app/challenge_solver.py` | Cloudflare + Incapsula bypass |
| `app/http_precheck.py` | Fast HTTP pre-check before browser launch |
| `app/agent/engine.py` | Bounded LLM loop (plan → execute → observe) |
| `app/agent/ghost.py` | Vision-based extraction for blocked pages |
| `app/jobs.py` | Async job queue (Cloud Tasks in prod, ThreadPool locally) |
| `app/mesh/coordinator.py` | P2P peer discovery and routing |
| `app/policy/gate.py` | Tool call policy enforcement |
| `app/tools/tool_registry.py` | AHP tool discovery and execution |
| `app/storage.py` | GCS (cloud) or local file storage |
| `app/core/middleware.py` | Auth, ContentType, 404-enrichment middleware |

---

## Running Locally

```bash
# Install dependencies
pip install -r requirements.txt
playwright install chromium

# Create .env (copy from template)
cp .env.example .env

# Run dev server
uvicorn app.main:app --reload --host 0.0.0.0 --port 6792
```

Minimal `.env` for local dev:
```env
DISABLE_AUTH=true
DEBUG=true
STORAGE_PATH=./storage
ENABLE_JAVASCRIPT=true
BROWSER_HEADLESS=true
```

---

## Authentication

All `/api/*` endpoints require a bearer token unless `DISABLE_AUTH=true`.

### nuts-auth tokens (`ahp_...`)

```bash
curl -H "Authorization: Bearer ahp_YOUR_TOKEN" \
     -X POST https://grub.nuts.services/api/crawl \
     -H "Content-Type: application/json" \
     -d '{"url": "https://example.com"}'
```

Token validation: the service POSTs to `https://auth.nuts.services/auth` with `token=ahp_...`, receives a JWT, and extracts the user email from the `sub` field.

### Internal HMAC tokens

Short-lived tokens used for AHP tool routes (`GET /{tool_name}?bearer_token=...`). Generated server-side, not for direct use.

### Unauthenticated / DISABLE_AUTH mode

When `DISABLE_AUTH=true`, all endpoints are publicly accessible. Pass `customer_id` in request bodies to partition storage.

```json
{
  "url": "https://example.com",
  "customer_id": "tenant-abc"
}
```

---

## Core Crawl Endpoints

### `POST /api/crawl`

Full crawl — returns HTML + markdown + metadata synchronously.

**Request:**
```json
{
  "url": "https://example.com",
  "session_id": "optional-uuid",
  "customer_id": "optional-tenant-id",
  "javascript_enabled": true,
  "options": {
    "javascript": true,
    "screenshot": false,
    "screenshot_mode": "full",
    "timeout": 30,
    "wait_until": "domcontentloaded",
    "wait_for_selector": null,
    "wait_after_load_ms": 1000,
    "retry_with_js_if_thin": false,
    "dedupe_tables": true,
    "proxy": null,
    "cookies": null
  }
}
```

**Response:** See [CrawlResult](#crawlresult).

---

### `POST /api/markdown`

Markdown-only crawl, supports 1–50 URLs in a single request. Multi-URL responses join content with `---` separators.

**Request:**
```json
{
  "url": "https://example.com",
  "urls": ["https://a.com", "https://b.com"],
  "options": { "timeout": 30 }
}
```

Provide either `url` (single) or `urls` (batch), not both.

**Header:** `X-Client-Timeout: 25` — caps total processing time to client budget (seconds).

**Response:** See [MarkdownResult](#markdownresult).

---

### `POST /api/raw`

Returns raw HTML without markdown conversion.

**Request:** Same as `/api/crawl`.

**Response:** See [RawHtmlResult](#rawhtmlresult).

---

### `POST /api/batch`

Synchronous batch crawl for up to 50 URLs with configurable concurrency.

**Request:**
```json
{
  "urls": ["https://a.com", "https://b.com"],
  "concurrent": 3,
  "options": { "timeout": 30 }
}
```

**Response:** See [BatchResult](#batchresult).

---

## Agent Endpoints

Requires `AGENT_ENABLED=true`. Returns `503` otherwise.

### `POST /api/agent/run`

Submit a natural language task to the autonomous agent. Blocks until the agent finishes or hits a stop condition.

**Request:**
```json
{
  "task": "Find the pricing page at example.com and extract all plan names and prices.",
  "session_id": "optional-uuid",
  "customer_id": "optional-tenant",
  "max_steps": 12,
  "max_wall_time_ms": 90000,
  "allowed_domains": ["example.com"],
  "allowed_tools": ["crawl", "markdown"]
}
```

**Response:**
```json
{
  "success": true,
  "run_id": "uuid",
  "stop_reason": "completed",
  "response": "The extracted markdown content...",
  "steps": 4,
  "wall_time_ms": 8200,
  "trace": [...],
  "artifacts": [],
  "error": null
}
```

**`stop_reason` values:** `completed` `max_steps` `max_wall_time` `max_failures` `no_op_loop` `policy_denied`

---

### `GET /api/agent/status/{run_id}?session_id=...`

Poll for a completed agent run stored in session.

**Response:**
```json
{
  "run_id": "uuid",
  "found": true,
  "success": true,
  "stop_reason": "completed",
  "response": "...",
  "steps": 4,
  "wall_time_ms": 8200,
  "error": null
}
```

---

### `POST /api/agent/ghost`

Requires `AGENT_GHOST_ENABLED=true`. Takes a screenshot of a URL and extracts content via vision LLM — useful for pages that block headless crawlers.

**Request:**
```json
{
  "url": "https://blocked-site.com",
  "timeout": 30,
  "prompt": "Extract the main article text.",
  "proxy": null
}
```

**Response:**
```json
{
  "success": true,
  "url": "https://blocked-site.com",
  "content": "Extracted markdown...",
  "render_mode": "ghost",
  "block_signal": "cloudflare_challenge",
  "block_reason": "challenge page detected",
  "capture_ms": 4200,
  "extraction_ms": 1800,
  "total_ms": 6000,
  "provider": "anthropic",
  "blocked_content": false,
  "error": null
}
```

---

## Job Queue Endpoints

Async job submission — results retrieved via session status polling.

### `POST /api/jobs/crawl`

```json
{
  "url": "https://example.com",
  "javascript": true,
  "screenshot": false,
  "timeout": 30,
  "callback_url": "https://your-service.com/webhook"
}
```

### `POST /api/jobs/batch-crawl`

```json
{
  "urls": ["https://a.com", "https://b.com"],
  "javascript": true,
  "max_concurrent": 3,
  "callback_url": "https://your-service.com/webhook"
}
```

### `POST /api/jobs/markdown`

```json
{
  "url": "https://example.com",
  "javascript": true,
  "callback_url": "https://your-service.com/webhook"
}
```

All job endpoints return:
```json
{
  "job_id": "uuid",
  "session_id": "uuid",
  "message": "Job submitted"
}
```

### `GET /api/sessions/{session_id}/status`

Poll job progress:

```json
{
  "session_id": "uuid",
  "stages": {
    "crawling": {
      "status": "complete",
      "total_urls": 5,
      "urls_processed": 5,
      "progress_percent": 100,
      "is_running": false
    }
  },
  "updated_at": "2026-04-21T00:00:00"
}
```

### Callback payloads

On completion:
```json
{
  "session_id": "uuid",
  "status": "completed",
  "data": { "url": "...", "markdown": "..." }
}
```

On failure:
```json
{
  "session_id": "uuid",
  "status": "failed",
  "data": { "error": "..." }
}
```

---

## Cache Endpoints

Semantic search cache for crawled content.

### `POST /api/cache/search`

```json
{
  "query": "pricing plans enterprise",
  "domain": "example.com",
  "min_similarity": 0.4,
  "max_results": 20,
  "quality_in": ["sufficient", "rich"]
}
```

### `GET /api/cache/list?domain=example.com&quality=sufficient&limit=50&offset=0`

### `GET /api/cache/doc/{doc_id}`

### `POST /api/cache/upsert`

```json
{
  "url": "https://example.com/pricing",
  "markdown": "# Pricing\n...",
  "quality": "sufficient",
  "metadata": {}
}
```

### `POST /api/cache/prune`

```json
{
  "domain": "example.com",
  "ttl_hours": 168,
  "dry_run": true
}
```

---

## Session & File Endpoints

### `GET /api/sessions/{session_id}/files?prefix=results&customer_id=...`

Lists files stored in a session.

### `GET /api/sessions/{session_id}/file?path=results/abc.json&customer_id=...`

Retrieves a stored file. Content-type inferred from extension (`.json`, `.png`, `.md`, `.txt`).

### `GET /api/sessions/{session_id}/results`

Returns session metadata and crawl results.

### `GET /api/sessions/{session_id}/screenshots`

Lists screenshot files for a session.

---

## Utility Endpoints

| Endpoint | Auth | Description |
|----------|------|-------------|
| `GET /health` | None | Health check + tool count + mesh peer info |
| `GET /tools` | None | List registered AHP crawl tools |
| `GET /view?url=...` | None | Proxy-render a page as HTML in the browser |
| `GET /download?url=...&save=false&download=false` | None | Fetch a binary file |
| `GET /docs` | None | API documentation page |
| `GET /` or `/site` | None | Landing page |

**`/view` params:** `url`, `javascript` (bool, default: true), `timeout` (int, default: 30)

**`/download` params:** `url`, `use_browser` (bool), `javascript` (bool), `timeout` (int), `session_id`, `save` (bool), `filename`, `download` (bool — triggers `Content-Disposition: attachment`)

---

## Request Models

### CrawlOptions

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `javascript` | bool | `true` | Enable JS rendering |
| `screenshot` | bool | `false` | Take screenshot |
| `screenshot_mode` | string | `"full"` | `"full"` or `"viewport"` |
| `timeout` | int (5–300) | `30` | Seconds |
| `wait_until` | string | `"domcontentloaded"` | `"domcontentloaded"` \| `"networkidle"` \| `"selector"` |
| `wait_for_selector` | string | `null` | CSS selector to wait for |
| `wait_after_load_ms` | int (0–60000) | `1000` | Extra wait after load |
| `retry_with_js_if_thin` | bool | `false` | Auto-retry with JS if content thin |
| `dedupe_tables` | bool | `true` | Deduplicate repeated table rows |
| `full_content` | bool | `true` | Include full page content |
| `proxy` | ProxyConfig | `null` | Per-request proxy override |
| `cookies` | dict | `null` | Pre-solved cookies to inject |

### ProxyConfig

```json
{
  "server": "http://proxy.example.com:8080",
  "username": "user",
  "password": "pass",
  "bypass": "localhost"
}
```

### Cookie injection

Pass Incapsula or other challenge cookies to bypass bot detection:

```json
{
  "url": "https://protected-site.com",
  "options": {
    "cookies": {
      "visid_incap_1879042": "abc123...",
      "incap_ses_1700_1879042": "xyz456..."
    }
  }
}
```

---

## Response Models

### CrawlResult

| Field | Type | Description |
|-------|------|-------------|
| `success` | bool | Whether crawl succeeded |
| `url` | string | Requested URL |
| `final_url` | string | URL after redirects |
| `html` | string | Raw HTML (only on `/api/crawl`) |
| `markdown` | string | Converted markdown |
| `markdown_plain` | string | Markdown without formatting |
| `content` | string | Extracted text content |
| `status_code` | int | HTTP status code |
| `blocked` | bool | Anti-bot block detected |
| `block_reason` | string | e.g. `"cloudflare_challenge"` |
| `captcha_detected` | bool | CAPTCHA detected |
| `http_error_family` | string | e.g. `"4xx"`, `"5xx"` |
| `render_mode` | string | `"http"` \| `"browser"` \| `"ghost"` |
| `wait_strategy` | string | Strategy used |
| `timings_ms` | dict | Timing breakdown |
| `body_char_count` | int | Character count |
| `body_word_count` | int | Word count |
| `visible_char_count` | int | Visible text chars |
| `visible_word_count` | int | Visible text words |
| `visible_similarity` | float | Similarity score |
| `content_quality` | string | `"minimal"` \| `"sufficient"` \| `"rich"` |
| `content_hash` | string | 8-char content hash |
| `quarantined` | bool | Flagged by policy |
| `quarantine_reason` | string | Policy flag reason |
| `policy_flags` | list[string] | Applied policy flags |
| `screenshot_url` | string | Path to screenshot if taken |
| `metadata` | dict | Extra metadata (title, timings, etc.) |
| `crawled_at` | datetime | Timestamp |
| `error` | string | Error message on failure |

### MarkdownResult

Same as CrawlResult minus the `html` field.

### RawHtmlResult

`success`, `url`, `html`, `metadata`, `crawled_at`, `error`

### BatchResult

```json
{
  "success": true,
  "job_id": "uuid",
  "total_urls": 3,
  "message": "Batch complete",
  "results": [
    {
      "url": "https://a.com",
      "success": true,
      "markdown": "...",
      "error": null
    }
  ],
  "summary": {
    "total": 3,
    "success": 2,
    "failed": 1
  }
}
```

---

## Environment Variables

### Server

| Variable | Default | Description |
|----------|---------|-------------|
| `HOST` | `0.0.0.0` | Bind host |
| `PORT` | `6792` | Bind port |
| `DEBUG` | `false` | Debug logging |

### Storage

| Variable | Default | Description |
|----------|---------|-------------|
| `STORAGE_PATH` | `./storage` | Local storage root |
| `RUNNING_IN_CLOUD` | `false` | Use GCS instead of local |
| `GCS_BUCKET_NAME` | — | GCS bucket name |
| `GOOGLE_CLOUD_PROJECT` | — | GCP project ID |

### Authentication

| Variable | Default | Description |
|----------|---------|-------------|
| `DISABLE_AUTH` | `false` | Bypass all auth (⚠️ dangerous in prod) |
| `GNOSIS_AUTH_URL` | `https://auth.nuts.services` | nuts-auth endpoint |

### Browser & Crawling

| Variable | Default | Description |
|----------|---------|-------------|
| `BROWSER_ENGINE` | `camoufox` | `camoufox` or `chromium` |
| `BROWSER_HEADLESS` | `true` | Headless mode |
| `BROWSER_TIMEOUT` | `30000` | Browser timeout (ms) |
| `MAX_CONCURRENT_CRAWLS` | `5` | Semaphore limit |
| `CRAWL_TIMEOUT` | `30` | Default crawl timeout (sec) |
| `ENABLE_JAVASCRIPT` | `true` | JS rendering default |
| `ENABLE_SCREENSHOTS` | `false` | Screenshots default |
| `STEALTH_ENABLED` | `true` | Stealth patches |
| `BLOCK_TRACKING_DOMAINS` | `true` | Block ad/tracker domains |
| `HTTP_PRECHECK_ENABLED` | `false` | HTTP pre-check before browser |
| `HTTP_PRECHECK_TIMEOUT` | `15` | Pre-check timeout (sec) |
| `HTTP_PRECHECK_IMPERSONATE` | `chrome135` | TLS impersonation profile |

### Agent (Mode B)

| Variable | Default | Description |
|----------|---------|-------------|
| `AGENT_ENABLED` | `false` | Enable agent endpoints |
| `AGENT_MAX_STEPS` | `12` | Default max steps |
| `AGENT_MAX_WALL_TIME_MS` | `90000` | Default max wall time |
| `AGENT_MAX_FAILURES` | `3` | Max tool failures |
| `AGENT_ALLOWED_TOOLS` | `` | Comma-separated tool allowlist (empty = all) |
| `AGENT_ALLOWED_DOMAINS` | `` | Comma-separated domain allowlist (empty = all) |
| `AGENT_BLOCK_PRIVATE_RANGES` | `true` | Block private IP ranges |
| `AGENT_REDACT_SECRETS` | `true` | Redact secrets in traces |
| `AGENT_PROVIDER` | `openai` | LLM provider (`openai` \| `anthropic` \| `ollama`) |
| `OPENAI_API_KEY` | — | OpenAI API key |
| `OPENAI_MODEL` | `gpt-4.1-mini` | OpenAI model |
| `ANTHROPIC_API_KEY` | — | Anthropic API key |
| `ANTHROPIC_MODEL` | `claude-3-5-sonnet-latest` | Anthropic model |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama endpoint |
| `OLLAMA_MODEL` | `llama3.1:8b-instruct` | Ollama model |

### Ghost Protocol

| Variable | Default | Description |
|----------|---------|-------------|
| `AGENT_GHOST_ENABLED` | `false` | Enable Ghost Protocol |
| `AGENT_GHOST_AUTO_TRIGGER` | `true` | Auto-trigger on blocked pages |
| `AGENT_GHOST_VISION_PROVIDER` | `` | Provider override (inherits AGENT_PROVIDER) |
| `AGENT_GHOST_MAX_IMAGE_WIDTH` | `1280` | Screenshot width cap (px) |
| `AGENT_GHOST_MAX_RETRIES` | `1` | Retry attempts |

### Proxy

| Variable | Default | Description |
|----------|---------|-------------|
| `PROXY_SERVER` | — | Proxy URL (e.g. `http://host:port`) |
| `PROXY_USERNAME` | — | Proxy username |
| `PROXY_PASSWORD` | — | Proxy password |
| `PROXY_BYPASS` | `` | Bypass list |
| `PROXY_SESSION_DURATION_MINUTES` | `10` | Rotate proxy session interval |
| `PROXY_RESTART_AFTER_FAILURES` | `3` | Failures before proxy rotation |

### Async Jobs

| Variable | Default | Description |
|----------|---------|-------------|
| `CLOUD_TASKS_QUEUE` | `crawl-jobs` | Cloud Tasks queue name |
| `CLOUD_TASKS_LOCATION` | `us-central1` | Queue region |
| `WORKER_SERVICE_URL` | — | Cloud Run worker URL for task dispatch |

### Mesh

| Variable | Default | Description |
|----------|---------|-------------|
| `MESH_ENABLED` | `false` | Enable P2P mesh |
| `MESH_PEERS` | `` | Comma-separated seed peer URLs |
| `MESH_NODE_NAME` | `` | Node name (defaults to hostname) |
| `MESH_SECRET` | `` | Shared HMAC secret for mesh auth |
| `MESH_ADVERTISE_URL` | `` | Public URL this node advertises |
| `MESH_HEARTBEAT_INTERVAL_S` | `15` | Heartbeat interval |
| `MESH_PEER_TIMEOUT_S` | `45` | Mark peer unhealthy after |
| `MESH_PEER_REMOVE_S` | `120` | Remove dead peer after |

### Browser Pool / Live Stream

| Variable | Default | Description |
|----------|---------|-------------|
| `BROWSER_STREAM_ENABLED` | `false` | Enable live browser stream |
| `BROWSER_POOL_SIZE` | `1` | Pool size |
| `BROWSER_STREAM_QUALITY` | `25` | MJPEG quality (1–100) |
| `BROWSER_STREAM_MAX_WIDTH` | `854` | Stream width cap (px) |
| `BROWSER_STREAM_MAX_LEASE_SECONDS` | `300` | Max lease per session |

---

## Deployment

### ⚠️ Correct service name

Always deploy to **`grubcrawler`** — this is the service with domain mappings.
There is a stale duplicate service `grub-crawl` with no domain mappings. Do not deploy to it.

```bash
# Deploy to production
gcloud run deploy grubcrawler \
  --source . \
  --region us-central1 \
  --project gnosis-459403 \
  --allow-unauthenticated \
  --port 6792
```

### Local Docker

```bash
./deploy.ps1 -Target local
# Test at http://localhost:6792
```

### Cloud Run services

| Service | Domains | Notes |
|---------|---------|-------|
| `grubcrawler` | `grub.nuts.services`, `grubcrawler.dev` | Production — deploy here |
| `grub-crawl` | none | Stale duplicate — can be deleted |

### Delete the stale duplicate when convenient

```bash
gcloud run services delete grub-crawl --region us-central1 --project gnosis-459403
```

---

## Error Handling

All error responses follow this format:

```json
{
  "error": "http_error",
  "status": 404,
  "details": { "message": "Not Found" },
  "grub": {
    "endpoints": { ... },
    "hint": "All POST endpoints accept JSON. See /docs for full schema."
  }
}
```

Unknown endpoints return 404 with the `grub.endpoints` map so callers know what's available.

### Common errors

| Status | `error` | Cause |
|--------|---------|-------|
| 401 | `"Missing bearer token"` | No `Authorization` header |
| 401 | `"Invalid or expired token"` | Token rejected by nuts-auth |
| 422 | `"validation_error"` | Request body failed Pydantic validation |
| 500 | `"internal_error"` | Unhandled exception |
| 503 | `"agent_disabled"` | `AGENT_ENABLED` not set |

---

## Storage Partitioning

Files are stored at `{customer_hash}/{session_id}/` where `customer_hash` is the first 12 chars of SHA256 of the `customer_id` or authenticated user email.

```
storage/
└── a3f9b2c1d8e4/          ← customer hash
    └── {session_uuid}/
        ├── metadata.json
        ├── session_status.json
        └── results/
            ├── {url_hash}.json
            └── screenshots/
```

---

*Last updated: April 21, 2026*
