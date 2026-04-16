# CLAUDE.md - Development Context for grub-crawl

## Project Overview

**grub-crawl** is an agentic web crawling service with mesh P2P. It provides HTML extraction, markdown conversion, batch processing, and an internal LLM-backed agent (Mode B) that can autonomously execute crawl tool chains. Deployed as `grub-crawl` on Cloud Run at `grub.nuts.services`.

## Architecture

### Core Components

```
app/
├── main.py               # FastAPI app, AHP tool catch-all, /view /download routes
├── config.py             # Environment-based configuration
├── auth.py               # nuts-auth token exchange + internal HMAC tokens
├── models.py             # Pydantic request/response models
├── routes.py             # REST API endpoint handlers (/api/*)
├── job_routes.py         # Async job queue endpoints (/api/jobs/*)
├── agent_routes.py       # Agent run endpoints (/api/agent/*)
├── storage.py            # User-partitioned storage service (local/GCS)
├── crawler.py            # Playwright-based crawling engine
├── markdown.py           # HTML to markdown conversion
├── browser.py            # Browser automation utilities
├── browser_pool.py       # Browser pool for live streaming
├── cache_store.py        # Result cache store
├── cookie_store.py       # Cookie persistence store
├── proxy.py              # Proxy management
├── proxy_pool.py         # Proxy pool with burnout detection
├── stealth.py            # Browser stealth patches
├── human_behavior.py     # Human-like mouse/scroll simulation
├── warmup_navigator.py   # Browser warmup before crawl
├── behavior_profile.py   # Configurable browser behavior profiles
├── stream.py             # Live browser streaming routes (optional)
├── http_precheck.py      # Fast HTTP pre-check before launching browser
├── challenge_solver.py   # Cloudflare + Incapsula challenge solver
├── jobs.py               # JobManager + JobProcessor (Cloud Tasks / local executor)
│
├── agent/                # Bounded LLM agent loop (Mode B)
│   ├── engine.py         # AgentEngine: plan→execute→observe loop + Ghost fallback
│   ├── dispatcher.py     # Tool call dispatcher with policy gate
│   ├── ghost.py          # Ghost Protocol: vision-based extraction for blocked pages
│   ├── types.py          # RunConfig, RunContext, RunResult, StopReason, etc.
│   ├── errors.py         # PolicyDeniedError, ProviderError, StopConditionError
│   └── providers/        # LLM adapters: Anthropic, OpenAI, Ollama
│
├── core/
│   ├── auth_client.py    # Auth HTTP client
│   └── middleware.py     # ContentTypeMiddleware, AuthMiddleware
│
├── mesh/                 # P2P mesh networking
│   ├── coordinator.py    # MeshCoordinator: heartbeat, peer discovery
│   ├── router.py         # Route crawl jobs to healthy peers
│   ├── dispatcher.py     # Dispatch tool calls across mesh
│   ├── client.py         # Peer HTTP client
│   ├── models.py         # Mesh data models
│   └── routes.py         # Mesh API routes (conditional, mesh_enabled only)
│
├── observability/
│   ├── events.py         # EventBus + lifecycle events (RunStart/End, ToolDispatch, etc.)
│   └── trace.py          # TraceCollector, RunSummary, persist_trace()
│
├── policy/
│   ├── gate.py           # check_tool_call() → PolicyVerdict (allow/deny)
│   ├── domain.py         # Domain allowlist/blocklist policy
│   ├── injection.py      # Prompt injection detection
│   └── redaction.py      # Secret redaction in traces
│
└── tools/
    ├── tool_registry.py  # ToolRegistry: discover, register, execute tools
    ├── crawl_tools.py    # Crawl tool implementations
    └── base.py           # BaseTool interface
```

### Storage Structure

```
storage/
└── {customer_hash}/        # 12-char SHA256 hash of customer_id or email
    └── {session_id}/       # UUID for grouping related crawls
        ├── metadata.json
        └── results/
            ├── {url_hash}.json
            └── screenshots/
```

### Authentication Modes

1. **With Auth (default)**: Uses gnosis-auth HMAC JWT tokens, extracts user email
2. **Without Auth (`DISABLE_AUTH=true`)**: Requires `customer_id` in requests
3. **Hybrid**: Can provide `customer_id` even with auth to override storage partition

## API Endpoints

### Core Crawling
- `POST /api/crawl` - Single URL crawl (HTML + markdown + optional cookies)
- `POST /api/markdown` - Markdown-only crawl (optimized)
- `POST /api/batch` - Batch crawl multiple URLs

### Async Jobs
- `POST /api/jobs/crawl` - Submit async crawl job
- `POST /api/jobs/batch` - Submit async batch job
- `GET /api/jobs/{session_id}/status` - Get job/session status

### Agent (Mode B)
- `POST /api/agent/run` - Run an LLM-backed agent task (bounded loop)
- `GET /api/agent/runs/{run_id}` - Get agent run result + trace

### Session Management
- `GET /api/sessions/{session_id}/files` - List stored files
- `GET /api/sessions/{session_id}/file` - Retrieve specific file

### Utilities
- `GET /view?url=...` - Render a page through the crawler and return HTML (browser proxy)
- `GET /download?url=...` - Fetch + optionally save a binary file
- `GET /tools` - List all registered AHP crawl tools
- `GET /{tool_name}?bearer_token=...` - Execute tool via AHP protocol (catch-all, requires internal HMAC token)

### System
- `GET /health` - Health check (includes mesh peer info if enabled)
- `GET /` or `/site` - Embedded landing page
- `GET /docs` - API documentation page

## Agent System (Mode B)

The agent is a bounded LLM loop in `app/agent/engine.py`. It runs plan→execute→observe cycles up to `max_steps` / `max_wall_time_ms` / `max_failures` limits.

**Ghost Protocol** (`app/agent/ghost.py`): When a crawl tool returns blocked/thin content, the engine auto-triggers a vision-based fallback — takes a screenshot and asks the LLM to extract content from the image. Controlled by `AGENT_GHOST_ENABLED` and `AGENT_GHOST_AUTO_TRIGGER` settings.

**Policy gate** (`app/policy/gate.py`): Every tool call passes through `check_tool_call()` before dispatch. Policies: domain allowlist/blocklist, prompt injection detection, secret redaction in traces.

**Providers**: Anthropic (default), OpenAI, Ollama — all implement `LLMAdapter` protocol in `app/agent/providers/`.

**Observability**: Every run creates an `EventBus` + `TraceCollector`. Events emitted at each lifecycle point. Trace persisted to storage via `persist_trace()`.

## Key Design Decisions

### Why Optional customer_id?
- **Flexibility**: Support both authenticated SaaS and unauthenticated self-hosted deployments
- **Multi-tenancy**: Allow custom storage partitioning even with auth
- **Self-hosted**: Enable deployments without gnosis-auth dependency

### Why Hash Customer IDs?
- **Privacy**: Doesn't expose actual email/customer_id in file paths
- **Consistency**: Same hash every time = predictable storage location
- **Safety**: File system safe characters only

### Why Session IDs?
- **Grouping**: Related crawls can be organized together
- **Retrieval**: Easy to list all results from a batch operation
- **Optional**: Auto-generated if not provided

## Configuration

### Environment Variables

**Server:**
- `HOST` - Server host (default: 0.0.0.0)
- `PORT` - Server port (default: 6792)
- `DEBUG` - Debug mode (default: false)

**Storage:**
- `STORAGE_PATH` - Local storage path (default: ./storage)
- `RUNNING_IN_CLOUD` - Use GCS instead of local (default: false)
- `GCS_BUCKET_NAME` - GCS bucket name (cloud mode only)

**Authentication:**
- `DISABLE_AUTH` - Bypass all authentication (default: false) ⚠️
- `GNOSIS_AUTH_URL` - Auth service URL (default: http://gnosis-auth:5000)

**Crawling:**
- `MAX_CONCURRENT_CRAWLS` - Max parallel crawls (default: 5)
- `CRAWL_TIMEOUT` - Timeout in seconds (default: 30)
- `ENABLE_JAVASCRIPT` - JavaScript rendering (default: true)
- `ENABLE_SCREENSHOTS` - Take screenshots (default: false)
- `BROWSER_HEADLESS` - Headless mode (default: true)
- `BROWSER_TIMEOUT` - Browser timeout in ms (default: 30000)

### Deployment Configs

- **.env** - Local development (not in git)
- **.env.mesh** - Mesh coordinator configuration reference
- **.env.example** - Template for new deployments

## Common Development Tasks

### Running Locally
```bash
# Install dependencies
pip install -r requirements.txt

# Run with uvicorn
uvicorn app.main:app --reload --host 0.0.0.0 --port 6792
```

### Building Docker Image
```bash
docker build -t gnosis-crawl:latest .
```

### Testing Changes
```bash
# Update test script with deployed URL
python test_remote_api.py
```

### Adding New Endpoints
1. Add Pydantic models to `app/models.py`
2. Create route handler in `app/routes.py`
3. Use `get_optional_user_email()` dependency for auth flexibility
4. Call `get_customer_identifier()` to resolve customer ID
5. Update README.md with new endpoint documentation

## Known Issues & Gotchas

### Pydantic V2 Warning
```
Valid config keys have changed in V2: 'fields' has been removed
```
**Status**: Harmless warning from config.py Config class
**Fix**: Update to Pydantic V2 config pattern (low priority)

### Auth Middleware Order
The middleware checks for `disable_auth` flag BEFORE attempting to load auth_client.
**Critical**: Must check `settings.disable_auth` before any auth client operations.

### File Versioning
The `*_versions/` directories are created by file-diff-writer tool for local version tracking.
**Status**: Added to .gitignore, should not be committed

## Dependencies

### Core
- **FastAPI** - Web framework
- **Pydantic** - Data validation
- **Uvicorn** - ASGI server
- **Playwright** - Browser automation

### Storage
- **google-cloud-storage** - GCS support (optional)

### Processing
- **BeautifulSoup4** - HTML parsing
- **html2text** - Markdown conversion
- **httpx** - HTTP client

## Testing

### Remote API Test Script
`test_remote_api.py` - Comprehensive test suite for deployed instances

**Tests:**
- Health check
- Single URL crawl with customer_id
- Markdown-only crawl
- Batch crawl
- Session file listing
- No customer_id fallback

**Usage:**
```python
# Update configuration
API_BASE_URL = "https://your-deployed-url.com"
CUSTOMER_ID = "test-client-123"
BEARER_TOKEN = None  # or "your-token" for auth testing

# Run tests
python test_remote_api.py
```

## Deployment

### Local Docker
```bash
./deploy.sh local            # or ./deploy.ps1 -Target local
```

### 2-Node Mesh
```bash
./deploy.sh mesh             # or ./deploy.ps1 -Target mesh
```

### Google Cloud Run
```bash
./deploy.sh cloudrun v1.0.0  # or ./deploy.ps1 -Target cloudrun -Tag v1.0.0
```

### Cloud Run + Mesh
```bash
./deploy.sh cloudrun v1.0.0 --mesh-peer http://your-ip:6792 --mesh-secret mykey
```

## Security Considerations

### DISABLE_AUTH Flag
⚠️ **WARNING**: Setting `DISABLE_AUTH=true` makes ALL endpoints publicly accessible.

**Safe Use Cases:**
- Private internal networks
- Behind corporate firewall
- Trusted Kubernetes cluster with network policies
- Development/testing environments

**Unsafe Use Cases:**
- Public internet exposure
- Multi-tenant SaaS without auth
- Untrusted networks

### Customer ID Validation
Currently NO validation on customer_id format. Consider adding:
- Length limits
- Character restrictions
- Rate limiting per customer_id
- Usage quotas

## Future Enhancements

### Potential Improvements
- [ ] Add customer_id validation/sanitization
- [ ] Rate limiting per customer_id
- [ ] Usage metrics and quotas
- [ ] Webhook notifications for batch completion
- [ ] Priority queue for crawl requests
- [ ] Retry logic for failed crawls
- [ ] Browser pool optimization
- [ ] Cost tracking per customer_id

### Phase 3 Roadmap
- [ ] Comprehensive test suite
- [ ] Error handling improvements
- [ ] Monitoring and alerting
- [ ] Performance optimization
- [ ] Documentation improvements

## Contact & References

- **Documentation**: README.md
- **Master Plan**: MASTER_PLAN.md
- **Customer ID Details**: CUSTOMER_ID_IMPLEMENTATION.md
- **Remote Testing**: test_remote_api.py
- **Gnosis Standards**: Follows gnosis deployment patterns

## Tips for AI Assistants

1. **Always check imports** when adding new FastAPI features (Header, Query, etc.)
2. **Use `get_optional_user_email()`** for new endpoints to support both auth modes
3. **Call `get_customer_identifier()`** to resolve customer ID from multiple sources
4. **Test both modes**: with and without DISABLE_AUTH flag
5. **Storage paths**: Always use customer_hash/session_id structure
6. **Backward compatibility**: Never break existing auth-based flows
7. **File versions**: Don't commit *_versions/ directories

## Quick Reference

### Testing Authenticated Request
```bash
curl -H "Authorization: Bearer <token>" \
     -X POST http://localhost:6792/api/crawl \
     -H "Content-Type: application/json" \
     -d '{"url": "https://example.com"}'
```

### Testing Unauthenticated Request
```bash
curl -X POST http://localhost:6792/api/crawl \
     -H "Content-Type: application/json" \
     -d '{
       "url": "https://example.com",
       "customer_id": "test-client-123"
     }'
```

### Getting Session Files
```bash
curl "http://localhost:6792/api/sessions/{session_id}/files?customer_id=test-client-123"
```

---

## Session Updates (April 16, 2026)

### Auth: Switched from gnosis-auth HMAC to nuts-auth token exchange

`app/auth.py` was fully rewritten. Token validation now POSTs to `https://auth.nuts.services/auth` with form data `token=ahp_...`. Response is a JWT; email is extracted from the `sub` field in the payload.

```python
resp = await client.post(f"{self.auth_url}/auth", data={"token": token})
jwt_token = resp.json().get("access_token", "")
payload = json.loads(base64.urlsafe_b64decode(jwt_token.split(".")[1] + "=="))
email = payload.get("sub", "unknown@grub-crawl.local")
```

`validate_token_from_query()` was preserved — it handles internal short-lived HMAC tokens used by `verify_internal_token` in `main.py` (line 26 imports it, line 138 uses it). Do NOT remove it.

**Config changes:**
- `app/config.py`: `gnosis_auth_url` default changed to `"https://auth.nuts.services"`
- `.env.cloud`: `GNOSIS_AUTH_URL=https://auth.nuts.services`
- `deploy.ps1`: hardcoded `GNOSIS_AUTH_URL=https://auth.nuts.services` in env vars

**Production Cloud Run service**: `grub-crawl` at `grub.nuts.services`
**Test token**: stored in `.env` (not committed) — obtain from nuts.services dashboard

### Challenge Solver: Added Incapsula/Imperva support

`app/challenge_solver.py` was extended. Previously Cloudflare-only. Now also handles Incapsula/Imperva.

**New in challenge_solver.py:**
- `ChallengeType.INCAPSULA` added to enum
- `INCAPSULA_HTML_MARKERS` and `INCAPSULA_COOKIE_NAMES` constants
- `detect_incapsula(page)` — checks for `/_incapsula_resource` in page HTML
- `resolve_incapsula(page, url, timeout_ms)` — simulates mouse movement + scrolling via `human_behavior.py` while waiting for `visid_incap_*`/`incap_ses_*` cookies to be set, then re-navigates
- `resolve_challenge()` now checks Incapsula first before the Cloudflare pipeline

**`app/http_precheck.py`**: Added `_incapsula_resource`, `incapsula_resource`, `visid_incap_` to `_BROWSER_NEEDED_MARKERS`.

**`app/crawler.py`**: Added Incapsula detection to `_detect_block_signals()` — checks for `/_incapsula_resource` in HTML and returns `(True, "incapsula_challenge", False)` immediately (no substantial-content exemption since Incapsula pages are always tiny).

**Known limitation**: Sites with Incapsula in full lockdown mode (e.g. myronsprime.com) block even the `/_Incapsula_Resource` script from loading for datacenter IPs, so `page.goto()` times out before the solver can run. Residential proxy required for those.

### Cookie Injection: New `options.cookies` field

Users can now pass pre-solved cookies (e.g. from a browser session) to bypass challenges:

```json
{
  "url": "https://example.com",
  "options": {
    "cookies": {
      "visid_incap_1879042": "...",
      "incap_ses_1700_1879042": "..."
    }
  }
}
```

**Files changed:**
- `app/models.py`: Added `cookies: Optional[Dict[str, str]] = None` to `CrawlOptions`
- `app/browser.py`: `crawl_with_context()` accepts `cookies` param; injects them into the browser context via `context.add_cookies()` before `page.goto()`
- `app/crawler.py`: `crawl_url()` accepts and threads `cookies` through to `crawl_with_context()`
- `app/routes.py`: Both `/api/crawl` and `/api/markdown` endpoints pass `request.options.cookies` to `crawl_url()`

**These changes are local only — NOT yet deployed to Cloud Run as of April 16, 2026.**

### Deployment Protocol

**ALWAYS test locally before deploying to Cloud Run.**

```powershell
# Local test first
./deploy.ps1 -Target local
# Then test against http://localhost:6792

# Only then deploy to Cloud Run
./deploy.ps1 -Target cloudrun -Tag v1.x.x
# OR: gcloud run deploy grubcrawler --source . --region us-central1 --project gnosis-459403 --allow-unauthenticated --port 6792
```

Do NOT run `gcloud run deploy` without being asked. Do NOT run long deploys in Bash tool background processes. Use Hyperia terminal for long-running builds so the user can see progress.

### Related Services

| Service | URL | Cloud Run Name |
|---------|-----|----------------|
| grub-crawl (prod) | grub.nuts.services | grub-crawl |
| nuts-auth | auth.nuts.services | gnosis-auth |
| DeepBlue Dynamics site | deepbluedynamics.com | dbd-site |

### DeepBlue Dynamics Site

Static nginx site at `C:\Users\kordl\Code\DeepBlueDynamics\site`.

Pages: `/`, `/hyperia`, `/nemesis`, `/ferricula`, `/blog`, `/privacy`, `/terms`, `/lead`

**Critical nginx.conf fix** (already deployed): Must include `absolute_redirect off;` or sub-pages return `301` redirects to `http://...:8080/path/` which Cloud Run doesn't expose, causing browser hangs.

```nginx
server {
    listen 8080;
    root /usr/share/nginx/html;
    index index.html;
    absolute_redirect off;
    location / {
        try_files $uri $uri/ =404;
    }
}
```

Deploy: `gcloud run deploy deepblue-site --source . --region us-central1 --project gnosis-459403 --allow-unauthenticated --port 8080` from `C:\Users\kordl\Code\DeepBlueDynamics\site`.

---

**Last Updated**: April 16, 2026
**Current Version**: v1.0.0 (grub-crawl-00003-q9c)
**Status**: Production Ready ✅ — deployed to Cloud Run; cookie injection deployed with this revision
