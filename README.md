<div align="center">

<img src="https://img.shields.io/badge/🪱-GRUB_CRAWLER-black?style=for-the-badge&labelColor=0d1117" alt="Grub Crawler" />

<br/>

[![License](https://img.shields.io/badge/license-proprietary-red?style=flat-square)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.11+-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688?style=flat-square&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Playwright](https://img.shields.io/badge/Playwright-latest-2EAD33?style=flat-square&logo=playwright&logoColor=white)](https://playwright.dev)
[![MCP](https://img.shields.io/badge/MCP-compatible-blueviolet?style=flat-square)](https://modelcontextprotocol.io)
[![Ghost Protocol](https://img.shields.io/badge/Ghost_Protocol-active-ff6b6b?style=flat-square)](#ghost-protocol)
[![Live Stream](https://img.shields.io/badge/Live_Stream-CDP/WebSocket-00d4ff?style=flat-square)](#live-stream)
[![Camoufox](https://img.shields.io/badge/Camoufox-anti--detect-ff8c00?style=flat-square)](#anti-detection)
[![Proxy](https://img.shields.io/badge/Proxy-per--request-8b5cf6?style=flat-square)](#anti-detection)

<br/>

**Agentic web crawler with anti-detection, vision fallback, and peer-to-peer mesh.**

<br/>

<a href="#api-endpoints">Endpoints</a> · <a href="#mesh">Mesh</a> · <a href="#anti-detection">Anti-Detection</a> · <a href="#ghost-protocol">Ghost Protocol</a> · <a href="#live-stream">Live Stream</a> · <a href="#mcp-tools-grub-crawlpy">MCP Tools</a> · <a href="#quick-start">Quick Start</a> · <a href="#benchmarks">Benchmarks</a> · <a href="MASTER_PLAN.md">Architecture</a>

---

Full-stack web crawling engine with JavaScript rendering, Camoufox anti-detect browser, per-request proxy routing, and autonomous agent loops. Converts pages to clean markdown with a native Rust extraction engine. When standard crawling is blocked by Cloudflare, CAPTCHAs, or JavaScript walls, Ghost Protocol captures a screenshot and extracts content via vision AI (Claude, GPT-4o, or Ollama). Supports multi-provider LLM orchestration across OpenAI, Anthropic, and Ollama in a single session. Nodes coordinate over a gossip-based peer-to-peer mesh for distributed crawling.

---

</div>

## Why Grub

We integrated features from every major crawler — then added what none of them have.

### Self-Hosted Crawlers

| Feature | Crawl4AI | Firecrawl | Scrapy | **Grub** |
|---|---|---|---|---|
| **JS rendering** | ✅ Playwright | ✅ Playwright | ❌ HTTP only | ✅ **Playwright** |
| **Anti-detect browser** | stealth plugin | ❌ | ❌ | ✅ **Camoufox** |
| **Ghost Protocol** | ❌ | ❌ | ❌ | ✅ **auto fallback** |
| **Per-request proxy** | ✅ escalation | ❌ | middleware | ✅ **per-request** |
| **Stealth patches** | ✅ | ❌ | ❌ | ✅ **opt-in** |
| **Agent loop** | ✅ agentic | ✅ /agent | ❌ spiders | ✅ **bounded SM** |
| **Live browser stream** | ✅ WebSocket | ✅ Live View | ❌ | ✅ **WS + MJPEG** |
| **Markdown output** | ✅ Fit Markdown | ✅ core | ❌ | ✅ **Rust engine** |
| **MCP tools** | ✅ community | ✅ official | ⚠️ community | ✅ **15 tools** |
| **Multi-provider LLM** | ✅ all LLMs | ⚠️ Gemini | ❌ | ✅ **OpenAI/Anthropic/Ollama** |
| **Mesh P2P** | ❌ | ❌ | ❌ | ✅ **gossip protocol** |
| **Policy enforcement** | ❌ | ❌ | ❌ | ✅ **domain gates + redaction** |
| **Prompt injection defense** | ❌ | ❌ | ❌ | ✅ **quarantine + visible-text diff** |
| **License** | Apache 2.0 | AGPL-3.0 | BSD | Proprietary |
| **Pricing** | Free | Free–$333/mo | Free | Self-hosted |

### Cloud / Managed Crawlers

| Feature | Browserbase | Scrapfly | Firecrawl Cloud | **Grub** |
|---|---|---|---|---|
| **JS rendering** | ✅ custom Chromium | ✅ proprietary | ✅ Playwright | ✅ **Playwright** |
| **Anti-detect browser** | ✅ custom Chromium | ✅ proprietary | ✅ cloud stealth | ✅ **Camoufox** |
| **Ghost Protocol** | ❌ | ❌ | ❌ | ✅ **auto fallback** |
| **Per-request proxy** | ✅ managed | ✅ 130M+ IPs | ✅ cloud-managed | ✅ **per-request** |
| **Stealth patches** | ✅ built-in | ✅ built-in | ✅ built-in | ✅ **opt-in** |
| **Agent loop** | ✅ Stagehand | ⚠️ via integrations | ✅ /agent | ✅ **bounded SM** |
| **Live browser stream** | ✅ iFrame + CDP | ✅ CDP | ✅ Live View | ✅ **WS + MJPEG** |
| **Markdown output** | ✅ via MCP | ✅ built-in | ✅ core | ✅ **Rust engine** |
| **MCP tools** | ✅ official | ✅ official | ✅ official | ✅ **15 tools** |
| **Mesh P2P** | ❌ | ❌ | ❌ | ✅ **gossip protocol** |
| **Policy enforcement** | ❌ | ❌ | ❌ | ✅ **domain gates + redaction** |
| **Prompt injection defense** | ❌ | ❌ | ❌ | ✅ **quarantine + visible-text diff** |
| **Self-hostable** | ❌ cloud only | ❌ cloud only | ⚠️ limited OSS | ✅ **full + Cloud Run** |
| **Pricing** | Free–$99/mo | Usage-based | Free–$333/mo | Self-hosted |

**Only Grub has Ghost Protocol** — automatic vision-based fallback that screenshots blocked pages and extracts content via LLM when standard crawling fails. Prevention (Camoufox + proxy + stealth) handles 95% of blocks. Ghost Protocol handles the rest.

## API Endpoints

### Core Crawling
| Method | Path | Description | Status |
|--------|------|-------------|--------|
| `POST` | `/api/crawl` | Single URL crawl (HTML + markdown) | Live |
| `POST` | `/api/markdown` | Single or multi-URL markdown extraction | Live |
| `POST` | `/api/batch` | Batch crawl with job tracking | Live |
| `POST` | `/api/raw` | Raw HTML extraction (no markdown) | Live |
| `GET`  | `/view` | Browser-rendered HTML viewer | Live |
| `GET`  | `/download` | File download (PDFs, etc.) through crawler | Live |

### Agent (Mode B)
| Method | Path | Description | Status |
|--------|------|-------------|--------|
| `POST` | `/api/agent/run` | Submit task to autonomous agent loop | Live |
| `GET`  | `/api/agent/status/{run_id}` | Check agent run status / load trace | Live |
| `POST` | `/api/agent/ghost` | Ghost Protocol: screenshot + vision extract | Live |

### Job Management
| Method | Path | Description | Status |
|--------|------|-------------|--------|
| `POST` | `/api/jobs/create` | Generic job submission | Live |
| `POST` | `/api/jobs/crawl` | Submit single URL crawl job | Live |
| `POST` | `/api/jobs/batch-crawl` | Submit batch crawl job | Live |
| `POST` | `/api/jobs/markdown` | Submit markdown-only job | Live |
| `POST` | `/api/jobs/process-job` | Cloud Tasks worker endpoint | Live |
| `POST` | `/api/wraith` | AI-driven crawl workflow | Placeholder |

### Remote Cache
| Method | Path | Description | Status |
|--------|------|-------------|--------|
| `POST` | `/api/cache/search` | Fuzzy search cached content | Live |
| `GET`  | `/api/cache/list` | List cached document metadata | Live |
| `GET`  | `/api/cache/doc/{doc_id}` | Fetch one cached document | Live |
| `POST` | `/api/cache/upsert` | Upsert cache entries | Live |
| `POST` | `/api/cache/prune` | Prune cache entries by TTL/domain | Live |

### Session Management
| Method | Path | Description | Status |
|--------|------|-------------|--------|
| `GET`  | `/api/sessions/{session_id}/files` | List session files | Live |
| `GET`  | `/api/sessions/{session_id}/file` | Get specific file | Live |
| `GET`  | `/api/sessions/{session_id}/status` | Session progress status | Live |
| `GET`  | `/api/sessions/{session_id}/results` | All crawl results | Live |
| `GET`  | `/api/sessions/{session_id}/screenshots` | List screenshots | Live |

### Live Stream
| Method | Path | Description | Status |
|--------|------|-------------|--------|
| `WS`   | `/stream/{session_id}` | WebSocket viewport stream | Live |
| `GET`  | `/stream/{session_id}/mjpeg` | MJPEG fallback stream | Live |
| `GET`  | `/stream/{session_id}/status` | Stream session status | Live |
| `GET`  | `/stream/pool/status` | Browser pool status | Live |

### Mesh
| Method | Path | Description | Status |
|--------|------|-------------|--------|
| `POST` | `/mesh/join` | Peer join + gossip discovery | Live |
| `POST` | `/mesh/heartbeat` | Peer heartbeat with load metrics | Live |
| `POST` | `/mesh/execute` | Cross-node tool execution (1-hop max) | Live |
| `POST` | `/mesh/leave` | Peer departure notification | Live |
| `GET`  | `/mesh/peers` | List known peers + health status | Live |
| `GET`  | `/mesh/status` | This node's mesh status + load | Live |

### System
| Method | Path | Description | Status |
|--------|------|-------------|--------|
| `GET`  | `/health` | Health check + tool count + mesh info | Live |
| `GET`  | `/tools` | List registered AHP tools | Live |
| `GET`  | `/site` | Embedded landing page | Live |
| `GET`  | `/{tool_name}` | Execute AHP tool (catch-all) | Live |

## MCP Tools (grub-crawl.py)

The MCP bridge exposes all capabilities to any MCP-compatible host:

| Tool | Description | Status |
|------|-------------|--------|
| `crawl_url` | Single URL markdown extraction with JS injection | Live |
| `crawl_batch` | Batch processing up to 50 URLs with collation | Live |
| `raw_html` | Raw HTML fetch without conversion | Live |
| `download_file` | Download files (PDFs, etc.) through crawler | Live |
| `crawl_validate` | Content quality assessment | Live |
| `crawl_search` | Fuzzy search local crawl cache | Live |
| `crawl_cache_list` | List local cached files | Live |
| `crawl_remote_search` | Search remote crawler cache | Live |
| `crawl_remote_cache_list` | List remote cache entries | Live |
| `crawl_remote_cache_doc` | Fetch remote cached document | Live |
| `agent_run` | Submit task to autonomous agent (Mode B) | Live |
| `agent_status` | Check agent run status | Live |
| `ghost_extract` | Ghost Protocol: screenshot + vision AI extraction | Live |
| `mesh_peers` | List mesh peers and their health/load status | Live |
| `mesh_status` | Get this node's mesh status and load metrics | Live |
| `set_auth_token` | Save auth token to .wraithenv | Live |
| `crawl_status` | Report configuration and connection | Live |

## Internal Modules

### Agent Core (`app/agent/`)
| File | Purpose | Status |
|------|---------|--------|
| `types.py` | `RunState` enum, `StopReason`, `ToolCall`, `ToolResult`, `AssistantAction`, `RunConfig`, `RunContext`, `StepTrace`, `RunResult` | Done |
| `errors.py` | Typed errors: `validation_error`, `policy_denied`, `tool_timeout`, `tool_unavailable`, `execution_error`, `provider_error`, `stop_condition` | Done |
| `dispatcher.py` | Tool validation, timeout enforcement (30s), retry (1x), typed error normalization | Done |
| `engine.py` | Bounded loop: `plan -> execute -> observe -> stop`. EventBus integration. Returns `(RunResult, RunSummary)` | Done |
| `ghost.py` | Ghost Protocol: block detection, screenshot capture, vision extraction, auto-trigger | Done |

### Provider Adapters (`app/agent/providers/`)
| File | Purpose | Status |
|------|---------|--------|
| `base.py` | `LLMAdapter` ABC, `FallbackAdapter` (rotate on failure), factory functions | Done |
| `openai_adapter.py` | OpenAI tool_calls mapping, GPT-4o vision | Done |
| `anthropic_adapter.py` | Anthropic tool_use/tool_result blocks, Claude Sonnet vision | Done |
| `ollama_adapter.py` | Ollama HTTP `/api/chat`, llava vision | Done |

### Policy Gates (`app/policy/`)
| File | Purpose | Status |
|------|---------|--------|
| `domain.py` | Domain allowlist, RFC-1918/loopback/link-local deny | Done |
| `gate.py` | Pre-tool and pre-fetch policy checks with `PolicyVerdict` | Done |
| `redaction.py` | Secret pattern redaction (API keys, JWTs, private keys) | Done |

### Observability (`app/observability/`)
| File | Purpose | Status |
|------|---------|--------|
| `events.py` | `EventBus` + 7 typed events: `run_start`, `step_start`, `tool_dispatch`, `tool_result`, `policy_denied`, `step_end`, `run_end` | Done |
| `trace.py` | `TraceCollector`, `RunSummary` JSON serialization, `persist_trace()` / `load_trace()` via storage | Done |

### API Layer
| File | Purpose | Status |
|------|---------|--------|
| `agent_routes.py` | `POST /api/agent/run`, `GET /api/agent/status/{run_id}`. 503 when disabled | Done |
| `routes.py` | Core crawl/markdown/batch/cache REST endpoints | Done |
| `job_routes.py` | Job CRUD, session status, Cloud Tasks worker | Done |
| `jobs.py` | `JobType` enum (incl. `AGENT_RUN`), `JobManager`, `JobProcessor` | Done |
| `models.py` | All Pydantic models incl. `AgentRunRequest/Response` | Done |

### Anti-Detection (`app/`)
| File | Purpose | Status |
|------|---------|--------|
| `stealth.py` | playwright-stealth patches, tracker domain blocking | Done |
| `proxy.py` | Per-request proxy resolution with env fallback | Done |

### Mesh (`app/mesh/`)
| File | Purpose | Status |
|------|---------|--------|
| `models.py` | Wire protocol models: NodeInfo, NodeLoad, MeshToolRequest/Response, PeerState | Done |
| `auth.py` | HMAC-SHA256 token signing/verification with 60s TTL | Done |
| `client.py` | httpx async client for join, heartbeat, leave, execute_tool | Done |
| `coordinator.py` | Lifecycle, peer table, heartbeat loop with seed retry | Done |
| `routes.py` | `/mesh/*` endpoints — join, heartbeat, execute, leave, peers, status | Done |
| `router.py` | Load scoring + target selection (pure logic, no I/O) | Done |
| `dispatcher.py` | MeshDispatcher wrapping local Dispatcher for transparent routing | Done |

### Infrastructure
| File | Purpose | Status |
|------|---------|--------|
| `config.py` | All env vars incl. agent + provider + ghost + proxy + stealth config | Done |
| `storage.py` | User-partitioned storage (local filesystem / GCS) | Done |
| `crawler.py` | Playwright crawling engine with proxy support | Done |
| `markdown.py` | HTML to markdown conversion | Done |
| `browser.py` | Browser automation — Chromium + Camoufox engines | Done |
| `browser_pool.py` | Persistent browser pool with lease/return pattern | Done |
| `stream.py` | CDP screencast → WebSocket/MJPEG relay + interactive commands | Done |

## Agent State Machine

```
INIT -> PLAN -> EXECUTE_TOOL -> OBSERVE -> PLAN -> ... -> RESPOND -> STOP
                     |                                        |
                     +-- policy_denied ---------------------->+
                     +-- max_steps / max_wall_time / max_failures -> STOP
                     +-- no_op_loop (3x empty) ------------> STOP
                     +-- blocked (ghost trigger) -----------> GHOST -> OBSERVE
```

Stop conditions enforced every iteration:
- `max_steps` (default: 12)
- `max_wall_time` (default: 90s)
- `max_failures` (default: 3)
- `no_op_loop` (3 consecutive empty responses)
- `policy_denied` (blocked tool/domain)
- `completed` (agent responds with text)

## Anti-Detection

Three layers of anti-detection that stack together. Prevention stops blocks before they happen. Ghost Protocol handles them after.

### Camoufox Engine

Pluggable anti-detect browser with C++-level fingerprint spoofing. No manual user-agent tricks — Camoufox generates realistic fingerprints per context at the browser level, including canvas, WebGL, fonts, and navigator properties.

```bash
# Switch engine (default: chromium)
BROWSER_ENGINE=camoufox
```

### Per-Request Proxy

Route crawl traffic through residential, datacenter, or custom proxy pools. Per-request override with env-based defaults. Full Playwright-compatible proxy config.

```bash
# Env-based default
PROXY_SERVER=http://proxy.example.com:10001
PROXY_USERNAME=your_username
PROXY_PASSWORD=your_password

# Or per-request
curl -X POST http://localhost:6792/api/crawl \
  -H "Content-Type: application/json" \
  -d '{
    "url": "https://example.com",
    "options": {
      "proxy": {
        "server": "http://proxy.example.com:10001",
        "username": "your_username",
        "password": "your_password"
      }
    }
  }'
```

### Stealth Mode

Opt-in `playwright-stealth` patches for Chromium (skipped for Camoufox where it's built-in). Blocks 20+ tracking/analytics domains (Google Analytics, DataDome, PerimeterX, etc.) to reduce fingerprint surface.

```bash
STEALTH_ENABLED=true
BLOCK_TRACKING_DOMAINS=true
```

## Ghost Protocol

When a crawl result signals an anti-bot block (Cloudflare challenge, CAPTCHA,
empty SPA shell), the agent can switch to cloak mode:

1. Take a full-page screenshot via Playwright
2. Send the image to a vision-capable LLM (Claude Sonnet or GPT-4o)
3. Extract content from the rendered pixels
4. Return extracted text with `render_mode: "ghost"` in the trace

This bypasses DOM-based anti-bot detection entirely.

Requires `AGENT_GHOST_ENABLED=true`. Auto-triggers on detected blocks when `AGENT_GHOST_AUTO_TRIGGER=true`.

## Mesh

Agents talking to agents. Every Grub instance is both a worker and a coordinator. Local node offloads to cloud, cloud delegates to local. Tool calls cross the wire transparently.

```
Node A (local)                    Node B (cloud)
┌─────────────┐                  ┌─────────────┐
│ AgentEngine  │                  │ AgentEngine  │
│     ↓        │                  │     ↓        │
│ MeshDispatcher ──── HTTP ────→ MeshDispatcher │
│     ↓        │                  │     ↓        │
│ Dispatcher   │                  │ Dispatcher   │
│     ↓        │                  │     ↓        │
│ ToolRegistry │                  │ ToolRegistry │
└─────────────┘                  └─────────────┘
       ↕ heartbeat (15s)                ↕
       └────────────────────────────────┘
```

**How it works:**
- **Discovery** — nodes join via seed peer list, then gossip (1-hop) to learn about others
- **Heartbeat** — every 15s, nodes exchange load metrics. 3 missed = unhealthy. 2 min = removed
- **Routing** — MeshDispatcher scores all nodes by load, locality, and affinity, then routes tool calls to the best node
- **1-hop max** — Node A → B only, never A → B → C. Prevents routing loops
- **Local fallback** — if remote execution fails, falls back to local Dispatcher
- **HMAC auth** — all mesh traffic is signed with a shared secret (SHA-256, 60s TTL)

### Run a 2-Node Mesh Locally

```bash
# Docker Compose (recommended)
./deploy.sh mesh           # Linux/Mac
./deploy.ps1 -Target mesh  # Windows

# Verify
curl http://localhost:6792/mesh/peers  # Node A sees Node B
curl http://localhost:6793/mesh/peers  # Node B sees Node A
```

### Connect Local to Cloud Run

```bash
# Deploy to Cloud Run with mesh
./deploy.sh cloudrun latest --mesh-peer http://your-local-ip:6792 --mesh-secret mysecret

# Start local node
MESH_ENABLED=true MESH_SECRET=mysecret MESH_PEERS=https://your-cloud-run-url \
  MESH_ADVERTISE_URL=http://your-local-ip:6792 \
  uvicorn app.main:app --port 6792
```

### Manual Setup

```bash
# Node A
MESH_ENABLED=true MESH_NODE_NAME=local MESH_SECRET=test123 \
  MESH_ADVERTISE_URL=http://localhost:6792 \
  uvicorn app.main:app --port 6792

# Node B
MESH_ENABLED=true MESH_NODE_NAME=cloud MESH_SECRET=test123 \
  MESH_PEERS=http://localhost:6792 \
  MESH_ADVERTISE_URL=http://localhost:8081 \
  uvicorn app.main:app --port 8081
```

When mesh is disabled (`MESH_ENABLED=false`, the default), Grub operates as a normal single-node crawler with zero mesh overhead.

## Live Stream

Watch the crawler work in real-time. A persistent pool of warm Chromium instances streams viewport frames over WebSocket or MJPEG.

**WebSocket** — connect and send interactive commands:
```javascript
const ws = new WebSocket("ws://localhost:6792/stream/my-session?url=https://example.com");
ws.onmessage = (e) => {
  const msg = JSON.parse(e.data);
  if (msg.type === "frame") document.getElementById("viewport").src = "data:image/jpeg;base64," + msg.data;
};
// Navigate, click, scroll, type — all over the same socket
ws.send(JSON.stringify({ action: "navigate", url: "https://example.com/pricing" }));
ws.send(JSON.stringify({ action: "click", selector: "#signup-btn" }));
ws.send(JSON.stringify({ action: "scroll", direction: "down" }));
```

**MJPEG** — drop it in an `<img>` tag, instant video:
```html
<img src="http://localhost:6792/stream/my-session/mjpeg?url=https://example.com" />
```

Requires `BROWSER_STREAM_ENABLED=true`. Each Chromium instance uses ~150-300MB RAM.

## Quick Start

### Local Development

```bash
git clone <repo>
cd grub-crawl
cp .env.example .env
pip install -r requirements.txt
uvicorn app.main:app --reload --host 0.0.0.0 --port 6792
```

### Enable Agent Mode B

```bash
# Add to .env
AGENT_ENABLED=true
OPENAI_API_KEY=sk-...
# or
ANTHROPIC_API_KEY=sk-ant-...
AGENT_PROVIDER=anthropic
```

### Submit an Agent Task

```bash
curl -X POST http://localhost:6792/api/agent/run \
  -H "Content-Type: application/json" \
  -d '{
    "task": "Find the pricing page on example.com and extract plan details",
    "max_steps": 10,
    "allowed_domains": ["example.com"]
  }'
```

### Docker

```bash
# Single node
./deploy.sh local            # or ./deploy.ps1 -Target local

# 2-node mesh
./deploy.sh mesh             # or ./deploy.ps1 -Target mesh

# Cloud Run
./deploy.sh cloudrun v1.0.0  # or ./deploy.ps1 -Target cloudrun -Tag v1.0.0

# Cloud Run + mesh (connect to local node)
./deploy.sh cloudrun v1.0.0 --mesh-peer http://your-ip:6792 --mesh-secret mykey
```

### Anti-Detection (Camoufox + Proxy)

```bash
# Add to .env
BROWSER_ENGINE=camoufox
STEALTH_ENABLED=true
BLOCK_TRACKING_DOMAINS=true

# Optional: proxy
PROXY_SERVER=http://proxy.example.com:10001
PROXY_USERNAME=your_username
PROXY_PASSWORD=your_password
```

### Ghost Protocol (anti-bot bypass)

```bash
# Add to .env
AGENT_GHOST_ENABLED=true

curl -X POST http://localhost:6792/api/agent/ghost \
  -H "Content-Type: application/json" \
  -d '{"url": "https://blocked-site.com"}'
```

### Live Browser Stream

```bash
# Add to .env
BROWSER_STREAM_ENABLED=true
BROWSER_POOL_SIZE=2

# MJPEG (open in browser)
open "http://localhost:6792/stream/demo/mjpeg?url=https://example.com"
```

## Configuration

### Server
- `HOST` (default: 0.0.0.0)
- `PORT` (default: 6792)
- `DEBUG` (default: false)

### Storage
- `STORAGE_PATH` (default: ./storage)
- `RUNNING_IN_CLOUD` (default: false)
- `GCS_BUCKET_NAME`
- `GOOGLE_CLOUD_PROJECT`

### Authentication
- `DISABLE_AUTH` (default: false)
- `GNOSIS_AUTH_URL` (default: http://gnosis-auth:5000)

### Browser Engine
- `BROWSER_ENGINE` — chromium | camoufox (default: chromium)

### Crawling
- `MAX_CONCURRENT_CRAWLS` (default: 5)
- `CRAWL_TIMEOUT` (default: 30)
- `ENABLE_JAVASCRIPT` (default: true)
- `ENABLE_SCREENSHOTS` (default: false)

### Proxy
- `PROXY_SERVER` — proxy URL (e.g. http://proxy:10001)
- `PROXY_USERNAME`
- `PROXY_PASSWORD`
- `PROXY_BYPASS` — comma-separated bypass list

### Stealth
- `STEALTH_ENABLED` (default: false) — playwright-stealth patches
- `BLOCK_TRACKING_DOMAINS` (default: false) — block analytics/tracking requests

### Agent (Mode B)
- `AGENT_ENABLED` (default: false)
- `AGENT_MAX_STEPS` (default: 12)
- `AGENT_MAX_WALL_TIME_MS` (default: 90000)
- `AGENT_MAX_FAILURES` (default: 3)
- `AGENT_ALLOWED_TOOLS` — comma-separated allowlist
- `AGENT_ALLOWED_DOMAINS` — comma-separated allowlist
- `AGENT_BLOCK_PRIVATE_RANGES` (default: true)
- `AGENT_REDACT_SECRETS` (default: true)

### LLM Providers
- `AGENT_PROVIDER` — openai | anthropic | ollama (default: openai)
- `OPENAI_API_KEY`
- `OPENAI_MODEL` (default: gpt-4.1-mini)
- `ANTHROPIC_API_KEY`
- `ANTHROPIC_MODEL` (default: claude-3-5-sonnet-latest)
- `OLLAMA_BASE_URL` (default: http://localhost:11434)
- `OLLAMA_MODEL` (default: llama3.1:8b-instruct)

### Ghost Protocol
- `AGENT_GHOST_ENABLED` (default: false)
- `AGENT_GHOST_AUTO_TRIGGER` (default: true)
- `AGENT_GHOST_VISION_PROVIDER` — inherits from AGENT_PROVIDER
- `AGENT_GHOST_MAX_IMAGE_WIDTH` (default: 1280)

### Mesh
- `MESH_ENABLED` (default: false) — master switch
- `MESH_PEERS` — comma-separated seed peer URLs
- `MESH_NODE_NAME` — human-readable name (default: hostname)
- `MESH_SECRET` — shared HMAC secret for inter-node auth
- `MESH_ADVERTISE_URL` — URL peers use to reach this node
- `MESH_PREFER_LOCAL` (default: true) — bias toward local execution
- `MESH_HEARTBEAT_INTERVAL_S` (default: 15)
- `MESH_PEER_TIMEOUT_S` (default: 45) — mark unhealthy after this
- `MESH_PEER_REMOVE_S` (default: 120) — remove from peer table after this
- `MESH_REMOTE_TIMEOUT_MS` (default: 35000) — timeout for remote tool calls

### Live Stream
- `BROWSER_POOL_SIZE` (default: 1)
- `BROWSER_STREAM_ENABLED` (default: false)
- `BROWSER_STREAM_QUALITY` (default: 25) — JPEG quality 1-100
- `BROWSER_STREAM_MAX_WIDTH` (default: 854)
- `BROWSER_STREAM_MAX_LEASE_SECONDS` (default: 300)

## Response Contract

`POST /api/markdown` returns:

`success`, `url`, `final_url`, `status_code`, `markdown`, `markdown_plain`, `content`, `render_mode`, `wait_strategy`, `timings_ms`, `blocked`, `block_reason`, `captcha_detected`, `http_error_family`, `body_char_count`, `body_word_count`, `visible_char_count`, `visible_word_count`, `visible_similarity`, `quarantined`, `quarantine_reason`, `policy_flags`, `content_quality`, `extractor_version`, `normalized_url`, `content_hash`

### Content Quality

- `blocked` — anti-bot/captcha/challenge
- `empty` — very low signal
- `minimal` — thin/error pages
- `sufficient` — usable for summarization

Do not summarize unless `content_quality == "sufficient"`.

### Prompt Injection Defense

- `quarantined=true` means the extractor detected instruction-like text in extracted content that was not present in the page's visible rendered text (common in `.sr-only`/visually-hidden abuse).
- When quarantined, `content_quality` is downgraded to `minimal`, `policy_flags` includes `hidden_text_suspected` and `quarantined`, and `content`/`markdown` outputs are blanked (fail-closed).

### Error Format

```json
{"error": "http_error|validation_error|internal_error", "status": 400, "details": {}}
```

## Benchmarks

Combat arena — head-to-head benchmarks against Crawl4AI, Firecrawl (self-hosted), and Scrapy. All tests run on the same machine, same URLs, same conditions. Grub runs first as baseline, remaining adapters in randomized order with 10s delay between each to prevent rate-limiting bias.

### Single-URL Speed (ms, lower is better)

| URL | Grub | Crawl4AI | Firecrawl | Scrapy | Winner |
|-----|------|----------|-----------|--------|--------|
| example.com | **391** | 1283 | 1513 | 831 | Grub |
| news.ycombinator.com | **423** | 2204 | 1401 | 2038 | Grub |
| en.wikipedia.org | **507** | 4866 | 2637 | 2368 | Grub |
| httpbin.org | 283 | 1170 | **176** | 769 | Firecrawl |
| quotes.toscrape.com | **275** | 1244 | 647 | 1145 | Grub |

Grub wins 4/5 single-URL speed races. Markdown conversion runs at 0-21ms via native Rust engine (`grub_md`).

### Grub Phase Breakdown (server-side ms)

| URL | nav | content | visible_text | markdown | total |
|-----|-----|---------|--------------|----------|-------|
| example.com | 358 | 11 | 14 | 0 | 391 |
| news.ycombinator.com | 384 | 19 | 13 | 2 | 423 |
| en.wikipedia.org | 369 | 48 | 38 | 21 | 507 |
| httpbin.org | 229 | 27 | 16 | 0 | 283 |
| quotes.toscrape.com | 230 | 10 | 27 | 1 | 275 |

Navigation dominates; markdown conversion is sub-millisecond on most pages thanks to the Rust engine.

### Batch Throughput (ms, lower is better)

| Batch | Grub | Crawl4AI | Firecrawl | Scrapy | Winner |
|-------|------|----------|-----------|--------|--------|
| 10 URLs | **1940** | 4670 | 3216 | 8680 | Grub |
| 25 URLs | **4085** | 10479 | 6692 | 25303 | Grub |
| 50 URLs | 15604 | 23870 | **12729** | 36232 | Firecrawl |

Grub wins 2/3 batch sizes. Per-URL cost: 163-312ms (Grub) vs 255-477ms (others).

### How to Run

```bash
# Start Grub
docker compose up -d

# Start Firecrawl (optional)
docker compose -f combat/firecrawl-compose.yaml up -d

# Install combat deps
pip install crawl4ai scrapy markdownify tabulate

# Run the arena
pytest combat/ -m combat -v

# Generate report
python -m combat.report
```

## Development Status

### Phase 1: Core Infrastructure ✅
### Phase 2: Crawling ✅
### Phase 3: Agent Module ✅
- [x] Agent core — state machine, types, errors (W1)
- [x] Unified tool contract — dispatcher with timeout/retry (W2)
- [x] Policy gates — domain allowlist, private-range deny, redaction (W3)
- [x] Observability — EventBus, TraceCollector, RunSummary persistence (W4)
- [x] API wiring — `/api/agent/run`, `/api/agent/status`, JobType.AGENT_RUN (W5)
- [x] Provider adapters — OpenAI, Anthropic, Ollama with fallback (W6)
- [x] Config flags — agent, provider, ghost, stream settings (W7)

### Phase 4: Ghost Protocol ✅
- [x] Cloak-mode trigger detection (W8)
- [x] Screenshot capture pipeline (W8)
- [x] Vision extraction via Claude/GPT-4o (W8)
- [x] Fallback chain in engine (W8)
- [x] Ghost tool for external callers (W8)
- [x] Ghost MCP tool + REST endpoint (W8)

### Phase 5: Live Browser Stream ✅
- [x] Persistent browser pool with lease/return (W9)
- [x] CDP screencast relay (W9)
- [x] WebSocket endpoint with interactive commands (W9)
- [x] MJPEG fallback stream (W9)
- [x] Stream status + pool status endpoints (W9)

### Phase 5.5: Anti-Detection ✅
- [x] Camoufox anti-detect browser engine (W10)
- [x] Per-request proxy with env fallback (W10)
- [x] Stealth patches for Chromium (W10)
- [x] Tracker/analytics domain blocking (W10)
- [x] Anthropic vision format detection fix (W10)

### Phase 6: Mesh Coordinator ✅
- [x] Peer discovery with gossip (1-hop) (W11)
- [x] HMAC-SHA256 inter-node auth (W11)
- [x] Heartbeat loop with load metrics + seed retry (W11)
- [x] MeshDispatcher — transparent cross-node tool routing (W12)
- [x] Load-based scoring with locality/affinity bonus (W12)
- [x] Deploy scripts — local, mesh, Cloud Run (W12)
- [x] Docker Compose 2-node mesh topology (W12)
- [x] Embedded landing page (grub-site) (W12)

### Phase 7: Performance + Hardening
- [x] Rust markdown engine (`grub_md`) — PyO3 native extension, sub-ms conversion
- [x] Combat arena — automated benchmarks vs Crawl4AI, Firecrawl, Scrapy
- [x] Unit test suite — 176 tests across all modules
- [ ] Error handling improvements
- [ ] Monitoring and alerting

See [MASTER_PLAN.md](MASTER_PLAN.md) for the full architecture plan.

## License

Grub Crawler Project License
