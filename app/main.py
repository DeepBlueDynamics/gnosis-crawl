"""
Grub Crawler — FastAPI Application
Agentic web crawling service with mesh P2P
"""
import os
import re
import html as html_lib
import logging
import uuid
import json
from contextlib import asynccontextmanager
from typing import Dict, Any, AsyncGenerator, Optional
from fastapi import FastAPI, Request, Response, HTTPException, Depends
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, StreamingResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware

from pathlib import Path

from app.config import settings
from app.routes import router
from app.job_routes import router as job_router
from app.agent_routes import router as agent_router
from app.tools.tool_registry import get_global_registry, ToolError
from app.core.middleware import ContentTypeMiddleware, AuthMiddleware
from app.auth import validate_token_from_query
from app.crawler import get_crawler_engine

# Resolve site directory (embedded grub-site landing page)
_SITE_DIR = Path(__file__).resolve().parent.parent / "site"
_SITE_INDEX = _SITE_DIR / "index.html" if _SITE_DIR.is_dir() else None

# Configure logging
logging.basicConfig(
    level=logging.INFO if not settings.debug else logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
CRAWL_SECRET_KEY = os.getenv("CRAWL_SECRET_KEY")

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    # Startup
    logger.info(f"Starting Grub Crawler service on {settings.host}:{settings.port}")
    logger.info(f"Cloud mode: {settings.is_cloud_environment()}")
    logger.info(f"Auth service: {settings.gnosis_auth_url}")

    # Discover and register tools
    tool_registry = get_global_registry()
    try:
        discovered_schemas = tool_registry.discover_tools("app/tools")
        logger.info(f"Discovered {len(discovered_schemas)} crawling tools")
        for schema in discovered_schemas:
            logger.info(f"  - {schema['name']}: {schema['description']}")
    except Exception as e:
        logger.error(f"Failed to discover tools: {e}")

    # Start mesh coordinator (if enabled)
    if settings.mesh_enabled:
        try:
            from app.mesh.coordinator import MeshCoordinator
            from app.tools.tool_registry import get_global_registry as _get_reg

            _reg = _get_reg()
            tool_names = [s["name"] for s in _reg.get_schemas()]

            coordinator = MeshCoordinator(
                node_name=settings.mesh_node_name,
                advertise_url=settings.mesh_advertise_url,
                secret=settings.mesh_secret,
                seed_peers=settings.get_mesh_peers(),
                heartbeat_interval_s=settings.mesh_heartbeat_interval_s,
                peer_timeout_s=settings.mesh_peer_timeout_s,
                peer_remove_s=settings.mesh_peer_remove_s,
                tools=tool_names,
                max_concurrent_crawls=settings.max_concurrent_crawls,
            )
            app.state.mesh_coordinator = coordinator
            await coordinator.start()
            logger.info("Mesh coordinator started: %s (%s)", coordinator.node_name, coordinator.node_id)
        except Exception as e:
            logger.error("Failed to start mesh coordinator: %s", e, exc_info=True)
            app.state.mesh_coordinator = None

    # Start browser pool for live streaming (if enabled)
    if settings.browser_stream_enabled:
        try:
            from app.browser_pool import get_browser_pool
            pool = await get_browser_pool()
            logger.info(f"Browser pool started (size={settings.browser_pool_size})")
        except Exception as e:
            logger.error(f"Failed to start browser pool: {e}")

    yield

    # Shutdown mesh coordinator
    if settings.mesh_enabled and getattr(app.state, "mesh_coordinator", None):
        try:
            await app.state.mesh_coordinator.stop()
            logger.info("Mesh coordinator stopped")
        except Exception as e:
            logger.error("Error stopping mesh coordinator: %s", e)

    # Shutdown
    if settings.browser_stream_enabled:
        try:
            from app.browser_pool import shutdown_browser_pool
            await shutdown_browser_pool()
            logger.info("Browser pool shut down")
        except Exception as e:
            logger.error(f"Error shutting down browser pool: {e}")

    logger.info("Shutting down Grub Crawler service")


# Auth Dependency for AHP Tool Routes
async def verify_internal_token(request: Request):
    """
    Dependency that validates the short-lived, internal HMAC token found in
    pre-signed tool URLs and populates request.state.actor with the payload.
    """
    # Skip auth if disabled globally (Porter/Kubernetes deployments)
    if settings.disable_auth:
        logger.debug("Auth disabled - skipping token verification")
        request.state.actor = {"sub": "anonymous", "agent_id": "porter"}
        request.state.bearer_token = "disabled"
        return

    bearer_token = request.query_params.get("bearer_token")
    if not bearer_token:
        raise HTTPException(status_code=401, detail="Missing bearer token")

    try:
        # Validates signature and expiration, returns the full payload
        token_payload = validate_token_from_query(bearer_token, CRAWL_SECRET_KEY)
        request.state.actor = token_payload  # Payload is {"sub": "...", "agent_id": "..."}
        request.state.bearer_token = bearer_token

    except HTTPException as e:
        # Re-raise HTTP exceptions
        raise e
    except Exception as e:
        logger.error(f"Internal token validation failed with an unexpected error: {e}", exc_info=True)
        raise HTTPException(status_code=401, detail="Token validation failed")


# Create FastAPI application
app = FastAPI(
    title="Grub Crawler",
    description="Agentic web crawling service with mesh P2P",
    version="1.0.0",
    docs_url=None,  # Disable default docs for AHP pattern
    redoc_url=None,
    lifespan=lifespan
)


@app.exception_handler(HTTPException)
async def http_exception_handler(_: Request, exc: HTTPException):
    detail = exc.detail
    if isinstance(detail, dict):
        details = detail
    else:
        details = {"message": str(detail)}
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": "http_error",
            "status": exc.status_code,
            "details": details,
        },
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(_: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=422,
        content={
            "error": "validation_error",
            "status": 422,
            "details": exc.errors(),
        },
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(_: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "internal_error",
            "status": 500,
            "details": {"message": str(exc)},
        },
    )

# Add middleware (order matters)
app.add_middleware(ContentTypeMiddleware)
app.add_middleware(AuthMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes (traditional REST endpoints)
app.include_router(router, prefix="/api")
app.include_router(job_router)  # Job routes already have /api prefix
app.include_router(agent_router)  # Agent routes already have /api/agent prefix

# Mesh routes (conditional)
if settings.mesh_enabled:
    from app.mesh.routes import router as mesh_router
    app.include_router(mesh_router)

# Live browser stream routes (conditional)
if settings.browser_stream_enabled:
    from app.stream import router as stream_router
    app.include_router(stream_router)

# Health check endpoint (no auth required) - MUST be before catch-all route
@app.get("/health")
async def health_check():
    """Basic health check endpoint"""
    logger.debug("Health endpoint called - starting execution")
    try:
        tool_registry = get_global_registry()
        result = {
            "status": "healthy",
            "service": "grub-crawl",
            "version": "1.0.0",
            "cloud_mode": settings.is_cloud_environment(),
            "tools_registered": len(tool_registry.tools),
        }
        # Add mesh info if enabled
        coordinator = getattr(app.state, "mesh_coordinator", None)
        if coordinator:
            result["mesh"] = {
                "node_id": coordinator.node_id,
                "node_name": coordinator.node_name,
                "peers": len(coordinator.get_peers()),
                "healthy_peers": len(coordinator.get_healthy_peers()),
            }
        logger.debug(f"Health endpoint returning: {result}")
        return result
    except Exception as e:
        logger.error(f"Health endpoint error: {e}")
        raise

def _inject_base_tag(html_content: str, base_url: str) -> str:
    if not html_content or not base_url:
        return html_content
    if re.search(r"<base\\s", html_content, flags=re.IGNORECASE):
        return html_content
    base_tag = f'<base href="{html_lib.escape(base_url, quote=True)}">'
    head_match = re.search(r"<head[^>]*>", html_content, flags=re.IGNORECASE)
    if head_match:
        insert_at = head_match.end()
        return html_content[:insert_at] + base_tag + html_content[insert_at:]
    html_match = re.search(r"<html[^>]*>", html_content, flags=re.IGNORECASE)
    if html_match:
        insert_at = html_match.end()
        return html_content[:insert_at] + "<head>" + base_tag + "</head>" + html_content[insert_at:]
    return base_tag + html_content

@app.get("/view", response_class=HTMLResponse)
async def view_page(url: str = "", javascript: bool = True, timeout: int = 30):
    if not url:
        return HTMLResponse(
            content=(
                "<!doctype html>"
                "<html><head><title>Grub Crawler Viewer</title></head>"
                "<body style=\"font-family: sans-serif; margin: 2rem;\">"
                "<h2>Grub Crawler Viewer</h2>"
                "<form method=\"get\" action=\"/view\" style=\"display:flex; gap:0.5rem;\">"
                "<input name=\"url\" placeholder=\"https://news.ycombinator.com\" "
                "style=\"flex:1; padding:0.5rem;\" />"
                "<label style=\"display:flex; align-items:center; gap:0.25rem;\">JS"
                "<select name=\"javascript\">"
                "<option value=\"true\" selected>on</option>"
                "<option value=\"false\">off</option>"
                "</select></label>"
                "<button type=\"submit\">Open</button>"
                "</form>"
                "<p style=\"color:#666;\">This view fetches the page through Grub Crawler "
                "and renders it with a base URL so links work.</p>"
                "</body></html>"
            ),
            status_code=200
        )
    crawler = await get_crawler_engine()
    result = await crawler.crawl_raw_html(
        url=url,
        javascript=javascript,
        timeout=timeout
    )
    if not result.get("success"):
        return HTMLResponse(
            content=f"<pre>Failed to load {html_lib.escape(url)}: {html_lib.escape(str(result.get('error')))}</pre>",
            status_code=502
        )
    html_content = _inject_base_tag(result.get("html", ""), url)
    return HTMLResponse(content=html_content, status_code=200)

@app.get("/download")
async def download_file(
    url: str,
    use_browser: bool = False,
    javascript: bool = True,
    timeout: int = 30,
    session_id: Optional[str] = None,
    save: bool = False,
    filename: Optional[str] = None,
    download: bool = False
):
    if save and not session_id:
        raise HTTPException(status_code=400, detail="session_id required when save=true")
    crawler = await get_crawler_engine()
    result = await crawler.fetch_binary(
        url=url,
        use_browser=use_browser,
        javascript=javascript,
        timeout=timeout,
        session_id=session_id if save else None,
        filename=filename
    )
    if not result.get("success"):
        raise HTTPException(status_code=502, detail=result.get("error") or "download failed")

    content = result.get("content", b"")
    content_type = result.get("content_type") or "application/octet-stream"
    response_headers: Dict[str, str] = {}

    if download:
        safe_name = (result.get("filename") or "download").replace("\"", "")
        response_headers["Content-Disposition"] = f"attachment; filename=\"{safe_name}\""
    elif result.get("content_disposition"):
        response_headers["Content-Disposition"] = result["content_disposition"]

    if result.get("saved_path"):
        response_headers["X-Saved-Path"] = result["saved_path"]
    response_headers["Content-Length"] = str(len(content))

    return StreamingResponse(iter([content]), media_type=content_type, headers=response_headers)

# Tools listing endpoint
@app.get("/tools")
async def list_tools():
    """List all available crawling tools"""
    tool_registry = get_global_registry()
    schemas = tool_registry.get_schemas()
    return {"tools": schemas}

# TODO: Agent search/suggestion endpoint - needs documentation and design review
# This endpoint will provide AI-driven tool discovery and suggestions for AHP integration
# @app.get("/@search")
# async def agent_search(q: str = ""):
#     """
#     Agent search and tool suggestion endpoint for AHP integration
#     
#     This endpoint enables AI agents to:
#     - Discover available crawling tools dynamically
#     - Get usage suggestions based on natural language queries
#     - Find appropriate tools for specific crawling tasks
#     
#     Future features:
#     - Semantic search across tool descriptions
#     - Usage pattern recommendations
#     - Tool chaining suggestions
#     - Context-aware tool selection
#     """
#     pass

# Frontend error reporting endpoint (no auth required)
@app.post("/api/site/error")
async def site_error_report(request: Request):
    """Receive frontend error reports from the landing page."""
    try:
        body = await request.json()
    except Exception:
        return JSONResponse(content={"ok": False}, status_code=400)
    logger.warning(
        "Frontend error: %s at %s:%s — %s",
        body.get("type", "unknown"),
        body.get("filename", "?"),
        body.get("lineno", "?"),
        body.get("message", "(no message)"),
    )
    return {"ok": True}

# Embedded landing page (grub-site) — served at / and /site
if _SITE_INDEX and _SITE_INDEX.is_file():
    _site_html = _SITE_INDEX.read_text(encoding="utf-8")

    @app.get("/", response_class=HTMLResponse)
    async def serve_root():
        """Serve the landing page at root."""
        return HTMLResponse(content=_site_html, status_code=200)

    @app.get("/site", response_class=HTMLResponse)
    async def serve_site():
        """Serve the embedded Grub landing page."""
        return HTMLResponse(content=_site_html, status_code=200)

# API documentation page
_DOCS_INDEX = _SITE_DIR / "docs.html" if _SITE_DIR and _SITE_DIR.is_dir() else None
if _DOCS_INDEX and _DOCS_INDEX.is_file():
    _docs_html = _DOCS_INDEX.read_text(encoding="utf-8")

    @app.get("/docs", response_class=HTMLResponse)
    async def serve_docs():
        """Serve the API documentation page."""
        return HTMLResponse(content=_docs_html, status_code=200)

# AHP Tool Routes - Dynamic tool execution (CATCH-ALL - must be last)
@app.get("/{tool_name}")
async def execute_tool(
    tool_name: str, 
    request: Request,
    _auth: Dict = Depends(verify_internal_token)
):
    """Execute a tool via AHP protocol"""
    tool_registry = get_global_registry()
    
    try:
        tool_instance = tool_registry.get_tool(tool_name)
    except ToolError:
        raise HTTPException(status_code=404, detail=f"Tool '{tool_name}' not found")
    
    # Extract parameters from query string
    params = dict(request.query_params)
    # Remove bearer_token from params as it's handled by auth
    params.pop("bearer_token", None)
    
    try:
        # Execute the tool
        result = await tool_instance.execute(**params)
        
        if result.success:
            return {"success": True, "data": result.data, "metadata": result.metadata}
        else:
            return {"success": False, "error": result.error}
            
    except Exception as e:
        logger.error(f"Tool execution failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Tool execution failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug
    )
