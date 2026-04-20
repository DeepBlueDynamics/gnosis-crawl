"""
FastAPI Middleware for Grub Crawler - Authentication and Content-Type enforcement.
"""
import os
import json
import logging
from fastapi import Request, HTTPException
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

from app.tools.tool_registry import get_global_registry

logger = logging.getLogger(__name__)

_ENDPOINT_HINT = {
    "endpoints": {
        "crawl":    "POST /api/crawl          {url, options?}",
        "markdown": "POST /api/markdown       {url, options?}",
        "batch":    "POST /api/batch          {urls: [...]}",
        "jobs":     "POST /api/jobs/crawl     {url, session_id?}",
        "agent":    "POST /api/agent/run      {task, config?}",
        "tools":    "GET  /tools              list available AHP tools",
        "view":     "GET  /view?url=...       render page as HTML",
        "download": "GET  /download?url=...   fetch binary/file",
        "health":   "GET  /health",
        "docs":     "GET  /docs",
    },
    "hint": "All POST endpoints accept JSON. See /docs for full schema.",
}


class NotFoundEnricherMiddleware(BaseHTTPMiddleware):
    """Rewrites bare Starlette 404s to include the endpoint hint."""

    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        if response.status_code != 404:
            return response
        # Read the body — only rewrite the exact default Starlette 404
        body = b""
        async for chunk in response.body_iterator:
            body += chunk
        if body.strip() == b'{"detail":"Not Found"}':
            enriched = {
                "error": "http_error",
                "status": 404,
                "details": {"message": "Not Found"},
                "grub": _ENDPOINT_HINT,
            }
            return JSONResponse(status_code=404, content=enriched)
        # Not a bare 404 — return original response with body re-streamed
        from starlette.responses import Response
        return Response(
            content=body,
            status_code=response.status_code,
            headers=dict(response.headers),
            media_type=response.media_type,
        )


class ContentTypeMiddleware(BaseHTTPMiddleware):
    """
    Ensures that every response has a Content-Type header.
    Defaults to application/json if no other content type is set.
    """
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        if "content-type" not in response.headers:
            response.headers["Content-Type"] = "application/json"
        return response


class AuthMiddleware(BaseHTTPMiddleware):
    """
    Authentication middleware that validates JWT tokens with auth service.
    """
    
    def __init__(self, app: ASGIApp):
        super().__init__(app)
        self.auth_client = None  # Lazy load to avoid init issues
    
    def _get_auth_client(self):
        """Lazy load auth client to avoid initialization issues"""
        if self.auth_client is None:
            try:
                from app.auth import auth_client
                self.auth_client = auth_client
            except Exception as e:
                logger.error(f"Failed to initialize auth client: {e}")
                raise HTTPException(status_code=500, detail="Authentication service unavailable")
        return self.auth_client
    
    async def dispatch(self, request: Request, call_next):
        # Check if auth is disabled globally (for Porter/Kubernetes deployments)
        from app.config import settings
        if settings.disable_auth:
            logger.debug("Auth disabled globally - skipping authentication")
            response = await call_next(request)
            return response

        # Skip auth for certain paths - CHECK THIS FIRST before any auth client access
        logger.debug(f"AuthMiddleware checking path: '{request.url.path}'")
        if request.url.path in ["/", "/health", "/tools", "/@search", "/auth", "/docs", "/redoc", "/openapi.json", "/view", "/download", "/site", "/api/site/error"]:
            logger.debug(f"Skipping auth for path: {request.url.path}")
            try:
                response = await call_next(request)
                logger.debug(f"Auth-skipped path {request.url.path} returned status: {response.status_code}")
                return response
            except Exception as e:
                logger.error(f"Error in auth-skipped path {request.url.path}: {e}")
                raise
        logger.debug(f"Requiring auth for path: {request.url.path}")
        
        # Extract bearer token from Authorization header
        auth_header = request.headers.get("Authorization", "")
        
        if not auth_header.startswith("Bearer "):
            # Check query params as fallback
            bearer_token = request.query_params.get("bearer_token")
            if not bearer_token:
                return JSONResponse(
                    status_code=401,
                    content={"error": "Missing bearer token"}
                )
        else:
            bearer_token = auth_header[7:]  # Remove "Bearer " prefix
        
        try:
            # Validate JWT token with auth service
            auth_client = self._get_auth_client()
            user_data = await auth_client.validate_token(bearer_token)
            
            # Attach auth context to request
            request.state.user = user_data
            request.state.bearer_token = bearer_token
            
            logger.debug(f"Authenticated user {user_data.get('subject', 'unknown')}")
            
        except Exception as e:
            logger.error(f"Authentication failed: {e}")
            return JSONResponse(
                status_code=401,
                content={"error": "Invalid or expired token"}
            )
        
        # Proceed with authenticated request
        response = await call_next(request)
        return response


# Create the combined auth middleware function for use in main.py
async def auth_middleware(request: Request, call_next):
    """
    Combined authentication middleware for backwards compatibility.
    Validates JWT tokens with auth service.
    """
    from app.config import settings

    # Check if auth is disabled globally (for Porter/Kubernetes deployments)
    if settings.disable_auth:
        logger.debug("Auth disabled globally - skipping authentication")
        return await call_next(request)

    from app.auth import auth_client

    # Skip auth for certain paths
    if request.url.path in ["/", "/health", "/tools", "/@search", "/auth", "/docs", "/redoc", "/openapi.json", "/site", "/api/site/error"]:
        return await call_next(request)
    
    # Extract bearer token from Authorization header
    auth_header = request.headers.get("Authorization", "")
    
    if not auth_header.startswith("Bearer "):
        # Check query params as fallback
        bearer_token = request.query_params.get("bearer_token")
        if not bearer_token:
            return JSONResponse(
                status_code=401,
                content={"error": "Missing bearer token"}
            )
    else:
        bearer_token = auth_header[7:]  # Remove "Bearer " prefix
    
    try:
        # Validate JWT token with auth service
        user_data = await auth_client.validate_token(bearer_token)
        
        # Attach auth context to request
        request.state.user = user_data
        request.state.bearer_token = bearer_token
        
        logger.debug(f"Authenticated user {user_data.get('subject', 'unknown')}")
        
    except Exception as e:
        logger.error(f"Authentication failed: {e}")
        return JSONResponse(
            status_code=401,
            content={"error": "Invalid or expired token"}
        )
    
    # Proceed with authenticated request
    response = await call_next(request)
    return response
