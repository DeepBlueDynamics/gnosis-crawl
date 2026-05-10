"""Agent API routes for Grub Crawler Mode B.

POST /api/agent/run      — submit a task to the internal agent loop
GET  /api/agent/status/{run_id} — check status of a run (via persisted trace)
POST /api/agent/ghost    — Ghost Protocol: screenshot + vision extract

Returns 503 when AGENT_ENABLED is false.
"""

import logging
import uuid
from typing import Optional

from fastapi import APIRouter, Depends, Header, HTTPException, Request

from app.routes import get_optional_user_email
from app.config import settings
from app.models import (
    AgentRunRequest, AgentRunResponse, AgentStatusResponse, AgentTraceEntry,
    GhostExtractRequest, GhostExtractResponse,
)
from app.storage import CrawlStorageService
from app.proxy import resolve_proxy

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/agent", tags=["agent"])


def _require_agent_enabled():
    """Dependency that gates all agent routes."""
    if not settings.agent_enabled:
        raise HTTPException(
            status_code=503,
            detail="Agent is disabled. Set AGENT_ENABLED=true to enable Mode B.",
        )


# ---------------------------------------------------------------------------
# POST /api/agent/run — synchronous for now (< 90s wall time by default)
# ---------------------------------------------------------------------------

@router.post("/run", response_model=AgentRunResponse, dependencies=[Depends(_require_agent_enabled)])
async def agent_run(
    request: AgentRunRequest,
    raw_request: Request,
    user_email: Optional[str] = Depends(get_optional_user_email),
):
    """Submit a task to the internal agent loop and wait for the result."""
    from app.agent.types import RunConfig
    from app.agent.dispatcher import Dispatcher
    from app.agent.engine import AgentEngine
    from app.agent.providers import create_provider_from_config
    from app.observability.trace import persist_trace
    from app.tools.tool_registry import get_global_registry

    session_id = request.session_id or uuid.uuid4().hex[:16]

    # Build run config
    run_config = settings.build_run_config()
    run_config.max_steps = request.max_steps
    run_config.max_wall_time_ms = request.max_wall_time_ms
    if request.allowed_domains:
        run_config.allowed_domains = request.allowed_domains
    if request.allowed_tools:
        run_config.allowed_tools = request.allowed_tools

    # Wire engine
    registry = get_global_registry()
    provider = create_provider_from_config()
    local_dispatcher = Dispatcher(registry, run_config)

    # Wrap with MeshDispatcher when mesh is active
    coordinator = getattr(raw_request.app.state, "mesh_coordinator", None)
    if coordinator and settings.mesh_enabled:
        from app.mesh.dispatcher import MeshDispatcher
        dispatcher = MeshDispatcher(
            local_dispatcher,
            coordinator,
            prefer_local=settings.mesh_prefer_local,
            customer_id=request.customer_id,
            session_id=session_id,
            run_id=None,  # run_id is created inside engine
        )
    else:
        dispatcher = local_dispatcher

    tool_schemas = registry.get_schemas()
    engine = AgentEngine(provider, dispatcher, tool_schemas)

    # Execute
    result, summary = await engine.run_task(request.task, run_config)

    # Persist trace (best-effort)
    try:
        await persist_trace(summary, session_id, user_email=user_email)
    except Exception as exc:
        logger.error("Failed to persist trace for run %s: %s", result.run_id, exc)

    # Build response
    trace_entries = [
        AgentTraceEntry(
            event=t.get("event"),
            step_id=t.get("step_id"),
            tool_name=t.get("tool_name"),
            duration_ms=t.get("duration_ms"),
            status=t.get("status"),
            error_code=t.get("error_code"),
            timestamp_ms=t.get("timestamp_ms"),
        )
        for t in summary.trace
    ]

    return AgentRunResponse(
        success=result.success,
        run_id=result.run_id,
        stop_reason=result.stop_reason.value,
        response=result.response,
        steps=result.steps,
        wall_time_ms=result.wall_time_ms,
        trace=trace_entries,
        artifacts=result.artifacts,
        error=result.error,
    )


# ---------------------------------------------------------------------------
# GET /api/agent/status/{run_id}
# ---------------------------------------------------------------------------

@router.get("/status/{run_id}", response_model=AgentStatusResponse, dependencies=[Depends(_require_agent_enabled)])
async def agent_status(
    run_id: str,
    session_id: Optional[str] = None,
    user_email: Optional[str] = Depends(get_optional_user_email),
):
    """Check the status of a completed agent run by loading its persisted trace."""
    from app.observability.trace import load_trace

    if not session_id:
        # Without session_id we can't locate the trace in storage.
        # Return a helpful error.
        raise HTTPException(
            status_code=400,
            detail="session_id query parameter is required to look up a trace.",
        )

    summary = await load_trace(run_id, session_id, user_email=user_email)
    if summary is None:
        return AgentStatusResponse(run_id=run_id, found=False)

    return AgentStatusResponse(
        run_id=summary.run_id,
        found=True,
        success=summary.success,
        stop_reason=summary.stop_reason,
        response=summary.response,
        steps=summary.steps,
        wall_time_ms=summary.wall_time_ms,
        error=summary.error,
    )


# ---------------------------------------------------------------------------
# POST /api/agent/ghost — Ghost Protocol: screenshot + diagnose block
# ---------------------------------------------------------------------------

def _require_ghost_enabled():
    """Dependency that gates the ghost endpoint."""
    if not settings.agent_ghost_enabled:
        raise HTTPException(
            status_code=503,
            detail="Ghost Protocol is disabled. Set AGENT_GHOST_ENABLED=true to enable.",
        )


@router.post("/ghost", response_model=GhostExtractResponse, dependencies=[Depends(_require_ghost_enabled)])
async def agent_ghost(
    request: GhostExtractRequest,
    x_client_timeout: Optional[str] = Header(None, alias="X-Client-Timeout"),
):
    """Ghost Protocol: screenshot a blocked URL and diagnose what's blocking it.

    Takes a screenshot of what the browser actually rendered (the block page,
    challenge, cookie wall, etc.) and uses vision AI to identify the block
    type and recommend the next action to take.
    """
    from app.agent.ghost import run_ghost_protocol, create_ghost_provider

    try:
        provider = create_ghost_provider()
    except Exception as exc:
        logger.error("Failed to create ghost provider: %s", exc)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to initialize vision provider: {exc}",
        )

    proxy = resolve_proxy(getattr(request, 'proxy', None))

    effective_timeout = request.timeout
    if x_client_timeout and x_client_timeout.isdigit():
        client_budget = int(x_client_timeout)
        if client_budget > 0 and (effective_timeout is None or client_budget < effective_timeout):
            effective_timeout = client_budget
            logger.info("Ghost timeout capped to client budget: %ds", client_budget)

    result = await run_ghost_protocol(
        request.url,
        provider=provider,
        max_width=settings.agent_ghost_max_image_width,
        timeout=effective_timeout,
        proxy=proxy,
    )

    return GhostExtractResponse(
        success=result.success,
        url=result.url,
        block_type=result.block_type,
        description=result.description,
        action=result.action,
        action_reason=result.action_reason,
        capture_ms=result.capture_ms,
        diagnosis_ms=result.diagnosis_ms,
        total_ms=result.total_ms,
        provider=result.provider,
        error=result.error,
    )
