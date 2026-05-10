"""HTTP client for mesh peer communication.

All inter-node RPCs go through this client: join, heartbeat, leave,
and tool execution. Uses httpx with configurable timeouts.
"""

from __future__ import annotations

import logging
from typing import Optional

import httpx

from app.mesh.auth import sign_mesh_token
from app.mesh.models import (
    HeartbeatRequest,
    HeartbeatResponse,
    JoinRequest,
    JoinResponse,
    LeaveRequest,
    LeaveResponse,
    MeshContext,
    MeshToolCall,
    MeshToolRequest,
    MeshToolResponse,
    NodeInfo,
    NodeLoad,
)

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT_S = 10
EXECUTE_TIMEOUT_S = 35


class MeshClient:
    """HTTP client for talking to mesh peers."""

    def __init__(
        self,
        secret: str,
        timeout_s: float = DEFAULT_TIMEOUT_S,
        *,
        require_nonce: bool = False,
    ):
        self.secret = secret
        self.timeout_s = timeout_s
        self._require_nonce = require_nonce
        self._client: Optional[httpx.AsyncClient] = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(self.timeout_s, connect=5.0),
                follow_redirects=True,
            )
        return self._client

    async def close(self) -> None:
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None

    # ------------------------------------------------------------------
    # Join
    # ------------------------------------------------------------------

    async def join(self, peer_url: str, self_info: NodeInfo) -> Optional[JoinResponse]:
        """Send a join request to a peer. Returns None on failure."""
        try:
            client = await self._get_client()
            body = JoinRequest(
                node_info=self_info,
                mesh_token=sign_mesh_token(self.secret, require_nonce=self._require_nonce),
            )
            resp = await client.post(
                f"{peer_url.rstrip('/')}/mesh/join",
                json=body.model_dump(),
            )
            if resp.status_code == 200:
                return JoinResponse.model_validate(resp.json())
            logger.warning("Join to %s failed: %d %s", peer_url, resp.status_code, resp.text[:200])
            return None
        except Exception as exc:
            logger.warning("Join to %s error: %s", peer_url, exc)
            return None

    # ------------------------------------------------------------------
    # Heartbeat
    # ------------------------------------------------------------------

    async def heartbeat(self, peer_url: str, load: NodeLoad) -> Optional[HeartbeatResponse]:
        """Send heartbeat to a peer. Returns None on failure."""
        try:
            client = await self._get_client()
            body = HeartbeatRequest(
                node_load=load,
                mesh_token=sign_mesh_token(self.secret, require_nonce=self._require_nonce),
            )
            resp = await client.post(
                f"{peer_url.rstrip('/')}/mesh/heartbeat",
                json=body.model_dump(),
            )
            if resp.status_code == 200:
                return HeartbeatResponse.model_validate(resp.json())
            return None
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Leave
    # ------------------------------------------------------------------

    async def leave(self, peer_url: str, node_id: str) -> bool:
        """Notify a peer that we're leaving. Returns success."""
        try:
            client = await self._get_client()
            body = LeaveRequest(
                node_id=node_id,
                mesh_token=sign_mesh_token(self.secret, require_nonce=self._require_nonce),
            )
            resp = await client.post(
                f"{peer_url.rstrip('/')}/mesh/leave",
                json=body.model_dump(),
            )
            return resp.status_code == 200
        except Exception as exc:
            logger.debug("Leave notification to %s failed: %s", peer_url, exc)
            return False

    # ------------------------------------------------------------------
    # Remote tool execution
    # ------------------------------------------------------------------

    async def execute_tool(
        self,
        peer_url: str,
        tool_call: MeshToolCall,
        context: Optional[MeshContext] = None,
        timeout_s: float = EXECUTE_TIMEOUT_S,
    ) -> Optional[MeshToolResponse]:
        """Execute a tool call on a remote peer. Returns None on failure."""
        try:
            client = await self._get_client()
            body = MeshToolRequest(
                tool_call=tool_call,
                context=context or MeshContext(),
                mesh_token=sign_mesh_token(self.secret, require_nonce=self._require_nonce),
                hop_count=0,  # origin request; receiver enforces 1-hop max
            )
            resp = await client.post(
                f"{peer_url.rstrip('/')}/mesh/execute",
                json=body.model_dump(),
                timeout=timeout_s,
            )
            if resp.status_code == 200:
                return MeshToolResponse.model_validate(resp.json())
            logger.warning(
                "Remote execute on %s failed: %d %s",
                peer_url, resp.status_code, resp.text[:200],
            )
            return None
        except Exception as exc:
            logger.warning("Remote execute on %s error: %s", peer_url, exc)
            return None
