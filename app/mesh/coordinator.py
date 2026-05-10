"""MeshCoordinator: lifecycle, peer management, heartbeat loop.

Owns the peer table, runs the background heartbeat task, and handles
join/leave/discovery. The coordinator is created at app startup when
MESH_ENABLED=true and stopped at shutdown.
"""

from __future__ import annotations

import asyncio
import logging
import platform
import time
import uuid
from typing import Dict, List, Optional

from app.mesh.auth import verify_mesh_token
from app.mesh.client import MeshClient
from app.mesh.models import (
    HeartbeatResponse,
    JoinResponse,
    NodeInfo,
    NodeLoad,
    PeerState,
)

logger = logging.getLogger(__name__)


class MeshCoordinator:
    """Manages the mesh lifecycle for this node."""

    def __init__(
        self,
        *,
        node_name: str = "",
        advertise_url: str = "",
        secret: str,
        seed_peers: List[str] | None = None,
        heartbeat_interval_s: float = 15.0,
        peer_timeout_s: float = 45.0,
        peer_remove_s: float = 120.0,
        tools: List[str] | None = None,
        capabilities: List[str] | None = None,
        max_concurrent_crawls: int = 5,
        require_nonce: bool = False,
    ):
        self.node_id = uuid.uuid4().hex[:12]
        self.node_name = node_name or platform.node()
        self.advertise_url = advertise_url
        self.secret = secret
        self.seed_peers = seed_peers or []
        self.heartbeat_interval_s = heartbeat_interval_s
        self.peer_timeout_s = peer_timeout_s
        self.peer_remove_s = peer_remove_s
        self.max_concurrent_crawls = max_concurrent_crawls
        self.require_nonce = require_nonce

        self.node_info = NodeInfo(
            node_id=self.node_id,
            node_name=self.node_name,
            advertise_url=self.advertise_url,
            tools=tools or [],
            capabilities=capabilities or ["crawl", "markdown", "agent"],
        )

        # Peer table: node_id -> PeerState
        self._peers: Dict[str, PeerState] = {}
        self._client = MeshClient(secret, require_nonce=require_nonce)
        self._heartbeat_task: Optional[asyncio.Task] = None
        self._running = False

        # Load counters (updated by the app)
        self.active_crawls = 0
        self.active_agent_runs = 0
        self.browser_pool_free = 0

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Start the coordinator: join seed peers and begin heartbeating."""
        self._running = True
        logger.info(
            "Mesh starting: node=%s id=%s url=%s seeds=%s",
            self.node_name, self.node_id, self.advertise_url, self.seed_peers,
        )

        # Join seed peers concurrently
        if self.seed_peers:
            results = await asyncio.gather(
                *(self._join_peer(url) for url in self.seed_peers),
                return_exceptions=True,
            )
            joined = sum(1 for r in results if r is True)
            logger.info("Joined %d/%d seed peers", joined, len(self.seed_peers))

        # Start heartbeat loop
        self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())
        logger.info("Mesh coordinator started with %d peers", len(self._peers))

    async def stop(self) -> None:
        """Stop heartbeating and notify peers we're leaving."""
        self._running = False
        if self._heartbeat_task:
            self._heartbeat_task.cancel()
            try:
                await self._heartbeat_task
            except asyncio.CancelledError:
                pass

        # Notify peers
        leave_tasks = [
            self._client.leave(peer.info.advertise_url, self.node_id)
            for peer in self._peers.values()
            if peer.healthy
        ]
        if leave_tasks:
            await asyncio.gather(*leave_tasks, return_exceptions=True)

        await self._client.close()
        logger.info("Mesh coordinator stopped")

    # ------------------------------------------------------------------
    # Peer management
    # ------------------------------------------------------------------

    def get_peers(self) -> List[PeerState]:
        """Return all known peers (healthy or not)."""
        return list(self._peers.values())

    def get_healthy_peers(self) -> List[PeerState]:
        """Return only healthy peers."""
        return [p for p in self._peers.values() if p.healthy]

    def get_peer(self, node_id: str) -> Optional[PeerState]:
        """Look up a peer by node_id."""
        return self._peers.get(node_id)

    def register_peer(self, info: NodeInfo, load: Optional[NodeLoad] = None) -> PeerState:
        """Add or update a peer in the peer table."""
        if info.node_id == self.node_id:
            return PeerState(info=info, load=load)  # don't track self

        existing = self._peers.get(info.node_id)
        if existing:
            existing.info = info
            existing.last_heartbeat_ms = int(time.time() * 1000)
            existing.missed_heartbeats = 0
            existing.healthy = True
            if load:
                existing.load = load
            return existing

        peer = PeerState(info=info, load=load)
        self._peers[info.node_id] = peer
        logger.info("Peer registered: %s (%s) at %s", info.node_name, info.node_id, info.advertise_url)
        return peer

    def remove_peer(self, node_id: str) -> None:
        """Remove a peer from the peer table."""
        removed = self._peers.pop(node_id, None)
        if removed:
            logger.info("Peer removed: %s (%s)", removed.info.node_name, node_id)

    def update_peer_load(self, node_id: str, load: NodeLoad) -> None:
        """Update load metrics for a known peer."""
        peer = self._peers.get(node_id)
        if peer:
            peer.load = load
            peer.last_heartbeat_ms = int(time.time() * 1000)
            peer.missed_heartbeats = 0
            peer.healthy = True

    def verify_token(self, token: str) -> bool:
        """Verify an incoming mesh token."""
        return verify_mesh_token(token, self.secret, require_nonce=self.require_nonce)

    def get_self_load(self) -> NodeLoad:
        """Snapshot of this node's current load."""
        return NodeLoad(
            node_id=self.node_id,
            active_crawls=self.active_crawls,
            active_agent_runs=self.active_agent_runs,
            browser_pool_free=self.browser_pool_free,
            max_concurrent_crawls=self.max_concurrent_crawls,
        )

    def get_known_peer_infos(self) -> List[NodeInfo]:
        """Return NodeInfo for all known peers (for join responses)."""
        return [p.info for p in self._peers.values()]

    # ------------------------------------------------------------------
    # Client access (for MeshDispatcher)
    # ------------------------------------------------------------------

    @property
    def client(self) -> MeshClient:
        return self._client

    # ------------------------------------------------------------------
    # Internal: join a single peer
    # ------------------------------------------------------------------

    async def _join_peer(self, peer_url: str) -> bool:
        """Attempt to join a single peer. Returns True on success."""
        resp: Optional[JoinResponse] = await self._client.join(peer_url, self.node_info)
        if resp and resp.ok:
            self.register_peer(resp.node_info)
            # Gossip: learn about peers the remote knows
            for known in resp.known_peers:
                if known.node_id != self.node_id and known.node_id not in self._peers:
                    self.register_peer(known)
                    # Don't recursively join — 1-hop gossip only
            return True
        return False

    # ------------------------------------------------------------------
    # Internal: heartbeat loop
    # ------------------------------------------------------------------

    async def _heartbeat_loop(self) -> None:
        """Periodically heartbeat all known peers and cull stale ones.

        When the peer table is empty, re-attempt joining seed peers so
        that nodes that start simultaneously can eventually discover
        each other.
        """
        while self._running:
            try:
                await asyncio.sleep(self.heartbeat_interval_s)
                if not self._running:
                    break

                # Re-try seed joins if we have no peers
                if not self._peers and self.seed_peers:
                    logger.debug("No peers known — retrying seed joins")
                    results = await asyncio.gather(
                        *(self._join_peer(url) for url in self.seed_peers),
                        return_exceptions=True,
                    )
                    joined = sum(1 for r in results if r is True)
                    if joined:
                        logger.info("Seed retry: joined %d/%d peers", joined, len(self.seed_peers))

                await self._send_heartbeats()
                self._cull_stale_peers()
            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("Heartbeat loop error")

    async def _send_heartbeats(self) -> None:
        """Send heartbeat to all known peers."""
        load = self.get_self_load()
        tasks = [
            self._client.heartbeat(peer.info.advertise_url, load)
            for peer in self._peers.values()
        ]
        if not tasks:
            return

        results = await asyncio.gather(*tasks, return_exceptions=True)
        peers = list(self._peers.values())
        for peer, result in zip(peers, results):
            if isinstance(result, HeartbeatResponse) and result.ok:
                peer.last_heartbeat_ms = int(time.time() * 1000)
                peer.missed_heartbeats = 0
                peer.healthy = True
            elif isinstance(result, Exception) or result is None:
                peer.missed_heartbeats += 1
                if peer.missed_heartbeats * self.heartbeat_interval_s >= self.peer_timeout_s:
                    peer.healthy = False
                    logger.warning(
                        "Peer %s (%s) marked unhealthy after %d missed heartbeats",
                        peer.info.node_name, peer.info.node_id, peer.missed_heartbeats,
                    )

    def _cull_stale_peers(self) -> None:
        """Remove peers that have been unresponsive beyond peer_remove_s."""
        now = int(time.time() * 1000)
        to_remove = [
            nid for nid, peer in self._peers.items()
            if (now - peer.last_heartbeat_ms) > self.peer_remove_s * 1000
        ]
        for nid in to_remove:
            self.remove_peer(nid)
