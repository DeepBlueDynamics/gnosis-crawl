"""Agent engine: bounded loop runner for Mode B (internal micro-agent).

The engine implements:
  plan(ctx)  -> ask the LLM for the next action
  step(ctx)  -> execute that action (tool calls or respond)
  run_task() -> outer loop with stop-condition enforcement every iteration

Observability:
  Every run creates an EventBus + TraceCollector. Events are emitted at
  each lifecycle point so listeners (loggers, metrics, trace persistence)
  can observe the run without coupling to engine internals.
"""

from __future__ import annotations

import logging
import time
from typing import Optional, Protocol

from app.agent.types import (
    AssistantAction,
    Respond,
    RunConfig,
    RunContext,
    RunResult,
    RunState,
    StepResult,
    StepTrace,
    StopReason,
    ToolCall,
    ToolCalls,
    ToolResult,
)
from app.agent.errors import PolicyDeniedError, ProviderError, StopConditionError
from app.agent.dispatcher import Dispatcher
from app.agent.ghost import (
    detect_block,
    should_trigger_ghost,
    run_ghost_protocol,
    create_ghost_provider,
    GhostDiagnosis,
)
from app.observability.events import (
    EventBus,
    PolicyDeniedEvent,
    RunEndEvent,
    RunStartEvent,
    StepEndEvent,
    StepStartEvent,
    ToolDispatchEvent,
    ToolResultEvent,
)
from app.observability.trace import TraceCollector, RunSummary
from app.policy.gate import check_tool_call, PolicyVerdict

logger = logging.getLogger(__name__)

NO_OP_THRESHOLD = 3  # consecutive empty/no-op responses before forced stop


# ---------------------------------------------------------------------------
# Provider protocol (W6 will supply concrete implementations)
# ---------------------------------------------------------------------------

class LLMAdapter(Protocol):
    """Minimal interface that every provider adapter must satisfy."""

    async def complete(self, messages: list[dict], tools: list[dict]) -> AssistantAction:
        """Send messages + tool schemas → get back an AssistantAction."""
        ...


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class AgentEngine:
    """Bounded loop runner for Mode B."""

    def __init__(
        self,
        provider: LLMAdapter,
        dispatcher: Dispatcher,
        tool_schemas: list[dict],
    ):
        self.provider = provider
        self.dispatcher = dispatcher
        self.tool_schemas = tool_schemas

    # ------------------------------------------------------------------
    # Top-level entry point
    # ------------------------------------------------------------------

    async def run_task(
        self,
        task: str,
        config: Optional[RunConfig] = None,
    ) -> tuple[RunResult, RunSummary]:
        """Execute a bounded agent loop.

        Returns:
            (RunResult, RunSummary) — the result and the full trace summary.
        """
        config = config or RunConfig()
        ctx = RunContext(task=task, config=config)

        # --- set up observability ---
        bus = EventBus()
        collector = TraceCollector(
            run_id=ctx.run_id,
            redact=config.redact_secrets,
        )
        collector.attach(bus)

        bus.emit(RunStartEvent(run_id=ctx.run_id, task=task, config=config))

        # Seed the conversation with the task
        ctx.messages.append({"role": "user", "content": task})
        ctx.state = RunState.PLAN

        while ctx.state not in (RunState.STOP, RunState.ERROR):
            # --- stop-condition check (every iteration) ---
            stop = self._check_stop(ctx)
            if stop is not None:
                ctx.state = RunState.STOP
                result = self._finalize(ctx, stop)
                return self._emit_end(bus, collector, result, stop)

            try:
                step = await self._tick(ctx, bus)
            except StopConditionError as exc:
                ctx.state = RunState.STOP
                sr = StopReason(exc.code) if exc.code in StopReason.__members__ else StopReason.COMPLETED
                result = self._finalize(ctx, sr, error=str(exc))
                return self._emit_end(bus, collector, result, sr)
            except Exception as exc:
                logger.error("Engine tick failed: %s", exc, exc_info=True)
                ctx.state = RunState.ERROR
                result = self._finalize(ctx, StopReason.COMPLETED, error=str(exc))
                return self._emit_end(bus, collector, result, StopReason.COMPLETED, error=str(exc))

        result = self._finalize(ctx, StopReason.COMPLETED)
        return self._emit_end(bus, collector, result, StopReason.COMPLETED)

    # ------------------------------------------------------------------
    # Single tick: plan → execute → observe
    # ------------------------------------------------------------------

    async def _tick(self, ctx: RunContext, bus: EventBus) -> StepResult:
        """One full plan→execute→observe cycle."""
        ctx.step += 1
        step_start = time.monotonic()

        bus.emit(StepStartEvent(run_id=ctx.run_id, step_id=ctx.step, state=RunState.PLAN))

        # PLAN: ask the LLM
        ctx.state = RunState.PLAN
        try:
            action = await self.provider.complete(ctx.messages, self.tool_schemas)
        except Exception as exc:
            ctx.failures += 1
            raise ProviderError(str(exc))

        # RESPOND path: terminal
        if isinstance(action, Respond):
            ctx.state = RunState.RESPOND
            ctx.messages.append({"role": "assistant", "content": action.text})
            ctx.consecutive_no_ops = 0
            trace = StepTrace(run_id=ctx.run_id, step_id=ctx.step, state=RunState.RESPOND)
            ctx.trace.append(trace)
            duration = int((time.monotonic() - step_start) * 1000)
            bus.emit(StepEndEvent(run_id=ctx.run_id, step_id=ctx.step, duration_ms=duration))
            ctx.state = RunState.STOP
            return StepResult(action=action, stop_reason=StopReason.COMPLETED)

        # TOOL CALLS path
        if isinstance(action, ToolCalls):
            if not action.calls:
                ctx.consecutive_no_ops += 1
                duration = int((time.monotonic() - step_start) * 1000)
                bus.emit(StepEndEvent(run_id=ctx.run_id, step_id=ctx.step, duration_ms=duration))
                return StepResult(action=action)

            ctx.consecutive_no_ops = 0
            ctx.state = RunState.EXECUTE_TOOL

            # Record assistant message with tool calls
            ctx.messages.append({
                "role": "assistant",
                "tool_calls": [
                    {"id": c.id, "name": c.name, "args": c.args}
                    for c in action.calls
                ],
            })

            results: list[ToolResult] = []
            for call in action.calls:
                # Policy gate
                verdict: PolicyVerdict = check_tool_call(call, ctx.config)
                if not verdict.allowed:
                    logger.warning("Policy denied tool %s: %s", call.name, verdict.reason)
                    results.append(ToolResult(
                        tool_call_id=call.id,
                        ok=False,
                        error_code="policy_denied",
                        error_message=verdict.reason,
                    ))
                    ctx.trace.append(StepTrace(
                        run_id=ctx.run_id,
                        step_id=ctx.step,
                        state=RunState.EXECUTE_TOOL,
                        tool_name=call.name,
                        status="policy_denied",
                        policy_flags=verdict.flags,
                    ))
                    bus.emit(PolicyDeniedEvent(
                        run_id=ctx.run_id,
                        step_id=ctx.step,
                        tool_name=call.name,
                        reason=verdict.reason or "",
                        flags=verdict.flags or [],
                    ))
                    continue

                # Dispatch
                bus.emit(ToolDispatchEvent(
                    run_id=ctx.run_id,
                    step_id=ctx.step,
                    tool_call=call,
                ))
                result = await self.dispatcher.dispatch(call)
                results.append(result)

                bus.emit(ToolResultEvent(
                    run_id=ctx.run_id,
                    step_id=ctx.step,
                    tool_result=result,
                ))

                status = "ok" if result.ok else (result.error_code or "error")
                ctx.trace.append(StepTrace(
                    run_id=ctx.run_id,
                    step_id=ctx.step,
                    state=RunState.EXECUTE_TOOL,
                    tool_name=call.name,
                    args_hash=Dispatcher.args_hash(call.args),
                    duration_ms=result.duration_ms,
                    status=status,
                    error_code=result.error_code,
                ))

                if not result.ok:
                    ctx.failures += 1

                # Ghost Protocol fallback: if a crawl tool returned blocked
                # content, try vision-based extraction automatically.
                if result.ok and self._is_blocked_crawl_result(result):
                    ghost_result = await self._try_ghost_fallback(call, result, ctx, bus)
                    if ghost_result is not None:
                        results[-1] = ghost_result  # replace the blocked result

            # OBSERVE: feed results back into conversation
            ctx.state = RunState.OBSERVE
            for r in results:
                ctx.messages.append({
                    "role": "tool",
                    "tool_call_id": r.tool_call_id,
                    "content": r.payload if r.ok else f"ERROR [{r.error_code}]: {r.error_message}",
                })

            duration = int((time.monotonic() - step_start) * 1000)
            bus.emit(StepEndEvent(run_id=ctx.run_id, step_id=ctx.step, duration_ms=duration))
            return StepResult(action=action, tool_results=results)

        # Shouldn't reach here
        raise ProviderError(f"Unknown action type: {type(action)}")

    # ------------------------------------------------------------------
    # Ghost Protocol fallback
    # ------------------------------------------------------------------

    @staticmethod
    def _is_blocked_crawl_result(result: ToolResult) -> bool:
        """Check if a successful tool result contains blocked/anti-bot content."""
        if not result.ok or not isinstance(result.payload, dict):
            return False
        payload = result.payload
        # Check the crawl result's block indicators
        if payload.get("blocked") is True:
            return True
        if payload.get("content_quality") == "blocked":
            return True
        # Check for very thin content that might be a challenge page
        if (
            payload.get("body_word_count", 999) < 30
            and payload.get("body_char_count", 999) < 200
            and payload.get("html")
        ):
            return True
        return False

    async def _try_ghost_fallback(
        self,
        call: "ToolCall",
        original_result: ToolResult,
        ctx: RunContext,
        bus: EventBus,
    ) -> Optional[ToolResult]:
        """Run Ghost Protocol diagnosis on a blocked crawl result.

        Screenshots what the browser actually rendered, sends it to a vision
        LLM to identify the block type (Cloudflare, CAPTCHA, login wall, etc.),
        and returns a ToolResult carrying the diagnosis so the agent can decide
        what to try next (warmup, cookie injection, escalation, etc.).

        Returns a replacement ToolResult with ghost_diagnosis payload, or None
        if Ghost Protocol is disabled / no URL / screenshot failed.
        """
        from app.config import settings

        if not should_trigger_ghost(
            detect_block(
                html=str((original_result.payload or {}).get("html", "")),
                markdown=str((original_result.payload or {}).get("markdown", "")),
                status_code=(original_result.payload or {}).get("status_code"),
                body_char_count=(original_result.payload or {}).get("body_char_count", 0),
                body_word_count=(original_result.payload or {}).get("body_word_count", 0),
                content_quality=str((original_result.payload or {}).get("content_quality", "")),
            ),
            ghost_enabled=settings.agent_ghost_enabled,
            auto_trigger=settings.agent_ghost_auto_trigger,
        ):
            return None

        url = call.args.get("url") or (original_result.payload or {}).get("url")
        if not url:
            return None

        logger.info("Ghost Protocol auto-triggered for blocked URL: %s", url)

        bus.emit(ToolDispatchEvent(
            run_id=ctx.run_id,
            step_id=ctx.step,
            tool_call=ToolCall(id=f"{call.id}_ghost", name="ghost_diagnose", args={"url": url}),
        ))

        try:
            provider = create_ghost_provider()
            diagnosis: GhostDiagnosis = await run_ghost_protocol(
                url,
                provider=provider,
                max_width=settings.agent_ghost_max_image_width,
                timeout=30,
            )

            if not diagnosis.success:
                logger.warning("Ghost Protocol failed for %s: %s", url, diagnosis.error)
                return None

            # Build a replacement ToolResult that carries the diagnosis.
            # The agent sees this as a failed crawl with structured context about
            # what's blocking and what to try next.
            diag_payload = dict(original_result.payload or {})
            diag_payload["ghost_diagnosis"] = True
            diag_payload["block_type"] = diagnosis.block_type
            diag_payload["description"] = diagnosis.description
            diag_payload["recommended_action"] = diagnosis.action
            diag_payload["action_reason"] = diagnosis.action_reason
            diag_payload["ghost_provider"] = diagnosis.provider
            diag_payload["ghost_total_ms"] = diagnosis.total_ms
            diag_payload["blocked"] = True
            diag_payload["content_quality"] = "blocked"

            replacement = ToolResult(
                tool_call_id=original_result.tool_call_id,
                ok=False,  # still a failure — but now a diagnosed one
                payload=diag_payload,
                duration_ms=(original_result.duration_ms or 0) + diagnosis.total_ms,
            )

            bus.emit(ToolResultEvent(
                run_id=ctx.run_id,
                step_id=ctx.step,
                tool_result=replacement,
            ))

            ctx.trace.append(StepTrace(
                run_id=ctx.run_id,
                step_id=ctx.step,
                state=RunState.EXECUTE_TOOL,
                tool_name="ghost_diagnose",
                duration_ms=diagnosis.total_ms,
                status="ok",
            ))

            logger.info(
                "Ghost Protocol diagnosed %s: block_type=%s action=%s",
                url, diagnosis.block_type, diagnosis.action,
            )
            return replacement

        except Exception as exc:
            logger.error("Ghost Protocol error for %s: %s", url, exc, exc_info=True)
            return None

    # ------------------------------------------------------------------
    # Stop-condition checks (MASTER_PLAN §W1)
    # ------------------------------------------------------------------

    def _check_stop(self, ctx: RunContext) -> Optional[StopReason]:
        if ctx.step >= ctx.config.max_steps:
            return StopReason.MAX_STEPS
        if ctx.elapsed_ms >= ctx.config.max_wall_time_ms:
            return StopReason.MAX_WALL_TIME
        if ctx.failures >= ctx.config.max_failures:
            return StopReason.MAX_FAILURES
        if ctx.consecutive_no_ops >= NO_OP_THRESHOLD:
            return StopReason.NO_OP_LOOP
        return None

    # ------------------------------------------------------------------
    # Finalization
    # ------------------------------------------------------------------

    def _finalize(
        self,
        ctx: RunContext,
        stop_reason: StopReason,
        *,
        error: Optional[str] = None,
    ) -> RunResult:
        # Extract the last assistant text if any
        response = None
        for msg in reversed(ctx.messages):
            if msg.get("role") == "assistant" and isinstance(msg.get("content"), str):
                response = msg["content"]
                break

        return RunResult(
            run_id=ctx.run_id,
            success=stop_reason == StopReason.COMPLETED and error is None,
            stop_reason=stop_reason,
            response=response,
            artifacts=ctx.artifacts,
            trace=ctx.trace,
            steps=ctx.step,
            wall_time_ms=ctx.elapsed_ms,
            error=error,
        )

    @staticmethod
    def _emit_end(
        bus: EventBus,
        collector: TraceCollector,
        result: RunResult,
        stop_reason: StopReason,
        error: Optional[str] = None,
    ) -> tuple[RunResult, RunSummary]:
        """Emit RunEndEvent, finalize the collector, and return both."""
        bus.emit(RunEndEvent(
            run_id=result.run_id,
            success=result.success,
            stop_reason=stop_reason,
            steps=result.steps,
            wall_time_ms=result.wall_time_ms,
            error=error or result.error,
        ))
        summary = collector.finalize(result)
        return result, summary
