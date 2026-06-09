"""Core type primitives for the agent state machine and tool execution."""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence


# ---------------------------------------------------------------------------
# State machine
# ---------------------------------------------------------------------------

class RunState(str, Enum):
    """Agent loop states."""
    INIT = "init"
    PLAN = "plan"
    EXECUTE_TOOL = "execute_tool"
    OBSERVE = "observe"
    RESPOND = "respond"
    STOP = "stop"
    ERROR = "error"


class StopReason(str, Enum):
    """Why the agent loop terminated."""
    MAX_STEPS = "max_steps"
    MAX_WALL_TIME = "max_wall_time"
    MAX_FAILURES = "max_failures"
    NO_OP_LOOP = "no_op_loop"
    POLICY_DENIED = "policy_denied"
    COMPLETED = "completed"


# ---------------------------------------------------------------------------
# Tool call / result primitives (shared by Mode A and Mode B)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ToolCall:
    """A single tool invocation requested by the LLM."""
    id: str
    name: str
    args: Dict[str, Any]


@dataclass
class ToolResult:
    """Normalized result from executing a single tool call."""
    tool_call_id: str
    ok: bool
    payload: Any = None
    error_code: Optional[str] = None
    error_message: Optional[str] = None
    retriable: bool = False
    duration_ms: int = 0


# ---------------------------------------------------------------------------
# Assistant actions
# ---------------------------------------------------------------------------

@dataclass
class Respond:
    """The assistant wants to send a text response (terminal action)."""
    text: str


@dataclass
class ToolCalls:
    """The assistant wants to invoke one or more tools."""
    calls: List[ToolCall]


# Union of possible assistant actions
AssistantAction = Respond | ToolCalls


# ---------------------------------------------------------------------------
# Run configuration
# ---------------------------------------------------------------------------

@dataclass
class RunConfig:
    """Policy-bound limits for a single agent run."""
    max_steps: int = 12
    max_wall_time_ms: int = 90_000
    max_failures: int = 3
    allowed_tools: List[str] = field(default_factory=list)
    allowed_domains: List[str] = field(default_factory=list)
    block_private_ranges: bool = True
    redact_secrets: bool = True
    persist_raw_html: bool = False
    # When true, the engine inlines each successful tool payload into
    # RunResult.artifacts so callers don't have to make a second round-trip
    # to /api/sessions/.../file to read the raw markdown/HTML the agent saw.
    return_artifacts: bool = False
    # Bytes per inlined artifact field — keeps responses bounded when pages
    # are large. Applied to 'html' and 'markdown' payload fields; other
    # fields pass through untouched.
    artifact_max_field_bytes: int = 200_000


# ---------------------------------------------------------------------------
# Run context (mutable state carried across the loop)
# ---------------------------------------------------------------------------

@dataclass
class RunContext:
    """Mutable context threaded through the agent loop."""
    run_id: str = field(default_factory=lambda: uuid.uuid4().hex[:16])
    task: str = ""
    config: RunConfig = field(default_factory=RunConfig)
    state: RunState = RunState.INIT
    step: int = 0
    failures: int = 0
    consecutive_no_ops: int = 0
    messages: List[Dict[str, Any]] = field(default_factory=list)
    artifacts: List[Dict[str, Any]] = field(default_factory=list)
    trace: List[StepTrace] = field(default_factory=list)
    start_time_ms: int = field(default_factory=lambda: int(time.time() * 1000))

    @property
    def elapsed_ms(self) -> int:
        return int(time.time() * 1000) - self.start_time_ms


# ---------------------------------------------------------------------------
# Step / Run results
# ---------------------------------------------------------------------------

@dataclass
class StepTrace:
    """Single-step trace record."""
    run_id: str
    step_id: int
    state: RunState
    tool_name: Optional[str] = None
    args_hash: Optional[str] = None
    duration_ms: int = 0
    status: str = "ok"
    error_code: Optional[str] = None
    artifacts: List[str] = field(default_factory=list)
    policy_flags: List[str] = field(default_factory=list)


@dataclass
class StepResult:
    """Outcome of a single step."""
    action: AssistantAction
    tool_results: List[ToolResult] = field(default_factory=list)
    stop_reason: Optional[StopReason] = None


@dataclass
class RunResult:
    """Final outcome of a complete agent run."""
    run_id: str
    success: bool
    stop_reason: StopReason
    response: Optional[str] = None
    artifacts: List[Dict[str, Any]] = field(default_factory=list)
    trace: List[StepTrace] = field(default_factory=list)
    steps: int = 0
    wall_time_ms: int = 0
    error: Optional[str] = None
