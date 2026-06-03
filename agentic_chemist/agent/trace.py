"""Structured trace logger for agent runs."""

import json
import time
from dataclasses import dataclass, field, asdict
from typing import Any


@dataclass
class ToolCall:
    tool_name: str
    inputs: dict
    output: Any
    latency_ms: float
    step: int
    error: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Trace:
    """Full trace of an agent run."""
    task: str
    steps: list[ToolCall] = field(default_factory=list)
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_latency_ms: float = 0.0
    start_time: float = field(default_factory=time.time)
    end_time: float | None = None
    final_response: str | None = None

    def add_tool_call(self, tool_name: str, inputs: dict, output: Any,
                      latency_ms: float, error: str | None = None):
        """Record a tool call."""
        step = len(self.steps) + 1
        # Convert dataclass outputs to dicts for serialization
        if hasattr(output, "to_dict"):
            output = output.to_dict()
        self.steps.append(ToolCall(
            tool_name=tool_name,
            inputs=inputs,
            output=output,
            latency_ms=latency_ms,
            step=step,
            error=error,
        ))

    def add_usage(self, input_tokens: int, output_tokens: int):
        """Accumulate token usage from an API call."""
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens

    def finalize(self, final_response: str):
        """Mark the trace as complete."""
        self.end_time = time.time()
        self.total_latency_ms = (self.end_time - self.start_time) * 1000
        self.final_response = final_response

    @property
    def step_count(self) -> int:
        return len(self.steps)

    def to_dict(self) -> dict:
        return {
            "task": self.task,
            "step_count": self.step_count,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_latency_ms": round(self.total_latency_ms, 1),
            "steps": [s.to_dict() for s in self.steps],
            "final_response": self.final_response,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=str)
