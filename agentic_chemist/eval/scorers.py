"""Eval scorers: outcome, trajectory, and grounding evaluation."""

from dataclasses import dataclass, asdict
from agentic_chemist.agent.trace import Trace
from agentic_chemist.eval.tasks import EvalTask


@dataclass
class TrajectoryScore:
    """Scores for tool-call correctness and efficiency."""
    tool_precision: float  # fraction of actual calls that were expected
    tool_recall: float  # fraction of expected calls that were made
    redundant_calls: int  # calls beyond what was expected
    total_calls: int
    expected_calls: int
    missing_tools: list[str]  # expected tools not called
    extra_tools: list[str]  # unexpected tools called

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class GroundingScore:
    """Scores for whether claims are backed by tool results."""
    total_tool_results: int  # how many tool results were available
    tools_referenced_in_response: int  # how many seem referenced
    has_ungrounded_claims: bool  # heuristic flag

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class AbstentionScore:
    """Scores for whether the agent flagged OOD predictions."""
    should_warn: bool
    did_warn: bool
    correct: bool

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class EvalResult:
    task_id: str
    category: str
    trajectory: TrajectoryScore
    grounding: GroundingScore
    abstention: AbstentionScore | None = None

    def to_dict(self) -> dict:
        d = {
            "task_id": self.task_id,
            "category": self.category,
            "trajectory": self.trajectory.to_dict(),
            "grounding": self.grounding.to_dict(),
        }
        if self.abstention:
            d["abstention"] = self.abstention.to_dict()
        return d


def score_trajectory(trace: Trace, task: EvalTask) -> TrajectoryScore:
    """Score whether the agent called the right tools."""
    actual_calls = [(s.tool_name, s.inputs) for s in trace.steps]
    actual_tool_names = [name for name, _ in actual_calls]

    expected_tool_names = [tc["tool"] for tc in task.expected_tool_calls]

    # Match by (tool_name, key args) — order doesn't matter
    def _call_key(tool_name: str, args: dict) -> str:
        """Create a comparable key from tool name + core args."""
        # Only compare required args (smiles, endpoint), not optional ones
        core_keys = ["smiles", "smiles_a", "smiles_b", "endpoint"]
        core = {k: v for k, v in sorted(args.items()) if k in core_keys}
        return f"{tool_name}:{core}"

    expected_keys = {_call_key(tc["tool"], tc["args"]) for tc in task.expected_tool_calls}
    actual_keys = {_call_key(name, args) for name, args in actual_calls}

    matched = expected_keys & actual_keys
    missing = expected_keys - actual_keys
    extra = actual_keys - expected_keys

    precision = len(matched) / len(actual_keys) if actual_keys else 1.0
    recall = len(matched) / len(expected_keys) if expected_keys else 1.0

    missing_tools = [k.split(":")[0] for k in missing]
    extra_tools = [k.split(":")[0] for k in extra]
    redundant = max(0, len(actual_calls) - len(task.expected_tool_calls))

    return TrajectoryScore(
        tool_precision=round(precision, 3),
        tool_recall=round(recall, 3),
        redundant_calls=redundant,
        total_calls=len(actual_calls),
        expected_calls=len(task.expected_tool_calls),
        missing_tools=missing_tools,
        extra_tools=extra_tools,
    )


def score_grounding(trace: Trace) -> GroundingScore:
    """Score whether the final response is grounded in tool results.

    Heuristic: check if tool-specific keywords from results appear in the response.
    A more robust version would parse claims and match to trace entries.
    """
    if not trace.final_response:
        return GroundingScore(
            total_tool_results=len(trace.steps),
            tools_referenced_in_response=0,
            has_ungrounded_claims=True,
        )

    response_lower = trace.final_response.lower()
    referenced = 0

    for step in trace.steps:
        # Check if key values from this tool's output appear in the response
        output = step.output
        if isinstance(output, dict):
            found = False
            for key, val in output.items():
                if val is None:
                    continue
                val_str = str(val)
                # Check for numeric values (with some tolerance for rounding)
                if isinstance(val, (int, float)) and val_str in response_lower:
                    found = True
                    break
                # Check for string values
                if isinstance(val, str) and len(val) > 2 and val.lower() in response_lower:
                    found = True
                    break
            if found:
                referenced += 1

    return GroundingScore(
        total_tool_results=len(trace.steps),
        tools_referenced_in_response=referenced,
        has_ungrounded_claims=referenced < len(trace.steps) * 0.5,
    )


def score_abstention(trace: Trace, task: EvalTask) -> AbstentionScore | None:
    """Score whether the agent appropriately flagged OOD predictions."""
    should_warn = task.ground_truth.get("should_warn", False)
    if not should_warn and task.category != "abstention":
        return None

    if not trace.final_response:
        return AbstentionScore(should_warn=should_warn, did_warn=False, correct=not should_warn)

    response_lower = trace.final_response.lower()

    # Check for warning language
    warning_signals = [
        "out_of_domain", "out of domain", "borderline",
        "unreliable", "low confidence", "caution", "warning",
        "outside", "not reliable", "uncertain", "limited confidence",
        "applicability", "not in the training",
    ]
    did_warn = any(signal in response_lower for signal in warning_signals)

    return AbstentionScore(
        should_warn=should_warn,
        did_warn=did_warn,
        correct=should_warn == did_warn,
    )


def evaluate(trace: Trace, task: EvalTask) -> EvalResult:
    """Run all scorers on a trace."""
    return EvalResult(
        task_id=task.id,
        category=task.category,
        trajectory=score_trajectory(trace, task),
        grounding=score_grounding(trace),
        abstention=score_abstention(trace, task),
    )
