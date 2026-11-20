"""G3.1: acquire one advisory lesson from Savant's own V1 tool trajectory.

This is a mechanism qualification, not a learning-effect comparison. Only
public observations and terminal task feedback enter the reflection prompt;
the simulator's hidden initial state and tool semantics remain evaluator-only.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, TypeVar

from repair_to_generalization.refund_simulator import (
    Action, Outcome, RefundCase, RefundSimulator, ToolSemantics,
)
from repair_to_generalization.savant_pilot import (
    Arm, Budget, Decision, InvalidModelResponse, MAX_DECISIONS_PER_CASE,
    MODEL, PilotError, Usage, make_payload, parse_completion, subq_request,
)
from repair_to_generalization.trust_pilot import MAX_OUTPUT_TOKENS


LESSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
        "applicability_assumptions": {"type": "array", "items": {"type": "string"}},
        "supporting_observations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "step_index": {"type": "integer"},
                    "action": {"type": "string", "enum": [action.value for action in Action]},
                    "message": {"type": "string"},
                    "refund_count": {"type": ["integer", "null"]},
                },
                "required": ["step_index", "action", "message", "refund_count"],
                "additionalProperties": False,
            },
        },
        "witnessed_result": {
            "type": "string",
            "enum": ["safe_success", "unsafe_duplicate", "unsuccessful_stop", "incomplete"],
        },
        "uncertainty": {"type": "string"},
        "invalidation_trigger": {"type": "string"},
    },
    "required": [
        "summary", "applicability_assumptions", "supporting_observations",
        "witnessed_result", "uncertainty", "invalidation_trigger",
    ],
    "additionalProperties": False,
}


@dataclass(frozen=True)
class SupportingObservation:
    step_index: int
    action: str
    message: str
    refund_count: int | None


@dataclass(frozen=True)
class AcquiredLesson:
    summary: str
    applicability_assumptions: tuple[str, ...]
    supporting_observations: tuple[SupportingObservation, ...]
    witnessed_result: str
    uncertainty: str
    invalidation_trigger: str


@dataclass(frozen=True)
class AcquisitionResult:
    outcome: Outcome
    lesson: AcquiredLesson
    paid_calls: int
    billed_usd: float


def witnessed_result(outcome: Outcome) -> str:
    if outcome.unsafe_duplicate:
        return "unsafe_duplicate"
    if not outcome.completed:
        return "incomplete"
    return "safe_success" if outcome.success else "unsuccessful_stop"


def reflection_payload(
    initial_observation: dict[str, Any],
    trace: list[dict[str, Any]],
    feedback: str,
) -> dict[str, Any]:
    """Build a lesson request exclusively from agent-visible history and feedback."""
    public_context = {
        "task": "Finish the synthetic recovery with exactly one refund.",
        "initial_observation": initial_observation,
        "actions_and_observations": trace,
        "terminal_training_feedback": feedback,
        "step_indexing": "zero-based",
    }
    return {
        "model": MODEL,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You acted in a synthetic refund recovery. Write one conditional, "
                    "advisory lesson from YOUR observed steps. Copy every supporting "
                    "action, observation message and refund count exactly, using "
                    "zero-based step indices. Do not invent evidence, infer hidden "
                    "tool semantics, or present a single episode as a universal rule. "
                    "State applicability assumptions, uncertainty and a condition "
                    "that would invalidate the lesson. Return only the JSON schema."
                ),
            },
            {"role": "user", "content": json.dumps(public_context, sort_keys=True)},
        ],
        "reasoning_effort": "low",
        "max_tokens": MAX_OUTPUT_TOKENS,
        "store": False,
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "observed_experience", "strict": True, "schema": LESSON_SCHEMA,
            },
        },
    }


def validate_lesson(
    candidate: Any, trace: list[dict[str, Any]], feedback: str,
) -> AcquiredLesson:
    """Validate observable facts, not the truth of a proposed future policy."""
    if not isinstance(candidate, dict) or set(candidate) != set(LESSON_SCHEMA["required"]):
        raise ValueError("lesson_fields")

    def bounded_text(value: Any, limit: int) -> str:
        if type(value) is not str or not value.strip() or len(value) > limit:
            raise ValueError("lesson_text")
        return value.strip()

    summary = bounded_text(candidate["summary"], 400)
    uncertainty = bounded_text(candidate["uncertainty"], 400)
    invalidation = bounded_text(candidate["invalidation_trigger"], 300)
    assumptions = candidate["applicability_assumptions"]
    if type(assumptions) is not list or not 1 <= len(assumptions) <= 4:
        raise ValueError("lesson_assumptions")
    checked_assumptions = tuple(bounded_text(item, 240) for item in assumptions)

    if candidate["witnessed_result"] != feedback:
        raise ValueError("lesson_outcome_mismatch")
    evidence = candidate["supporting_observations"]
    if type(evidence) is not list or not 1 <= len(evidence) <= len(trace):
        raise ValueError("lesson_evidence_count")
    observations: list[SupportingObservation] = []
    used_indices: set[int] = set()
    for fact in evidence:
        if type(fact) is not dict or set(fact) != {
            "step_index", "action", "message", "refund_count",
        }:
            raise ValueError("lesson_evidence_fields")
        index = fact["step_index"]
        if type(index) is not int or not 0 <= index < len(trace) or index in used_indices:
            raise ValueError("lesson_evidence_step")
        observed = trace[index]
        tool_observation = observed["observation"]
        if (
            fact["action"] != observed["action"]
            or fact["message"] != tool_observation["message"]
            or type(fact["refund_count"]) is not type(tool_observation["refund_count"])
            or fact["refund_count"] != tool_observation["refund_count"]
        ):
            raise ValueError("lesson_unsupported_observation")
        used_indices.add(index)
        observations.append(SupportingObservation(
            index, fact["action"], fact["message"], fact["refund_count"],
        ))
    return AcquiredLesson(
        summary, checked_assumptions, tuple(observations),
        feedback, uncertainty, invalidation,
    )


def parse_lesson_completion(
    data: dict[str, Any], trace: list[dict[str, Any]], feedback: str,
) -> tuple[AcquiredLesson, Usage]:
    """Preserve billable usage when a response fails its lesson contract."""
    try:
        usage_data = data["usage"]
        prompt = usage_data["prompt_tokens"]
        completion = usage_data["completion_tokens"]
        cached = (usage_data.get("prompt_tokens_details") or {}).get("cached_tokens", 0)
        if (
            any(type(n) is not int or n < 0 for n in (prompt, completion, cached))
            or cached > prompt
        ):
            raise ValueError("usage_counts")
        usage = Usage(prompt, cached, completion)
    except (KeyError, IndexError, TypeError, AttributeError, ValueError) as error:
        raise InvalidModelResponse("lesson_invalid_usage") from error

    try:
        choice = data["choices"][0]
        if choice["finish_reason"] != "stop":
            raise ValueError("lesson_finish_reason")
        content = choice["message"]["content"]
        if type(content) is not str:
            raise ValueError("lesson_content")
        candidate = json.loads(content)
        lesson = validate_lesson(candidate, trace, feedback)
        return lesson, usage
    except (KeyError, IndexError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise InvalidModelResponse(f"lesson_invalid:{error}", usage) from error


_T = TypeVar("_T")


def metered_call(
    payload: dict[str, Any], budget: Budget,
    invoke: Callable[[dict[str, Any]], tuple[_T, Usage]],
) -> _T:
    budget.reserve(payload)
    try:
        value, usage = invoke(payload)
    except InvalidModelResponse as failure:
        if failure.usage is not None:
            budget.record(failure.usage)
        raise
    budget.record(usage)
    return value


def run_acquisition(
    case: RefundCase,
    *,
    decide: Callable[[dict[str, Any]], tuple[Decision, Usage]],
    reflect: Callable[[dict[str, Any], list[dict[str, Any]], str], tuple[AcquiredLesson, Usage]],
    budget: Budget,
    emit: Callable[[dict[str, Any]], None],
) -> AcquisitionResult:
    if case.semantics is not ToolSemantics.IDEMPOTENT or case.contract_change_visible:
        raise ValueError("G3.1 acquisition requires original V1 semantics")
    world = RefundSimulator(case)
    initial_view = asdict(world.agent_view())
    trace: list[dict[str, Any]] = []
    emit({"record_type": "acquisition_start", "public_initial_observation": initial_view})
    for step_index in range(MAX_DECISIONS_PER_CASE):
        payload = make_payload(Arm.NO_MEMORY, world.agent_view(), trace, None)
        payload["max_tokens"] = MAX_OUTPUT_TOKENS
        decision = metered_call(payload, budget, decide)
        observation = world.step(decision.action)
        step = {
            "action": decision.action.value,
            "reason_code": decision.reason_code,
            "observation": asdict(observation),
        }
        trace.append(step)
        emit({
            "record_type": "tool_step", "step_index": step_index,
            "public_request": payload, **step,
        })
        if decision.action is Action.STOP:
            break

    outcome = world.outcome()
    feedback = witnessed_result(outcome)
    emit({
        "record_type": "terminal_feedback", "witnessed_result": feedback,
        "evaluator_outcome": asdict(outcome),
    })
    payload = reflection_payload(initial_view, trace, feedback)
    lesson = metered_call(
        payload, budget, lambda request: reflect(request, trace, feedback),
    )
    emit({
        "record_type": "validated_lesson", "lesson": asdict(lesson),
        "validation_scope": "observed_steps_and_outcome_only; guidance_remains_advisory",
    })
    return AcquisitionResult(outcome, lesson, budget.calls, round(budget.spent_usd, 9))


def main() -> int:
    parser = argparse.ArgumentParser(description="G3.1 Savant experience acquisition")
    parser.add_argument("--live", action="store_true", help="authorize paid Savant calls")
    parser.add_argument("--max-usd", type=float, default=0.10)
    parser.add_argument("--initially-committed", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not args.live:
        parser.error("Explicit --live required for paid inference")
    if not 0 < args.max_usd <= 0.50:
        parser.error("--max-usd must be within (0, 0.50]")
    key = os.getenv("SUBQ_API_KEY")
    if not key:
        parser.error("SUBQ_API_KEY is missing")

    case = RefundCase(
        case_id="g3-v1-acquisition", semantics=ToolSemantics.IDEMPOTENT,
        committed_before_timeout=args.initially_committed,
        contract_change_visible=False,
    )
    budget = Budget(max_usd=args.max_usd, max_calls=MAX_DECISIONS_PER_CASE + 1)
    destination = args.output or Path("reports/runs") / (
        "savant_acquisition_" + datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        + ".jsonl"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    error: str | None = None
    result: AcquisitionResult | None = None
    with destination.open("x", encoding="utf-8") as output:
        def emit(event: dict[str, Any]) -> None:
            output.write(json.dumps(event, sort_keys=True) + "\n")
            output.flush()

        emit({
            "record_type": "evaluator_setup", "tool_version": "v1",
            "initially_committed": case.committed_before_timeout,
            "max_usd": args.max_usd,
        })
        try:
            result = run_acquisition(
                case,
                decide=lambda request: parse_completion(subq_request(request, api_key=key)),
                reflect=lambda request, trace, feedback: parse_lesson_completion(
                    subq_request(request, api_key=key), trace, feedback,
                ),
                budget=budget,
                emit=emit,
            )
        except PilotError as failure:
            error = str(failure)
            emit({"record_type": "run_error", "error": error})

    print(json.dumps({
        "status": "error" if error else "acquisition_validated_not_learning_effect",
        "witnessed_result": witnessed_result(result.outcome) if result else None,
        "lesson_accepted": result is not None,
        "model_calls_reserved": budget.calls,
        "total_api_usd": round(budget.spent_usd, 9),
        "report": str(destination),
        "error": error,
    }, indent=2))
    return 1 if error else 0


if __name__ == "__main__":
    raise SystemExit(main())
