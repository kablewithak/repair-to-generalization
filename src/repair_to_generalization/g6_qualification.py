"""G6 development-only response qualification; never a Cycle 1 rerun.

Three fresh episodes on the previously exposed replica-absent case. The only
request change from G5.2 is max_tokens. No cache, automatic retry, or learning.
Short-context evidence is not 10K/25K/50K qualification.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from repair_to_generalization.savant_pilot import (
    Budget, InvalidModelResponse, MODEL, PilotError, Usage, subq_request,
    INPUT_PER_MILLION, OUTPUT_PER_MILLION,
)
from repair_to_generalization.transfer_pilot import (
    FrozenSources, RECONCILIATION_V2, load_sources, parse_export_response,
    payload_for, world_for,
)
from repair_to_generalization.transfer_world import ExportAction

REVISION = "g6_output_8192_dev_v1"
OUTPUT_LIMIT = 8192
INPUT_BYTES_LIMIT = 24000
FRAMING_MARGIN = 1024
ARMS = ("original_lesson", "unrelated_memory", "repaired_lesson")
CASE = next(c for c in RECONCILIATION_V2.cases
            if c.case_id == "reconcile-replica-absent")
MAX_CALLS = len(ARMS) * RECONCILIATION_V2.max_actions
MAX_RUN_USD = 0.25
PROJECT_CAP_USD = 15.0
RESERVE_USD = 1.0
WORST_CASE_USD = MAX_CALLS * (
    (INPUT_BYTES_LIMIT + FRAMING_MARGIN) * INPUT_PER_MILLION
    + OUTPUT_LIMIT * OUTPUT_PER_MILLION
) / 1_000_000

Emit = Callable[[dict[str, Any]], None]
Request = Callable[[dict[str, Any]], dict[str, Any]]


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True,
                                    separators=(",", ":")).encode()).hexdigest()


def qualification_payload(arm: str, source: FrozenSources, world: Any,
                          history: list[dict[str, Any]]) -> dict[str, Any]:
    payload = payload_for(arm, source, world, history)
    return {**payload, "max_tokens": OUTPUT_LIMIT}


def reservation(payload: dict[str, Any]) -> tuple[int, float]:
    size = len(json.dumps(payload, ensure_ascii=True).encode("utf-8"))
    if size > INPUT_BYTES_LIMIT:
        raise PilotError("g6_development_input_limit")
    if (payload.get("model") != MODEL or payload.get("max_tokens") != OUTPUT_LIMIT
            or payload.get("reasoning_effort") != "low"):
        raise PilotError("g6_protocol_mismatch")
    upper = ((size + FRAMING_MARGIN) * INPUT_PER_MILLION
             + OUTPUT_LIMIT * OUTPUT_PER_MILLION) / 1_000_000
    return size, upper


def response_metadata(data: dict[str, Any]) -> tuple[str | None, int | None]:
    choices = data.get("choices")
    choice = choices[0] if isinstance(choices, list) and choices else None
    reason = choice.get("finish_reason") if isinstance(choice, dict) else None
    usage = data.get("usage")
    details = usage.get("completion_tokens_details") if isinstance(usage, dict) else None
    tokens = details.get("reasoning_tokens") if isinstance(details, dict) else None
    return (reason if isinstance(reason, str) else None,
            tokens if type(tokens) is int and tokens >= 0 else None)


class Meter:
    """Reserve before IO; unknown billed usage consumes the conservative bound."""

    def __init__(self, cap: float, request: Request, emit: Emit) -> None:
        self.budget = Budget(max_usd=cap, max_calls=MAX_CALLS)
        self.request = request
        self.emit = emit
        self.reported_usd = 0.0
        self.unknown_reserved_usd = 0.0

    def decide(self, payload: dict[str, Any], episode: int, step: int) -> Any:
        size, upper = reservation(payload)
        if self.budget.spent_usd + upper > self.budget.max_usd:
            raise PilotError("g6_dollar_limit")
        self.budget.reserve(payload)
        call_id = self.budget.calls
        self.emit({"record_type": "call_started", "call_id": call_id,
                   "episode": episode, "step": step, "request": payload,
                   "request_sha256": digest(payload), "input_bytes": size,
                   "reserved_usd": upper, "cache_hit": False})
        started = time.monotonic()
        usage: Usage | None = None
        decision = None
        failure: PilotError | None = None
        finish_reason = None
        reasoning_tokens = None
        try:
            data = self.request(payload)
            if not isinstance(data, dict):
                raise PilotError("invalid_http_body")
            finish_reason, reasoning_tokens = response_metadata(data)
            decision, usage = parse_export_response(data)
            if data.get("model", MODEL) != MODEL:
                raise InvalidModelResponse("g6_model_mismatch", usage)
            if (usage.completion_tokens > OUTPUT_LIMIT
                    or usage.prompt_tokens > size + FRAMING_MARGIN
                    or (reasoning_tokens is not None
                        and reasoning_tokens > usage.completion_tokens)):
                raise InvalidModelResponse("g6_usage_outside_bound", usage)
        except InvalidModelResponse as error:
            usage = error.usage
            failure = error
        except PilotError as error:
            failure = error

        charged = usage.usd if usage is not None else upper
        if usage is None:
            self.unknown_reserved_usd += charged
        else:
            self.reported_usd += charged
        accounting_error = None
        try:
            if usage is None:
                self.budget.spent_usd += charged
            else:
                self.budget.record(usage)
        except PilotError as error:
            accounting_error = error
        self.emit({"record_type": "call_finished", "call_id": call_id,
                   "episode": episode, "step": step,
                   "finish_reason": finish_reason, "reasoning_tokens": reasoning_tokens,
                   "latency_ms": round((time.monotonic() - started) * 1000, 3),
                   "usage": asdict(usage) if usage is not None else None,
                   "reported_usd": usage.usd if usage is not None else None,
                   "unknown_reserved_usd": upper if usage is None else 0.0,
                   "accounted_usd": charged, "error": str(failure or accounting_error)
                   if failure or accounting_error else None,
                   "decision": asdict(decision) if decision is not None and not failure else None})
        if accounting_error:
            raise accounting_error
        if failure:
            raise failure
        return decision


def run_qualification(source: FrozenSources, meter: Meter, emit: Emit) -> dict[str, Any]:
    scored = []
    for episode, arm in enumerate(ARMS, start=1):
        world = world_for(CASE)
        history: list[dict[str, Any]] = []
        emit({"record_type": "episode_started", "episode": episode,
              "case_id": CASE.case_id, "arm": arm})
        for step in range(world.MAX_ACTIONS):
            payload = qualification_payload(arm, source, world, history)
            decision = meter.decide(payload, episode, step)
            observation = world.step(decision.action)
            event = {"action": decision.action.value, "reason_code": decision.reason_code,
                     "observation": observation}
            history.append(event)
            emit({"record_type": "public_step", "episode": episode, "step": step, **event})
            if decision.action is ExportAction.FINISH:
                break
        result = asdict(world.outcome())
        scored.append(result)
        emit({"record_type": "episode_finished", "episode": episode, "arm": arm,
              "result": result})
    return {"episodes": len(scored),
            "finished_episodes": sum(r["completed"] for r in scored),
            "safe_successes_descriptive_only": sum(r["success"] for r in scored),
            "unsafe_episodes_descriptive_only": sum(
                r["unsafe_duplicate"] or r["unauthorized_dispatch"] for r in scored),
            "protocol_pass": all(r["completed"] for r in scored),
            "context_tier": "short_development_only"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experience", type=Path, required=True)
    parser.add_argument("--learning", type=Path, required=True)
    parser.add_argument("--repair", type=Path, required=True)
    parser.add_argument("--prior-spend-usd", type=float, required=True)
    parser.add_argument("--max-usd", type=float, default=0.20)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--live", action="store_true")
    args = parser.parse_args()
    if not args.live:
        parser.error("explicit --live required")
    if (not math.isfinite(args.prior_spend_usd) or args.prior_spend_usd < 0
            or not math.isfinite(args.max_usd)
            or not WORST_CASE_USD <= args.max_usd <= MAX_RUN_USD
            or args.prior_spend_usd + args.max_usd + RESERVE_USD > PROJECT_CAP_USD):
        parser.error("invalid expenditure envelope")
    key = os.getenv("SUBQ_API_KEY")
    if not key:
        parser.error("SUBQ_API_KEY missing")
    try:
        source = load_sources(args.experience, args.learning, args.repair)
        commit = subprocess.run(["git", "rev-parse", "HEAD"], check=True,
                                capture_output=True, text=True).stdout.strip()
    except (OSError, ValueError, TypeError, subprocess.CalledProcessError) as error:
        parser.error(f"source preflight: {type(error).__name__}")
    setup = {"record_type": "qualification_setup", "gate": "G6",
             "purpose": "development response protocol only; not a transfer comparison",
             "revision": REVISION, "commit": commit, "model": MODEL,
             "context_tier": "short_development_only", "reasoning_effort": "low",
             "max_output_tokens": OUTPUT_LIMIT, "max_input_bytes": INPUT_BYTES_LIMIT,
             "framing_margin": FRAMING_MARGIN, "max_calls": MAX_CALLS,
             "max_actions_per_episode": RECONCILIATION_V2.max_actions,
             "case_id": CASE.case_id, "arms": ARMS, "fresh_calls_only": True,
             "source_hashes": {k: v for k, v in asdict(source).items()
                               if k.endswith("sha256")},
             "project_cap_usd": PROJECT_CAP_USD, "project_reserve_usd": RESERVE_USD,
             "prior_spend_usd": args.prior_spend_usd, "run_cap_usd": args.max_usd,
             "projected_worst_case_usd": WORST_CASE_USD,
             "started_utc": datetime.now(timezone.utc).isoformat()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        def emit(row: dict[str, Any]) -> None:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
            stream.flush()

        emit(setup)
        meter = Meter(args.max_usd, lambda p: subq_request(p, api_key=key), emit)
        try:
            summary = run_qualification(source, meter, emit)
        except PilotError as error:
            summary = {"protocol_pass": False, "error": str(error)}
        summary.update({"record_type": "qualification_terminal",
                        "report": str(args.output), "calls": meter.budget.calls,
                        "reported_usd": meter.reported_usd,
                        "unknown_reserved_usd": meter.unknown_reserved_usd,
                        "accounted_usd": meter.budget.spent_usd})
        emit(summary)
    print(json.dumps(summary, indent=2))
    return 0 if summary["protocol_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
