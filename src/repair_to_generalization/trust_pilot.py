"""Savant pilot: when can an agent trust experience and its verifier?

Six matched refund cases cross cheap/expensive authoritative inspection with
five evidence treatments. The *evaluator* sets sandbox fidelity; the agent
receives only its result and an explicit warning that parity is unverified.

This is a controlled pilot, NOT independent task-family replication and NOT a
deployable method for knowing whether an unseen sandbox is faithful.

Run with PYTHONPATH=src:
    python -m repair_to_generalization.trust_pilot --live --max-usd 0.10
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable

from repair_to_generalization.comparison import benchmark_cases
from repair_to_generalization.cost_pilot import CostEpisode, run_cost_episode
from repair_to_generalization.refund_simulator import RefundCase, ToolSemantics
from repair_to_generalization.savant_pilot import (
    Arm, Budget, Decision, PilotError, Usage, parse_completion, subq_request,
)


INSPECTION_COSTS = (1, 25)
MAX_OUTPUT_TOKENS = 8192  # Includes Savant's private reasoning; no paid retries.


class Treatment(str, Enum):
    NO_MEMORY = "no_memory"
    RAW_LESSON = "raw_lesson"
    FAITHFUL_PROBE = "faithful_probe"
    LEGACY_PROBE = "legacy_probe"
    NO_PROBE = "no_probe"


def treatment_settings(case: RefundCase, treatment: Treatment) -> tuple[Arm, ToolSemantics | None, bool]:
    """Private evaluator configuration; the treatment name never enters prompts."""
    if treatment is Treatment.NO_MEMORY:
        return Arm.NO_MEMORY, None, True
    if treatment is Treatment.RAW_LESSON:
        return Arm.RAW_LESSON, None, True
    if treatment is Treatment.FAITHFUL_PROBE:
        return Arm.PROBE_EVIDENCE, case.semantics, True
    if treatment is Treatment.LEGACY_PROBE:
        return Arm.PROBE_EVIDENCE, ToolSemantics.IDEMPOTENT, True
    if treatment is Treatment.NO_PROBE:
        return Arm.PROBE_EVIDENCE, None, False
    raise ValueError("unsupported treatment")


@dataclass
class TrustEpisode:
    treatment: str
    environment_version: str
    pilot: CostEpisode


def run_trust_episode(
    case: RefundCase,
    treatment: Treatment,
    inspection_cost: int,
    *,
    decide: Callable[[dict[str, Any]], tuple[Decision, Usage]],
    budget: Budget,
) -> TrustEpisode:
    if inspection_cost not in INSPECTION_COSTS:
        raise ValueError("unsupported inspection cost")
    arm, sandbox_semantics, available = treatment_settings(case, treatment)
    result = run_cost_episode(
        case, arm, inspection_cost,
        decide=decide,
        budget=budget,
        probe_semantics=sandbox_semantics,
        probe_available=available,
        max_output_tokens=MAX_OUTPUT_TOKENS,
    )
    return TrustEpisode(
        treatment=treatment.value,
        environment_version=(
            "original_idempotent" if case.semantics is ToolSemantics.IDEMPOTENT
            else "changed_non_idempotent"
        ),
        pilot=result,
    )


def summarize(results: list[TrustEpisode]) -> list[dict[str, Any]]:
    """Report utility and safety separately, across all six matched cases."""
    summaries: list[dict[str, Any]] = []
    for inspection_cost in INSPECTION_COSTS:
        for treatment in Treatment:
            group = [
                row.pilot for row in results
                if row.pilot.inspection_cost == inspection_cost
                and row.treatment == treatment.value
                and row.pilot.error is None
            ]
            if not group:
                continue
            n = len(group)
            summaries.append({
                "inspect_cost": inspection_cost,
                "treatment": treatment.value,
                "episodes": n,
                "successes": sum(row.success for row in group),
                "unsafe_duplicates": sum(row.unsafe_duplicate for row in group),
                "mean_live_cost_units": round(sum(row.live_cost_units for row in group) / n, 3),
                "mean_probe_cost_units": round(sum(row.probe_cost_units for row in group) / n, 3),
                "mean_net_points": round(sum(row.net_points for row in group) / n, 3),
                "api_usd": round(sum(row.billed_usd for row in group), 6),
            })
    return summaries


def main() -> int:
    parser = argparse.ArgumentParser(description="Savant verifier-trust pilot")
    parser.add_argument("--live", action="store_true", help="authorize paid inference")
    parser.add_argument("--max-usd", type=float, default=0.10)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not args.live:
        parser.error("Explicit --live required for paid inference")
    if not 0 < args.max_usd <= 0.50:
        parser.error("--max-usd must be within (0, 0.50]")
    key = os.getenv("SUBQ_API_KEY")
    if not key:
        parser.error("SUBQ_API_KEY is missing")

    budget = Budget(max_usd=args.max_usd, max_calls=240)
    destination = args.output or Path("reports/runs") / (
        "savant_trust_pilot_" + datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S") + ".jsonl"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    results: list[TrustEpisode] = []
    error: str | None = None
    with destination.open("x", encoding="utf-8") as output:
        for case in benchmark_cases():
            for inspection_cost in INSPECTION_COSTS:
                for treatment in Treatment:
                    try:
                        result = run_trust_episode(
                            case, treatment, inspection_cost,
                            decide=lambda payload: parse_completion(subq_request(payload, api_key=key)),
                            budget=budget,
                        )
                    except PilotError as failure:
                        error = str(failure)
                        break
                    results.append(result)
                    output.write(json.dumps(asdict(result), sort_keys=True) + "\n")
                    output.flush()
                    if result.pilot.error:
                        error = result.pilot.error
                        break
                if error:
                    break
            if error:
                break

    valid = [row for row in results if row.pilot.error is None]
    print(json.dumps({
        "scope": "paired verifier-trust pilot; six case families; no independent replications",
        "completed_episodes": len(valid),
        "failed_episode_attempts": len(results) - len(valid),
        "expected_episodes": len(benchmark_cases()) * len(INSPECTION_COSTS) * len(Treatment),
        "model_calls_reserved": budget.calls,
        "total_api_usd": round(budget.spent_usd, 6),
        "report": str(destination),
        "error": error,
        "summary": summarize(valid),
    }, indent=2))
    return 1 if error else 0


if __name__ == "__main__":
    raise SystemExit(main())
