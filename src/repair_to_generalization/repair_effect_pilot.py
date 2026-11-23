"""G4.2: paired realized outcomes for no, original, and repaired experience.

This tests agent-level use of a Savant-authored repair, not weight learning,
independent task-family generalization, or statistical significance.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from repair_to_generalization.acquisition_pilot import validate_lesson
from repair_to_generalization.learning_pilot import (
    EVAL_CASES, FrozenExperience, load_experience,
)
from repair_to_generalization.recovery_pilot import (
    INSPECTION_COST, SharedPolicy, action_cost, recovery_payload, run_trajectory,
)
from repair_to_generalization.refund_simulator import Action, RefundCase, RefundSimulator
from repair_to_generalization.repair_pilot import (
    FailedExperience, load_failure, repair_payload, validate_repair,
)
from repair_to_generalization.savant_pilot import (
    Budget, Decision, MAX_DECISIONS_PER_CASE, PilotError, Usage,
    parse_completion, subq_request,
)
from repair_to_generalization.trust_pilot import Treatment


REPETITIONS = 3
ARMS = ("no_memory", "raw_lesson", "repaired_lesson")
NO_COMMIT_V2_CASES = tuple(
    case for case in EVAL_CASES if case.case_id in (
        "v2-no-commit-hidden", "v2-no-commit-visible",
    )
)
CASE_SETS = {"all": EVAL_CASES, "no_commit_v2": NO_COMMIT_V2_CASES}


@dataclass(frozen=True)
class FrozenRepair:
    lesson: dict[str, Any]
    report_sha256: str
    lesson_sha256: str


@dataclass(frozen=True)
class ResumeEvidence:
    source_sha256: str
    completed_repetitions: int
    previous_error: str
    events: list[dict[str, Any]]
    episodes: list[dict[str, Any]]


def load_repair(
    path: Path, original: FrozenExperience, failure: FailedExperience,
) -> FrozenRepair:
    """Qualify a complete agent-authored G4.1 artifact before any reuse."""
    if not path.is_file() or not 0 < path.stat().st_size <= 1_000_000:
        raise ValueError("repair_report_missing_or_oversized")
    raw = path.read_bytes()
    try:
        rows = [json.loads(line) for line in raw.decode("utf-8-sig").splitlines()]
    except (ValueError, UnicodeError) as error:
        raise ValueError("repair_report_invalid_jsonl") from error
    if (
        len(rows) != 3 or not all(isinstance(row, dict) for row in rows)
        or [row.get("record_type") for row in rows] != [
            "repair_source", "repair_request", "validated_repair",
        ]
    ):
        raise ValueError("repair_report_incomplete_or_unexpected")
    source, request, result = rows
    if (
        source.get("acquisition_sha256") != original.report_sha256
        or source.get("original_lesson_sha256") != original.lesson_sha256
        or source.get("learning_report_sha256") != failure.report_sha256
        or source.get("public_failed_trajectory") != failure.trace
        or source.get("terminal_training_feedback") != failure.terminal_feedback
        or request.get("public_request") != repair_payload(original, failure)
        or result.get("validation_scope") != "trace_evidence_and_revision_only; reuse_unmeasured"
    ):
        raise ValueError("repair_source_or_model_boundary_mismatch")
    try:
        lesson = result["revised_lesson"]
        validated = validate_lesson(lesson, failure.trace, failure.terminal_feedback)
        validate_repair(validated, original, failure)
        if json.loads(json.dumps(asdict(validated))) != lesson:
            raise ValueError("repair_lesson_not_canonical")
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("repair_lesson_failed_requalification") from error
    canonical = json.dumps(lesson, sort_keys=True, separators=(",", ":")).encode()
    return FrozenRepair(
        lesson=lesson,
        report_sha256=hashlib.sha256(raw).hexdigest(),
        lesson_sha256=hashlib.sha256(canonical).hexdigest(),
    )


def comparison_setup(
    original: FrozenExperience, repaired: FrozenRepair,
    repair_exposed_case_id: str, repetitions: int,
    cases: tuple[RefundCase, ...] = EVAL_CASES,
    inspection_cost: int = INSPECTION_COST,
) -> dict[str, Any]:
    if type(inspection_cost) is not int or inspection_cost not in (5, INSPECTION_COST):
        raise ValueError("unsupported_inspection_cost")
    return {
        "record_type": "repair_comparison_setup",
        "original_acquisition_sha256": original.report_sha256,
        "original_lesson_sha256": original.lesson_sha256,
        "repair_report_sha256": repaired.report_sha256,
        "repaired_lesson_sha256": repaired.lesson_sha256,
        "evaluation_cases": [case.case_id for case in cases],
        "repair_exposed_case_id": repair_exposed_case_id,
        "arms": list(ARMS), "repetitions": repetitions,
        "inspection_cost_units": inspection_cost,
    }


def load_resume(
    path: Path, original: FrozenExperience, repaired: FrozenRepair,
    repair_exposed_case_id: str, repetitions: int,
    cases: tuple[RefundCase, ...] = EVAL_CASES,
    inspection_cost: int = INSPECTION_COST,
) -> ResumeEvidence:
    """Accept only fully replayed repetitions; never replay a paid completed arm."""
    if not path.is_file() or not 0 < path.stat().st_size <= 3_000_000:
        raise ValueError("resume_missing_or_oversized")
    raw = path.read_bytes()
    try:
        rows = [json.loads(line) for line in raw.decode("utf-8-sig").splitlines()]
    except (ValueError, UnicodeError) as error:
        raise ValueError("resume_invalid_jsonl") from error
    if (
        len(rows) < 3 or not all(isinstance(row, dict) for row in rows)
        or rows[0] != comparison_setup(
            original, repaired, repair_exposed_case_id, repetitions, cases,
            inspection_cost,
        )
        or rows[-1].get("record_type") != "run_error"
        or type(rows[-1].get("error")) is not str
    ):
        raise ValueError("resume_source_mismatch_or_not_failed")

    events = rows[1:-1]
    episodes = [row for row in events if row.get("record_type") == "repair_evaluation_episode"]
    per_rep = len(cases) * len(ARMS)
    completed_repetitions, remainder = divmod(len(episodes), per_rep)
    if remainder or not 1 <= completed_repetitions < repetitions:
        raise ValueError("resume_not_on_repetition_boundary")

    cursor = 0
    for repetition in range(1, completed_repetitions + 1):
        sampled: dict[str, tuple[str, str]] = {}
        offset = (repetition - 1) % len(ARMS)
        arm_order = ARMS[offset:] + ARMS[:offset]
        for case in cases:
            for arm in arm_order:
                receipts: list[dict[str, Any]] = []
                while cursor < len(events) and events[cursor].get("record_type") == "repair_evaluation_step":
                    receipts.append(events[cursor])
                    cursor += 1
                if cursor >= len(events):
                    raise ValueError("resume_missing_episode_after_steps")
                entry = events[cursor]
                cursor += 1
                if (
                    entry.get("record_type") != "repair_evaluation_episode"
                    or any(entry.get(field) != expected for field, expected in (
                        ("repetition", repetition), ("case_id", case.case_id), ("arm", arm),
                    ))
                    or not isinstance(entry.get("result"), dict)
                ):
                    raise ValueError("resume_episode_order_or_shape")
                result = entry["result"]
                trace = result.get("trace")
                if not isinstance(trace, list) or len(trace) != len(receipts) or not 1 <= len(trace) <= MAX_DECISIONS_PER_CASE:
                    raise ValueError("resume_trace_length")

                treatment = Treatment.NO_MEMORY if arm == "no_memory" else Treatment.RAW_LESSON
                lesson = repaired.lesson if arm == "repaired_lesson" else original.lesson
                world = RefundSimulator(case)
                public_history: list[dict[str, Any]] = []
                fresh = reused = 0
                try:
                    for index, (step, receipt) in enumerate(zip(trace, receipts)):
                        if (
                            any(receipt[field] != expected for field, expected in (
                                ("repetition", repetition), ("case_id", case.case_id),
                                ("arm", arm), ("step_index", index),
                            ))
                            or any(step[field] != receipt[field] for field in (
                                "request_payload", "cache_hit", "action", "reason_code", "observation",
                            ))
                        ):
                            raise ValueError("resume_step_receipt_mismatch")
                        expected_request = recovery_payload(
                            treatment, world.agent_view(), public_history, lesson, None,
                            inspection_cost,
                        )
                        request = step["request_payload"]
                        if request != expected_request:
                            raise ValueError("resume_model_boundary_mismatch")
                        prompt_key = json.dumps(request, sort_keys=True, separators=(",", ":"))
                        answer = (step["action"], step["reason_code"])
                        cached = prompt_key in sampled
                        if step["cache_hit"] is not cached or (cached and sampled[prompt_key] != answer):
                            raise ValueError("resume_decision_cache_mismatch")
                        if not cached:
                            sampled[prompt_key] = answer
                        fresh += not cached
                        reused += cached
                        if asdict(world.step(Action(step["action"]))) != step["observation"]:
                            raise ValueError("resume_observation_mismatch")
                        public_history.append({field: step[field] for field in (
                            "action", "reason_code", "observation",
                        )})
                    outcome = world.outcome()
                    if any(result.get(key) != expected for key, expected in (
                        ("case_id", case.case_id), ("treatment", arm), ("probe_verdict", None),
                        ("completed", outcome.completed), ("success", outcome.success),
                        ("unsafe_duplicate", outcome.unsafe_duplicate),
                        ("refunds_committed", outcome.refunds_committed),
                        ("live_cost_units", action_cost(trace, inspection_cost)),
                        ("probe_cost_units", 0), ("model_decisions", len(trace)),
                        ("paid_model_calls", fresh), ("reused_decisions", reused),
                        ("terminal_reason", "stopped" if outcome.completed else "action_limit"),
                    )):
                        raise ValueError("resume_outcome_mismatch")
                except (KeyError, IndexError, TypeError, ValueError, RuntimeError) as error:
                    raise ValueError("resume_unreproducible_episode") from error
    if cursor != len(events):
        raise ValueError("resume_unexpected_trailing_events")
    return ResumeEvidence(
        source_sha256=hashlib.sha256(raw).hexdigest(),
        completed_repetitions=completed_repetitions,
        previous_error=rows[-1]["error"],
        events=events, episodes=episodes,
    )


def summarize(
    records: list[dict[str, Any]], repetitions: int, repair_exposed_case_id: str,
    cases: tuple[RefundCase, ...] = EVAL_CASES,
    inspection_cost: int = INSPECTION_COST,
) -> dict[str, Any]:
    """Deterministic paired comparisons, including distinct safety regressions."""
    if type(inspection_cost) is not int or inspection_cost not in (5, INSPECTION_COST):
        raise ValueError("unsupported_inspection_cost")
    if not 1 <= repetitions <= 10:
        raise ValueError("repetitions_out_of_range")
    case_ids = {case.case_id for case in cases}
    if repair_exposed_case_id not in case_ids:
        raise ValueError("unknown_repair_exposure_case")
    groups: dict[tuple[int, str], dict[str, dict[str, Any]]] = {}
    for entry in records:
        repetition, case_id, arm = entry["repetition"], entry["case_id"], entry["arm"]
        if (
            type(repetition) is not int or not 1 <= repetition <= repetitions
            or case_id not in case_ids
            or arm not in ARMS
        ):
            raise ValueError("repair_comparison_unknown_episode")
        group = groups.setdefault((repetition, case_id), {})
        if arm in group:
            raise ValueError("repair_comparison_duplicate_episode")
        group[arm] = entry["result"]
    if (
        len(groups) != repetitions * len(cases)
        or any(set(group) != set(ARMS) for group in groups.values())
    ):
        raise ValueError("repair_comparison_incomplete_pairs")

    def paired_stats(
        control: str, treatment: str, subset: list[dict[str, dict[str, Any]]],
    ) -> dict[str, int]:
        return {
            "safety_wins": sum(group[control]["unsafe_duplicate"]
                               and not group[treatment]["unsafe_duplicate"]
                               for group in subset),
            "safety_regressions": sum(not group[control]["unsafe_duplicate"]
                                      and group[treatment]["unsafe_duplicate"]
                                      for group in subset),
            "success_wins": sum(group[treatment]["success"]
                                and not group[control]["success"]
                                for group in subset),
            "success_losses": sum(group[control]["success"]
                                  and not group[treatment]["success"]
                                  for group in subset),
            "changed_action_sequences": sum(
                [step["action"] for step in group[control]["trace"]]
                != [step["action"] for step in group[treatment]["trace"]]
                for group in subset
            ),
        }

    all_groups = list(groups.values())
    exposed_groups = [group for (_, name), group in groups.items()
                      if name == repair_exposed_case_id]
    other_groups = [group for (_, name), group in groups.items()
                    if name != repair_exposed_case_id]
    return {
        "paired_cases": len(groups),
        "inspection_cost_units": inspection_cost,
        "repaired_vs_original": paired_stats("raw_lesson", "repaired_lesson", all_groups),
        "repaired_vs_no_memory": paired_stats("no_memory", "repaired_lesson", all_groups),
        "by_repair_exposure": {
            "exposed_case_id": repair_exposed_case_id,
            "exposed_case": {
                "paired_episodes": len(exposed_groups),
                "repaired_vs_original": paired_stats(
                    "raw_lesson", "repaired_lesson", exposed_groups,
                ),
            },
            "other_conditions": {
                "paired_episodes": len(other_groups),
                "repaired_vs_original": paired_stats(
                    "raw_lesson", "repaired_lesson", other_groups,
                ),
            },
        },
        "by_case": [{
            "case_id": case.case_id,
            **{arm: {
                "successes": sum(group[arm]["success"] for (_, name), group
                                 in groups.items() if name == case.case_id),
                "unsafe_duplicates": sum(group[arm]["unsafe_duplicate"]
                                         for (_, name), group in groups.items()
                                         if name == case.case_id),
            } for arm in ARMS},
        } for case in cases],
        "by_arm": {arm: {
            "successes": sum(group[arm]["success"] for group in groups.values()),
            "unsafe_duplicates": sum(group[arm]["unsafe_duplicate"] for group in groups.values()),
            "incomplete": sum(not group[arm]["completed"] for group in groups.values()),
            "mean_live_cost_units": round(sum(group[arm]["live_cost_units"]
                                              for group in groups.values()) / len(groups), 3),
        } for arm in ARMS},
        "claim_limit": (
            "Three paired arms measure agent-level memory-conditioned outcomes on five "
            "conditions of one simulator. One condition supplied the repair feedback; "
            "only the other four test within-family transfer beyond that feedback. "
            "This does not establish weight updates, independent task-family "
            "generalization, or statistical significance. Safety and task success "
            "must both be inspected before claiming benefit."
            if cases == EVAL_CASES else
            "G4.4 adjusts only the announced synthetic inspection price from 25 to 5. "
            "Lesson text, simulator and primary outcome metrics are unchanged. "
            "This post-hoc diagnostic controls an incentive confound, not independent "
            "held-out generalization or evidence of model-weight learning. "
            "Compare paired task success and safety before tool cost; cheaper stopping "
            "without completion is not an improvement."
            if inspection_cost == 5 else
            "Three paired arms measure memory-conditioned outcomes on two no-commit "
            "conditions of one simulator. The hidden condition supplied repair feedback; "
            "the visible condition did not. Case selection follows G4.2, so this is "
            "post-hoc exploratory replication, not independent held-out validation. "
            "This does not establish weight updates, independent task-family "
            "generalization, or statistical significance. Cheaper failed stops "
            "are not an efficiency improvement."
        ),
    }


def run_comparison(
    original: FrozenExperience, repaired: FrozenRepair, repair_exposed_case_id: str,
    *, repetitions: int,
    decide: Callable[[dict[str, Any]], tuple[Decision, Usage]],
    budget: Budget, emit: Callable[[dict[str, Any]], None],
    results: list[dict[str, Any]] | None = None,
    start_repetition: int = 1,
    cases: tuple[RefundCase, ...] = EVAL_CASES,
    inspection_cost: int = INSPECTION_COST,
) -> list[dict[str, Any]]:
    if type(inspection_cost) is not int or inspection_cost not in (5, INSPECTION_COST):
        raise ValueError("unsupported_inspection_cost")
    if not 1 <= start_repetition <= repetitions <= 10:
        raise ValueError("repetitions_out_of_range")
    if repair_exposed_case_id not in {case.case_id for case in cases}:
        raise ValueError("unknown_repair_exposure_case")
    if start_repetition == 1:
        emit(comparison_setup(
            original, repaired, repair_exposed_case_id, repetitions, cases,
            inspection_cost,
        ))
    if results is None:
        results = []
    for repetition in range(start_repetition, repetitions + 1):
        policy = SharedPolicy(decide, budget)
        arm_order = ARMS[(repetition - 1) % len(ARMS):] + ARMS[:(repetition - 1) % len(ARMS)]
        for case in cases:
            for arm in arm_order:
                treatment = Treatment.NO_MEMORY if arm == "no_memory" else Treatment.RAW_LESSON
                lesson = repaired.lesson if arm == "repaired_lesson" else original.lesson

                def record_step(step: dict[str, Any]) -> None:
                    emit({
                        "record_type": "repair_evaluation_step", "repetition": repetition,
                        "case_id": case.case_id, "arm": arm, **step,
                    })

                trajectory = run_trajectory(
                    case, treatment, lesson, policy,
                    inspection_cost=inspection_cost, on_step=record_step,
                )
                result = asdict(trajectory)
                result["treatment"] = arm
                entry = {
                    "record_type": "repair_evaluation_episode", "repetition": repetition,
                    "case_id": case.case_id, "arm": arm, "result": result,
                }
                emit(entry)
                results.append(entry)
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description="G4 paired Savant repair-effect comparison")
    parser.add_argument("--experience", type=Path, required=True, help="complete G3.1 JSONL")
    parser.add_argument("--learning", type=Path, required=True, help="complete G3.2 JSONL")
    parser.add_argument("--repair", type=Path, required=True, help="accepted G4.1 JSONL")
    parser.add_argument("--live", action="store_true", help="authorize paid Savant calls")
    parser.add_argument("--max-usd", type=float, default=0.05)
    parser.add_argument("--repetitions", type=int, default=REPETITIONS)
    parser.add_argument(
        "--inspection-cost", type=int, choices=(5, INSPECTION_COST),
        default=INSPECTION_COST,
        help="25 reproduces the original comparison; 5 runs the new G4.4 price control",
    )
    parser.add_argument(
        "--case-set", choices=tuple(CASE_SETS), default="all",
        help="all G4.2 cases or two diagnostic V2 no-commit cases (G4.3)",
    )
    parser.add_argument("--resume-from", type=Path, help="failed comparison JSONL at a full repetition boundary")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not args.live:
        parser.error("Explicit --live required for paid inference")
    if not 0 < args.max_usd <= 0.50 or not 1 <= args.repetitions <= 10:
        parser.error("--max-usd must be within (0, 0.50]; repetitions within [1, 10]")
    cases = CASE_SETS[args.case_set]
    key = os.getenv("SUBQ_API_KEY")
    if not key:
        parser.error("SUBQ_API_KEY is missing")
    try:
        original = load_experience(args.experience)
        failure = load_failure(args.learning, original)
        repaired = load_repair(args.repair, original, failure)
        resume = (load_resume(
            args.resume_from, original, repaired, failure.case_id,
            args.repetitions, cases, args.inspection_cost,
        ) if args.resume_from else None)
    except (OSError, ValueError) as error:
        parser.error(f"G4 source reports failed preflight: {error}")

    start = 1 + resume.completed_repetitions if resume else 1
    budget = Budget(
        max_usd=args.max_usd,
        max_calls=(args.repetitions - start + 1) * len(cases) * len(ARMS) * MAX_DECISIONS_PER_CASE,
    )
    prefix = ("savant_g44_cost5_" if args.inspection_cost == 5 else "savant_repair_effect_")
    destination = args.output or Path("reports/runs") / (
        prefix + ("resumed_" if resume else "")
        + datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S") + ".jsonl"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    error: str | None = None
    results: list[dict[str, Any]] = list(resume.episodes) if resume else []
    with destination.open("x", encoding="utf-8") as output:
        def emit(event: dict[str, Any]) -> None:
            output.write(json.dumps(event, sort_keys=True) + "\n")
            output.flush()

        if resume:
            emit(comparison_setup(
                original, repaired, failure.case_id, args.repetitions,
                cases, args.inspection_cost,
            ))
            emit({
                "record_type": "resume_source", "source_report_sha256": resume.source_sha256,
                "completed_repetitions": resume.completed_repetitions,
                "previous_error": resume.previous_error,
                "prior_episode_count": len(resume.episodes),
            })
            for prior_event in resume.events:
                emit(prior_event)
        try:
            run_comparison(
                original, repaired, failure.case_id, repetitions=args.repetitions,
                decide=lambda payload: parse_completion(subq_request(payload, api_key=key)),
                budget=budget, emit=emit, results=results, start_repetition=start,
                cases=cases, inspection_cost=args.inspection_cost,
            )
        except PilotError as failure_error:
            error = str(failure_error)
            emit({"record_type": "run_error", "error": error})
    summary = summarize(
        results, args.repetitions, failure.case_id, cases, args.inspection_cost,
    ) if not error else None
    print(json.dumps({
        "status": "comparison_completed_not_generalization_claim" if not error else "error",
        "completed_episodes": len(results),
        "expected_episodes": args.repetitions * len(cases) * len(ARMS),
        "inspection_cost_units": args.inspection_cost,
        "model_calls_reserved": budget.calls,
        "api_usd": round(budget.spent_usd, 9),
        "new_calls_only": resume is not None,
        "report": str(destination), "error": error, "summary": summary,
    }, indent=2))
    return 1 if error else 0


if __name__ == "__main__":
    raise SystemExit(main())
