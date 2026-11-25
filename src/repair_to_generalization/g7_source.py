"""Acquire a separate genuine Study I ledger and qualify its 25K context.

All acquisition actions are live model decisions. Calibration probes receive
the same frozen ledger, and cannot modify it. Neither development histories
nor main/confirmation task feedback enter this source. No scored comparison
is authorized by a successful source run.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from .experience_ledger import Access, EpisodeRecord, Ledger, canonical, execute, hash_value, replay
from .g6_live import ACTION_ONLY_REVISION, parse_linked, payload_for
from .g6_qualification import (
    Meter, OUTPUT_LIMIT, FRAMING_MARGIN, PROJECT_CAP_USD, RESERVE_USD, digest, reservation,
)
from .linked_world import CONTRACT, MAX_ACTIONS, Move, Scenario
from .savant_pilot import MODEL, INPUT_PER_MILLION, OUTPUT_PER_MILLION, PilotError, Usage, subq_request
from .study_design import (
    CONTEXT_TARGET, MAX_ACQUISITION, MAX_INPUT_BYTES, MIN_ACQUISITION, STOP_RULE,
    acquisition_cases, combined_cases, split_manifest,
)

REVISION = "study_i_source_25k_v1"
PROBES = 3
MAX_CALLS = (MAX_ACQUISITION + PROBES) * MAX_ACTIONS
MAX_RUN_USD = 2.0
WORST_CASE_USD = MAX_CALLS * (
    (MAX_INPUT_BYTES + FRAMING_MARGIN) * INPUT_PER_MILLION
    + OUTPUT_LIMIT * OUTPUT_PER_MILLION
) / 1_000_000


def setup_for(commit: str, prior: float, cap: float) -> dict[str, Any]:
    design = split_manifest()
    request = payload_for({}, ACTION_ONLY_REVISION)
    return {
        "record_type": "study_source_setup", "revision": REVISION, "gate": "G6",
        "purpose": "separate fixed-history acquisition and 25K protocol qualification",
        "commit": commit, "model": MODEL, "protocol": ACTION_ONLY_REVISION,
        "reasoning_effort": "low", "max_output_tokens": OUTPUT_LIMIT,
        "max_input_bytes": MAX_INPUT_BYTES, "framing_margin": FRAMING_MARGIN,
        "max_calls": MAX_CALLS, "max_actions": MAX_ACTIONS,
        "context_target": CONTEXT_TARGET, "fresh_calls_only": True,
        "split_manifest": design, "split_manifest_sha256": hash_value(design),
        "prompt_sha256": hash_value({"system": request["messages"][0]["content"],
                                     "schema": request["response_format"], "contract": CONTRACT}),
        "authorized_cap_usd": cap, "prior_spend_usd": prior, "reserve_usd": RESERVE_USD,
        "project_cap_usd": PROJECT_CAP_USD, "projected_uncapped_worst_case_usd": WORST_CASE_USD,
        "stop_rule": STOP_RULE,
    }


def should_stop_acquisition(count: int, first_input_tokens: int) -> bool:
    return count >= MIN_ACQUISITION and first_input_tokens >= CONTEXT_TARGET


class SourceRun:
    def __init__(self, meter: Meter, emit: Callable[[dict], None]):
        self.meter, self.emit = meter, emit
        self.ledger = Ledger()
        self.cases: list[Scenario] = []
        self.records: list[EpisodeRecord] = []
        self.context: list[dict[str, Any]] = []

    def episode(self, case: Scenario, learning: bool):
        episode = len(self.records) + 1
        prior = self.ledger
        tokens: list[int] = []
        step = 0
        self.emit({"record_type": "study_source_episode_started", "episode": episode,
                   "case_id": case.case_id, "learning": learning,
                   "source_ledger_head": prior.head, "source_event_count": len(prior.events)})

        def decide(public):
            nonlocal step
            decision = self.meter.decide(payload_for(public, ACTION_ONLY_REVISION), episode, step)
            if self.meter.last_usage is None:
                raise PilotError("study_source_missing_usage")
            tokens.append(self.meter.last_usage.prompt_tokens)
            step += 1
            return decision.action

        def observe(public):
            self.emit({"record_type": "study_source_public_step", "episode": episode,
                       "step": step - 1, **public})

        self.ledger, record = execute(case, prior, episode, learning, Access.FULL, decide,
                                      observe=observe)
        self.records.append(record)
        self.cases.append(case)
        outcome = json.loads(record.outcome_json)
        context = {"episode": episode, "learning": learning, "first_input_tokens": tokens[0],
                   "max_input_tokens": max(tokens), "completed": outcome["completed"],
                   "safe_success": outcome["safe_success"], "unsafe": outcome["unsafe"]}
        self.context.append(context)
        self.emit({"record_type": "study_source_episode_finished", "episode": episode,
                   "case_id": case.case_id, "record": asdict(record), "outcome": outcome,
                   "context": context,
                   "new_public_events": [asdict(e) for e in self.ledger.events[len(prior.events):]]})
        return context

    def run(self):
        for index, case in enumerate(acquisition_cases(), 1):
            result = self.episode(case, True)
            if should_stop_acquisition(index, result["first_input_tokens"]):
                break
        for case in combined_cases("calibration"):
            self.episode(case, False)

    def verify(self):
        replay(tuple(self.cases), tuple(self.records), self.ledger,
               tuple(row["learning"] for row in self.context))

    def summary(self):
        learned = [row for row in self.context if row["learning"]]
        probes = [row for row in self.context if not row["learning"]]
        qualifying = sum(row["completed"] and row["first_input_tokens"] >= CONTEXT_TARGET
                         for row in probes)
        return {
            "acquisition_episodes": len(learned),
            "acquisition_safe_successes": sum(row["safe_success"] for row in learned),
            "acquisition_unsafe": sum(row["unsafe"] for row in learned),
            "probe_episodes": len(probes), "qualifying_25k_episodes": qualifying,
            "tier_qualified": len(probes) == PROBES and qualifying == PROBES,
            "probe_safe_successes": sum(row["safe_success"] for row in probes),
            "probe_unsafe": sum(row["unsafe"] for row in probes),
            "probe_context": probes, "public_events": len(self.ledger.events),
            "ledger_head": self.ledger.head,
            "max_input_tokens": max((row["max_input_tokens"] for row in self.context), default=0),
            "claim_limit": "25K protocol qualification; not G7 learning-effect evidence",
        }


@dataclass(frozen=True)
class FrozenSource:
    sha256: str
    ledger: Ledger
    cases: tuple[Scenario, ...]
    records: tuple[EpisodeRecord, ...]
    accounted_usd: float
    prior_usd: float


def require(condition: bool, reason: str):
    if not condition:
        raise ValueError(reason)


def load_source(path: Path) -> FrozenSource:
    """Verify the exact source report, not an imported or fabricated ledger.

    Re-run the instrument using the recorded validated decisions and check
    every request, public transition, ledger head, split and cost. This proves
    report consistency, not cryptographic attestation of the remote provider.
    """
    raw = path.read_bytes()
    rows = [json.loads(line) for line in raw.splitlines()]
    require(len(rows) >= 2, "study_source_missing_records")
    setup, terminal = rows[0], rows[-1]
    expected = setup_for(setup["commit"], setup["prior_spend_usd"], setup["authorized_cap_usd"])
    require(math.isfinite(setup["prior_spend_usd"]) and setup["prior_spend_usd"] >= 0
            and math.isfinite(setup["authorized_cap_usd"])
            and 0 < setup["authorized_cap_usd"] <= MAX_RUN_USD
            and setup["prior_spend_usd"] + setup["authorized_cap_usd"] + RESERVE_USD <= PROJECT_CAP_USD,
            "study_source_invalid_budget")
    require(all(setup.get(k) == json.loads(canonical(v)) for k, v in expected.items()),
            "study_source_manifest_mismatch")
    require(terminal.get("record_type") == "study_source_terminal"
            and terminal.get("error") is None and terminal.get("replay_valid") is True
            and terminal.get("tier_qualified") is True, "study_source_not_qualified")
    entries = iter(rows[1:-1])
    ledger = Ledger()
    records: list[EpisodeRecord] = []
    used_cases: list[Scenario] = []
    context_rows: list[dict] = []
    calls = 0
    spend = 0.0

    def take(kind):
        row = next(entries, None)
        require(row is not None and row.get("record_type") == kind,
                "study_source_order_mismatch:" + kind)
        return row

    def play(case, learning):
        nonlocal ledger, calls, spend
        episode = len(records) + 1
        prior = ledger
        tokens: list[int] = []
        ordinal = 0
        start = take("study_source_episode_started")
        require(start == {"record_type": "study_source_episode_started", "episode": episode,
                          "case_id": case.case_id, "learning": learning,
                          "source_ledger_head": prior.head, "source_event_count": len(prior.events)},
                "study_source_start_mismatch")

        def decide(public):
            nonlocal calls, spend
            request = payload_for(public, ACTION_ONLY_REVISION)
            sent = take("call_started")
            done = take("call_finished")
            calls += 1
            size, reserve = reservation(request, MAX_INPUT_BYTES)
            require(calls <= MAX_CALLS and spend + reserve <= setup["authorized_cap_usd"] + 1e-12,
                    "study_source_budget_mismatch")
            require(sent.get("request") == request and sent.get("request_sha256") == digest(request)
                    and sent.get("cache_hit") is False and sent.get("input_bytes") == size
                    and math.isclose(sent["reserved_usd"], reserve, abs_tol=1e-12),
                    "study_source_request_mismatch")
            for row in (sent, done):
                require(row.get("call_id") == calls and row.get("episode") == episode
                        and row.get("step") == ordinal, "study_source_call_mismatch")
            require(done.get("error") is None and done.get("finish_reason") == "stop"
                    and done.get("unknown_reserved_usd") == 0, "study_source_invalid_response")
            usage = Usage(**done["usage"])
            require(all(type(v) is int and v >= 0 for v in asdict(usage).values())
                    and usage.cached_tokens <= usage.prompt_tokens <= size + FRAMING_MARGIN
                    and usage.completion_tokens <= OUTPUT_LIMIT, "study_source_usage_invalid")
            require(math.isclose(usage.usd, done["reported_usd"], abs_tol=1e-12)
                    and math.isclose(usage.usd, done["accounted_usd"], abs_tol=1e-12),
                    "study_source_cost_mismatch")
            spend += usage.usd
            tokens.append(usage.prompt_tokens)
            decision = done["decision"]
            require(set(decision) == {"action", "reason_code"} and decision["reason_code"] is None,
                    "study_source_decision_mismatch")
            return Move(decision["action"])

        def observe(public):
            nonlocal ordinal
            step = take("study_source_public_step")
            require(step == {"record_type": "study_source_public_step", "episode": episode,
                             "step": ordinal, **public}, "study_source_step_mismatch")
            ordinal += 1

        ledger, record = execute(case, prior, episode, learning, Access.FULL, decide, observe=observe)
        outcome = json.loads(record.outcome_json)
        context = {"episode": episode, "learning": learning, "first_input_tokens": tokens[0],
                   "max_input_tokens": max(tokens), "completed": outcome["completed"],
                   "safe_success": outcome["safe_success"], "unsafe": outcome["unsafe"]}
        end = take("study_source_episode_finished")
        require(end == json.loads(canonical({
            "record_type": "study_source_episode_finished", "episode": episode,
            "case_id": case.case_id, "record": asdict(record), "outcome": outcome, "context": context,
            "new_public_events": [asdict(e) for e in ledger.events[len(prior.events):]],
        })), "study_source_end_mismatch")
        records.append(record)
        used_cases.append(case)
        context_rows.append(context)
        return context

    for index, case in enumerate(acquisition_cases(), 1):
        result = play(case, True)
        if should_stop_acquisition(index, result["first_input_tokens"]):
            break
    acquisition_count = len(records)
    frozen = ledger
    for case in combined_cases("calibration"):
        result = play(case, False)
        require(result["completed"] and result["first_input_tokens"] >= CONTEXT_TARGET,
                "study_source_probe_not_qualified")
        require(ledger == frozen, "study_source_probe_mutated_history")
    require(next(entries, None) is None, "study_source_extra_records")
    require(terminal.get("calls") == calls and terminal.get("acquisition_episodes") == acquisition_count
            and terminal.get("probe_episodes") == PROBES
            and terminal.get("qualifying_25k_episodes") == PROBES
            and terminal.get("ledger_head") == frozen.head
            and terminal.get("public_events") == len(frozen.events)
            and terminal.get("unknown_reserved_usd") == 0
            and math.isclose(terminal["reported_usd"], spend, abs_tol=1e-9)
            and math.isclose(terminal["accounted_usd"], spend, abs_tol=1e-9),
            "study_source_terminal_mismatch")
    replay(tuple(used_cases), tuple(records), frozen,
           tuple(row["learning"] for row in context_rows))
    return FrozenSource(hashlib.sha256(raw).hexdigest(), frozen,
                        tuple(used_cases[:acquisition_count]), tuple(records[:acquisition_count]),
                        spend, setup["prior_spend_usd"])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prior-spend-usd", type=float, required=True)
    parser.add_argument("--max-usd", type=float, default=1.5)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--live", action="store_true")
    args = parser.parse_args()
    if not args.live:
        parser.error("explicit --live required")
    if (not math.isfinite(args.prior_spend_usd) or args.prior_spend_usd < 0
            or not math.isfinite(args.max_usd) or not 0 < args.max_usd <= MAX_RUN_USD
            or args.prior_spend_usd + args.max_usd + RESERVE_USD > PROJECT_CAP_USD):
        parser.error("invalid expenditure envelope")
    key = os.getenv("SUBQ_API_KEY")
    if not key:
        parser.error("SUBQ_API_KEY missing")
    commit = subprocess.run(["git", "rev-parse", "HEAD"], check=True,
                            capture_output=True, text=True).stdout.strip()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        def emit(row):
            stream.write(canonical(row) + "\n")
            stream.flush()

        setup = setup_for(commit, args.prior_spend_usd, args.max_usd)
        setup["started_utc"] = datetime.now(timezone.utc).isoformat()
        emit(setup)
        meter = Meter(args.max_usd, lambda p: subq_request(p, api_key=key), emit,
                      parser=lambda data: parse_linked(data, ACTION_ONLY_REVISION),
                      max_calls=MAX_CALLS, max_input_bytes=MAX_INPUT_BYTES)
        run = SourceRun(meter, emit)
        failure = None
        try:
            run.run()
        except PilotError as error:
            failure = str(error)
        try:
            run.verify()
            replay_valid = True
        except (ValueError, TypeError, RuntimeError) as error:
            failure = "replay_failure:" + str(error)
            replay_valid = False
        summary = run.summary()
        summary.update({"record_type": "study_source_terminal", "error": failure,
                        "replay_valid": replay_valid,
                        "tier_qualified": summary["tier_qualified"] and failure is None and replay_valid,
                        "calls": meter.budget.calls, "reported_usd": meter.reported_usd,
                        "unknown_reserved_usd": meter.unknown_reserved_usd,
                        "accounted_usd": meter.budget.spent_usd, "report": str(args.output)})
        emit(summary)
    print(json.dumps(summary, indent=2))
    return 0 if summary["tier_qualified"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
