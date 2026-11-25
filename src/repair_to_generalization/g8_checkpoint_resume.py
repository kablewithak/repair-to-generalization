"""One final, explicit recovery at the frozen G8 evaluation boundary.

Both existing reports remain immutable. Validate the original and its
continuation, including the incomplete evaluation trajectory. No acquisition,
reflection, or accepted decision is sampled again. This is the same development
experiment, not an independent replication. A further interruption stops it.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from .experience_ledger import canonical
from .g6_qualification import Meter, PROJECT_CAP_USD, RESERVE_USD
from .g8_closed_loop_dev import MAX_CALLS, MAX_INPUT_BYTES, Run, order, setup
from .g8_network_resume import (
    NETWORK_FAILURE, REVISION as FIRST_REVISION, Prefix, ReplayBoundary,
    load_prefix, require, same,
)
from .savant_pilot import PilotError, subq_request

REVISION = "g8_frozen_checkpoint_recovery_v1"
ACQUISITION_EPISODES = 12


@dataclass(frozen=True)
class Chain:
    original: Prefix
    continuation: Prefix

    @property
    def calls(self):
        return self.original.calls + self.continuation.calls

    @property
    def spent_usd(self):
        return self.original.spent_usd + self.continuation.spent_usd

    def total(self, key):
        return self.original.terminal[key] + self.continuation.terminal[key]

    def adapter(self, *, live=None, emit=None):
        second = ReplayBoundary(self.continuation, live=live, emit=emit)
        first = ReplayBoundary(self.original, live=second, emit=second.emit)
        return first, second


def first_manifest(parent, commit, prior, cap):
    return dict(setup(commit, prior, cap), resume_revision=FIRST_REVISION,
                parent_report_sha256=parent.sha256, parent_accounted_usd=parent.spent_usd,
                parent_calls=parent.calls, parent_scored_episodes=parent.terminal["episodes"],
                max_new_calls=MAX_CALLS-parent.calls)


def load_chain(original_path: Path, continuation_path: Path):
    original = load_prefix(original_path, setup, Run)
    raw = continuation_path.read_bytes()
    rows = [json.loads(line) for line in raw.splitlines()]
    require(len(rows) >= 4, "checkpoint_missing_records")
    manifest, terminal = rows[0], rows[-1]
    prior, cap = manifest["prior_spend_usd"], manifest["authorized_cap_usd"]
    require(math.isfinite(prior) and math.isfinite(cap)
            and prior + 1e-12 >= original.setup["prior_spend_usd"] + original.spent_usd
            and cap > 0 and cap + original.spent_usd <= original.setup["authorized_cap_usd"] + 1e-12
            and prior + cap + RESERVE_USD <= PROJECT_CAP_USD, "checkpoint_parent_budget")
    expected = first_manifest(original, manifest["commit"], prior, cap)
    require(set(manifest) == set(expected) | {"started_utc"}
            and all(same(manifest[k], v) for k, v in expected.items()), "checkpoint_setup")
    require(terminal.get("record_type") == "g8_closed_loop_terminal"
            and terminal.get("error") == NETWORK_FAILURE
            and terminal.get("replay_valid") is True
            and terminal.get("development_positive") is False
            and terminal.get("g8_pass") is False
            and terminal.get("episodes") == ACQUISITION_EPISODES
            and terminal.get("parent_report_sha256") == original.sha256
            and terminal.get("parent_scored_episodes") == original.terminal["episodes"]
            and original.terminal["episodes"] < ACQUISITION_EPISODES
            and 0 < terminal["calls"] < manifest["max_new_calls"]
            and 0 < terminal["accounted_usd"] < cap, "checkpoint_terminal")
    # Preserve the original segment's local call counter and its own ceiling.
    continuation = Prefix(hashlib.sha256(raw).hexdigest(),
                          dict(manifest, max_calls=manifest["max_new_calls"]),
                          tuple(rows[1:-1]), terminal)
    chain = Chain(original, continuation)
    require(chain.calls < MAX_CALLS and chain.spent_usd < original.setup["authorized_cap_usd"],
            "checkpoint_chain_budget")
    require(terminal.get("chain_calls") == chain.calls, "checkpoint_chain_calls")
    for key in ("accounted_usd", "reported_usd", "unknown_reserved_usd"):
        require(math.isclose(terminal["chain_"+key], chain.total(key), abs_tol=1e-12),
                "checkpoint_chain_accounting")
    first, second = chain.adapter()
    run = Run(first, first.emit)
    try:
        run.run()
    except PilotError as exc:
        require(str(exc) == NETWORK_FAILURE and first.crossed
                and second.calls == continuation.calls, "checkpoint_replay_boundary")
    else:
        raise ValueError("checkpoint_missing_network_failure")
    run.verify()
    require(all(same(terminal.get(k), v) for k,v in run.summary().items()), "checkpoint_summary")
    require(len(run.rows) == ACQUISITION_EPISODES and not any(r["checkpoint"] for r in run.rows)
            and len(run.reflections) == 4, "checkpoint_acquisition_boundary")
    failed = continuation.events[-1]
    first_checkpoint_episode, first_checkpoint_arm = order()[ACQUISITION_EPISODES]
    started = next(r for r in reversed(continuation.events)
                   if r["record_type"] == "g8_cl_episode_started")
    require(failed.get("record_type") == "call_finished"
            and failed.get("episode") == first_checkpoint_episode
            and started.get("episode") == first_checkpoint_episode
            and started.get("arm") == first_checkpoint_arm
            and started.get("checkpoint") is True and started.get("learning") is False,
            "checkpoint_not_first_evaluation")
    return chain


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--continuation", type=Path, required=True)
    parser.add_argument("--prior-spend-usd", type=float, required=True)
    parser.add_argument("--max-usd", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--live", action="store_true")
    args = parser.parse_args()
    if not args.live:
        parser.error("explicit --live required")
    chain = load_chain(args.parent, args.continuation)
    require(math.isfinite(args.prior_spend_usd) and math.isfinite(args.max_usd)
            and args.prior_spend_usd + 1e-12 >= max(
                chain.original.setup["prior_spend_usd"] + chain.spent_usd,
                chain.continuation.setup["prior_spend_usd"] + chain.continuation.spent_usd)
            and 0 < args.max_usd
            and args.max_usd + chain.spent_usd <= chain.original.setup["authorized_cap_usd"] + 1e-12
            and args.max_usd + chain.continuation.spent_usd <= chain.continuation.setup["authorized_cap_usd"] + 1e-12
            and args.prior_spend_usd + args.max_usd + RESERVE_USD <= PROJECT_CAP_USD,
            "checkpoint_new_budget")
    key = os.getenv("SUBQ_API_KEY")
    if not key:
        parser.error("SUBQ_API_KEY missing")
    commit = subprocess.run(["git", "rev-parse", "HEAD"], check=True,
                            capture_output=True, text=True).stdout.strip()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        def emit(row):
            stream.write(canonical(row)+"\n")
            stream.flush()

        emit(dict(setup(commit, args.prior_spend_usd, args.max_usd),
                  resume_revision=REVISION,
                  parent_report_sha256=chain.original.sha256,
                  continuation_report_sha256=chain.continuation.sha256,
                  prior_chain_calls=chain.calls, prior_chain_accounted_usd=chain.spent_usd,
                  max_new_calls=MAX_CALLS-chain.calls, acquisition_frozen=True,
                  started_utc=datetime.now(timezone.utc).isoformat(),
                  recovery_limit="final explicit checkpoint recovery; no further continuation"))
        meter = Meter(args.max_usd, lambda p: subq_request(p, api_key=key), emit,
                      max_calls=MAX_CALLS-chain.calls, max_input_bytes=MAX_INPUT_BYTES)
        first, second = chain.adapter(live=meter, emit=emit)
        run = Run(first, first.emit)
        failure = None
        try:
            run.run()
        except PilotError as exc:
            failure = str(exc)
        try:
            run.verify()
            replay_valid = first.crossed and second.crossed
        except (ValueError, TypeError, RuntimeError) as exc:
            replay_valid = False
            failure = "replay_failure:" + str(exc)
        result = run.summary()
        result.update(record_type="g8_closed_loop_terminal", error=failure,
                      replay_valid=replay_valid, development_positive=result["development_positive"]
                      and failure is None and replay_valid, calls=meter.budget.calls,
                      reported_usd=meter.reported_usd, unknown_reserved_usd=meter.unknown_reserved_usd,
                      accounted_usd=meter.budget.spent_usd, report=str(args.output),
                      parent_report_sha256=chain.original.sha256,
                      continuation_report_sha256=chain.continuation.sha256,
                      chain_calls=chain.calls+meter.budget.calls,
                      chain_reported_usd=chain.total("reported_usd")+meter.reported_usd,
                      chain_unknown_reserved_usd=chain.total("unknown_reserved_usd")+meter.unknown_reserved_usd,
                      chain_accounted_usd=chain.spent_usd+meter.budget.spent_usd,
                      acquisition_frozen=True)
        emit(result)
    print(json.dumps(result, indent=2))
    return 0 if failure is None and replay_valid and result["episodes"] == len(order()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
