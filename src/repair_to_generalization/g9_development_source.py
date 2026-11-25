"""Replay the interrupted G8 replication as a source, not a passed gate.

The first (and only executed) A4 acquisition stream is selected by fixed
index, without filtering its actions, claims or failures by outcome.
Checkpoints are excluded by the original learning mask.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

from .experience_ledger import Ledger, canonical, hash_value
from .g6_qualification import digest, reservation, FRAMING_MARGIN, OUTPUT_LIMIT
from .g8_closed_loop_claims import active_claims
from .g8_closed_loop_dev import Run
from .g8_independent import CONFIGS, MASK, design, setup, stream_summary, main_screen
from .g8_network_resume import Prefix, ReplayBoundary, require, same
from .savant_pilot import PilotError, Usage

SOURCE_SHA256 = "b38b3159c8a93914d5dabf015850175bc108138a41091c5f9b471f2b07461e0c"
FAILURE = "invalid_model_response:output_limit"


@dataclass(frozen=True)
class FrozenSource:
    sha256: str
    ledger: Ledger
    claims: tuple
    accounted_usd: float
    prior_usd: float

    def snapshot(self, arm):
        if arm == "A0":
            return []
        history = self.ledger.public()
        if arm == "A3":
            return history
        if arm == "A4_STALE":
            return history + active_claims(tuple(c for c in self.claims if c.claim.version == "v1"))
        if arm == "A4":
            return history + active_claims(self.claims)
        raise ValueError("unknown_fulfillment_arm")


class FailedSourceReplay(ReplayBoundary):
    """The accepted-call replay is unchanged; separately audit truncation."""

    def decide(self, payload, episode, step):
        if self.calls + 1 != self.prefix.calls:
            return super().decide(payload, episode, step)
        sent, done = self.take("call_started"), self.take("call_finished")
        size, upper = reservation(payload, self.prefix.setup["max_input_bytes"])
        self.calls += 1
        require(all(r["call_id"] == self.calls and r["episode"] == episode and
                    r["step"] == step for r in (sent, done)), "source_failed_call_identity")
        require(sent["request"] == payload and sent["request_sha256"] == digest(payload)
                and sent["input_bytes"] == size and sent["cache_hit"] is False
                and math.isclose(sent["reserved_usd"], upper, abs_tol=1e-12)
                and self.spent_usd + upper <= self.prefix.setup["authorized_cap_usd"],
                "source_failed_request")
        require(done["error"] == FAILURE and done["finish_reason"] == "length"
                and done["decision"] is None and done["content_shape"] == {"kind": "null"}
                and done["unknown_reserved_usd"] == 0, "source_failure_shape")
        usage = done["usage"]
        require(set(usage) == {"prompt_tokens", "cached_tokens", "completion_tokens"}
                and all(type(n) is int and n >= 0 for n in usage.values())
                and usage["cached_tokens"] <= usage["prompt_tokens"] <= size + FRAMING_MARGIN
                and usage["completion_tokens"] == done["reasoning_tokens"] == OUTPUT_LIMIT,
                "source_failure_usage")
        u = Usage(usage["prompt_tokens"], usage["cached_tokens"], usage["completion_tokens"])
        require(math.isclose(done["accounted_usd"], u.usd, abs_tol=1e-12)
                and math.isclose(done["reported_usd"], u.usd, abs_tol=1e-12),
                "source_failure_cost")
        self.spent_usd += u.usd
        self.reported_usd += u.usd
        require(next(self.events, None) is None and self.calls == self.prefix.calls
                and math.isclose(self.spent_usd, self.prefix.spent_usd, abs_tol=1e-12)
                and math.isclose(self.reported_usd, self.prefix.terminal["reported_usd"], abs_tol=1e-12)
                and self.prefix.terminal["unknown_reserved_usd"] == 0,
                "source_terminal_accounting")
        raise PilotError(FAILURE)


def replay_source(rows, sha):
    """Reconstruct the entire reported first-stream prefix offline."""
    require(len(rows) >= 5, "source_short_report")
    manifest, terminal = rows[0], rows[-1]
    prior, cap = manifest["prior_spend_usd"], manifest["authorized_cap_usd"]
    expected = setup(manifest["commit"], prior, cap)
    require(set(manifest) == set(expected) | {"started_utc"}
            and all(same(manifest[k], v) for k,v in expected.items()), "source_setup")
    seed = CONFIGS[0][0]
    tagged = rows[1:-1]
    require(all(r["seed"] == seed and r["study_stage"] == "main" for r in tagged),
            "source_only_first_stream")
    events = [{k:v for k,v in r.items() if k not in ("seed","study_stage")} for r in tagged]
    require(events[0] == {"record_type":"g8_rep_stream_started","fresh_state":True}
            and events[-1]["record_type"] == "g8_rep_stream_finished", "source_stream_boundary")
    require(terminal["record_type"] == "g8_rep_terminal" and terminal["error"] == FAILURE
            and terminal["scored_main_episodes"] == 17 and terminal["transport_recoveries"] == 0
            and terminal["ablation_status"] == "not_run" and terminal["g8_pass"] is False
            and terminal["replication_and_ablation_positive"] is False, "source_terminal")
    prefix = Prefix(sha, dict(manifest,max_calls=manifest["max_block_calls"]),
                    tuple(events[1:-1]), terminal)
    meter = FailedSourceReplay(prefix)
    cohort,schedule,_,_ = design(0)
    run = Run(meter,meter.emit,cohort=cohort,schedule=schedule,learning_mask=MASK)
    try:
        run.run()
    except PilotError as error:
        require(str(error) == FAILURE, "source_wrong_failure")
    else:
        raise ValueError("source_missing_failure")
    run.verify()
    summary = dict(stream_summary(run,0),error=FAILURE,replay_valid=True,
                   accounted_usd=meter.spent_usd,transport_recoveries=0)
    require(same(terminal["streams"],[summary]) and same(terminal["main"],main_screen([summary])),
            "source_summary")
    require(same(events[-1],dict(summary,record_type="g8_rep_stream_finished",
                    frozen_ledgers={a:l.public() for a,l in run.ledgers.items()},
                    frozen_claims=[asdict(c) for c in run.claims])), "source_frozen_state")
    require(len(run.rows)==17 and len(run.claims)==4 and
            {e.episode for e in run.ledgers["A4"].events}=={1,2,5,6},
            "source_acquisition_incomplete")
    # This reports the mechanism; it never converts the failed study to a pass.
    return FrozenSource(sha, run.ledgers["A4"], run.claims, meter.spent_usd, prior)


def load_source(path: Path):
    raw = path.read_bytes()
    sha = hashlib.sha256(raw).hexdigest()
    require(sha == SOURCE_SHA256, "source_identity")
    return replay_source([json.loads(line) for line in raw.splitlines()], sha)
