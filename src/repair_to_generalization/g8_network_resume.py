"""Explicit recovery from a single network interruption in closed-loop development.

Replay recorded model decisions and every realized public transition offline,
including the unscored episode prefix. Resume at the failed call, never repeat
an accepted action or claim. No response-contract retries or chained resumes.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

from .experience_ledger import canonical
from .g6_qualification import FRAMING_MARGIN, OUTPUT_LIMIT, digest, reservation
from .savant_pilot import PilotError, Usage

REVISION = "g8_closed_loop_network_resume_v1"
NETWORK_FAILURE = "api_network_failure"


def require(condition, message):
    if not condition:
        raise ValueError("g8_network_resume:" + message)


def same(a, b):
    return canonical(a) == canonical(b)


@dataclass(frozen=True)
class Prefix:
    sha256: str
    setup: dict
    events: tuple[dict, ...]
    terminal: dict

    @property
    def spent_usd(self):
        return self.terminal["accounted_usd"]

    @property
    def calls(self):
        return self.terminal["calls"]


class ReplayBoundary:
    """Meter/emit adapter for the original runner; switches only at the failed call."""

    def __init__(self, prefix: Prefix, *, live=None, emit=None):
        self.prefix, self.live, self.output = prefix, live, emit
        self.events = iter(prefix.events)
        self.parser = None
        self.last_usage = None
        self.calls = 0
        self.reported_usd = 0.0
        self.unknown_reserved_usd = 0.0
        self.spent_usd = 0.0
        self.crossed = False

    def take(self, kind):
        row = next(self.events, None)
        require(row is not None and row.get("record_type") == kind, "event_order:" + kind)
        return row

    def emit(self, row):
        if self.crossed:
            require(self.output is not None, "missing_live_output")
            self.output(row)
        else:
            require(same(self.take(row["record_type"]), row), "public_event_mismatch:" + row["record_type"])

    def fresh(self, payload, episode, step):
        self.live.parser = self.parser
        decision = self.live.decide(payload, episode, step)
        self.last_usage = self.live.last_usage
        return decision

    def decide(self, payload, episode, step):
        self.last_usage = None
        if self.crossed:
            return self.fresh(payload, episode, step)
        sent, done = self.take("call_started"), self.take("call_finished")
        self.calls += 1
        size, upper = reservation(payload, self.prefix.setup["max_input_bytes"])
        for row in (sent, done):
            require(row.get("call_id") == self.calls and row.get("episode") == episode
                    and row.get("step") == step, "call_identity")
        require(sent.get("request") == payload and sent.get("request_sha256") == digest(payload)
                and sent.get("input_bytes") == size and sent.get("cache_hit") is False
                and math.isclose(sent["reserved_usd"], upper, abs_tol=1e-12),
                "request_mismatch")
        require(self.calls <= self.prefix.setup["max_calls"]
                and self.spent_usd + upper <= self.prefix.setup["authorized_cap_usd"] + 1e-12,
                "reservation_mismatch")
        if done.get("error") is not None:
            require(done["error"] == NETWORK_FAILURE and done.get("usage") is None
                    and done.get("decision") is None and done.get("reported_usd") is None
                    and done.get("finish_reason") is None
                    and done.get("reasoning_tokens") is None and done.get("content_shape") is None
                    and math.isclose(done["accounted_usd"], upper, abs_tol=1e-12)
                    and math.isclose(done["unknown_reserved_usd"], upper, abs_tol=1e-12),
                    "not_unknown_network_failure")
            self.unknown_reserved_usd += upper
            self.spent_usd += upper
            require(next(self.events, None) is None, "records_after_failure")
            require(self.calls == self.prefix.calls
                    and math.isclose(self.reported_usd, self.prefix.terminal["reported_usd"], abs_tol=1e-12)
                    and math.isclose(self.unknown_reserved_usd, self.prefix.terminal["unknown_reserved_usd"], abs_tol=1e-12)
                    and math.isclose(self.spent_usd, self.prefix.spent_usd, abs_tol=1e-12),
                    "terminal_accounting_mismatch")
            if self.live is None:
                raise PilotError(NETWORK_FAILURE)
            self.crossed = True
            self.output(dict(record_type="g8_cl_network_resume_boundary", parent_report_sha256=self.prefix.sha256,
                             parent_call_id=self.calls, episode=episode, step=step,
                             request_sha256=digest(payload), prior_accounted_usd=self.spent_usd,
                             accepted_calls_replayed=self.calls - 1))
            return self.fresh(payload, episode, step)
        require(done.get("finish_reason") == "stop" and done.get("unknown_reserved_usd") == 0,
                "invalid_accepted_response")
        usage = done["usage"]
        require(set(usage) == {"prompt_tokens", "cached_tokens", "completion_tokens"}
                and all(type(n) is int and n >= 0 for n in usage.values())
                and usage["cached_tokens"] <= usage["prompt_tokens"] <= size + FRAMING_MARGIN
                and usage["completion_tokens"] <= OUTPUT_LIMIT, "usage_mismatch")
        reasoning = done.get("reasoning_tokens")
        require(reasoning is None or (type(reasoning) is int and
                0 <= reasoning <= usage["completion_tokens"]), "reasoning_usage")
        u = Usage(usage["prompt_tokens"], usage["cached_tokens"], usage["completion_tokens"])
        require(math.isclose(done["accounted_usd"], u.usd, abs_tol=1e-12)
                and math.isclose(done["reported_usd"], u.usd, abs_tol=1e-12),
                "cost_mismatch")
        self.spent_usd += u.usd
        self.reported_usd += u.usd
        decision = done["decision"]
        if payload["response_format"]["json_schema"]["name"] == "closed_loop_receipt_prediction":
            body = {"prediction": decision["prediction"], "advice": decision["advice"],
                    "evidence": decision["anchor"]}
        else:
            body = {"action": decision["action"]}
        # Revalidate parsed fields, not fabricated raw responses or provider attestations.
        response = {"choices": [{"finish_reason": "stop", "message": {"content": canonical(body)}}],
                    "usage": {"prompt_tokens": u.prompt_tokens, "completion_tokens": u.completion_tokens,
                              "prompt_tokens_details": {"cached_tokens": u.cached_tokens}}}
        parsed, parsed_usage = self.parser(response)
        require(same(asdict(parsed), decision) and parsed_usage == u, "decision_replay_mismatch")
        self.last_usage = u
        return parsed


def load_prefix(path: Path, setup_for, run_factory):
    raw = path.read_bytes()
    rows = [json.loads(line) for line in raw.splitlines()]
    require(len(rows) >= 4, "missing_records")
    setup, terminal = rows[0], rows[-1]
    prior, cap = setup["prior_spend_usd"], setup["authorized_cap_usd"]
    require(math.isfinite(prior) and prior >= 0 and math.isfinite(cap) and 0 < cap <= .75
            and prior + cap + setup["reserve_usd"] <= setup["project_cap_usd"], "invalid_parent_budget")
    expected = setup_for(setup["commit"], prior, cap)
    require(set(setup) == set(expected) | {"started_utc"} and
            all(same(setup[k], v) for k, v in expected.items()), "setup_mismatch")
    require(terminal.get("record_type") == "g8_closed_loop_terminal"
            and terminal.get("error") == NETWORK_FAILURE
            and terminal.get("replay_valid") is True
            and terminal.get("development_positive") is False
            and terminal.get("g8_pass") is False
            and 0 < terminal["calls"] < setup["max_calls"]
            and 0 < terminal["accounted_usd"] < cap, "parent_not_resumable")
    prefix = Prefix(hashlib.sha256(raw).hexdigest(), setup, tuple(rows[1:-1]), terminal)
    adapter = ReplayBoundary(prefix)
    run = run_factory(adapter, adapter.emit)
    try:
        run.run()
    except PilotError as error:
        require(str(error) == NETWORK_FAILURE, "unexpected_replay_failure")
    else:
        raise ValueError("g8_network_resume:missing_failure_boundary")
    run.verify()
    require(all(same(terminal.get(k), v) for k, v in run.summary().items()), "terminal_summary_mismatch")
    return prefix
