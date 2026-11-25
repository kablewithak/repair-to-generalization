"""One explicit continuation of the interrupted G8 reflection-only prefix.

Reconstruct all nine accepted claims against the same source and verify the
failed tenth call and accounting. No scored episode may have started. Only
the last review is resampled, with a shorter requested answer; the original
1,000-character acceptance bound and semantic schema are unchanged.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

from .experience_ledger import canonical, hash_value
from .g6_qualification import FRAMING_MARGIN, OUTPUT_LIMIT, digest, reservation
from .g8_claims import ReceiptClaim, claim_payload, diagnose, parse_claim, snapshot
from .savant_pilot import Usage
from .study_design import MAX_INPUT_BYTES, SERVICES

REVISION = "g8_last_review_concise_v1"
FAILURE = "invalid_model_response:invalid_claim:claim_advice"
SUFFIX = (
    " For this review, keep operational advice to at most three short sentences "
    "and target 400 to 500 characters. State only the essential conditional "
    "procedure and uncertainty. The original JSON fields and validation rules "
    "remain unchanged."
)


@dataclass(frozen=True)
class ReflectionPrefix:
    report_sha256: str
    initial: tuple[ReceiptClaim, ...]
    reviewed: tuple[ReceiptClaim, ...]
    diagnostics: tuple[dict, ...]
    prior_usd: float
    cap_usd: float
    reflection_cap_usd: float
    spent_usd: float
    calls: int


def require(condition, reason):
    if not condition:
        raise ValueError("g8_resume:" + reason)


def same(a, b):
    return canonical(a) == canonical(b)


def load_prefix(source, path: Path, setup_for):
    raw = path.read_bytes()
    rows = [json.loads(line) for line in raw.splitlines()]
    # Setup + nine successful call pairs and claim rows + failed call pair + terminal.
    require(len(rows) == 31, "unexpected_record_count")
    setup, terminal = rows[0], rows[-1]
    expected = setup_for(source, setup["commit"], setup["prior_spend_usd"],
                         setup["authorized_cap_usd"])
    require(set(setup) == set(expected) | {"started_utc"}
            and all(same(setup[k], v) for k, v in expected.items()), "setup_mismatch")
    require(terminal.get("record_type") == "g8_diagnostic_terminal"
            and terminal.get("status") == "interrupted"
            and terminal.get("error") == FAILURE
            and terminal.get("scored_episodes") == 0
            and terminal.get("planned_episodes") == 24
            and terminal.get("diagnostic_complete") is False
            and terminal.get("calls") == 10, "not_last_review_interruption")
    initial, reviewed, diagnostics = [], [], []
    spent = 0.0
    cursor = 1
    for i in range(1, 11):
        service = SERVICES[(i-1) % len(SERVICES)]
        prior = initial[i-6] if i > 5 else None
        payload = claim_payload(source.ledger, service, prior)
        started, done = rows[cursor:cursor+2]
        cursor += 2
        size, reserve = reservation(payload, MAX_INPUT_BYTES)
        identity = {"meter_phase": "reflection", "call_id": i, "episode": i, "step": 0}
        require(all(r.get(k) == v for r in (started, done) for k, v in identity.items()),
                "call_identity")
        require(started.get("record_type") == "call_started"
                and started.get("request") == payload
                and started.get("request_sha256") == digest(payload)
                and started.get("input_bytes") == size
                and started.get("cache_hit") is False
                and math.isclose(started["reserved_usd"], reserve, abs_tol=1e-12),
                "request_mismatch")
        require(done.get("record_type") == "call_finished"
                and done.get("finish_reason") == "stop"
                and done.get("unknown_reserved_usd") == 0, "response_metadata")
        usage = done["usage"]
        require(set(usage) == {"prompt_tokens", "cached_tokens", "completion_tokens"}
                and all(type(n) is int and n >= 0 for n in usage.values())
                and usage["cached_tokens"] <= usage["prompt_tokens"] <= size + FRAMING_MARGIN
                and usage["completion_tokens"] <= OUTPUT_LIMIT, "usage")
        reasoning = done.get("reasoning_tokens")
        require(reasoning is None or (type(reasoning) is int and
                0 <= reasoning <= usage["completion_tokens"]), "reasoning_usage")
        u = Usage(usage["prompt_tokens"], usage["cached_tokens"], usage["completion_tokens"])
        require(math.isclose(done["accounted_usd"], u.usd, abs_tol=1e-12)
                and math.isclose(done["reported_usd"], u.usd, abs_tol=1e-12),
                "cost_mismatch")
        require(spent + reserve <= setup["reflection_cap_usd"] + 1e-12, "reservation")
        spent += u.usd
        if i == 10:
            require(done.get("error") == FAILURE and done.get("decision") is None,
                    "last_review_not_advice_failure")
            continue
        require(done.get("error") is None and isinstance(done.get("decision"), dict),
                "accepted_call_failure")
        decision = done["decision"]
        body = {"prediction": decision["prediction"], "advice": decision["advice"],
                "evidence": decision["anchor"]}
        if prior is not None:
            body["operation"] = decision["operation"]
        # Revalidate accepted parsed fields. Raw provider output is intentionally
        # not retained; this proves consistency of recorded decisions, not attestation.
        response = {"choices": [{"finish_reason": "stop", "message": {"content": canonical(body)}}],
                    "usage": {"prompt_tokens": u.prompt_tokens, "completion_tokens": u.completion_tokens,
                              "prompt_tokens_details": {"cached_tokens": u.cached_tokens}}}
        claim, parsed_usage = parse_claim(response, source.ledger, service, prior)
        require(same(asdict(claim), decision) and parsed_usage == u, "claim_replay")
        record = rows[cursor]
        cursor += 1
        expected_record = {"record_type": "g8_claim_acquired" if prior is None else "g8_claim_reviewed",
                           "episode": i, "claim": claim.public()}
        if prior is None:
            initial.append(claim)
        else:
            reviewed.append(claim)
            diagnostics.append(diagnose(source.ledger, prior, claim))
            expected_record["diagnostic"] = diagnostics[-1]
        require(same(record, expected_record), "claim_record_mismatch")
    require(cursor == len(rows)-1 and len(initial) == 5 and len(reviewed) == 4,
            "prefix_boundary")
    require(all(math.isclose(terminal[k], spent, abs_tol=1e-12)
                for k in ("accounted_usd", "reported_usd", "reflection_usd"))
            and terminal["unknown_reserved_usd"] == 0, "terminal_accounting")
    require(all(row["episodes"] == 0 for row in terminal["by_arm"].values()),
            "scored_prefix_not_empty")
    return ReflectionPrefix(hashlib.sha256(raw).hexdigest(), tuple(initial), tuple(reviewed),
                            tuple(diagnostics), setup["prior_spend_usd"],
                            setup["authorized_cap_usd"], setup["reflection_cap_usd"], spent, 10)


def retry_payload(source, prefix):
    old = prefix.initial[-1]
    payload = claim_payload(source.ledger, old.claim.service, old)
    payload["messages"][0]["content"] += SUFFIX
    return payload


def finish_reflection(source, prefix, meter, emit, arms):
    old = prefix.initial[-1]
    service = old.claim.service
    meter.parser = lambda data: parse_claim(data, source.ledger, service, old)
    result = meter.decide(retry_payload(source, prefix), 10, 0)
    diagnostics = list(prefix.diagnostics) + [diagnose(source.ledger, old, result)]
    reviewed = prefix.reviewed + (result,)
    emit({"record_type": "g8_claim_reviewed", "episode": 10, "claim": result.public(),
          "diagnostic": diagnostics[-1]})
    snapshots = {arm: snapshot(prefix.initial, reviewed, arm) for arm in arms}
    emit({"record_type": "g8_claims_frozen", "source_ledger_head": source.ledger.head,
          "snapshot_sha256": {a: hash_value(s) for a,s in snapshots.items()},
          "snapshots": snapshots, "diagnostics": diagnostics})
    return snapshots, diagnostics
