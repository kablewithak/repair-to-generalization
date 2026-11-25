"""Development-only closed-loop claims: semantic prediction, derived bookkeeping.

A4 learns only from its own realized observations. The model authors the
prediction and advice; code derives the transition label, never the prediction.
The failed fixed-history diagnostic remains closed and is not replayed here.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Any

from .experience_ledger import Claim, Ledger, Uncertainty, Validity, canonical, validate_claim
from .g6_live import ACTION_ONLY_REVISION, payload_for, unique_object
from .g8_claims import PREDICTIONS, ReceiptClaim
from .savant_pilot import InvalidModelResponse, Usage

REVISION = "g8_derived_claim_transition_v1"
SYSTEM = (
    "Infer a receipt-source prediction and a short conditional operational lesson "
    "from your own public interactions for this service. Use only observed "
    "evidence; an earlier claim can be wrong. If insufficient evidence, choose "
    "unknown. Select one actual supporting step alias from this episode. "
    "Write at most three short sentences of advice, target 400 characters, "
    "never more than 1000. Include uncertainty or applicability as needed. "
    "Return only the required prediction, advice and evidence fields. "
    "Do not emit a retain or revise operation."
)


def scope(ledger: Ledger, episode: int, service: str, version: str):
    """Only actual receipt observations in the current learning episode qualify."""
    public = []
    anchors = {}
    observed = {}
    start = next((e for e in ledger.events if e.episode == episode and e.kind == "start"), None)
    if start is None:
        raise ValueError("missing_learning_episode")
    state = json.loads(start.body_json)
    if (state["service"], state["version"]) != (service, version):
        raise ValueError("claim_scope_mismatch")
    scopes = {}
    for event in ledger.events:
        if event.kind == "start":
            scopes[event.episode] = json.loads(event.body_json)["service"]
        if scopes[event.episode] != service:
            continue
        alias = f"e{event.episode}_{event.seq}"
        row = event.public()
        public.append({**row, "event_id": alias})
        if event.episode != episode or event.kind != "step":
            continue
        body = row["body"]
        obs = body["observation"]
        value = (obs.get("receipt_source") if body["action"] == "inspect_contract"
                 else obs.get("source") if body["action"] == "audit_export" else None)
        if value in PREDICTIONS[:2]:
            anchors[alias] = event.event_id
            observed[alias] = value
    return public, anchors, observed


def derive_transition(previous: str | None, prediction: str) -> str:
    if prediction not in PREDICTIONS or previous not in (None, *PREDICTIONS):
        raise ValueError("invalid_prediction")
    if previous is None:
        return "acquire"
    if prediction == "unknown":
        return "uncertain" if previous == "unknown" else "invalidate"
    return "retain" if prediction == previous else "revise"


def reflection_payload(ledger, episode, service, version, prior):
    public, anchors, _ = scope(ledger, episode, service, version)
    if not anchors:
        raise ValueError("no_receipt_evidence")
    context = {"service": service, "current_version": version, "public_interactions": public,
               "prior_claim": prior.public() if prior else None}
    request = payload_for(context, ACTION_ONLY_REVISION)
    request["messages"][0]["content"] = SYSTEM
    request["response_format"]["json_schema"] = {
        "name": "closed_loop_receipt_prediction", "strict": True,
        "schema": {"type": "object", "properties": {
            "prediction": {"type": "string", "enum": list(PREDICTIONS)},
            "advice": {"type": "string"},
            "evidence": {"type": "string", "enum": sorted(anchors)},
        }, "required": ["prediction", "advice", "evidence"], "additionalProperties": False},
    }
    return request


def parse_reflection(data: dict[str, Any], ledger: Ledger, episode: int, service: str,
                     version: str, previous: tuple[ReceiptClaim, ...]):
    try:
        row = data["usage"]
        prompt, completion = row["prompt_tokens"], row["completion_tokens"]
        cached = (row.get("prompt_tokens_details") or {}).get("cached_tokens", 0)
        if any(type(n) is not int or n < 0 for n in (prompt, completion, cached)) or cached > prompt:
            raise ValueError("invalid_usage")
        usage = Usage(prompt, cached, completion)
    except (KeyError, TypeError, AttributeError, ValueError) as error:
        raise InvalidModelResponse("closed_loop_invalid_usage") from error
    try:
        choices = data["choices"]
        if type(choices) is not list or len(choices) != 1:
            raise ValueError("choice_count")
        choice = choices[0]
        if choice["finish_reason"] == "length":
            raise InvalidModelResponse("output_limit", usage)
        if choice["finish_reason"] != "stop":
            raise ValueError("finish_reason")
        body = json.loads(choice["message"]["content"], object_pairs_hook=unique_object)
        if type(body) is not dict or set(body) != {"prediction", "advice", "evidence"}:
            raise ValueError("fields")
        prediction, advice, alias = body["prediction"], body["advice"], body["evidence"]
        _, anchors, _ = scope(ledger, episode, service, version)
        if type(prediction) is not str or prediction not in PREDICTIONS:
            raise ValueError("prediction")
        if type(advice) is not str or not advice.strip() or len(advice) > 1000:
            raise ValueError("advice")
        if type(alias) is not str or alias not in anchors:
            raise ValueError("evidence")
        prior = next((r for r in reversed(previous) if r.claim.service == service), None)
        # The operational label is derived from two model-authored predictions.
        # Do not compute or substitute the observed correct receipt source.
        operation = derive_transition(prior.prediction if prior else None, prediction)
        claim = Claim(
            claim_id=f"{service}/receipt/{episode}", service=service, version=version,
            applicability="Authorized export with uncertain prior acknowledgement",
            predicted_action_effect=prediction, evidence_ids=(anchors[alias],),
            uncertainty=Uncertainty.INSUFFICIENT_EVIDENCE if prediction == "unknown" else Uncertainty.HIGH,
            validity=Validity.UNKNOWN if prediction == "unknown" else Validity.SUPPORTED,
            revises=prior.claim.claim_id if prior else None,
        )
        validate_claim(claim, ledger, tuple(r.claim for r in previous))
        return ReceiptClaim(claim, prediction, advice, operation, alias), usage
    except InvalidModelResponse:
        raise
    except (KeyError, TypeError, IndexError, ValueError) as error:
        raise InvalidModelResponse("closed_loop_claim:" + str(error), usage) from error


def active_claims(claims: tuple[ReceiptClaim, ...]):
    latest = {r.claim.service: r for r in claims}
    return [{"kind": "persistent_agent_claim", "body": r.public()}
            for _, r in sorted(latest.items())]
