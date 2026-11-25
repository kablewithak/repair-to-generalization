"""G8 fixed-history diagnostic: model-authored, evidence-linked receipt claims.

The source is genuine A3 acquisition, not the G7 scored report. Initial
claims see only v1 observations. Review sees the original claim plus later
public evidence and independently chooses retain/revise/invalidate/unknown.
No evaluator outcome, correct recommendation or hidden state is provided.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Any

from .experience_ledger import Claim, Ledger, Uncertainty, Validity, validate_claim, hash_value
from .g6_live import ACTION_ONLY_REVISION, payload_for, unique_object
from .savant_pilot import InvalidModelResponse, Usage
from .study_design import SERVICES
from .study_memory import episodes

REVISION = "g8_receipt_claim_v1"
PREDICTIONS = ("provider_ledger", "delivery_replica", "unknown")
OPERATIONS = ("retain", "revise", "invalidate", "uncertain")
MAX_ADVICE = 1000
SYSTEM = (
    "You are reviewing your own public export interactions. Form an advisory "
    "claim about the receipt source used after scheduler advancement, and "
    "what action sequence your observations support when earlier effects "
    "are uncertain. The prediction must describe the observed source, not "
    "a desired outcome. Your operational advice is your own inference; "
    "include conditions and uncertainty, and never invent hidden facts. "
    "A later observation may contradict a prior claim or leave it unchanged. "
    "When reviewing, decide whether to retain, revise, invalidate or remain "
    "uncertain about its receipt-source prediction. Refine the advice only "
    "as warranted. Retain means the same prediction; revise means a "
    "different concrete prediction; invalidate and uncertain mean unknown. "
    "Select a supporting step alias from the allowed current-version steps. "
    "Keep advice at most 1000 characters. Return only the JSON schema."
)


def public_scope(ledger: Ledger, service: str, version: str):
    """Return detached export-only histories and scoped current-step aliases."""
    if service not in SERVICES or version not in ("v1", "v2"):
        raise ValueError("invalid_claim_scope")
    rows = []
    anchors = {}
    observed = set()
    for group in episodes(ledger):
        start = group[0]["body"]
        if start["service"] != service or start["version"] != version:
            continue
        if start["requested"] != ["export"]:
            continue
        for index, event in enumerate(group):
            alias = f"e{event['episode']}_{index}"
            rows.append({**event, "event_id": alias})
            if event["kind"] != "step":
                continue
            body = event["body"]
            observation = body["observation"]
            if body["action"] == "inspect_contract":
                value = observation.get("receipt_source")
            elif body["action"] == "audit_export":
                value = observation.get("source")
            else:
                continue
            if value in PREDICTIONS[:2]:
                anchors[alias] = event["event_id"]
                observed.add(value)
    if not anchors or len(observed) != 1:
        raise ValueError("ambiguous_or_missing_public_receipt_evidence")
    return rows, anchors, next(iter(observed))


@dataclass(frozen=True)
class ReceiptClaim:
    claim: Claim
    prediction: str
    advice: str
    operation: str
    anchor: str

    def public(self):
        return {
            "claim_id": self.claim.claim_id, "service": self.claim.service,
            "version": self.claim.version, "applicability": self.claim.applicability,
            "prediction": self.prediction, "operational_advice": self.advice,
            "uncertainty": self.claim.uncertainty.value,
            "agent_asserted_validity": self.claim.validity.value,
            "evidence_ids": list(self.claim.evidence_ids), "revises": self.claim.revises,
            "operation": self.operation,
        }


def schema_for(anchors: dict[str, str], review: bool):
    properties = {
        "prediction": {"type": "string", "enum": list(PREDICTIONS)},
        "advice": {"type": "string"},
        "evidence": {"type": "string", "enum": sorted(anchors)},
    }
    if review:
        properties["operation"] = {"type": "string", "enum": list(OPERATIONS)}
    return {"type": "object", "properties": properties,
            "required": list(properties), "additionalProperties": False}


def claim_payload(ledger: Ledger, service: str, prior: ReceiptClaim | None = None):
    version = "v1" if prior is None else "v2"
    public, anchors, _ = public_scope(ledger, service, version)
    context: dict[str, Any] = {"service": service, "current_version": version,
                              "public_interactions": public}
    if prior is not None:
        earlier, _, _ = public_scope(ledger, service, "v1")
        context["earlier_public_interactions"] = earlier
        context["prior_claim"] = prior.public()
        context["prior_evidence_alias"] = prior.anchor
    payload = payload_for(context, ACTION_ONLY_REVISION)
    payload["messages"][0]["content"] = SYSTEM
    payload["response_format"]["json_schema"] = {
        "name": "receipt_claim_review" if prior else "receipt_claim_initial",
        "strict": True, "schema": schema_for(anchors, prior is not None),
    }
    return payload


def parse_claim(data: dict[str, Any], ledger: Ledger, service: str,
                prior: ReceiptClaim | None = None) -> tuple[ReceiptClaim, Usage]:
    try:
        row = data["usage"]
        prompt, completion = row["prompt_tokens"], row["completion_tokens"]
        cached = (row.get("prompt_tokens_details") or {}).get("cached_tokens", 0)
        if any(type(n) is not int or n < 0 for n in (prompt, completion, cached)) or cached > prompt:
            raise ValueError("usage")
        usage = Usage(prompt, cached, completion)
    except (KeyError, TypeError, AttributeError, ValueError) as error:
        raise InvalidModelResponse("claim_invalid_usage") from error
    try:
        choices = data["choices"]
        if not isinstance(choices, list) or len(choices) != 1:
            raise ValueError("claim_choice_count")
        choice = choices[0]
        if choice["finish_reason"] == "length":
            raise InvalidModelResponse("output_limit", usage)
        if choice["finish_reason"] != "stop":
            raise ValueError("claim_finish_reason")
        body = json.loads(choice["message"]["content"], object_pairs_hook=unique_object)
        version = "v1" if prior is None else "v2"
        _, anchors, _ = public_scope(ledger, service, version)
        schema = schema_for(anchors, prior is not None)
        if not isinstance(body, dict) or set(body) != set(schema["required"]):
            raise ValueError("claim_fields")
        prediction, advice, anchor = body["prediction"], body["advice"], body["evidence"]
        if type(prediction) is not str or prediction not in PREDICTIONS:
            raise ValueError("claim_prediction")
        if type(advice) is not str or not advice.strip() or len(advice) > MAX_ADVICE:
            raise ValueError("claim_advice")
        if type(anchor) is not str or anchor not in anchors:
            raise ValueError("claim_evidence_scope")
        operation = body.get("operation", "acquire")
        if prior is not None:
            if prior.claim.service != service or prior.claim.version != "v1":
                raise ValueError("claim_prior_scope")
            if type(operation) is not str or operation not in OPERATIONS:
                raise ValueError("claim_operation")
            if operation == "retain" and prediction != prior.prediction:
                raise ValueError("claim_retain_changed_prediction")
            if operation == "revise" and (prediction == prior.prediction or prediction == "unknown"):
                raise ValueError("claim_revision_not_distinct")
            if operation in ("invalidate", "uncertain") and prediction != "unknown":
                raise ValueError("claim_uncertainty_mismatch")
        # Scope and lineage are deterministic metadata. Prediction/advice are
        # entirely model-authored. The validator never supplies a correct answer.
        claim = Claim(
            claim_id=f"{service}/export-receipt/{version}", service=service, version=version,
            applicability="Requested, authorized export; prior acknowledgement uncertain",
            predicted_action_effect=prediction,
            evidence_ids=(anchors[anchor],),
            uncertainty=Uncertainty.INSUFFICIENT_EVIDENCE if prediction == "unknown" else Uncertainty.LOW,
            validity=Validity.UNKNOWN if prediction == "unknown" else Validity.SUPPORTED,
            revises=prior.claim.claim_id if prior is not None else None,
        )
        validate_claim(claim, ledger, (prior.claim,) if prior is not None else ())
        return ReceiptClaim(claim, prediction, advice, operation, anchor), usage
    except InvalidModelResponse:
        raise
    except (KeyError, TypeError, ValueError, IndexError) as error:
        raise InvalidModelResponse("invalid_claim:" + str(error), usage) from error


def diagnose(ledger: Ledger, initial: ReceiptClaim, review: ReceiptClaim):
    """Literal observable agreement, computed for reporting, never supplied to policy."""
    _, _, old = public_scope(ledger, initial.claim.service, "v1")
    _, _, new = public_scope(ledger, review.claim.service, "v2")
    correct_initial = initial.prediction == old
    correct_review = review.prediction == new
    changed = old != new
    return {
        "service": initial.claim.service, "observed_v1": old, "observed_v2": new,
        "receipt_changed": changed, "initial_agrees": correct_initial,
        "review_agrees": correct_review, "operation": review.operation,
        "detected_and_repaired": changed and correct_initial and correct_review
                                 and review.operation == "revise",
        "retained_unaffected_prediction": not changed and correct_initial and correct_review
                                         and review.operation == "retain",
        "false_invalidation": not changed and correct_initial and review.prediction == "unknown",
    }


def snapshot(initial: tuple[ReceiptClaim, ...], reviewed: tuple[ReceiptClaim, ...], arm: str):
    """All policies retain identical original interactions; only claim state differs."""
    if tuple(c.claim.service for c in initial) != SERVICES or tuple(c.claim.service for c in reviewed) != SERVICES:
        raise ValueError("incomplete_claim_snapshot")
    if any(r.claim.revises != i.claim.claim_id for i, r in zip(initial, reviewed)):
        raise ValueError("invalid_claim_lineage")
    if arm == "A3":
        return []
    if arm == "A4_STALE":
        return [{"kind": "persistent_agent_claim", "body": c.public()} for c in initial]
    if arm == "A4_INVALIDATE_ALL":
        return [{"kind": "invalidated_claim", "body": {"claim_id": c.claim.claim_id,
                   "service": c.claim.service, "validity": "invalidated"}} for c in initial]
    if arm == "A4":
        return [{"kind": "persistent_agent_claim", "body": c.public()} for c in reviewed]
    raise ValueError("unknown_claim_arm")


def snapshot_hash(initial, reviewed, arm):
    return hash_value(snapshot(initial, reviewed, arm))
