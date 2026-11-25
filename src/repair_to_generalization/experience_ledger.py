"""Append-only public experience and deterministic replay for linked tasks.

Evaluation outcomes are separate from experience. Only observations generated
by the instrument are eligible for learning; never import arbitrary JSON as
trusted history. Replay from the private case manifest before reuse.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Callable

from .linked_world import CONTRACT, MAX_ACTIONS, VERSION, LinkedWorld, Move, Scenario


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def hash_value(value):
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class PublicEvent:
    seq: int
    episode: int
    kind: str
    body_json: str
    previous: str
    event_id: str

    def public(self):
        return {"event_id": self.event_id, "episode": self.episode, "kind": self.kind,
                "body": json.loads(self.body_json)}


@dataclass(frozen=True)
class Ledger:
    events: tuple[PublicEvent, ...] = ()

    @property
    def head(self):
        return self.events[-1].event_id if self.events else "0" * 64

    def append(self, episode, kind, body):
        if type(episode) is not int or episode < 1 or kind not in ("start", "step", "end"):
            raise ValueError("invalid_event")
        seq = len(self.events) + 1
        body_json = canonical(body)
        identity = hash_value([seq, episode, kind, body_json, self.head])
        return Ledger(self.events + (PublicEvent(seq, episode, kind, body_json,
                                                self.head, identity),))

    def public(self):
        return [event.public() for event in self.events]


class Access(str, Enum):
    CURRENT = "current_only"
    FULL = "full_history"


@dataclass(frozen=True)
class EpisodeRecord:
    actions: tuple[Move, ...]
    outcome_json: str
    ledger_head: str
    learning: bool


def request_for(world, history, prior, access):
    if not isinstance(access, Access):
        raise TypeError("invalid_access")
    # Canonical round-trip gives the policy a detached public snapshot.
    return json.loads(canonical({
        "instrument": VERSION, "contract": CONTRACT, "current": world.view(),
        "current_interactions": history,
        "prior_interactions": prior.public() if access is Access.FULL else [],
    }))


def execute(case: Scenario, prior: Ledger, episode: int, learning: bool, access: Access,
            decide: Callable[[dict], Move]):
    if type(learning) is not bool or type(episode) is not int or episode < 1:
        raise ValueError("invalid_episode_configuration")
    if prior.events and episode <= prior.events[-1].episode:
        raise ValueError("episode_not_monotonic")
    world = LinkedWorld(case)
    result_ledger = prior
    if learning:
        result_ledger = result_ledger.append(episode, "start", world.view())
    history = []
    actions = []
    while not world.finished and world.actions < MAX_ACTIONS:
        move = decide(request_for(world, history, prior, access))
        if not isinstance(move, Move):
            raise TypeError("invalid_policy_action")
        observation = world.step(move)
        row = {"action": move.value, "observation": observation}
        history.append(row)
        actions.append(move)
        if learning:
            result_ledger = result_ledger.append(episode, "step", row)
    if learning:
        result_ledger = result_ledger.append(episode, "end", {
            "terminal": "finished" if world.finished else "action_limit",
            "remaining_actions": MAX_ACTIONS - world.actions,
        })
    return result_ledger, EpisodeRecord(tuple(actions), canonical(world.outcome()),
                                        result_ledger.head, learning)


def replay(cases: tuple[Scenario, ...], records: tuple[EpisodeRecord, ...], expected: Ledger):
    """Validate each actual transition, evaluator outcome and learning exclusion."""
    if len(cases) != len(records):
        raise ValueError("replay_length_mismatch")
    ledger = Ledger()
    for episode, (case, record) in enumerate(zip(cases, records), 1):
        moves = iter(record.actions)

        def decide(_):
            try:
                return next(moves)
            except StopIteration as error:
                raise ValueError("replay_missing_action") from error

        ledger, rebuilt = execute(case, ledger, episode, record.learning, Access.CURRENT, decide)
        if next(moves, None) is not None or rebuilt != record:
            raise ValueError("replay_record_mismatch")
    if ledger != expected:
        raise ValueError("replay_ledger_mismatch")
    return ledger


class Validity(str, Enum):
    SUPPORTED = "supported"
    CONTRADICTED = "contradicted"
    UNKNOWN = "unknown"
    SUPERSEDED = "superseded"


class Uncertainty(str, Enum):
    LOW = "low"
    HIGH = "high"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"


@dataclass(frozen=True)
class Claim:
    claim_id: str
    service: str
    version: str
    applicability: str
    predicted_action_effect: str
    evidence_ids: tuple[str, ...]
    uncertainty: Uncertainty
    validity: Validity
    revises: str | None = None


def validate_claim(claim: Claim, ledger: Ledger, previous: tuple[Claim, ...]):
    """Check shape, prior public evidence and revision lineage, not semantic truth."""
    for field in (claim.claim_id, claim.service, claim.version, claim.applicability,
                  claim.predicted_action_effect):
        if type(field) is not str or not field.strip() or len(field) > 1000:
            raise ValueError("invalid_claim_field")
    if not isinstance(claim.uncertainty, Uncertainty) or not isinstance(claim.validity, Validity):
        raise ValueError("invalid_claim_state")
    if type(claim.evidence_ids) is not tuple or not claim.evidence_ids:
        raise ValueError("missing_claim_evidence")
    if any(type(e) is not str for e in claim.evidence_ids):
        raise ValueError("invalid_claim_evidence")
    if len(set(claim.evidence_ids)) != len(claim.evidence_ids):
        raise ValueError("duplicate_claim_evidence")
    scopes = {}
    evidence = {}
    for event in ledger.events:
        body = json.loads(event.body_json)
        if event.kind == "start":
            scopes[event.episode] = (body["service"], body["version"])
        elif event.kind == "step":
            evidence[event.event_id] = scopes[event.episode]
    if any(e not in evidence or evidence[e][0] != claim.service for e in claim.evidence_ids):
        raise ValueError("unknown_or_out_of_scope_evidence")
    if not any(evidence[e][1] == claim.version for e in claim.evidence_ids):
        raise ValueError("missing_current_version_evidence")
    existing = {c.claim_id: c for c in previous}
    if claim.claim_id in existing:
        raise ValueError("duplicate_claim_id")
    if claim.revises is not None:
        if claim.revises not in existing or existing[claim.revises].service != claim.service:
            raise ValueError("invalid_revision_lineage")
        old = existing[claim.revises]
        if any(c.revises == old.claim_id for c in previous):
            raise ValueError("revision_already_superseded")
        if not set(claim.evidence_ids) - set(old.evidence_ids):
            raise ValueError("revision_missing_new_evidence")
        if (claim.predicted_action_effect, claim.validity, claim.applicability, claim.version) == (
                old.predicted_action_effect, old.validity, old.applicability, old.version):
            raise ValueError("trivial_revision")
    return previous + (claim,)
