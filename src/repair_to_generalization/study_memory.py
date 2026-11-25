"""Study I frozen memory policies; only A1 uses model-authored compression.

All evidence comes from the same verified source ledger. Retrieval is a
deterministic lexical baseline, not a task-solving oracle.
"""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from typing import Any

from .experience_ledger import Ledger, canonical, hash_value
from .g6_live import ACTION_ONLY_REVISION, payload_for, unique_object
from .savant_pilot import InvalidModelResponse, Usage
from .study_design import RETRIEVAL, SERVICES

MEMORY_REVISION = "study_i_memory_v1"
COMPRESSION_SCHEMA = {
    "type": "object",
    "properties": {
        "lesson": {"type": "string"},
        "evidence": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["lesson", "evidence"],
    "additionalProperties": False,
}
COMPRESSION_SYSTEM = (
    "Compress your own public interaction history for this service into one short "
    "advisory lesson (at most 1600 characters). Include version-specific "
    "applicability, evidence of changed assumptions, uncertainty and observed "
    "failure patterns where supported. Never invent hidden facts or evaluator "
    "feedback. Cite 1 to 12 distinct step aliases from the supplied history "
    "including evidence from each observed version. Return only the JSON schema."
)


def episodes(ledger: Ledger) -> tuple[tuple[dict, ...], ...]:
    groups: dict[int, list[dict]] = {}
    for event in ledger.events:
        groups.setdefault(event.episode, []).append(event.public())
    return tuple(tuple(group) for group in groups.values())


def compression_input(ledger: Ledger, service: str):
    if service not in SERVICES:
        raise ValueError("unknown_compression_service")
    public = []
    aliases = {}
    versions = {}
    for group in episodes(ledger):
        start = group[0]
        if start["body"]["service"] != service:
            continue
        version = start["body"]["version"]
        for index, event in enumerate(group):
            row = dict(event)
            alias = f"e{event['episode']}_{index}"
            row["event_id"] = alias
            public.append(row)
            if event["kind"] == "step":
                aliases[alias] = event["event_id"]
                versions[alias] = version
    if not aliases:
        raise ValueError("empty_compression_scope")
    return {"service": service, "public_interactions": public}, aliases, versions


def compression_payload(ledger: Ledger, service: str):
    public, _, _ = compression_input(ledger, service)
    request = payload_for(public, ACTION_ONLY_REVISION)
    request["messages"][0]["content"] = COMPRESSION_SYSTEM
    request["response_format"]["json_schema"] = {
        "name": "compressed_experience", "strict": True, "schema": COMPRESSION_SCHEMA,
    }
    return request


@dataclass(frozen=True)
class CompressedLesson:
    service: str
    lesson: str
    evidence_ids: tuple[str, ...]

    def public(self):
        return {"kind": "agent_authored_compressed_lesson", "body": asdict(self)}


def parse_compression(data: dict[str, Any], ledger: Ledger,
                      service: str) -> tuple[CompressedLesson, Usage]:
    try:
        row = data["usage"]
        prompt, completion = row["prompt_tokens"], row["completion_tokens"]
        cached = (row.get("prompt_tokens_details") or {}).get("cached_tokens", 0)
        if any(type(v) is not int or v < 0 for v in (prompt, completion, cached)) or cached > prompt:
            raise ValueError("invalid_usage")
        usage = Usage(prompt, cached, completion)
    except (KeyError, TypeError, AttributeError, ValueError) as error:
        raise InvalidModelResponse("compression_invalid_usage") from error
    try:
        choices = data["choices"]
        if not isinstance(choices, list) or len(choices) != 1:
            raise ValueError("invalid_choice_count")
        choice = choices[0]
        if choice["finish_reason"] == "length":
            raise InvalidModelResponse("output_limit", usage)
        if choice["finish_reason"] != "stop":
            raise InvalidModelResponse("unexpected_finish_reason", usage)
        body = json.loads(choice["message"]["content"], object_pairs_hook=unique_object)
        if not isinstance(body, dict) or set(body) != {"lesson", "evidence"}:
            raise ValueError("invalid_compression_fields")
        lesson, refs = body["lesson"], body["evidence"]
        if type(lesson) is not str or not lesson.strip() or len(lesson) > 1600:
            raise ValueError("invalid_compression_length")
        _, aliases, versions = compression_input(ledger, service)
        if (type(refs) is not list or not 1 <= len(refs) <= 12
                or any(type(ref) is not str or ref not in aliases for ref in refs)
                or len(set(refs)) != len(refs)):
            raise ValueError("invalid_compression_evidence")
        if {versions[ref] for ref in refs} != set(versions.values()):
            raise ValueError("missing_version_evidence")
        return CompressedLesson(service, lesson, tuple(aliases[ref] for ref in refs)), usage
    except InvalidModelResponse:
        raise
    except (KeyError, TypeError, ValueError, IndexError) as error:
        raise InvalidModelResponse("invalid_compression:" + str(error), usage) from error


def tokens(value) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", canonical(value).lower()))


class MemoryPolicies:
    def __init__(self, ledger: Ledger, lessons: tuple[CompressedLesson, ...]):
        if tuple(lesson.service for lesson in lessons) != SERVICES:
            raise ValueError("incomplete_compression_set")
        self.ledger = ledger
        self.lessons = lessons
        self._episodes = episodes(ledger)
        self._terms = tuple(tokens(group) for group in self._episodes)
        evidence_by_service = {}
        for service in SERVICES:
            _, aliases, _ = compression_input(ledger, service)
            evidence_by_service[service] = set(aliases.values())
        for lesson in lessons:
            if (type(lesson.lesson) is not str or not lesson.lesson.strip() or len(lesson.lesson) > 1600
                    or not lesson.evidence_ids
                    or len(lesson.evidence_ids) > 12
                    or len(set(lesson.evidence_ids)) != len(lesson.evidence_ids)
                    or not set(lesson.evidence_ids) <= evidence_by_service[lesson.service]):
                raise ValueError("invalid_frozen_compression")

    def retrieve(self, public: dict) -> list[dict]:
        query = tokens([public["current"], public["current_interactions"]])
        ranked = sorted(range(len(self._episodes)),
                        key=lambda i: (-len(query & self._terms[i]), i))
        selected = []
        for i in ranked:
            if len(selected) == RETRIEVAL["top_k"]:
                break
            candidate = sorted(selected + [i])
            rows = [event for j in candidate for event in self._episodes[j]]
            if len(canonical(rows).encode("utf-8")) <= RETRIEVAL["max_serialized_bytes"]:
                selected = candidate
        return [event for i in selected for event in self._episodes[i]]

    def memory(self, arm: str, public: dict) -> list[dict]:
        if arm == "A0":
            return []
        if arm == "A1":
            return [lesson.public() for lesson in self.lessons]
        if arm == "A2":
            return self.retrieve(public)
        if arm == "A3":
            return self.ledger.public()
        raise ValueError("unknown_study_arm")

    def request(self, arm: str, public: dict) -> dict:
        if public["prior_interactions"]:
            raise ValueError("unexpected_existing_memory")
        return payload_for({**public, "prior_interactions": self.memory(arm, public)},
                           ACTION_ONLY_REVISION)

    def freeze(self):
        return {
            "revision": MEMORY_REVISION,
            "source_ledger_head": self.ledger.head,
            "source_events": len(self.ledger.events),
            "source_sha256": hash_value(self.ledger.public()),
            "compression_prompt_sha256": hash_value({"system": COMPRESSION_SYSTEM,
                                                       "schema": COMPRESSION_SCHEMA}),
            "compressed_lessons": [asdict(lesson) for lesson in self.lessons],
            "compressed_sha256": hash_value([asdict(lesson) for lesson in self.lessons]),
            "retrieval_sha256": hash_value(RETRIEVAL),
        }
