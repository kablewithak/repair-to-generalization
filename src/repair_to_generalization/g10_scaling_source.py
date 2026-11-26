"""Read-only replay of qualified nested histories and existing frozen memory."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

from .experience_ledger import canonical
from .g7_source import load_source
from .g7_compare import setup_for as g7_setup
from .g8_diagnostic import setup_for as g8_setup
from .g8_reflection_resume import load_prefix as load_claim_prefix
from .g8_network_resume import Prefix, ReplayBoundary, require, same
from .g10_50k_qualification import ExtensionRun, setup, BASE_SHA256
from .study_memory import CompressedLesson, MemoryPolicies

EXTENSION_SHA = "fbc3ea5efcbcea16e8e00f83385dc44689e18b95ff1b01d19f69f1730b370947"
COMPRESSION_SHA = "6b7e76bd5fe51f1a13e6aefea3d2c6dd212b81ba999e83dc2f4270548accb24e"
CLAIMS_SHA = "30a253cf8b864fdbc79f1ce69d36768c33adc14f2a0e29b841eb690ff6a3f0a0"


def read_report(path, expected):
    raw = path.read_bytes()
    require(hashlib.sha256(raw).hexdigest() == expected, "scale_report_identity")
    return [json.loads(line) for line in raw.splitlines()]


def replay_extension(base, rows, sha):
    manifest, terminal = rows[0], rows[-1]
    expected = setup(base, manifest["commit"], manifest["prior_spend_usd"],
                     manifest["authorized_cap_usd"])
    require(set(manifest) == set(expected) | {"started_utc"}
            and all(same(manifest[k],v) for k,v in expected.items()), "scale_extension_setup")
    require(terminal["record_type"] == "g10_50k_terminal" and terminal["tier_qualified"]
            and terminal["replay_valid"] and terminal["error"] is None
            and terminal["unknown_reserved_usd"] == 0 and terminal["transport_recoveries"] == 0,
            "scale_extension_unqualified")
    prefix = Prefix(sha,manifest,tuple(rows[1:-1]),terminal)
    meter = ReplayBoundary(prefix)
    run = ExtensionRun(base,meter,meter.emit)
    run.run()
    run.verify()
    require(next(meter.events,None) is None and meter.calls == terminal["calls"]
            and math.isclose(meter.spent_usd,terminal["accounted_usd"],abs_tol=1e-12)
            and math.isclose(meter.reported_usd,terminal["reported_usd"],abs_tol=1e-12),
            "scale_extension_accounting")
    require(all(same(terminal[k],v) for k,v in run.summary().items()), "scale_extension_summary")
    return run.ledger


def load_inputs(base_path: Path, extension_path: Path, compression_path: Path, claim_path: Path):
    base = load_source(base_path)
    require(base.sha256 == BASE_SHA256, "scale_base_identity")
    extension = read_report(extension_path,EXTENSION_SHA)
    full = replay_extension(base,extension,EXTENSION_SHA)
    compression = read_report(compression_path,COMPRESSION_SHA)
    meta = compression[0]
    require(all(same(meta[k],v) for k,v in g7_setup(
        base,meta["commit"],meta["prior_spend_usd"],meta["authorized_cap_usd"]).items()),
        "scale_compression_setup")
    freezes = [r for r in compression if r["record_type"]=="g7_memory_frozen"]
    require(len(freezes)==1,"scale_compression_freeze")
    frozen = freezes[0]
    lessons = tuple(CompressedLesson(r["service"],r["lesson"],tuple(r["evidence_ids"]))
                    for r in frozen["compressed_lessons"])
    policies = MemoryPolicies(base.ledger,lessons)
    require(same(frozen,dict(policies.freeze(),record_type="g7_memory_frozen",
                            source_report_sha256=base.sha256)), "scale_compression_identity")
    # Hash-locked original report; no failed or held-out evaluation rows enter memory.
    prefix = load_claim_prefix(base,claim_path,g8_setup)
    require(prefix.report_sha256==CLAIMS_SHA,"scale_claim_identity")
    claims = [{"kind":"persistent_agent_claim","body":c.public()} for c in prefix.reviewed]
    require(len(claims)==4,"scale_partial_claims")
    return base,full,lessons,claims,extension[-1]["accounted_usd"]
