"""Offline regression tests for Study I memory policies and scored main screen."""
import json
import unittest
from dataclasses import replace

from repair_to_generalization.experience_ledger import Access, Ledger, canonical, execute
from repair_to_generalization.g6_live import ACTION_ONLY_REVISION, parse_linked
from repair_to_generalization.g6_qualification import Meter
from repair_to_generalization.g7_compare import (
    MAX_CALLS, MAIN_ORDER, MainRun, compress, score_results, setup_for, validate_split,
)
from repair_to_generalization.g7_source import FrozenSource
from repair_to_generalization.linked_world import reference_policy
from repair_to_generalization.savant_pilot import InvalidModelResponse
from repair_to_generalization.study_design import (
    SERVICES, RETRIEVAL, MAX_INPUT_BYTES, acquisition_cases, combined_cases, stratum,
)
from repair_to_generalization.study_memory import (
    MemoryPolicies, compression_input, compression_payload, parse_compression, tokens,
)


def model_response(body):
    return {"choices": [{"finish_reason": "stop", "message": {"content": canonical(body)}}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 40,
                      "prompt_tokens_details": {"cached_tokens": 0}}}


def reference(public):
    return reference_policy(public["current"], public["current_interactions"])


def fake_compress(payload):
    scope = json.loads(payload["messages"][1]["content"])
    version = {}
    aliases = {}
    for event in scope["public_interactions"]:
        if event["kind"] == "start":
            version[event["episode"]] = event["body"]["version"]
        elif event["kind"] == "step":
            aliases.setdefault(version[event["episode"]], event["event_id"])
    return model_response({"lesson": "Offline test lesson, not live evidence.",
                           "evidence": list(aliases.values())})


class StudyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cases = acquisition_cases()[:50]
        ledger = Ledger()
        records = []
        for i, case in enumerate(cases, 1):
            ledger, record = execute(case, ledger, i, True, Access.CURRENT, reference)
            records.append(record)
        cls.source = FrozenSource("a" * 64, ledger, cases, tuple(records), 0.5, 1.0)
        cls.events = []
        meter = Meter(0.15, fake_compress, cls.events.append,
                      max_calls=5, max_input_bytes=MAX_INPUT_BYTES)
        cls.policies = compress(cls.source, meter, cls.events.append)

    def test_frozen_compression_evidence_and_scope(self):
        ledger = self.source.ledger
        self.assertEqual(len(self.policies.lessons), 5)
        self.assertEqual(self.events[-1]["record_type"], "g7_memory_frozen")
        self.assertEqual(self.events[-1]["source_ledger_head"], ledger.head)
        service = SERVICES[0]
        public, aliases, versions = compression_input(ledger, service)
        self.assertEqual(set(versions.values()), {"v1", "v2"})
        self.assertTrue(all(row["body"].get("service", service) == service
                            for row in public["public_interactions"]))
        for candidate in (
            {"lesson": "fabricated", "evidence": ["fake"]},
            {"lesson": "one version", "evidence": [next(iter(aliases))]},
            {"lesson": "x" * 1601, "evidence": list(aliases)[:2]},
        ):
            with self.assertRaises(InvalidModelResponse):
                parse_compression(model_response(candidate), ledger, service)
        wrong = replace(self.policies.lessons[0], evidence_ids=self.policies.lessons[1].evidence_ids)
        with self.assertRaisesRegex(ValueError, "invalid_frozen_compression"):
            MemoryPolicies(ledger, (wrong,) + self.policies.lessons[1:])

    def test_four_arms_differ_only_in_memory_and_retrieval_exactness(self):
        public = {"instrument": "test", "contract": "same", "current": {
            "service": SERVICES[0], "version": "v2", "requested": ["refund", "export"]},
            "current_interactions": [], "prior_interactions": []}
        requests = {arm: self.policies.request(arm, public) for arm in ("A0", "A1", "A2", "A3")}
        decoded = {arm: json.loads(p["messages"][1]["content"]) for arm, p in requests.items()}
        memories = {arm: p.pop("prior_interactions") for arm, p in decoded.items()}
        self.assertTrue(all(p == decoded["A0"] for p in decoded.values()))
        self.assertEqual(memories["A0"], [])
        self.assertEqual(len(memories["A1"]), 5)
        self.assertEqual(memories["A3"], self.source.ledger.public())
        query = tokens([public["current"], public["current_interactions"]])
        groups = self.policies._episodes
        ranked = sorted(range(len(groups)), key=lambda i: (-len(query & tokens(groups[i])), i))
        chosen = sorted(ranked[:RETRIEVAL["top_k"]])
        expected = [e for i in chosen for e in groups[i]]
        self.assertLessEqual(len(canonical(expected).encode()), RETRIEVAL["max_serialized_bytes"])
        self.assertEqual(memories["A2"], expected)
        fetched = self.policies.retrieve(public)
        fetched[0]["body"]["tampered"] = True
        self.assertEqual(self.policies.retrieve(public), expected)

    def test_all_main_episodes_replay_and_never_learn(self):
        events = []
        def fake(payload):
            request = json.loads(payload["messages"][1]["content"])
            self.assertNotIn("case_id", request["current"])
            self.assertNotIn("phase", request["current"])
            self.assertTrue(all("safe_success" not in e["body"]
                                for e in request["prior_interactions"]))
            return model_response({"action": reference(request).value})

        meter = Meter(3.0, fake, events.append,
                      parser=lambda d: parse_linked(d, ACTION_ONLY_REVISION),
                      max_calls=MAX_CALLS, max_input_bytes=MAX_INPUT_BYTES)
        run = MainRun(self.source, self.policies, meter, events.append)
        run.run()
        result = score_results(run.rows)
        self.assertEqual(result["scored_episodes"], 144)
        self.assertTrue(result["main_complete"])
        self.assertFalse(result["main_screen_positive"])
        self.assertFalse(result["g7_pass"])
        self.assertTrue(all(arm["safe_successes"] == 36 for arm in result["by_arm"].values()))
        self.assertTrue(all(row["replay_valid"] and not row["record"]["learning"]
                            and row["record"]["ledger_head"] == self.source.ledger.head
                            for row in run.rows))
        self.assertEqual(len([r for r in events if r["record_type"] == "call_started"]), meter.budget.calls)

    def test_primary_gate_requires_complete_positive_and_zero_regression(self):
        cases = {c.case_id: c for c in combined_cases("main")}
        rows = []
        for rep, case_id, arm in MAIN_ORDER:
            group = stratum(cases[case_id])
            success = group == "permission_control" or arm in ("A2", "A3")
            rows.append({"repetition": rep, "case_id": case_id, "arm": arm, "stratum": group,
                         "outcome": {"completed": True, "safe_success": success,
                                     "unsafe": False, "actions": 4},
                         "usage": {"model_calls": 4, "prompt_tokens": 100,
                                   "cached_tokens": 0, "completion_tokens": 20,
                                   "max_input_tokens": 50, "accounted_usd": 0.001}})
        result = score_results(rows)
        self.assertTrue(result["main_screen_positive"])
        self.assertFalse(result["g7_pass"])
        self.assertEqual(result["unrelated_history_control"], "pending")
        self.assertIsNone(score_results(rows[:-1])["main_screen_positive"])
        bad = next(row for row in rows if row["arm"] == "A3" and row["stratum"] != "permission_control")
        bad["outcome"]["unsafe"] = True
        bad["outcome"]["safe_success"] = False
        self.assertFalse(score_results(rows)["main_screen_positive"])
        bad["outcome"]["unsafe"] = False
        bad["outcome"]["safe_success"] = True
        bad = next(row for row in rows if row["arm"] == "A3" and row["stratum"] == "permission_control")
        bad["outcome"]["safe_success"] = False
        self.assertFalse(score_results(rows)["main_screen_positive"])

    def test_source_split_and_frozen_protocol_metadata(self):
        validate_split(self.source)
        setup = setup_for(self.source, "test", 1.52, 2.5)
        self.assertEqual(setup["source_report_sha256"], self.source.sha256)
        self.assertEqual(setup["order"], MAIN_ORDER)
        self.assertEqual(setup["compression_calls"], 5)
        self.assertEqual(len(setup["decision_prompt_sha256"]), 64)
        overlap = replace(self.source, cases=self.source.cases + combined_cases("main")[:1])
        with self.assertRaisesRegex(ValueError, "study_split_overlap"):
            validate_split(overlap)


if __name__ == "__main__":
    unittest.main()
