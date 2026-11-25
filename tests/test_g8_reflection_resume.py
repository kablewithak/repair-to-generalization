"""Verify reflection-only checkpoint replay and a single bounded continuation."""
import json
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path

from repair_to_generalization.experience_ledger import Access, Ledger, canonical, execute
from repair_to_generalization.g6_qualification import Meter
from repair_to_generalization.g7_source import FrozenSource
from repair_to_generalization.g8_diagnostic import ARMS, reflect, score, setup_for
from repair_to_generalization.g8_reflection_resume import (
    FAILURE, SUFFIX, finish_reflection, load_prefix, retry_payload,
)
from repair_to_generalization.linked_world import reference_policy
from repair_to_generalization.savant_pilot import InvalidModelResponse
from repair_to_generalization.study_design import acquisition_cases, MAX_INPUT_BYTES
from test_g8_diagnostic import reflection_model


class ResumeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cases = acquisition_cases()[:50]
        ledger, records = Ledger(), []
        for i, case in enumerate(cases, 1):
            ledger, record = execute(case, ledger, i, True, Access.CURRENT,
                lambda public: reference_policy(public["current"], public["current_interactions"]))
            records.append(record)
        cls.source = FrozenSource("a"*64, ledger, cases, tuple(records), 0.5, 1.0)
        cls.rows = [dict(setup_for(cls.source, "test", 2.06, 1.25), started_utc="test")]
        calls = 0
        def fail_last(payload):
            nonlocal calls
            calls += 1
            result = reflection_model(payload)
            if calls == 10:
                body = json.loads(result["choices"][0]["message"]["content"])
                body["advice"] = "x" * 1001
                result["choices"][0]["message"]["content"] = canonical(body)
            return result
        meter = Meter(0.20, fail_last, lambda r: cls.rows.append({"meter_phase": "reflection", **r}),
                      max_calls=10, max_input_bytes=MAX_INPUT_BYTES)
        try:
            reflect(cls.source, meter, cls.rows.append)
        except InvalidModelResponse as error:
            failure = str(error)
        else:
            raise AssertionError("expected failure")
        cls.rows.append(dict(score([], []), record_type="g8_diagnostic_terminal", status="interrupted",
            error=failure, calls=meter.budget.calls, accounted_usd=meter.budget.spent_usd,
            reported_usd=meter.reported_usd, reflection_usd=meter.budget.spent_usd,
            unknown_reserved_usd=meter.unknown_reserved_usd))

    def load(self, rows):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "parent.jsonl"
            path.write_text("".join(canonical(row)+"\n" for row in rows), encoding="utf-8")
            return load_prefix(self.source, path, setup_for)

    def test_replay_preserves_nine_and_retries_only_last(self):
        prefix = self.load(self.rows)
        self.assertEqual(len(prefix.initial), 5)
        self.assertEqual(len(prefix.reviewed), 4)
        self.assertEqual(prefix.calls, 10)
        self.assertEqual(prefix.spent_usd, self.rows[-1]["accounted_usd"])
        original = self.rows[-3]["request"]
        retry = retry_payload(self.source, prefix)
        self.assertEqual(retry["response_format"], original["response_format"])
        self.assertEqual(retry["messages"][1], original["messages"][1])
        self.assertEqual(retry["messages"][0]["content"], original["messages"][0]["content"]+SUFFIX)
        seen = []
        def invoke(payload):
            seen.append(payload)
            return reflection_model(payload)
        events = []
        meter = Meter(0.20-prefix.spent_usd, invoke, events.append, max_calls=1,
                      max_input_bytes=MAX_INPUT_BYTES)
        snapshots, diagnostics = finish_reflection(self.source, prefix, meter, events.append, ARMS)
        self.assertEqual(seen, [retry])
        self.assertEqual(snapshots["A4_STALE"], [
            {"kind": "persistent_agent_claim", "body": c.public()} for c in prefix.initial])
        self.assertEqual([r["body"] for r in snapshots["A4"][:4]], [c.public() for c in prefix.reviewed])
        self.assertEqual(len(diagnostics), 5)
        self.assertTrue(score([], diagnostics)["claim_mechanism_observed"])
        self.assertFalse(score([], diagnostics)["g8_pass"])

    def test_reject_tampering_or_scored_prefix(self):
        for mutation in ("decision", "request", "cost", "terminal", "extra"):
            with self.subTest(mutation=mutation):
                rows = json.loads(canonical(self.rows))
                if mutation == "decision":
                    rows[2]["decision"]["advice"] += " altered"
                elif mutation == "request":
                    rows[1]["request"]["messages"][1]["content"] = "{}"
                elif mutation == "cost":
                    rows[2]["accounted_usd"] += 0.001
                elif mutation == "terminal":
                    rows[-1]["scored_episodes"] = 1
                else:
                    rows.insert(-1, {"record_type": "g8_episode_started"})
                with self.assertRaises(ValueError):
                    self.load(rows)

    def test_failed_retry_does_not_freeze_or_start_decisions(self):
        prefix = self.load(self.rows)
        def invalid(payload):
            result = reflection_model(payload)
            body = json.loads(result["choices"][0]["message"]["content"])
            body["advice"] = "x"*1001
            result["choices"][0]["message"]["content"] = canonical(body)
            return result
        events = []
        meter = Meter(0.20-prefix.spent_usd, invalid, events.append, max_calls=1,
                      max_input_bytes=MAX_INPUT_BYTES)
        with self.assertRaisesRegex(InvalidModelResponse, "claim_advice"):
            finish_reflection(self.source, prefix, meter, events.append, ARMS)
        self.assertEqual(meter.budget.calls, 1)
        self.assertFalse(any(r["record_type"] == "g8_claims_frozen" for r in events))
        self.assertEqual(events[-1]["error"], FAILURE)


if __name__ == "__main__":
    unittest.main()
