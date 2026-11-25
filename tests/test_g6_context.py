"""Offline tests for source revalidation and 10K protocol qualification."""
import json
import tempfile
import unittest
from pathlib import Path

from repair_to_generalization.experience_ledger import canonical
from repair_to_generalization.g6_context import (
    ACQUISITION_COUNT, MAX_CALLS, PROBE_COUNT, ContextRun, additional_cases, load_source,
)
from repair_to_generalization.g6_live import (
    ACTION_ONLY_REVISION, MAX_INPUT_BYTES, SEED, DevelopmentRun, manifest, parse_linked,
)
from repair_to_generalization.g6_qualification import Meter
from repair_to_generalization.linked_world import development_stream, reference_policy
from repair_to_generalization.savant_pilot import InvalidModelResponse


def fake_provider(payload, tokens=100):
    public = json.loads(payload["messages"][1]["content"])
    action = reference_policy(public["current"], public["current_interactions"])
    return {"choices": [{"finish_reason": "stop", "message": {
        "content": json.dumps({"action": action.value})}}],
        "usage": {"prompt_tokens": tokens, "completion_tokens": 40,
                  "prompt_tokens_details": {"cached_tokens": 0}}}


def meter_for(request, events, max_calls=288):
    return Meter(2.0, request, events.append,
                 parser=lambda data: parse_linked(data, ACTION_ONLY_REVISION),
                 max_calls=max_calls, max_input_bytes=MAX_INPUT_BYTES)


class ContextTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cases = development_stream(SEED)
        cls.events = [manifest(cases, "test-commit", 0.81, 1.5, ACTION_ONLY_REVISION)]
        meter = meter_for(fake_provider, cls.events)
        run = DevelopmentRun(meter, cls.events.append, cases, ACTION_ONLY_REVISION)
        run.run()
        run.verify()
        cls.events.append({
            "record_type": "g6_linked_terminal", "status": "development_complete",
            "replay_valid": True, "error": None, "calls": meter.budget.calls,
            "reported_usd": meter.reported_usd, "accounted_usd": meter.budget.spent_usd,
            "unknown_reserved_usd": meter.unknown_reserved_usd, **run.summary(),
        })
        cls.expected_ledger = run.ledgers["A3"]

    def read(self, rows):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "source.jsonl"
            path.write_text("".join(canonical(row) + "\n" for row in rows), encoding="utf-8")
            return load_source(path)

    def test_source_replays_every_call_and_event(self):
        source = self.read(self.events)
        self.assertEqual(source.ledger, self.expected_ledger)
        self.assertEqual(len(source.records), 12)
        self.assertEqual(len(source.sha256), 64)
        self.assertGreater(source.accounted_usd, 0)

    def test_tampered_public_observation_rejected(self):
        rows = json.loads(canonical(self.events))
        step = next(row for row in rows if row["record_type"] == "linked_public_step")
        step["observation"]["invented"] = True
        with self.assertRaisesRegex(ValueError, "source_step_mismatch"):
            self.read(rows)

    def test_tampered_prompt_or_terminal_rejected(self):
        rows = json.loads(canonical(self.events))
        call = next(row for row in rows if row["record_type"] == "call_started")
        call["request"]["messages"][1]["content"] = "{}"
        with self.assertRaisesRegex(ValueError, "source_request_mismatch"):
            self.read(rows)
        rows = json.loads(canonical(self.events))
        rows[-1]["accounted_usd"] += 0.01
        with self.assertRaisesRegex(ValueError, "source_terminal_mismatch"):
            self.read(rows)

    def test_extension_replays_with_frozen_nonlearning_probes(self):
        source = self.read(self.events)
        events = []
        # Simulated provider counts test qualification logic only, never real 10K evidence.
        meter = meter_for(lambda p: fake_provider(p, 100), events, MAX_CALLS)
        run = ContextRun(source, meter, events.append)
        run.run()
        run.verify()
        result = run.summary()
        self.assertEqual(result["acquisition_episodes"], ACQUISITION_COUNT)
        self.assertEqual(result["probe_episodes"], PROBE_COUNT)
        self.assertEqual(result["probe_safe_successes"], PROBE_COUNT)
        self.assertFalse(result["tier_qualified"])
        self.assertEqual(result["qualifying_10k_episodes"], 0)
        starts = [r for r in events if r["record_type"] == "context_episode_started" and not r["learning"]]
        self.assertEqual(len({r["source_ledger_head"] for r in starts}), 1)
        self.assertEqual(starts[0]["source_ledger_head"], run.ledger.head)
        self.assertEqual(len(run.cases), 11)
        self.assertTrue(all(not record.learning for record in run.records[-3:]))
        self.assertTrue(all(json.loads(record.outcome_json)["safe_success"] for record in run.records))

    def test_measured_threshold_and_completion_are_both_required(self):
        source = self.read(self.events)
        run = ContextRun(source, None, lambda _: None)
        run.context = [{"episode": i, "learning": False, "first_input_tokens": 10_000,
                        "max_input_tokens": 10_200, "completed": True,
                        "safe_success": False, "unsafe": True} for i in range(3)]
        self.assertTrue(run.summary()["tier_qualified"])
        run.context[0]["first_input_tokens"] = 9999
        self.assertFalse(run.summary()["tier_qualified"])
        run.context[0]["first_input_tokens"] = 10_000
        run.context[1]["completed"] = False
        self.assertFalse(run.summary()["tier_qualified"])

    def test_truncation_preserves_source_and_replay(self):
        source = self.read(self.events)
        events = []
        def fail(payload):
            data = fake_provider(payload)
            data["choices"][0]["finish_reason"] = "length"
            data["usage"]["completion_tokens"] = 8192
            return data
        run = ContextRun(source, meter_for(fail, events, MAX_CALLS), events.append)
        with self.assertRaisesRegex(InvalidModelResponse, "output_limit"):
            run.run()
        run.verify()
        self.assertEqual(run.ledger, source.ledger)
        self.assertEqual(run.records, [])
        self.assertFalse(run.summary()["tier_qualified"])


if __name__ == "__main__":
    unittest.main()
