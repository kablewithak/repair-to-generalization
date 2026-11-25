"""Offline tests for the frozen Study I design and source replay."""
import json
import tempfile
import unittest
from pathlib import Path

from repair_to_generalization.experience_ledger import canonical
from repair_to_generalization.g6_live import ACTION_ONLY_REVISION, parse_linked
from repair_to_generalization.g6_qualification import Meter
from repair_to_generalization.g7_source import (
    MAX_CALLS, SourceRun, load_source, setup_for, should_stop_acquisition,
)
from repair_to_generalization.linked_world import LinkedWorld, reference_policy, MAX_ACTIONS
from repair_to_generalization.study_design import (
    ARMS, CONTEXT_TARGET, MAX_INPUT_BYTES, MIN_ACQUISITION, MAX_ACQUISITION,
    acquisition_cases, combined_cases, case_signature, episode_order, split_manifest,
)


def fake_provider(payload):
    public = json.loads(payload["messages"][1]["content"])
    action = reference_policy(public["current"], public["current_interactions"])
    # Synthetic token counts exercise the threshold code, not provider capacity.
    tokens = min(CONTEXT_TARGET, len(json.dumps(payload)) // 2)
    return {"choices": [{"finish_reason": "stop", "message": {
        "content": json.dumps({"action": action.value})}}],
        "usage": {"prompt_tokens": tokens, "completion_tokens": 40,
                  "prompt_tokens_details": {"cached_tokens": 0}}}


class DesignTests(unittest.TestCase):
    def test_splits_disjoint_and_solvable(self):
        source = acquisition_cases()
        groups = [source] + [combined_cases(name) for name in
                             ("calibration", "main", "confirmation")]
        cases = [case for group in groups for case in group]
        self.assertEqual([len(g) for g in groups], [60, 3, 12, 12])
        self.assertEqual(len({case.case_id for case in cases}), len(cases))
        self.assertEqual(len({case_signature(case) for case in cases}), len(cases))
        self.assertTrue(all(case.request_refund != case.request_export for case in source))
        self.assertTrue(all(case.request_refund and case.request_export
                            for group in groups[1:] for case in group))
        self.assertEqual(len({(case.service, case.version) for case in source[:MIN_ACQUISITION]}), 10)
        for case in cases:
            world = LinkedWorld(case)
            history = []
            while not world.finished and world.actions < MAX_ACTIONS:
                action = reference_policy(world.view(), history)
                observation = world.step(action)
                history.append({"action": action.value, "observation": observation})
            self.assertTrue(world.outcome()["safe_success"], case.case_id)

    def test_deterministic_split_hashes_and_rotating_order(self):
        a, b = split_manifest(), split_manifest()
        self.assertEqual(a, b)
        self.assertEqual(len(set(a["split_sha256"].values())), 4)
        for split in ("main", "confirmation"):
            order = episode_order(split)
            self.assertEqual(len(order), 144)
            self.assertEqual(len(set(order)), 144)
            self.assertEqual({row[2] for row in order[:4]}, set(ARMS))
            self.assertNotEqual(order[0][2], order[48][2])

    def test_source_stop_rule_uses_context_not_outcomes(self):
        self.assertFalse(should_stop_acquisition(MIN_ACQUISITION - 1, CONTEXT_TARGET))
        self.assertFalse(should_stop_acquisition(MIN_ACQUISITION, CONTEXT_TARGET - 1))
        self.assertTrue(should_stop_acquisition(MIN_ACQUISITION, CONTEXT_TARGET))


class SourceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.events = [setup_for("test-commit", 1.0, 2.0)]
        meter = Meter(2.0, fake_provider, cls.events.append,
                      parser=lambda data: parse_linked(data, ACTION_ONLY_REVISION),
                      max_calls=MAX_CALLS, max_input_bytes=MAX_INPUT_BYTES)
        cls.run_result = SourceRun(meter, cls.events.append)
        cls.run_result.run()
        cls.run_result.verify()
        summary = cls.run_result.summary()
        cls.events.append({"record_type": "study_source_terminal", "error": None,
                           "replay_valid": True, "calls": meter.budget.calls,
                           "reported_usd": meter.reported_usd,
                           "unknown_reserved_usd": meter.unknown_reserved_usd,
                           "accounted_usd": meter.budget.spent_usd, **summary})

    def load(self, events):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "source.jsonl"
            path.write_text("".join(canonical(row) + "\n" for row in events), encoding="utf-8")
            return load_source(path)

    def test_complete_qualified_ledger_replays(self):
        source = self.load(self.events)
        self.assertEqual(source.ledger, self.run_result.ledger)
        self.assertTrue(MIN_ACQUISITION <= len(source.cases) <= MAX_ACQUISITION)
        self.assertTrue(all(record.learning for record in source.records))
        self.assertEqual(self.run_result.summary()["qualifying_25k_episodes"], 3)
        starts = [row for row in self.events if row["record_type"] ==
                  "study_source_episode_started" and not row["learning"]]
        self.assertEqual(len({row["source_ledger_head"] for row in starts}), 1)
        self.assertEqual(starts[0]["source_ledger_head"], source.ledger.head)
        self.assertTrue(all(event.episode <= len(source.cases) for event in source.ledger.events))

    def test_no_evaluation_labels_in_prompt(self):
        for event in self.events:
            if event["record_type"] == "call_started":
                request = json.loads(event["request"]["messages"][1]["content"])
                self.assertNotIn("case_id", request["current"])
                self.assertNotIn("phase", request["current"])
                self.assertTrue(all("safe_success" not in e["body"] for e in request["prior_interactions"]))

    def test_tampered_trace_and_manifest_rejected(self):
        rows = list(self.events)
        index = next(i for i, row in enumerate(rows)
                     if row["record_type"] == "study_source_public_step")
        rows[index] = {**rows[index], "observation": {"invented": True}}
        with self.assertRaisesRegex(ValueError, "study_source_step_mismatch"):
            self.load(rows)
        rows = list(self.events)
        rows[0] = {**rows[0], "split_manifest_sha256": "invalid"}
        with self.assertRaisesRegex(ValueError, "study_source_manifest_mismatch"):
            self.load(rows)

    def test_probe_threshold_not_inferred_from_source_length(self):
        rows = list(self.events)
        rows[-1] = {**rows[-1], "tier_qualified": False}
        with self.assertRaisesRegex(ValueError, "study_source_not_qualified"):
            self.load(rows)


if __name__ == "__main__":
    unittest.main()
