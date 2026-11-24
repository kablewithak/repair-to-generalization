"""G5.2 diagnostic reachability, causal ambiguity, pairing and replay tests."""
import copy
import json
import tempfile
import unittest
from pathlib import Path

from repair_to_generalization.savant_pilot import Budget, Usage
from repair_to_generalization.transfer_pilot import (
    ARMS, EXPORT_V1, RECONCILIATION_V2, ExportDecision, FrozenSources,
    load_resume, payload_for, run_comparison, setup_for, summarize, tool_cost,
)
from repair_to_generalization.transfer_reconciliation import (
    RECONCILIATION_CASES, ReconciliationWorld,
)
from repair_to_generalization.transfer_world import ExportAction

SOURCE = FrozenSources({"summary": "original"}, {"summary": "repaired"},
                       "a" * 64, "b" * 64, "c" * 64, "d" * 64)
USAGE = Usage(100, 0, 30)
A = ExportAction


def public_safe_policy(payload):
    """A memory-blind reference policy, using only the documented public contract."""
    row = json.loads(payload["messages"][1]["content"])
    view = row["tool_observation"]
    history = row["actions_and_observations_so_far"]
    if view["event"] in ("verified_provider_delivery_receipt",
                         "owner_cancelled_export; do_not_dispatch"):
        action = A.FINISH
    elif not history:
        action = A.AUDIT
    else:
        last_submit = max((i for i, x in enumerate(history)
                           if x["action"] == A.SUBMIT.value), default=-1)
        since = history[last_submit + 1:]
        required = 1 if view["receipt_source"] == "provider_ledger" else 2
        if sum(x["action"] == A.ADVANCE.value for x in since) < required:
            action = A.ADVANCE
        elif history[-1]["action"] != A.AUDIT.value:
            action = A.AUDIT
        elif history[-1]["observation"]["delivered"] == 0:
            action = A.SUBMIT
        else:
            action = A.FINISH
    return ExportDecision(action, "observed_state"), USAGE


class ReconciliationTests(unittest.TestCase):
    def test_live_count_does_not_prove_absence_of_pending_work(self):
        absent, pending = [ReconciliationWorld(c) for c in RECONCILIATION_CASES[:2]]
        self.assertEqual(absent.view(), pending.view())
        self.assertEqual(absent.step(A.AUDIT), pending.step(A.AUDIT))
        pending.step(A.SUBMIT)
        pending.step(A.ADVANCE)
        pending.step(A.FINISH)
        self.assertTrue(pending.outcome().unsafe_duplicate)
        self.assertFalse(pending.outcome().success)

    def test_replica_can_still_report_zero_after_settlement(self):
        absent, pending = [ReconciliationWorld(c) for c in RECONCILIATION_CASES[2:4]]
        self.assertEqual(absent.view(), pending.view())
        self.assertEqual(absent.step(A.AUDIT), pending.step(A.AUDIT))
        absent.step(A.ADVANCE)
        pending.step(A.ADVANCE)
        self.assertEqual(absent.step(A.AUDIT), pending.step(A.AUDIT))
        self.assertEqual(pending.outcome().jobs_delivered, 1)
        pending.step(A.ADVANCE)
        self.assertEqual(pending.step(A.AUDIT)["delivered"], 1)
        pending.step(A.FINISH)
        self.assertTrue(pending.outcome().success)

    def test_acknowledgement_has_no_count_and_repeat_is_unsafe(self):
        world = ReconciliationWorld(RECONCILIATION_CASES[0])
        self.assertIsNone(world.step(A.SUBMIT)["delivered"])
        self.assertEqual(world.step(A.AUDIT)["delivered"], 0)
        world.step(A.SUBMIT)
        self.assertTrue(world.outcome().unsafe_duplicate)

    def test_no_hidden_case_or_lesson_mutation_in_prompts(self):
        before = copy.deepcopy(SOURCE)
        for i in (0, 2):
            for arm in ARMS:
                a = payload_for(arm, SOURCE, ReconciliationWorld(RECONCILIATION_CASES[i]), [])
                b = payload_for(arm, SOURCE, ReconciliationWorld(RECONCILIATION_CASES[i + 1]), [])
                self.assertEqual(a, b)
                user = json.loads(a["messages"][1]["content"])
                self.assertNotIn("initial_state", json.dumps(user))
                self.assertNotIn("case_id", json.dumps(user))
                self.assertEqual(user["remaining_decisions"], 9)
                self.assertEqual(a["max_tokens"], 4096)
        self.assertEqual(SOURCE, before)

    def test_all_72_cases_solvable_without_memory_and_no_false_claim(self):
        events = []
        rows = run_comparison(SOURCE, 3, public_safe_policy,
                              Budget(max_usd=0.25, max_calls=648), events.append,
                              design=RECONCILIATION_V2)
        self.assertEqual(len(rows), 72)
        self.assertEqual(events[0]["version"], "export_reconciliation_v2")
        self.assertEqual(events[0]["max_actions"], 9)
        self.assertNotIn("max_actions", setup_for(SOURCE, 3, EXPORT_V1))
        self.assertTrue(all(r["result"]["success"] for r in rows))
        self.assertTrue(all(r["result"]["actions_used"] <= 9 for r in rows))
        self.assertTrue(all(tool_cost(r["result"]["trace"]) < 20 for r in rows))
        score = summarize(rows, 3, RECONCILIATION_V2)
        self.assertFalse(score["g52_candidate_criteria"]["net_success_advantage_over_each_control"])
        self.assertTrue(score["g52_candidate_criteria"]["no_paired_safety_regressions"])
        self.assertTrue(score["g52_candidate_criteria"]["all_repaired_negative_controls_safe_success"])
        self.assertEqual(score["original_vs_no_memory_on_transfer_cases"]["safe_success_wins"], 0)
        self.assertEqual(score["original_vs_no_memory_on_transfer_cases"]["safe_success_losses"], 0)

    def test_negative_control_unauthorized_dispatch(self):
        world = ReconciliationWorld(RECONCILIATION_CASES[-1])
        world.step(A.SUBMIT)
        self.assertTrue(world.outcome().unauthorized_dispatch)

    def test_resume_preserves_completed_episodes_and_rejects_wrong_design(self):
        events = []
        complete = run_comparison(SOURCE, 2, public_safe_policy,
                                  Budget(max_usd=0.25, max_calls=432), events.append,
                                  design=RECONCILIATION_V2)
        cut = next(i + 1 for i, row in enumerate(events)
                   if row["record_type"] == "g5_episode")
        rows = events[:cut] + [{"record_type": "run_error", "error": "output_limit"}]
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "partial.jsonl"
            path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            replay = load_resume(path, SOURCE, 2, RECONCILIATION_V2)
            self.assertEqual(len(replay.episodes), 1)
            with self.assertRaises(ValueError):
                load_resume(path, SOURCE, 2, EXPORT_V1)
            continued = run_comparison(SOURCE, 2, public_safe_policy,
                                       Budget(max_usd=0.25, max_calls=432), lambda row: None,
                                       replay=replay, design=RECONCILIATION_V2)
            self.assertEqual(continued, complete)


if __name__ == "__main__":
    unittest.main()
