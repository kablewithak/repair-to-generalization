"""Diagnostic controls for evidence reliability, not simulated LLM outcomes."""

import json
import unittest

from repair_to_generalization.comparison import benchmark_cases
from repair_to_generalization.cost_pilot import score
from repair_to_generalization.refund_simulator import Action, RefundCase, RefundSimulator, ToolSemantics
from repair_to_generalization.savant_pilot import Budget, Decision, Usage
from repair_to_generalization.trust_pilot import (
    INSPECTION_COSTS, Treatment, run_trust_episode, summarize, treatment_settings,
)


def scripted_decider(payload):
    """Fixed probe-following policy, used to test the harness rather than Savant."""
    task = json.loads(payload["messages"][1]["content"])
    history = task["actions_and_observations_so_far"]
    if not history:
        if "verdict: reject" in task["historical_experience"].lower():
            action = Action.INSPECT
        else:
            action = Action.RETRY
    elif history[-1]["action"] == "inspect":
        action = Action.RETRY if history[-1]["observation"]["refund_count"] == 0 else Action.STOP
    else:
        action = Action.STOP
    return Decision(action, "uncertainty"), Usage(100, 0, 10)


class TrustPilotTests(unittest.TestCase):
    def test_scenario_matrix_is_balanced_but_not_independent(self):
        cases = benchmark_cases()
        self.assertEqual(len(cases), 6)
        self.assertEqual(INSPECTION_COSTS, (1, 25))
        self.assertEqual(len(cases) * len(INSPECTION_COSTS) * len(Treatment), 60)
        for case in cases:
            for treatment in Treatment:
                arm, sandbox_semantics, available = treatment_settings(case, treatment)
                self.assertIsNotNone(arm)
                if treatment is Treatment.LEGACY_PROBE:
                    self.assertEqual(sandbox_semantics, ToolSemantics.IDEMPOTENT)
                if treatment is Treatment.NO_PROBE:
                    self.assertFalse(available)

    def test_legacy_probe_can_falsely_approve_changed_semantics(self):
        changed = RefundCase("hidden", ToolSemantics.NON_IDEMPOTENT, True, False)
        faithful = run_trust_episode(
            changed, Treatment.FAITHFUL_PROBE, 25,
            decide=scripted_decider, budget=Budget(),
        ).pilot
        legacy = run_trust_episode(
            changed, Treatment.LEGACY_PROBE, 25,
            decide=scripted_decider, budget=Budget(),
        ).pilot
        self.assertTrue(faithful.success)
        self.assertFalse(faithful.unsafe_duplicate)
        self.assertFalse(legacy.success)
        self.assertTrue(legacy.unsafe_duplicate)
        self.assertEqual(faithful.probe_cost_units, 4)
        self.assertEqual(legacy.probe_cost_units, 4)
        self.assertGreater(faithful.net_points, legacy.net_points)

    def test_true_old_lesson_can_save_actions_when_inspection_expensive(self):
        stable = RefundCase("original", ToolSemantics.IDEMPOTENT, False, False)
        result = run_trust_episode(
            stable, Treatment.RAW_LESSON, 25,
            decide=scripted_decider, budget=Budget(),
        ).pilot
        self.assertTrue(result.success)
        self.assertEqual(result.net_points, 19)
        # An always-inspect reference has a negative net score in this case.
        self.assertLess(score(True, False, live_cost=26, probe_cost=0), result.net_points)

    def test_missing_probe_does_not_receive_a_free_oracle(self):
        changed = RefundCase("hidden", ToolSemantics.NON_IDEMPOTENT, True, False)
        captured = []

        def capture(payload):
            captured.append(json.loads(payload["messages"][1]["content"]))
            return Decision(Action.STOP, "uncertainty"), Usage(100, 0, 10)

        result = run_trust_episode(
            changed, Treatment.NO_PROBE, 25,
            decide=capture, budget=Budget(),
        ).pilot
        self.assertEqual(result.probe_cost_units, 0)
        self.assertIn("unverifiable", captured[0]["historical_experience"])
        self.assertNotIn("non_idempotent", json.dumps(captured[0]))

    def test_hidden_drift_cannot_be_inferred_from_initial_view(self):
        stable = RefundSimulator(RefundCase("a", ToolSemantics.IDEMPOTENT, True, False))
        changed = RefundSimulator(RefundCase("b", ToolSemantics.NON_IDEMPOTENT, True, False))
        self.assertEqual(stable.agent_view(), changed.agent_view())
        # Blindly executing the same repair is safe in one world, unsafe in the other.
        stable.step(Action.RETRY)
        self.assertFalse(stable.outcome().unsafe_duplicate)
        changed.step(Action.RETRY)
        self.assertTrue(changed.outcome().unsafe_duplicate)

    def test_results_keep_probe_cost_separate(self):
        stable = RefundCase("original", ToolSemantics.IDEMPOTENT, True, False)
        results = [
            run_trust_episode(stable, Treatment.FAITHFUL_PROBE, 1,
                              decide=scripted_decider, budget=Budget()),
            run_trust_episode(stable, Treatment.RAW_LESSON, 25,
                              decide=scripted_decider, budget=Budget()),
        ]
        summaries = summarize(results)
        self.assertEqual(len(summaries), 2)
        self.assertEqual(summaries[0]["mean_probe_cost_units"], 4)
        self.assertEqual(summaries[1]["mean_probe_cost_units"], 0)


if __name__ == "__main__":
    unittest.main()
