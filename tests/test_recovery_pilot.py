"""Behavioral controls for realized paired trajectories, not model performance."""

import json
import unittest

from repair_to_generalization.recovery_pilot import (
    SharedPolicy, acquire_experience, recovery_payload, run_pair, summarize,
)
from repair_to_generalization.refund_simulator import Action, RefundCase, RefundSimulator, ToolSemantics
from repair_to_generalization.savant_pilot import Budget, Decision, InvalidModelResponse, Usage
from repair_to_generalization.trust_pilot import Treatment


USAGE = Usage(50, 0, 20)


def inspect_after_retry(payload):
    prompt = json.loads(payload["messages"][1]["content"])
    history = prompt["actions_and_observations_so_far"]
    action = (Action.RETRY, Action.INSPECT, Action.STOP)[len(history)]
    return Decision(action, "uncertainty"), USAGE


class RecoveryPilotTests(unittest.TestCase):
    def setUp(self):
        self.lesson = acquire_experience()

    def test_experience_is_from_executed_successful_traces_without_hidden_labels(self):
        self.assertEqual(self.lesson["source"], "guided successful tool executions")
        self.assertEqual(len(self.lesson["successful_examples"]), 2)
        for example in self.lesson["successful_examples"]:
            self.assertEqual(example["witnessed_outcome"], "completed_with_exactly_one_refund")
            self.assertEqual(
                [row["action"] for row in example["actions_and_observations"]],
                ["retry", "inspect", "stop"],
            )
            self.assertEqual(example["actions_and_observations"][1]["observation"]["refund_count"], 1)
        self.assertNotIn("committed_before_timeout", json.dumps(self.lesson))
        self.assertNotIn("non_idempotent", json.dumps(self.lesson))

    def test_hidden_drift_shares_samples_until_feedback_branches(self):
        budget = Budget(max_usd=0.25)
        decisions = []

        def decide(payload):
            decisions.append(payload)
            return inspect_after_retry(payload)

        policy = SharedPolicy(decide, budget)
        pair = run_pair(
            repetition=1, initially_committed=True, drift_visible=False,
            treatment=Treatment.RAW_LESSON, lesson=self.lesson, policy=policy,
        )
        self.assertTrue(pair.same_initial_prompt)
        self.assertEqual(pair.shared_decision_prefix, 2)
        self.assertEqual(pair.shared_observation_prefix, 1)
        self.assertTrue(pair.old_world.success)
        self.assertFalse(pair.changed_world.success)
        self.assertTrue(pair.changed_world.unsafe_duplicate)
        self.assertEqual(pair.changed_world.refunds_committed, 2)
        self.assertEqual(pair.old_world.model_decisions + pair.changed_world.model_decisions, 6)
        self.assertEqual(len(decisions), 4)  # first two shared; branches after inspect
        self.assertEqual(policy.cache_hits, 2)
        self.assertEqual(budget.calls, 4)
        self.assertGreater(budget.spent_usd, 0)
        self.assertNotIn("committed_before_timeout", json.dumps(decisions))

    def test_noncommitted_pair_reuses_full_trajectory_and_stays_safe(self):
        budget = Budget(max_usd=0.25)
        policy = SharedPolicy(inspect_after_retry, budget)
        pair = run_pair(
            repetition=1, initially_committed=False, drift_visible=False,
            treatment=Treatment.NO_MEMORY, lesson=self.lesson, policy=policy,
        )
        self.assertEqual(pair.shared_decision_prefix, 3)
        self.assertEqual(pair.shared_observation_prefix, 3)
        self.assertTrue(pair.old_world.success)
        self.assertTrue(pair.changed_world.success)
        self.assertFalse(pair.changed_world.unsafe_duplicate)
        self.assertEqual(budget.calls, 3)

    def test_faithful_probe_changes_visible_prompt_but_stale_probe_does_not(self):
        for treatment, expect_same, verdict in (
            (Treatment.FAITHFUL_PROBE, False, "reject"),
            (Treatment.LEGACY_PROBE, True, "accept"),
            (Treatment.NO_PROBE, True, "unverifiable"),
        ):
            with self.subTest(treatment=treatment):
                policy = SharedPolicy(inspect_after_retry, Budget(max_usd=0.25))
                pair = run_pair(
                    repetition=1, initially_committed=True, drift_visible=False,
                    treatment=treatment, lesson=self.lesson, policy=policy,
                )
                self.assertEqual(pair.same_initial_prompt, expect_same)
                self.assertEqual(pair.changed_world.probe_verdict, verdict)
                self.assertEqual(
                    pair.changed_world.probe_cost_units, 0 if treatment is Treatment.NO_PROBE else 4
                )
                for step in pair.changed_world.trace:
                    prompt = json.loads(step["request_payload"]["messages"][1]["content"])
                    self.assertEqual(prompt["historical_experience"]["sandbox_probe"]["parity_with_current_tool"], "unverified")

    def test_visible_drift_is_a_public_prompt_difference(self):
        policy = SharedPolicy(inspect_after_retry, Budget(max_usd=0.25))
        pair = run_pair(
            repetition=1, initially_committed=True, drift_visible=True,
            treatment=Treatment.NO_MEMORY, lesson=self.lesson, policy=policy,
        )
        self.assertFalse(pair.same_initial_prompt)
        self.assertEqual(pair.shared_decision_prefix, 0)

    def test_summary_separates_realized_failure_and_cost(self):
        policy = SharedPolicy(inspect_after_retry, Budget(max_usd=0.25))
        pair = run_pair(
            repetition=1, initially_committed=True, drift_visible=False,
            treatment=Treatment.LEGACY_PROBE, lesson=self.lesson, policy=policy,
        )
        result = summarize([pair])[0]
        self.assertEqual(result["v1_successes"], 1)
        self.assertEqual(result["v2_successes"], 0)
        self.assertEqual(result["v2_unsafe_duplicates"], 1)
        self.assertEqual(result["v2_incomplete"], 0)
        self.assertEqual(result["v2_stopped_unsuccessfully"], 1)
        self.assertEqual(result["mean_v2_probe_cost_units"], 4)
        self.assertEqual(result["mean_v2_live_cost_units"], 26)
        self.assertNotIn("mean_net_points", result)

    def test_invalid_model_response_is_billed_once_not_cached_or_retried(self):
        budget = Budget(max_usd=0.25)
        calls = []

        def invalid(_payload):
            calls.append(1)
            raise InvalidModelResponse("output_limit", Usage(200, 0, 8192))

        policy = SharedPolicy(invalid, budget)
        view = RefundSimulator(RefundCase("x", ToolSemantics.IDEMPOTENT, True, False)).agent_view()
        payload = recovery_payload(Treatment.NO_MEMORY, view, [], self.lesson, None)
        with self.assertRaisesRegex(InvalidModelResponse, "output_limit"):
            policy.choose(payload)
        self.assertEqual(calls, [1])
        self.assertEqual(budget.calls, 1)
        self.assertGreater(budget.spent_usd, 0)
        self.assertEqual(policy.cache_hits, 0)


if __name__ == "__main__":
    unittest.main()
