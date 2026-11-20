"""Controls for a stochastic policy experiment, not model-quality claims."""

import unittest

from repair_to_generalization.decision_pilot import (
    INSPECTION_COST, REPLICATIONS, Sample, make_conditions, run_sample, summarize,
)
from repair_to_generalization.savant_pilot import Budget, InvalidModelResponse, Usage


class DecisionPilotTests(unittest.TestCase):
    def test_exact_prompt_dedup_eliminates_hidden_state_sampling_confound(self):
        conditions = make_conditions()
        self.assertEqual(INSPECTION_COST, 25)
        self.assertEqual(REPLICATIONS, 20)
        self.assertEqual(len(conditions), 10)
        all_assignments = [a for c in conditions for a in c.assignments]
        self.assertEqual(len(all_assignments), 30)
        shared = [
            c for c in conditions if any(a.case_id == "original-committed" and
                                          a.treatment == "no_memory" for a in c.assignments)
        ]
        self.assertEqual(len(shared), 1)
        self.assertTrue(any(a.case_id == "hidden-drift-not-committed" and
                            a.treatment == "no_memory" for a in shared[0].assignments))
        self.assertNotIn("case_id", str(shared[0].payload))

    def test_identical_probe_prompts_share_samples_across_treatments(self):
        conditions = make_conditions()
        shared = [
            c for c in conditions if any(a.case_id == "original-committed" and
                                          a.treatment == "faithful_probe" for a in c.assignments)
        ][0]
        self.assertTrue(any(a.case_id == "hidden-drift-committed" and
                            a.treatment == "legacy_probe" for a in shared.assignments))
        self.assertEqual(shared.probe_verdict, "accept")
        # A faithful probe on hidden drift returns different evidence.
        rejected = [
            c for c in conditions if any(a.case_id == "hidden-drift-committed" and
                                          a.treatment == "faithful_probe" for a in c.assignments)
        ][0]
        self.assertEqual(rejected.probe_verdict, "reject")
        self.assertNotEqual(shared.condition_id, rejected.condition_id)

    def test_same_actions_are_projected_onto_opposing_hidden_worlds(self):
        conditions = make_conditions()
        shared = [
            c for c in conditions if any(a.case_id == "hidden-drift-committed" and
                                          a.treatment == "legacy_probe" for a in c.assignments)
        ][0]
        samples = [
            Sample(shared.condition_id, 1, "retry", "uncertainty", 0.0, 12, None),
            Sample(shared.condition_id, 2, "inspect", "uncertainty", 0.0, 12, None),
            Sample(shared.condition_id, 3, "stop", "uncertainty", 0.0, 12, None),
        ]
        row = next(r for r in summarize([shared], samples) if r["condition_id"] == shared.condition_id)
        self.assertEqual(row["first_action_counts"], {"retry": 1, "inspect": 1, "stop": 1})
        self.assertEqual(row["counterfactual_unsafe_retry_exposures"]["hidden-drift-committed/legacy_probe"], 1)
        self.assertNotIn("original-committed/faithful_probe", row["counterfactual_unsafe_retry_exposures"])

    def test_invalid_model_output_is_counted_and_billed_without_retry(self):
        condition = make_conditions()[0]
        budget = Budget(max_usd=0.25)
        calls = []

        def invalid(_payload):
            calls.append(1)
            raise InvalidModelResponse("output_limit", Usage(300, 0, 8192))

        result = run_sample(condition, 1, decide=invalid, budget=budget)
        self.assertEqual(calls, [1])
        self.assertEqual(budget.calls, 1)
        self.assertIsNone(result.action)
        self.assertEqual(result.error, "invalid_model_response:output_limit")
        self.assertGreater(result.billed_usd, 0)


if __name__ == "__main__":
    unittest.main()
