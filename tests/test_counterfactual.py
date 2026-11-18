"""Only invariants diagnostic of experience transfer and probe fidelity."""

import unittest

from repair_to_generalization.counterfactual import (
    RefundProbeSandbox,
    Verdict,
    run_gated_trial,
    run_matrix,
    summarize,
    validate_retry_lesson,
)
from repair_to_generalization.refund_simulator import RefundCase, ToolSemantics


class CounterfactualGateTests(unittest.TestCase):
    def test_previously_safe_retry_passes_both_counterfactual_probes(self):
        result = validate_retry_lesson(RefundProbeSandbox(ToolSemantics.IDEMPOTENT))
        self.assertEqual(result.verdict, Verdict.ACCEPT)
        self.assertEqual(result.probe_episodes, 2)
        self.assertEqual(result.probe_actions, 4)
        self.assertFalse(result.failed_conditions)

    def test_changed_semantics_rejects_unconditional_retry_without_signal(self):
        evidence = validate_retry_lesson(RefundProbeSandbox(ToolSemantics.NON_IDEMPOTENT))
        self.assertEqual(evidence.verdict, Verdict.REJECT)
        self.assertEqual(evidence.failed_conditions, ("prior_commit=True",))
        case = RefundCase("hidden", ToolSemantics.NON_IDEMPOTENT, True, False)
        trial = run_gated_trial(case, RefundProbeSandbox(ToolSemantics.NON_IDEMPOTENT))
        self.assertTrue(trial.success)
        self.assertFalse(trial.unsafe_duplicate)
        self.assertEqual(trial.live_actions, 2)

    def test_stale_sandbox_falsely_approves_and_is_not_hidden(self):
        case = RefundCase("stale", ToolSemantics.NON_IDEMPOTENT, True, False)
        trial = run_gated_trial(case, RefundProbeSandbox(ToolSemantics.IDEMPOTENT))
        self.assertEqual(trial.verdict, Verdict.ACCEPT.value)
        self.assertTrue(trial.unsafe_duplicate)
        self.assertFalse(trial.success)

    def test_missing_sandbox_falls_back_to_live_inspection(self):
        case = RefundCase("missing", ToolSemantics.NON_IDEMPOTENT, False, False)
        trial = run_gated_trial(case, None)
        self.assertEqual(trial.verdict, Verdict.UNVERIFIABLE.value)
        self.assertEqual(trial.probe_actions, 0)
        self.assertEqual(trial.live_actions, 3)
        self.assertTrue(trial.success)

    def test_faithful_matrix_reports_outcomes_and_probe_overhead(self):
        trials = run_matrix()
        self.assertEqual(len(trials), 24)
        report = {row.strategy: row for row in summarize(trials)}
        self.assertEqual(report["reuse"].successes, 4)
        self.assertEqual(report["reuse"].unsafe_duplicates, 2)
        self.assertEqual(report["signal_gated_recheck"].successes, 5)
        self.assertEqual(report["counterfactual_gate"].successes, 6)
        self.assertEqual(report["counterfactual_gate"].unsafe_duplicates, 0)
        self.assertEqual(report["counterfactual_gate"].mean_probe_actions, 4.0)
        self.assertGreater(
            report["counterfactual_gate"].mean_total_actions,
            report["invalidate_all"].mean_total_actions,
        )


if __name__ == "__main__":
    unittest.main()
