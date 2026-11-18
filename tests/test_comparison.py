"""Tests protect the causal comparison, not demo presentation details."""

import unittest

from repair_to_generalization.comparison import Strategy, run_matrix, run_trial, summarize
from repair_to_generalization.refund_simulator import RefundCase, ToolSemantics


class StrategyComparisonTests(unittest.TestCase):
    def test_visible_drift_prevents_duplicate_when_rechecked(self):
        case = RefundCase("visible", ToolSemantics.NON_IDEMPOTENT, True, True)
        reuse = run_trial(case, Strategy.REUSE)
        gated = run_trial(case, Strategy.SIGNAL_GATED_RECHECK)
        self.assertTrue(reuse.unsafe_duplicate)
        self.assertFalse(reuse.success)
        self.assertTrue(gated.success)
        self.assertTrue(gated.inspected)

    def test_unannounced_drift_remains_a_failure(self):
        case = RefundCase("hidden", ToolSemantics.NON_IDEMPOTENT, True, False)
        gated = run_trial(case, Strategy.SIGNAL_GATED_RECHECK)
        invalidated = run_trial(case, Strategy.INVALIDATE_ALL)
        self.assertTrue(gated.unsafe_duplicate)
        self.assertFalse(gated.inspected)
        self.assertTrue(invalidated.success)

    def test_preserves_useful_reuse_and_exposes_cost_of_invalidate_all(self):
        case = RefundCase("unchanged", ToolSemantics.IDEMPOTENT, False, False)
        reuse = run_trial(case, Strategy.REUSE)
        gated = run_trial(case, Strategy.SIGNAL_GATED_RECHECK)
        invalidated = run_trial(case, Strategy.INVALIDATE_ALL)
        self.assertTrue(reuse.success)
        self.assertTrue(gated.success)
        self.assertTrue(invalidated.success)
        self.assertEqual(gated.actions_used, reuse.actions_used)
        self.assertGreater(invalidated.actions_used, gated.actions_used)

    def test_incomplete_refund_recovered_even_after_visible_drift(self):
        case = RefundCase("changed-uncommitted", ToolSemantics.NON_IDEMPOTENT, False, True)
        gated = run_trial(case, Strategy.SIGNAL_GATED_RECHECK)
        self.assertTrue(gated.success)
        self.assertEqual(gated.actions_used, 3)

    def test_summary_counts_match_trial_outcomes(self):
        trials = run_matrix()
        summaries = summarize(trials)
        self.assertEqual(len(trials), 18)
        self.assertEqual(len(summaries), 3)
        for row in summaries:
            selected = [trial for trial in trials if trial.strategy == row.strategy]
            self.assertEqual(row.successes, sum(trial.success for trial in selected))
            self.assertEqual(row.unsafe_duplicates, sum(trial.unsafe_duplicate for trial in selected))


if __name__ == "__main__":
    unittest.main()
