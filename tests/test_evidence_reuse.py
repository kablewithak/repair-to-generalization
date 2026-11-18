"""Diagnostic tests for amortized lesson validation and silent drift."""

import unittest

from repair_to_generalization.evidence_reuse import (
    EvidenceCache,
    run_cached_sequence,
    scenario_cases,
)
from repair_to_generalization.counterfactual import RefundProbeSandbox
from repair_to_generalization.refund_simulator import (
    RefundCase,
    RefundSimulator,
    ToolSemantics,
)


class EvidenceReuseTests(unittest.TestCase):
    def test_evidence_is_reused_until_audit_due(self):
        cases = tuple(
            RefundCase(str(i), ToolSemantics.IDEMPOTENT, i % 2 == 0, False)
            for i in range(10)
        )
        trials = run_cached_sequence(
            cases,
            audit_every=4,
            sandbox_for_case=lambda case: RefundProbeSandbox(case.semantics),
        )
        self.assertEqual(sum(t.probe_actions > 0 for t in trials), 3)
        self.assertEqual(sum(t.probe_actions for t in trials), 12)
        self.assertTrue(all(t.success for t in trials))

    def test_announced_drift_forces_revalidation_before_unsafe_retry(self):
        trials = run_cached_sequence(
            scenario_cases(announced=True),
            audit_every=4,
            sandbox_for_case=lambda case: RefundProbeSandbox(case.semantics),
        )
        self.assertEqual(len(trials), 24)
        self.assertTrue(all(t.success for t in trials))
        self.assertEqual(sum(t.unsafe_duplicate for t in trials), 0)
        self.assertGreater(trials[10].probe_actions, 0)
        self.assertEqual(trials[10].verdict, "reject")

    def test_silent_drift_causes_harm_until_next_audit(self):
        cases = scenario_cases(announced=False)
        trials = run_cached_sequence(
            cases,
            audit_every=4,
            sandbox_for_case=lambda case: RefundProbeSandbox(case.semantics),
        )
        self.assertTrue(trials[10].unsafe_duplicate)
        self.assertEqual(trials[10].probe_actions, 0)
        self.assertFalse(trials[12].unsafe_duplicate)
        self.assertEqual(trials[12].verdict, "reject")
        self.assertEqual(trials[12].probe_actions, 4)

    def test_stale_sandbox_cannot_detect_drift_and_missing_probe_fails_closed(self):
        cases = scenario_cases(announced=False)
        stale = run_cached_sequence(
            cases,
            audit_every=4,
            sandbox_for_case=lambda _: RefundProbeSandbox(ToolSemantics.IDEMPOTENT),
        )
        self.assertTrue(stale[12].unsafe_duplicate)
        self.assertEqual(stale[12].verdict, "accept")
        controller = EvidenceCache(audit_every=4)
        no_probe = controller.execute(RefundSimulator(cases[10]), None)
        self.assertTrue(no_probe.success)
        self.assertEqual(no_probe.verdict, "unverifiable")
        self.assertEqual(no_probe.probe_actions, 0)

    def test_invalid_interval_is_rejected(self):
        for invalid in (0, -1, True, 1.5):
            with self.subTest(interval=invalid):
                with self.assertRaises(ValueError):
                    EvidenceCache(invalid)


if __name__ == "__main__":
    unittest.main()
