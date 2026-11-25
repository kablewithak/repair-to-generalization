"""G6 development world and public-only ledger regression tests."""
import json
import unittest
from dataclasses import replace

from repair_to_generalization.experience_ledger import (
    Access, Claim, Ledger, Uncertainty, Validity, execute, replay, validate_claim,
)
from repair_to_generalization.linked_world import (
    MAX_ACTIONS, LinkedWorld, Move, development_stream, reference_policy,
)
from repair_to_generalization.transfer_world import InitialState


def reference(request):
    return reference_policy(request["current"], request["current_interactions"])


class InstrumentTests(unittest.TestCase):
    def test_reference_solves_seeded_development_cases(self):
        for seed in range(32):
            cases = development_stream(seed)
            self.assertEqual(cases, development_stream(seed))
            self.assertEqual(len({c.case_id for c in cases}), len(cases))
            ledger = Ledger()
            records = []
            for i, case in enumerate(cases, 1):
                ledger, record = execute(case, ledger, i, True, Access.CURRENT, reference)
                result = json.loads(record.outcome_json)
                self.assertTrue(result["safe_success"], (case.case_id, result))
                self.assertLessEqual(result["actions"], MAX_ACTIONS)
                records.append(record)
            self.assertEqual(replay(cases, tuple(records), ledger, (True,) * len(cases)), ledger)

    def test_private_initial_state_is_not_public(self):
        case = development_stream(3)[0]
        a = LinkedWorld(replace(case, refund_committed=False))
        b = LinkedWorld(replace(case, refund_committed=True))
        self.assertEqual(a.view(), b.view())
        self.assertNotEqual(a.step(Move.REFUND_COUNT), b.step(Move.REFUND_COUNT))
        self.assertNotIn("case_id", a.view())
        self.assertNotIn("phase", a.view())
        self.assertNotIn("allow_refund", a.view())

    def test_permission_dependency_and_duplicates_are_realized(self):
        cases = development_stream(7)
        denied = LinkedWorld(cases[-1])
        self.assertEqual(denied.step(Move.REFUND)["event"], "permission_denied")
        self.assertEqual(denied.step(Move.SUBMIT)["event"], "dependency_rejected")
        denied.step(Move.FINISH)
        result = denied.outcome()
        self.assertTrue(result["unsafe"])
        self.assertFalse(result["safe_success"])
        self.assertEqual(result["jobs_created"], 0)
        self.assertEqual(result["refunds"], 0)
        duplicate = LinkedWorld(replace(cases[5], refund_committed=False))
        duplicate.step(Move.REFUND)
        duplicate.step(Move.REFUND)
        duplicate.step(Move.FINISH)
        self.assertTrue(duplicate.outcome()["unsafe_duplicate"])
        export = LinkedWorld(replace(cases[6], export_initial=InitialState.ABSENT))
        export.step(Move.SUBMIT)
        export.step(Move.SUBMIT)
        export.step(Move.TICK)
        export.step(Move.FINISH)
        self.assertTrue(export.outcome()["unsafe_duplicate"])

    def test_no_learning_from_evaluation_and_no_mutable_aliases(self):
        cases = development_stream(1)[:3]
        empty = Ledger()
        ledger, first = execute(cases[0], empty, 1, True, Access.CURRENT, reference)
        before = ledger.public()
        self.assertEqual(empty.events, ())
        observed = []

        def probe(request):
            observed.append(request["prior_interactions"])
            request["prior_interactions"].clear()
            return reference(request)

        unchanged, second = execute(cases[1], ledger, 2, False, Access.FULL, probe)
        self.assertEqual(unchanged, ledger)
        self.assertEqual(ledger.public(), before)
        seen = []

        def capture(request):
            seen.append(request["prior_interactions"])
            return reference(request)

        final, third = execute(cases[2], unchanged, 3, True, Access.FULL, capture)
        self.assertEqual(seen[0], before)
        self.assertTrue(all(e.episode != 2 for e in final.events))
        self.assertEqual(replay(cases, (first, second, third), final,
                                (True, False, True)), final)
        with self.assertRaisesRegex(ValueError, "learning_split"):
            replay(cases, (first, second, third), final, (True, True, True))

    def test_tampered_ledger_outcome_or_actions_fail_replay(self):
        case = development_stream(2)[0]
        ledger, record = execute(case, Ledger(), 1, True, Access.CURRENT, reference)
        corrupt = Ledger((replace(ledger.events[0], body_json="{}"),) + ledger.events[1:])
        with self.assertRaisesRegex(ValueError, "ledger_mismatch"):
            replay((case,), (record,), corrupt, (True,))
        with self.assertRaisesRegex(ValueError, "record_mismatch"):
            replay((case,), (replace(record, outcome_json="{}"),), ledger, (True,))
        with self.assertRaisesRegex(ValueError, "missing_action"):
            replay((case,), (replace(record, actions=()),), ledger, (True,))

    def test_action_limit_and_abandonment_are_not_success(self):
        case = development_stream(2)[0]
        _, record = execute(case, Ledger(), 1, True, Access.CURRENT,
                            lambda _: Move.PERMISSIONS)
        result = json.loads(record.outcome_json)
        self.assertEqual(result["actions"], MAX_ACTIONS)
        self.assertFalse(result["completed"])
        self.assertFalse(result["safe_success"])
        _, stopped = execute(replace(case, refund_committed=False), Ledger(), 1,
                             True, Access.CURRENT, lambda _: Move.FINISH)
        self.assertFalse(json.loads(stopped.outcome_json)["safe_success"])

    def test_claim_provenance_and_nontrivial_revision(self):
        cases = development_stream(4)
        ledger, _ = execute(cases[0], Ledger(), 1, True, Access.CURRENT, reference)
        old_evidence = next(e.event_id for e in ledger.events if e.kind == "step")
        claim = Claim("c1", cases[0].service, "v1", "refund recovery", "retry deduplicates",
                      (old_evidence,), Uncertainty.HIGH, Validity.SUPPORTED)
        prior = validate_claim(claim, ledger, ())
        with self.assertRaisesRegex(ValueError, "out_of_scope"):
            validate_claim(replace(claim, service="another"), ledger, ())
        with self.assertRaisesRegex(ValueError, "out_of_scope"):
            validate_claim(replace(claim, evidence_ids=("invented",)), ledger, ())
        ledger2, _ = execute(cases[5], ledger, 2, True, Access.CURRENT, reference)
        new_evidence = next(e.event_id for e in ledger2.events
                            if e.kind == "step" and e.episode == 2)
        revision = replace(claim, claim_id="c2", version="v2", revises="c1",
                           predicted_action_effect="retry may duplicate",
                           evidence_ids=(old_evidence, new_evidence))
        self.assertEqual(validate_claim(revision, ledger2, prior), (claim, revision))
        with self.assertRaisesRegex(ValueError, "out_of_scope"):
            validate_claim(revision, ledger, prior)
        with self.assertRaisesRegex(ValueError, "new_evidence"):
            validate_claim(replace(claim, claim_id="c3", revises="c1"), ledger2, prior)


if __name__ == "__main__":
    unittest.main()
