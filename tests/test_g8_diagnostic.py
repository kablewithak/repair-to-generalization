"""Offline tests for scoped claim revision and paired diagnostic."""
import json
import unittest
from dataclasses import replace

from repair_to_generalization.experience_ledger import Access, Ledger, execute, hash_value
from repair_to_generalization.g6_live import ACTION_ONLY_REVISION, parse_linked
from repair_to_generalization.g6_qualification import Meter
from repair_to_generalization.g7_source import FrozenSource
from repair_to_generalization.g8_claims import claim_payload, parse_claim, public_scope, diagnose, snapshot
from repair_to_generalization.g8_diagnostic import (
    ARMS, MAX_CALLS, REFLECTION_CALLS, DiagnosticRun, cases, order, population,
    reflect, score, setup_for, validate_design,
)
from repair_to_generalization.linked_world import reference_policy
from repair_to_generalization.savant_pilot import InvalidModelResponse, PilotError
from repair_to_generalization.study_design import SERVICES, MAX_INPUT_BYTES, acquisition_cases


def response(body, finish="stop"):
    return {"choices": [{"finish_reason": finish, "message": {"content": json.dumps(body)}}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 40}}


def ref(public):
    return reference_policy(public["current"], public["current_interactions"])


def reflection_model(payload):
    public = json.loads(payload["messages"][1]["content"])
    schema = payload["response_format"]["json_schema"]["schema"]
    alias = schema["properties"]["evidence"]["enum"][0]
    event = next(row for row in public["public_interactions"] if row["event_id"] == alias)
    obs = event["body"]["observation"]
    prediction = obs.get("receipt_source", obs.get("source"))
    result = dict(prediction=prediction, advice="Synthetic test advice.", evidence=alias)
    if "prior_claim" in public:
        result["operation"] = "retain" if prediction == public["prior_claim"]["prediction"] else "revise"
    return response(result)


class DiagnosticTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        source_cases = acquisition_cases()[:50]
        ledger, records = Ledger(), []
        for i, case in enumerate(source_cases, 1):
            ledger, record = execute(case, ledger, i, True, Access.CURRENT, ref)
            records.append(record)
        cls.source = FrozenSource("a"*64, ledger, source_cases, tuple(records), 0.5, 1.0)
        cls.reflection_events = []
        meter = Meter(0.20, reflection_model, cls.reflection_events.append,
                      max_calls=REFLECTION_CALLS, max_input_bytes=MAX_INPUT_BYTES)
        cls.snapshots, cls.diagnostics = reflect(cls.source, meter, cls.reflection_events.append)

    def test_design_disjoint_and_solvable(self):
        validate_design(self.source)
        self.assertEqual(len(cases()), 6)
        self.assertEqual(len(set(order())), 24)
        self.assertEqual([population(c) for c in cases()].count("changed_receipt"), 3)
        for i, case in enumerate(cases(), 1):
            _, r = execute(case, Ledger(), i, False, Access.CURRENT, ref)
            self.assertTrue(json.loads(r.outcome_json)["safe_success"])
        self.assertEqual(setup_for(self.source, "test", 2.06, 1.25)["order"], order())

    def test_initial_and_review_scope(self):
        ledger = self.source.ledger
        service = SERVICES[1]
        initial_payload = claim_payload(ledger, service)
        initial_public = json.loads(initial_payload["messages"][1]["content"])
        self.assertNotIn("prior_claim", initial_public)
        self.assertNotIn("earlier_public_interactions", initial_public)
        self.assertTrue(all(e["body"]["version"] == "v1"
                            for e in initial_public["public_interactions"] if e["kind"] == "start"))
        old, _ = parse_claim(reflection_model(initial_payload), ledger, service)
        reviewed_payload = claim_payload(ledger, service, old)
        reviewed_public = json.loads(reviewed_payload["messages"][1]["content"])
        self.assertTrue(all(e["body"]["version"] == "v2"
                            for e in reviewed_public["public_interactions"] if e["kind"] == "start"))
        new, _ = parse_claim(reflection_model(reviewed_payload), ledger, service, old)
        self.assertEqual(new.claim.revises, old.claim.claim_id)
        self.assertTrue(diagnose(ledger, old, new)["detected_and_repaired"])
        body = json.loads(reflection_model(reviewed_payload)["choices"][0]["message"]["content"])
        body["evidence"] = old.anchor
        with self.assertRaisesRegex(InvalidModelResponse, "claim_evidence_scope"):
            parse_claim(response(body), ledger, service, old)
        body = json.loads(reflection_model(reviewed_payload)["choices"][0]["message"]["content"])
        body["operation"] = "retain"
        with self.assertRaisesRegex(InvalidModelResponse, "claim_retain_changed_prediction"):
            parse_claim(response(body), ledger, service, old)

    def test_wrong_prediction_is_measured_not_replaced(self):
        ledger = self.source.ledger
        service = SERVICES[1]
        p = claim_payload(ledger, service)
        old, _ = parse_claim(reflection_model(p), ledger, service)
        body = json.loads(reflection_model(claim_payload(ledger, service, old))
                          ["choices"][0]["message"]["content"])
        body.update(prediction=old.prediction, operation="retain")
        new, _ = parse_claim(response(body), ledger, service, old)
        self.assertEqual(new.prediction, old.prediction)
        self.assertFalse(diagnose(ledger, old, new)["review_agrees"])
        self.assertFalse(diagnose(ledger, old, new)["detected_and_repaired"])

    def test_frozen_snapshots_and_full_replay(self):
        self.assertEqual(self.snapshots["A3"], [])
        self.assertTrue(all(e["body"]["version"] == "v1" for e in self.snapshots["A4_STALE"]))
        self.assertTrue(all(e["body"]["version"] == "v2" for e in self.snapshots["A4"]))
        self.assertTrue(all(e["kind"] == "invalidated_claim" for e in self.snapshots["A4_INVALIDATE_ALL"]))
        self.assertEqual(sum(d["detected_and_repaired"] for d in self.diagnostics), 3)
        self.assertEqual(sum(d["retained_unaffected_prediction"] for d in self.diagnostics), 2)
        events = []
        public_source = self.source.ledger.public()
        hashes = {a: hash_value(s) for a,s in self.snapshots.items()}

        def request(payload):
            public = json.loads(payload["messages"][1]["content"])
            self.assertEqual(public["prior_interactions"][:len(public_source)], public_source)
            self.assertIn(public["prior_interactions"][len(public_source):], list(self.snapshots.values()))
            self.assertNotIn("case_id", public["current"])
            return response({"action": ref(public).value})

        meter = Meter(1.25, request, events.append,
                      parser=lambda d: parse_linked(d, ACTION_ONLY_REVISION),
                      max_calls=MAX_CALLS, max_input_bytes=MAX_INPUT_BYTES)
        run = DiagnosticRun(self.source, self.snapshots, meter, events.append)
        run.run()
        result = score(run.rows, self.diagnostics)
        self.assertEqual(result["scored_episodes"], 24)
        self.assertTrue(result["claim_mechanism_observed"])
        self.assertFalse(result["behavior_screen_positive"])
        self.assertFalse(result["g8_pass"])
        self.assertTrue(all(v["safe_successes"] == 6 for v in result["by_arm"].values()))
        self.assertTrue(all(row["replay_valid"] and not row["record"]["learning"]
                            and row["record"]["ledger_head"] == self.source.ledger.head for row in run.rows))
        self.assertEqual(hashes, {a: hash_value(s) for a,s in self.snapshots.items()})

    def test_score_requires_distinct_benefit_no_regressions(self):
        rows = []
        for case in cases():
            for arm in ARMS:
                pop = population(case)
                success = pop != "changed_receipt" or arm == "A4"
                rows.append(dict(case_id=case.case_id, arm=arm, population=pop,
                                 outcome=dict(completed=True, safe_success=success, unsafe=False),
                                 usage=dict(model_calls=3, accounted_usd=0.001)))
        result = score(rows, self.diagnostics)
        self.assertTrue(result["diagnostic_positive"])
        self.assertFalse(result["g8_pass"])
        self.assertIsNone(score(rows[:-1], self.diagnostics)["behavior_screen_positive"])
        row = next(r for r in rows if r["arm"] == "A4" and r["population"] == "changed_receipt")
        row["outcome"].update(unsafe=True, safe_success=False)
        self.assertFalse(score(rows, self.diagnostics)["diagnostic_positive"])

    def test_truncation_stops_and_charges_call(self):
        events = []
        meter = Meter(1.25, lambda _: response({}, "length"), events.append,
                      parser=lambda d: parse_linked(d, ACTION_ONLY_REVISION),
                      max_calls=MAX_CALLS, max_input_bytes=MAX_INPUT_BYTES)
        run = DiagnosticRun(self.source, self.snapshots, meter, events.append)
        with self.assertRaisesRegex(PilotError, "output_limit"):
            run.run()
        self.assertEqual(meter.budget.calls, 1)
        self.assertGreater(meter.budget.spent_usd, 0)
        self.assertEqual(run.rows, [])
        self.assertFalse(score(run.rows, self.diagnostics)["diagnostic_positive"])


if __name__ == "__main__":
    unittest.main()
