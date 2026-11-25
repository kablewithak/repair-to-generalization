"""Development contract, causal isolation and outcome scoring regressions."""
import json
import unittest
from dataclasses import replace

from repair_to_generalization.experience_ledger import Access, Ledger, execute
from repair_to_generalization.g6_qualification import Meter
from repair_to_generalization.g8_closed_loop_claims import (
    derive_transition, parse_reflection, reflection_payload, scope, active_claims,
)
from repair_to_generalization.g8_closed_loop_dev import (
    ARMS, MAX_CALLS, MAX_INPUT_BYTES, Run, cases, order, setup,
)
from repair_to_generalization.linked_world import Move, reference_policy
from repair_to_generalization.savant_pilot import InvalidModelResponse
from repair_to_generalization.study_design import acquisition_cases, case_signature, combined_cases


def response(body):
    return {"choices": [{"finish_reason": "stop", "message": {"content": json.dumps(body)}}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 40}}


def reference(public):
    return reference_policy(public["current"], public["current_interactions"])


def provider(payload):
    public = json.loads(payload["messages"][1]["content"])
    schema = payload["response_format"]["json_schema"]
    if schema["name"] == "closed_loop_receipt_prediction":
        alias = schema["schema"]["properties"]["evidence"]["enum"][0]
        obs = next(e["body"]["observation"] for e in public["public_interactions"]
                   if e["event_id"] == alias)
        return response(dict(prediction=obs.get("receipt_source", obs.get("source")),
                             advice="Synthetic test-only advice.", evidence=alias))
    return response({"action": reference(public).value})


class ClosedLoopTests(unittest.TestCase):
    def test_design_and_reference_solvability(self):
        cohort = cases()
        reserved = acquisition_cases() + tuple(c for name in ("main", "confirmation", "calibration")
                                                for c in combined_cases(name))
        self.assertEqual(len({case_signature(c) for c in reserved+cohort}), len(reserved+cohort))
        self.assertEqual(len(set(order())), 24)
        for i, c in enumerate(cohort, 1):
            _, r = execute(c, Ledger(), i, False, Access.CURRENT, reference)
            self.assertTrue(json.loads(r.outcome_json)["safe_success"])
        self.assertEqual(setup("test", 2.10, .5)["order"], order())

    def test_derived_transition_and_no_oracle_correction(self):
        self.assertEqual(derive_transition("provider_ledger", "delivery_replica"), "revise")
        self.assertEqual(derive_transition("delivery_replica", "delivery_replica"), "retain")
        self.assertEqual(derive_transition("delivery_replica", "unknown"), "invalidate")
        self.assertEqual(derive_transition("unknown", "unknown"), "uncertain")
        case = cases()[1]
        ledger, _ = execute(case, Ledger(), 2, True, Access.CURRENT, reference)
        _, anchors, observed = scope(ledger, 2, case.service, case.version)
        alias = next(iter(anchors))
        wrong = "delivery_replica" if observed[alias] == "provider_ledger" else "provider_ledger"
        body = dict(prediction=wrong, advice="Synthetic incorrect claim.", evidence=alias)
        claim, _ = parse_reflection(response(body), ledger, 2, case.service, case.version, ())
        self.assertEqual(claim.prediction, wrong)
        self.assertEqual(claim.operation, "acquire")
        self.assertEqual(active_claims((claim,))[0]["body"]["prediction_status"], "unverified_model_hypothesis")
        body["operation"] = "retain"
        with self.assertRaisesRegex(InvalidModelResponse, "fields"):
            parse_reflection(response(body), ledger, 2, case.service, case.version, ())
        del body["operation"]
        body["evidence"] = "invented"
        with self.assertRaisesRegex(InvalidModelResponse, "evidence"):
            parse_reflection(response(body), ledger, 2, case.service, case.version, ())

    def test_independent_streams_and_checkpoint_exclusion(self):
        events = []
        meter = Meter(.75, provider, events.append, max_calls=MAX_CALLS, max_input_bytes=MAX_INPUT_BYTES)
        run = Run(meter, events.append)
        run.run()
        run.verify()
        result = run.summary()
        self.assertEqual(result["episodes"], 24)
        self.assertEqual(len(run.claims), 4)
        self.assertTrue(result["claim_mechanism_observed"])
        self.assertFalse(result["development_positive"])
        self.assertFalse(result["g8_pass"])
        self.assertTrue(all(t["safe_successes"] == 4 for t in result["by_arm"].values()))
        self.assertEqual(run.ledgers["A0"], Ledger())
        for arm in ("A3","A4"):
            self.assertTrue(all(e.episode <= 4 for e in run.ledgers[arm].events))
        # Reference trajectories happen to agree, but run-level ledger ownership is separate.
        self.assertIsNot(run.ledgers["A3"], run.ledgers["A4"])
        call_arm = None
        for event in events:
            if event["record_type"] == "g8_cl_episode_started":
                call_arm = event["arm"]
            if event["record_type"] != "call_started":
                continue
            payload = event["request"]
            public = json.loads(payload["messages"][1]["content"])
            if payload["response_format"]["json_schema"]["name"] == "closed_loop_receipt_prediction":
                self.assertEqual(call_arm, "A4")
                self.assertTrue(all(e["episode"] <= event["episode"] for e in public["public_interactions"]))
                self.assertNotIn("outcome", public)
                continue
            history = public["prior_interactions"]
            if call_arm == "A0":
                self.assertEqual(history, [])
            if call_arm == "A3":
                self.assertFalse(any(e["kind"] == "persistent_agent_claim" for e in history))
            self.assertTrue(all(e["episode"] < event["episode"] for e in history if "episode" in e))
            self.assertNotIn("case_id", public["current"])
            self.assertNotIn("phase", public["current"])

    def test_missing_evidence_never_fabricates_claim(self):
        ledger, _ = execute(cases()[0], Ledger(), 1, True, Access.CURRENT, lambda _: Move.FINISH)
        meter = Meter(.75, lambda _: self.fail("unexpected model call"), lambda _: None,
                      max_calls=MAX_CALLS, max_input_bytes=MAX_INPUT_BYTES)
        run = Run(meter, lambda _: None)
        run.ledgers["A4"] = ledger
        run.reflect(cases()[0], 1)
        self.assertEqual(run.claims, ())
        self.assertEqual(run.reflections[0]["status"], "no_observed_receipt_evidence")
        self.assertEqual(meter.budget.calls, 0)

    def test_failed_reflection_preserves_replayable_prefix(self):
        def bad(payload):
            if payload["response_format"]["json_schema"]["name"] == "closed_loop_receipt_prediction":
                return response({})
            return provider(payload)
        meter = Meter(.75, bad, lambda _: None, max_calls=MAX_CALLS, max_input_bytes=MAX_INPUT_BYTES)
        run = Run(meter, lambda _: None)
        with self.assertRaisesRegex(InvalidModelResponse, "fields"):
            run.run()
        run.verify()
        self.assertEqual(len(run.rows), 3)
        self.assertEqual(run.claims, ())
        self.assertFalse(run.summary()["development_positive"])

    def test_future_events_excluded_from_old_scope(self):
        ledger = Ledger()
        for i, c in enumerate(cases()[:4], 1):
            ledger, _ = execute(c, ledger, i, True, Access.CURRENT, reference)
        public, _, _ = scope(ledger, 1, cases()[0].service, "v1")
        self.assertTrue(all(e["episode"] == 1 for e in public))


if __name__ == "__main__":
    unittest.main()
