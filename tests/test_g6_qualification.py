"""Offline response-boundary tests: no provider calls, no credentials."""
import json
import unittest

from repair_to_generalization.g6_qualification import (
    ARMS, CASE, INPUT_BYTES_LIMIT, MAX_CALLS, OUTPUT_LIMIT, WORST_CASE_USD,
    Meter, qualification_payload, reservation, run_qualification,
)
from repair_to_generalization.savant_pilot import InvalidModelResponse, PilotError, Usage
from repair_to_generalization.transfer_pilot import FrozenSources, payload_for, world_for


SOURCE = FrozenSources(
    {"summary": "old"}, {"summary": "revised"},
    "a" * 64, "b" * 64, "c" * 64, "d" * 64,
)


def response(action="finish", reason="stop", completion=40, reasoning=15):
    return {
        "choices": [{"finish_reason": reason, "message": {
            "content": json.dumps({"action": action, "reason_code": "observed_state"})
        }}],
        "usage": {"prompt_tokens": 100, "completion_tokens": completion,
                  "prompt_tokens_details": {"cached_tokens": 20},
                  "completion_tokens_details": {"reasoning_tokens": reasoning}},
    }


class QualificationTests(unittest.TestCase):
    def setUp(self):
        self.events = []
        self.payload = qualification_payload(ARMS[0], SOURCE, world_for(CASE), [])

    def test_only_request_change_is_output_cap(self):
        old = payload_for(ARMS[0], SOURCE, world_for(CASE), [])
        self.assertEqual(old["max_tokens"], 4096)
        self.assertEqual(self.payload, {**old, "max_tokens": OUTPUT_LIMIT})
        self.assertEqual(MAX_CALLS, 27)
        self.assertLess(WORST_CASE_USD, 0.20)

    def test_usage_and_latency_for_valid_decision(self):
        meter = Meter(0.20, lambda _: response(), self.events.append)
        decision = meter.decide(self.payload, 1, 0)
        self.assertEqual(decision.action.value, "finish")
        self.assertEqual([e["record_type"] for e in self.events],
                         ["call_started", "call_finished"])
        self.assertEqual(self.events[-1]["reasoning_tokens"], 15)
        self.assertEqual(self.events[-1]["finish_reason"], "stop")
        self.assertGreaterEqual(self.events[-1]["latency_ms"], 0)
        self.assertAlmostEqual(meter.reported_usd, Usage(100, 20, 40).usd)
        self.assertEqual(meter.unknown_reserved_usd, 0)

    def test_truncation_charged_without_retry(self):
        calls = []
        def request(payload):
            calls.append(payload)
            return response(reason="length", completion=8192, reasoning=8192)
        meter = Meter(0.20, request, self.events.append)
        with self.assertRaisesRegex(InvalidModelResponse, "output_limit"):
            meter.decide(self.payload, 1, 0)
        self.assertEqual(len(calls), 1)
        self.assertAlmostEqual(meter.reported_usd, Usage(100, 20, 8192).usd)
        self.assertEqual(self.events[-1]["finish_reason"], "length")
        self.assertIsNone(self.events[-1]["decision"])

    def test_unknown_usage_and_network_failure_reserved_once(self):
        for request in (lambda _: {"choices": []}, self.network_failure):
            with self.subTest(request=request):
                events = []
                meter = Meter(0.20, request, events.append)
                with self.assertRaises(PilotError):
                    meter.decide(self.payload, 1, 0)
                self.assertEqual(meter.budget.calls, 1)
                self.assertAlmostEqual(meter.unknown_reserved_usd, reservation(self.payload)[1])
                self.assertEqual(meter.reported_usd, 0)
                self.assertIsNone(events[-1]["usage"])
                self.assertIsNotNone(events[-1]["error"])

    @staticmethod
    def network_failure(_):
        raise PilotError("api_network_failure")

    def test_malformed_decision_preserves_reported_usage(self):
        meter = Meter(0.20, lambda _: response(action="not_an_action"), self.events.append)
        with self.assertRaises(InvalidModelResponse):
            meter.decide(self.payload, 1, 0)
        self.assertAlmostEqual(meter.reported_usd, Usage(100, 20, 40).usd)
        self.assertIsNone(self.events[-1]["decision"])

    def test_oversized_usage_stops_after_accounting(self):
        meter = Meter(0.20, lambda _: response(completion=8193), self.events.append)
        with self.assertRaisesRegex(InvalidModelResponse, "usage_outside_bound"):
            meter.decide(self.payload, 1, 0)
        self.assertAlmostEqual(meter.reported_usd, Usage(100, 20, 8193).usd)

    def test_oversize_input_or_insufficient_budget_never_calls_provider(self):
        calls = []
        meter = Meter(0.0001, lambda p: calls.append(p), self.events.append)
        with self.assertRaisesRegex(PilotError, "dollar_limit"):
            meter.decide(self.payload, 1, 0)
        with self.assertRaisesRegex(PilotError, "input_limit"):
            meter.decide({**self.payload, "extra": "x" * INPUT_BYTES_LIMIT}, 1, 0)
        self.assertEqual(calls, [])
        self.assertEqual(meter.budget.calls, 0)

    def test_fresh_episodes_and_terminal_protocol_not_scientific_success(self):
        calls = []
        def request(payload):
            calls.append(payload)
            return response()
        meter = Meter(0.20, request, self.events.append)
        result = run_qualification(SOURCE, meter, self.events.append)
        self.assertEqual(len(calls), 3)
        self.assertEqual(result["finished_episodes"], 3)
        self.assertTrue(result["protocol_pass"])
        self.assertEqual(result["safe_successes_descriptive_only"], 0)
        self.assertEqual(result["context_tier"], "short_development_only")

    def test_action_limit_is_not_protocol_qualification_pass(self):
        meter = Meter(0.20, lambda _: response(action="audit_queue"), self.events.append)
        result = run_qualification(SOURCE, meter, self.events.append)
        self.assertEqual(meter.budget.calls, MAX_CALLS)
        self.assertFalse(result["protocol_pass"])
        self.assertEqual(result["finished_episodes"], 0)


if __name__ == "__main__":
    unittest.main()
