"""Offline tests for the G6 live runner. No provider network or credentials."""
import json
import unittest

from repair_to_generalization.g6_live import (
    ACTION_ONLY_REVISION, REVISION, ARMS, MAX_CALLS, MAX_INPUT_BYTES, OUTPUT_LIMIT, SEED, DevelopmentRun,
    case_order, manifest, parse_linked, payload_for,
)
from repair_to_generalization.g6_qualification import Meter
from repair_to_generalization.linked_world import Move, development_stream, reference_policy
from repair_to_generalization.savant_pilot import InvalidModelResponse, PilotError, Usage


def response(action, reason="stop", completion=40):
    return {
        "choices": [{"finish_reason": reason, "message": {
            "content": json.dumps({"action": action, "reason_code": "observed_state"})
        }}],
        "usage": {"prompt_tokens": 100, "completion_tokens": completion,
                  "prompt_tokens_details": {"cached_tokens": 0}},
    }


def fake_reference(payload):
    public = json.loads(payload["messages"][1]["content"])
    action = reference_policy(public["current"], public["current_interactions"])
    return response(action.value)


class LiveInstrumentTests(unittest.TestCase):
    def setUp(self):
        self.cases = development_stream(SEED)
        self.events = []
        self.requests = []

    def meter(self, request, cap=2.0):
        return Meter(cap, request, self.events.append, parser=parse_linked,
                     max_calls=MAX_CALLS, max_input_bytes=MAX_INPUT_BYTES)

    def test_strict_parse_preserves_billed_usage(self):
        decision, usage = parse_linked(response("inspect_permissions"))
        self.assertIs(decision.action, Move.PERMISSIONS)
        self.assertEqual(usage, Usage(100, 0, 40))
        for action in ("audit_queue", "invented"):
            with self.assertRaises(InvalidModelResponse) as error:
                parse_linked(response(action))
            self.assertEqual(error.exception.usage, usage)
        duplicate = response("finish")
        duplicate["choices"][0]["message"]["content"] = (
            '{"action":"finish","action":"retry_refund","reason_code":"observed_state"}')
        with self.assertRaises(InvalidModelResponse):
            parse_linked(duplicate)
        with self.assertRaisesRegex(InvalidModelResponse, "output_limit") as error:
            parse_linked(response("finish", "length", OUTPUT_LIMIT))
        self.assertEqual(error.exception.usage.completion_tokens, OUTPUT_LIMIT)

    def test_full_development_stream_replays_with_no_evaluator_feedback(self):
        def request(payload):
            self.requests.append(payload)
            return fake_reference(payload)

        meter = self.meter(request)
        run = DevelopmentRun(meter, self.events.append, self.cases)
        run.run()
        run.verify()
        summary = run.summary()
        self.assertEqual(summary["scored_episodes"], {"A0": 12, "A3": 12})
        self.assertEqual(summary["paired_cases"], 12)
        self.assertFalse(summary["tier_qualified"])
        for arm in ARMS:
            self.assertEqual(summary["by_arm"][arm]["safe_successes"], 12)
            self.assertEqual(summary["by_arm"][arm]["unsafe"], 0)
        self.assertEqual(run.ledgers["A0"].events, ())
        self.assertGreater(len(run.ledgers["A3"].events), 0)
        active = None
        requests = []
        for row in self.events:
            if row["record_type"] == "linked_episode_started":
                active = row
            elif row["record_type"] == "call_started":
                prompt = json.loads(row["request"]["messages"][1]["content"])
                requests.append(prompt)
                self.assertEqual(set(prompt), {"instrument", "contract", "current",
                                                "current_interactions", "prior_interactions"})
                self.assertNotIn("case_id", prompt["current"])
                self.assertNotIn("phase", prompt["current"])
                prior = prompt["prior_interactions"]
                if active["arm"] == "A0":
                    self.assertEqual(prior, [])
                else:
                    self.assertEqual(len(prior), active["source_event_count"])
                    self.assertTrue(all(event["episode"] < active["episode"] for event in prior))
                    self.assertTrue(all(event["kind"] in ("start", "step", "end") for event in prior))
                    self.assertTrue(all("safe_success" not in event["body"] for event in prior))
        self.assertEqual(len(requests), meter.budget.calls)
        self.assertTrue(any(p["prior_interactions"] for p in requests))
        self.assertEqual(len(self.requests), meter.budget.calls)

    def test_interrupted_episode_is_not_promoted_into_history(self):
        calls = 0
        def request(payload):
            nonlocal calls
            calls += 1
            if calls == 2:
                return response("finish", "length", OUTPUT_LIMIT)
            return response("finish")
        meter = self.meter(request)
        run = DevelopmentRun(meter, self.events.append, self.cases)
        with self.assertRaisesRegex(InvalidModelResponse, "output_limit"):
            run.run()
        run.verify()
        self.assertEqual(run.summary()["scored_episodes"], {"A0": 1, "A3": 0})
        self.assertEqual(run.ledgers["A3"].events, ())
        self.assertEqual(meter.budget.calls, 2)
        self.assertEqual(len([e for e in self.events if e["record_type"] == "call_finished"]), 2)

    def test_pre_call_reservation_stops_before_io(self):
        meter = self.meter(lambda payload: self.fail("unexpected network call"), cap=0.0001)
        run = DevelopmentRun(meter, self.events.append, self.cases)
        with self.assertRaisesRegex(PilotError, "dollar_limit"):
            run.run()
        run.verify()
        self.assertEqual(meter.budget.calls, 0)
        self.assertEqual(run.summary()["paired_cases"], 0)

    def test_manifest_has_frozen_order_and_actual_schema(self):
        setup = manifest(self.cases, "test-commit", 0.8, 1.5)
        self.assertEqual(setup["order"][:4], [(1, "A0"), (1, "A3"), (2, "A3"), (2, "A0")])
        self.assertEqual(setup["order"], case_order(self.cases))
        self.assertEqual(len(setup["prompt_sha256"]), 64)
        self.assertEqual(setup["max_calls"], 288)
        self.assertEqual(payload_for({})["max_tokens"], OUTPUT_LIMIT)
        self.assertEqual(setup["authorized_cap_usd"], 1.5)


    def test_action_only_changes_schema_not_prompt_or_limits(self):
        old = payload_for({"example": 1}, REVISION)
        new = payload_for({"example": 1}, ACTION_ONLY_REVISION)
        old_schema = old.pop("response_format")
        new_schema = new.pop("response_format")
        self.assertEqual(old, new)
        self.assertEqual(old_schema["json_schema"]["schema"]["required"], ["action", "reason_code"])
        self.assertEqual(new_schema["json_schema"]["schema"]["required"], ["action"])
        self.assertEqual(new_schema["json_schema"]["schema"]["properties"]["action"],
                         old_schema["json_schema"]["schema"]["properties"]["action"])
        a = manifest(self.cases, "test-commit", 0.8, 1.5, REVISION)
        b = manifest(self.cases, "test-commit", 0.8, 1.5, ACTION_ONLY_REVISION)
        self.assertEqual(b["revision"], ACTION_ONLY_REVISION)
        self.assertNotEqual(a["prompt_sha256"], b["prompt_sha256"])
        for key in ("order", "cases", "max_calls", "max_output_tokens", "max_actions"):
            self.assertEqual(a[key], b[key])

    def test_action_only_strict_parse_and_usage(self):
        data = response("finish")
        data["choices"][0]["message"]["content"] = '{"action":"finish"}'
        decision, usage = parse_linked(data, ACTION_ONLY_REVISION)
        self.assertIs(decision.action, Move.FINISH)
        self.assertIsNone(decision.reason_code)
        self.assertEqual(usage, Usage(100, 0, 40))
        with self.assertRaises(InvalidModelResponse):
            parse_linked(data, REVISION)
        for body in ('{"action":"invented"}', '{"action":"finish","reason_code":"goal_complete"}',
                     '{"action":"finish","action":"retry_refund"}', '{}', '{"action":null}', 'finish'):
            data["choices"][0]["message"]["content"] = body
            with self.assertRaises(InvalidModelResponse) as error:
                parse_linked(data, ACTION_ONLY_REVISION)
            self.assertEqual(error.exception.usage, usage)

    def test_action_only_stream_replays(self):
        def request(payload):
            data = fake_reference(payload)
            decision = json.loads(data["choices"][0]["message"]["content"])
            data["choices"][0]["message"]["content"] = json.dumps({"action": decision["action"]})
            return data
        meter = Meter(2.0, request, self.events.append,
                      parser=lambda data: parse_linked(data, ACTION_ONLY_REVISION),
                      max_calls=MAX_CALLS, max_input_bytes=MAX_INPUT_BYTES)
        run = DevelopmentRun(meter, self.events.append, self.cases, ACTION_ONLY_REVISION)
        run.run()
        run.verify()
        self.assertEqual(run.summary()["scored_episodes"], {"A0": 12, "A3": 12})
        self.assertEqual(run.summary()["by_arm"]["A3"]["safe_successes"], 12)
        self.assertTrue(all(e["decision"]["reason_code"] is None for e in self.events
                            if e["record_type"] == "call_finished"))

    def test_truncated_content_shape_without_raw_reasoning_or_action(self):
        data = response("finish", "length", OUTPUT_LIMIT)
        body = '{"action":"finish"}' + " " * 100
        data["choices"][0]["message"]["content"] = body
        data["choices"][0]["message"]["reasoning"] = "PRIVATE_REASONING"
        data["usage"]["completion_tokens_details"] = {"reasoning_tokens": 1001}
        meter = Meter(2.0, lambda _: data, self.events.append,
                      parser=lambda data: parse_linked(data, ACTION_ONLY_REVISION),
                      max_calls=MAX_CALLS, max_input_bytes=MAX_INPUT_BYTES)
        with self.assertRaisesRegex(InvalidModelResponse, "output_limit"):
            meter.decide(payload_for({}, ACTION_ONLY_REVISION), 1, 0)
        result = self.events[-1]
        self.assertIsNone(result["decision"])
        self.assertEqual(result["reasoning_tokens"], 1001)
        self.assertEqual(result["content_shape"]["trailing_whitespace_characters"], 100)
        self.assertEqual(result["content_shape"]["characters"], len(body))
        self.assertEqual(len(result["content_shape"]["sha256"]), 64)
        self.assertNotIn("PRIVATE_REASONING", json.dumps(self.events))
        self.assertNotIn(body, json.dumps(self.events))
        self.assertEqual(meter.budget.calls, 1)


if __name__ == "__main__":
    unittest.main()
