"""Deterministic G3.1 controls: model-owned actions, provenance, and budgets."""

import json
import unittest

from repair_to_generalization.acquisition_pilot import (
    parse_lesson_completion, run_acquisition, validate_lesson,
)
from repair_to_generalization.refund_simulator import Action, RefundCase, ToolSemantics
from repair_to_generalization.savant_pilot import (
    Budget, Decision, InvalidModelResponse, Usage,
)


USAGE = Usage(100, 0, 100)


def candidate_from(trace, feedback):
    first = trace[0]
    return {
        "summary": "Inspect the live refund count before deciding whether to retry.",
        "applicability_assumptions": ["Inspection remains authoritative."],
        "supporting_observations": [{
            "step_index": 0,
            "action": first["action"],
            "message": first["observation"]["message"],
            "refund_count": first["observation"]["refund_count"],
        }],
        "witnessed_result": feedback,
        "uncertainty": "One episode cannot establish future tool semantics.",
        "invalidation_trigger": "Inspection becomes stale or retry semantics change.",
    }


def completion(candidate, usage=USAGE):
    return {
        "usage": {
            "prompt_tokens": usage.prompt_tokens,
            "completion_tokens": usage.completion_tokens,
            "prompt_tokens_details": {"cached_tokens": usage.cached_tokens},
        },
        "choices": [{"finish_reason": "stop", "message": {"content": json.dumps(candidate)}}],
    }


def v1_case(committed=False):
    return RefundCase("g3-train", ToolSemantics.IDEMPOTENT, committed, False)


class AcquisitionPilotTests(unittest.TestCase):
    def test_agent_controls_trajectory_and_creates_grounded_lesson(self):
        events = []
        requests = []

        def decide(payload):
            requests.append(payload)
            step = len(requests) - 1
            return Decision((Action.INSPECT, Action.RETRY, Action.STOP)[step], "uncertainty"), USAGE

        def reflect(payload, trace, feedback):
            requests.append(payload)
            return parse_lesson_completion(completion(candidate_from(trace, feedback)), trace, feedback)

        budget = Budget(max_usd=0.20, max_calls=5)
        result = run_acquisition(
            v1_case(), decide=decide, reflect=reflect, budget=budget, emit=events.append,
        )
        self.assertTrue(result.outcome.success)
        self.assertEqual([event["record_type"] for event in events], [
            "acquisition_start", "tool_step", "tool_step", "tool_step",
            "terminal_feedback", "validated_lesson",
        ])
        self.assertEqual([event["action"] for event in events if event["record_type"] == "tool_step"],
                         ["inspect", "retry", "stop"])
        self.assertEqual(result.lesson.supporting_observations[0].refund_count, 0)
        self.assertEqual(result.lesson.witnessed_result, "safe_success")
        self.assertEqual(budget.calls, 4)
        self.assertGreater(budget.spent_usd, 0)
        self.assertNotIn("committed_before_timeout", json.dumps(requests))
        self.assertNotIn("initially_committed", json.dumps(requests))
        self.assertNotIn("g3-train", json.dumps(requests))
        self.assertNotIn('"semantics": "idempotent"', json.dumps(requests))

    def test_hidden_prior_commit_never_changes_first_public_prompt(self):
        first_prompts = []
        for committed in (False, True):
            requests = []

            def decide(payload):
                requests.append(payload)
                return Decision(Action.STOP, "uncertainty"), USAGE

            run_acquisition(
                v1_case(committed),
                decide=decide,
                reflect=lambda payload, trace, feedback: parse_lesson_completion(
                    completion(candidate_from(trace, feedback)), trace, feedback,
                ),
                budget=Budget(max_usd=0.20, max_calls=5),
                emit=lambda event: None,
            )
            first_prompts.append(requests[0])
        self.assertEqual(first_prompts[0], first_prompts[1])
        self.assertNotIn("committed_before_timeout", json.dumps(first_prompts))

    def test_rejects_fabricated_or_mismatched_observations(self):
        trace = [{
            "action": "inspect",
            "observation": {"action": "inspect", "message": "current_committed_refunds", "refund_count": 1},
        }]
        for change in (
            {"step_index": 4}, {"step_index": True}, {"action": "retry"},
            {"message": "fake"}, {"refund_count": 0}, {"refund_count": True},
        ):
            with self.subTest(change=change):
                candidate = candidate_from(trace, "safe_success")
                candidate["supporting_observations"][0].update(change)
                with self.assertRaises(ValueError):
                    validate_lesson(candidate, trace, "safe_success")

    def test_rejects_false_terminal_feedback_and_duplicate_steps(self):
        trace = [{
            "action": "stop",
            "observation": {"action": "stop", "message": "episode_finished", "refund_count": None},
        }]
        candidate = candidate_from(trace, "safe_success")
        with self.assertRaisesRegex(ValueError, "lesson_outcome_mismatch"):
            validate_lesson(candidate, trace, "unsuccessful_stop")
        candidate["supporting_observations"] *= 2
        with self.assertRaises(ValueError):
            validate_lesson(candidate, trace + trace, "safe_success")

    def test_malformed_reflection_is_billable_once(self):
        trace = [{
            "action": "stop",
            "observation": {"action": "stop", "message": "episode_finished", "refund_count": None},
        }]
        candidate = candidate_from(trace, "safe_success")
        candidate["supporting_observations"][0]["message"] = "invented"
        with self.assertRaises(InvalidModelResponse) as error:
            parse_lesson_completion(completion(candidate), trace, "safe_success")
        self.assertEqual(error.exception.usage, USAGE)

    def test_incomplete_four_action_episode_still_gets_truthful_feedback(self):
        events = []
        budget = Budget(max_usd=0.20, max_calls=5)
        result = run_acquisition(
            v1_case(),
            decide=lambda payload: (Decision(Action.INSPECT, "uncertainty"), USAGE),
            reflect=lambda payload, trace, feedback: parse_lesson_completion(
                completion(candidate_from(trace, feedback)), trace, feedback,
            ),
            budget=budget, emit=events.append,
        )
        self.assertFalse(result.outcome.completed)
        self.assertEqual(result.lesson.witnessed_result, "incomplete")
        self.assertEqual(budget.calls, 5)

    def test_partial_trace_survives_invalid_reflection(self):
        events = []
        budget = Budget(max_usd=0.20, max_calls=5)

        def invalid_reflection(payload, trace, feedback):
            raise InvalidModelResponse("lesson_invalid", USAGE)

        with self.assertRaises(InvalidModelResponse):
            run_acquisition(
                v1_case(),
                decide=lambda payload: (Decision(Action.STOP, "uncertainty"), USAGE),
                reflect=invalid_reflection, budget=budget, emit=events.append,
            )
        self.assertEqual([event["record_type"] for event in events], [
            "acquisition_start", "tool_step", "terminal_feedback",
        ])
        self.assertEqual(events[-1]["witnessed_result"], "unsuccessful_stop")
        self.assertEqual(budget.calls, 2)
        self.assertGreater(budget.spent_usd, USAGE.usd)

    def test_invalid_usage_is_not_silently_accepted(self):
        bad = completion({}, Usage(1, 0, 1))
        bad["usage"]["prompt_tokens"] = True
        with self.assertRaisesRegex(InvalidModelResponse, "lesson_invalid_usage"):
            parse_lesson_completion(bad, [], "safe_success")


if __name__ == "__main__":
    unittest.main()
