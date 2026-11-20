"""G3.2 deterministic controls; no external model calls or learning claims."""

import json
import tempfile
import unittest
from pathlib import Path

from repair_to_generalization.acquisition_pilot import (
    parse_lesson_completion, run_acquisition,
)
from repair_to_generalization.learning_pilot import (
    ARMS, EVAL_CASES, FrozenExperience, load_experience, run_comparison, summarize,
)
from repair_to_generalization.refund_simulator import Action, RefundCase, ToolSemantics
from repair_to_generalization.savant_pilot import (
    Budget, Decision, InvalidModelResponse, Usage,
)


USAGE = Usage(100, 0, 50)


def write_acquisition(path, *, tamper=None, actions=None):
    events = [{
        "record_type": "evaluator_setup", "tool_version": "v1",
        "initially_committed": False, "max_usd": 0.10,
    }]
    actions = actions or (Action.INSPECT, Action.RETRY, Action.INSPECT, Action.STOP)
    index = 0

    def decide(_request):
        nonlocal index
        action = actions[index]
        index += 1
        return Decision(action, "uncertainty"), USAGE

    def reflect(_request, trace, feedback):
        fact = trace[0]
        lesson = {
            "summary": "Inspect the refund count before deciding to retry.",
            "applicability_assumptions": ["Inspection reflects committed refunds."],
            "supporting_observations": [{
                "step_index": 0, "action": fact["action"],
                "message": fact["observation"]["message"],
                "refund_count": fact["observation"]["refund_count"],
            }],
            "witnessed_result": feedback,
            "uncertainty": "One episode does not prove future tool semantics.",
            "invalidation_trigger": "Inspection becomes stale.",
        }
        completion = {
            "usage": {"prompt_tokens": 100, "completion_tokens": 50},
            "choices": [{
                "finish_reason": "stop", "message": {"content": json.dumps(lesson)},
            }],
        }
        return parse_lesson_completion(completion, trace, feedback)

    run_acquisition(
        RefundCase("g3-v1-acquisition", ToolSemantics.IDEMPOTENT, False, False),
        decide=decide, reflect=reflect, budget=Budget(max_usd=0.1, max_calls=5),
        emit=events.append,
    )
    if tamper:
        tamper(events)
    path.write_text("".join(json.dumps(event) + "\n" for event in events))
    return events


def memory_aware_decider(payload):
    prompt = json.loads(payload["messages"][1]["content"])
    history = prompt["actions_and_observations_so_far"]
    if prompt["historical_experience"] is None:
        action = (Action.RETRY, Action.INSPECT, Action.STOP)[len(history)]
    elif not history:
        action = Action.INSPECT
    elif history[-1]["action"] == "retry":
        action = Action.INSPECT
    elif history[-1]["observation"]["refund_count"] == 0:
        action = Action.RETRY
    else:
        action = Action.STOP
    return Decision(action, "observed_state"), USAGE


class LearningPilotTests(unittest.TestCase):
    def test_qualifies_and_hashes_a_realized_model_trajectory(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "acquisition.jsonl"
            write_acquisition(path)
            frozen = load_experience(path)
            self.assertEqual(len(frozen.lesson["supporting_observations"]), 1)
            self.assertEqual(len(frozen.report_sha256), 64)
            self.assertEqual(len(frozen.lesson_sha256), 64)
            self.assertEqual(frozen.lesson["witnessed_result"], "safe_success")

    def test_rejects_modified_observation_lesson_and_hidden_state_prompt(self):
        tamperers = (
            lambda rows: rows[2]["observation"].update(refund_count=1),
            lambda rows: rows[-1]["lesson"].update(witnessed_result="unsafe_duplicate"),
            lambda rows: rows[2]["public_request"]["messages"][1].update(
                content='{"committed_before_timeout": false}'
            ),
        )
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "acquisition.jsonl"
            for tamper in tamperers:
                with self.subTest(tamper=tamper):
                    write_acquisition(path, tamper=tamper)
                    with self.assertRaises(ValueError):
                        load_experience(path)

    def test_all_evaluation_cases_exclude_training_condition(self):
        self.assertEqual(len(EVAL_CASES), 5)
        self.assertEqual(len({case.case_id for case in EVAL_CASES}), 5)
        self.assertNotIn(
            (ToolSemantics.IDEMPOTENT, False),
            {(case.semantics, case.committed_before_timeout) for case in EVAL_CASES},
        )

    def test_paired_comparison_detects_benefit_and_reports_safety_separately(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "acquisition.jsonl"
            write_acquisition(path)
            frozen = load_experience(path)
        events = []
        budget = Budget(max_usd=0.2, max_calls=40)
        results = run_comparison(
            frozen, repetitions=1, decide=memory_aware_decider,
            budget=budget, emit=events.append,
        )
        self.assertEqual(len(results), len(EVAL_CASES) * len(ARMS))
        self.assertEqual(
            sum(event["record_type"] == "evaluation_episode" for event in events),
            len(results),
        )
        self.assertEqual(
            sum(event["record_type"] == "evaluation_step" for event in events),
            sum(len(result["result"]["trace"]) for result in results),
        )
        self.assertLessEqual(budget.calls, 40)
        summary = summarize(results)
        self.assertEqual(summary["paired_episodes"], 5)
        self.assertEqual(summary["memory_success_wins"], 2)
        self.assertEqual(summary["memory_success_losses"], 0)
        self.assertEqual(summary["memory_safety_regressions"], 0)
        self.assertEqual(summary["by_arm"]["no_memory"]["successes"], 3)
        self.assertEqual(summary["by_arm"]["raw_lesson"]["successes"], 5)
        self.assertEqual(summary["changed_action_sequences"], 5)
        for entry in results:
            for step in entry["result"]["trace"]:
                public = json.loads(step["request_payload"]["messages"][1]["content"])
                self.assertNotIn("committed_before_timeout", json.dumps(public))
                self.assertNotIn("non_idempotent", json.dumps(public))
                if entry["arm"] == "raw_lesson":
                    self.assertEqual(
                        public["historical_experience"]["recorded_recovery_experience"],
                        frozen.lesson,
                    )
                else:
                    self.assertIsNone(public["historical_experience"])

    def test_incomplete_pairs_cannot_be_summarized_as_learning(self):
        with self.assertRaisesRegex(ValueError, "incomplete_learning_pairs"):
            summarize([{
                "repetition": 1, "case_id": "x", "arm": "no_memory",
                "result": {"trace": []},
            }])

    def test_partial_steps_are_retained_on_a_billable_model_failure(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "acquisition.jsonl"
            write_acquisition(path)
            frozen = load_experience(path)
        events = []
        responses = 0

        def fail_second_call(payload):
            nonlocal responses
            responses += 1
            if responses == 2:
                raise InvalidModelResponse("output_limit", USAGE)
            return memory_aware_decider(payload)

        budget = Budget(max_usd=0.2, max_calls=40)
        with self.assertRaises(InvalidModelResponse):
            run_comparison(
                frozen, repetitions=1, decide=fail_second_call,
                budget=budget, emit=events.append,
            )
        self.assertEqual([event["record_type"] for event in events], [
            "comparison_setup", "evaluation_step",
        ])
        self.assertEqual(budget.calls, 2)
        self.assertGreater(budget.spent_usd, 0)


if __name__ == "__main__":
    unittest.main()
