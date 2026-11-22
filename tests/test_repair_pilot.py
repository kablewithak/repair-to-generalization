"""G4.1: public-evidence replay and one-call repair, without paid inference."""

import copy
import json
import tempfile
import unittest
from pathlib import Path

from repair_to_generalization.acquisition_pilot import validate_lesson
from repair_to_generalization.learning_pilot import FrozenExperience, run_comparison
from repair_to_generalization.refund_simulator import Action
from repair_to_generalization.repair_pilot import (
    load_failure, repair_payload, run_repair, validate_repair,
)
from repair_to_generalization.savant_pilot import (
    Budget, Decision, InvalidModelResponse, PilotError, Usage,
)


USAGE = Usage(100, 0, 50)
ORIGINAL = FrozenExperience(
    lesson={
        "summary": "Inspect before retrying an ambiguous refund.",
        "applicability_assumptions": ["An inspection is authoritative."],
        "supporting_observations": [{
            "step_index": 0, "action": "inspect", "message": "Refund count: 0",
            "refund_count": 0,
        }],
        "witnessed_result": "safe_success",
        "uncertainty": "Tool semantics may change.",
        "invalidation_trigger": "Inspection is stale.",
    },
    report_sha256="a" * 64,
    lesson_sha256="b" * 64,
)


def failing_memory_decider(payload):
    """Cause one unsafe duplicate in the memory arm on hidden V2 drift."""
    prompt = json.loads(payload["messages"][1]["content"])
    history = prompt["actions_and_observations_so_far"]
    if prompt["historical_experience"] is None:
        action = Action.STOP
    elif not history:
        action = Action.INSPECT
    elif history[-1]["action"] == "retry":
        action = Action.RETRY if sum(row["action"] == "retry" for row in history) == 1 else Action.STOP
    elif (
        history[-1]["observation"]["refund_count"] == 0
        and prompt["tool_observation"]["contract_signal"] is None
    ):
        action = Action.RETRY
    else:
        action = Action.STOP
    return Decision(action, "observed_state"), USAGE


def make_report(path):
    events = []
    run_comparison(
        ORIGINAL, repetitions=1, decide=failing_memory_decider,
        budget=Budget(max_usd=0.2, max_calls=40), emit=events.append,
    )
    path.write_text("".join(json.dumps(row) + "\n" for row in events))
    return events


def revised_candidate(failure):
    return {
        "summary": "Repeated retries after a hidden acknowledgement can duplicate a refund; stop when uncertain.",
        "applicability_assumptions": ["The current tool may not deduplicate retries."],
        "supporting_observations": [{
            "step_index": index,
            "action": row["action"],
            "message": row["observation"]["message"],
            "refund_count": row["observation"]["refund_count"],
        } for index, row in enumerate(failure.trace) if row["action"] == "retry"],
        "witnessed_result": "unsafe_duplicate",
        "uncertainty": "One failed episode does not identify the hidden commit state.",
        "invalidation_trigger": "New observations establish safe deduplication under current semantics.",
    }


class RepairPilotTests(unittest.TestCase):
    def test_loads_unique_model_failure_and_replays_receipts(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "comparison.jsonl"
            make_report(path)
            failure = load_failure(path, ORIGINAL)
        self.assertEqual(failure.repetition, 1)
        self.assertEqual(failure.terminal_feedback, "unsafe_duplicate")
        self.assertEqual([row["action"] for row in failure.trace], [
            "inspect", "retry", "retry", "stop",
        ])
        self.assertEqual(len(failure.report_sha256), 64)

    def test_repair_payload_contains_only_agent_accessible_evidence(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "comparison.jsonl"
            make_report(path)
            failure = load_failure(path, ORIGINAL)
        payload = repair_payload(ORIGINAL, failure)
        context = json.loads(payload["messages"][1]["content"])
        self.assertEqual(context["prior_agent_authored_lesson"], ORIGINAL.lesson)
        self.assertEqual(context["failed_actions_and_observations"], failure.trace)
        self.assertEqual(context["terminal_training_feedback"], "unsafe_duplicate")
        self.assertFalse(payload["store"])
        for forbidden in ('"case_id"', '"semantics"', "committed_before_timeout", "evaluator_outcome"):
            self.assertNotIn(forbidden, json.dumps(payload))

    def test_one_metered_call_accepts_evidence_grounded_revision(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "comparison.jsonl"
            make_report(path)
            failure = load_failure(path, ORIGINAL)
        events = []
        seen = []

        def reflect(request, trace, feedback):
            seen.append(request)
            self.assertEqual(trace, failure.trace)
            self.assertEqual(feedback, "unsafe_duplicate")
            return validate_lesson(revised_candidate(failure), trace, feedback), USAGE

        budget = Budget(max_usd=0.02, max_calls=1)
        revised = run_repair(
            ORIGINAL, failure, reflect=reflect, budget=budget, emit=events.append,
        )
        self.assertEqual(len(seen), 1)
        self.assertEqual(budget.calls, 1)
        self.assertGreater(budget.spent_usd, 0)
        self.assertEqual([row["record_type"] for row in events], [
            "repair_source", "repair_request", "validated_repair",
        ])
        self.assertEqual(revised.witnessed_result, "unsafe_duplicate")
        self.assertIn("reuse_unmeasured", events[-1]["validation_scope"])

    def test_rejects_report_provenance_and_model_boundary_tampering(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "comparison.jsonl"
            original_events = make_report(path)
            target = next(row for row in original_events if row["record_type"] == "evaluation_episode"
                          and row["arm"] == "raw_lesson" and row["result"]["unsafe_duplicate"])
            repetition, case_id = target["repetition"], target["case_id"]

            def unsafe_episode(rows):
                return next(row for row in rows if row["record_type"] == "evaluation_episode"
                            and row["arm"] == "raw_lesson" and row["repetition"] == repetition
                            and row["case_id"] == case_id)

            def unsafe_receipt(rows):
                return next(row for row in rows if row["record_type"] == "evaluation_step"
                            and row["arm"] == "raw_lesson" and row["repetition"] == repetition
                            and row["case_id"] == case_id)

            def inject_hidden_state(rows):
                step = unsafe_episode(rows)["result"]["trace"][0]
                prompt = json.loads(step["request_payload"]["messages"][1]["content"])
                prompt["committed_before_timeout"] = False
                replacement = json.dumps(prompt)
                step["request_payload"]["messages"][1]["content"] = replacement
                unsafe_receipt(rows)["request_payload"]["messages"][1]["content"] = replacement

            def modify_initial_cost(rows):
                step = unsafe_episode(rows)["result"]["trace"][0]
                prompt = json.loads(step["request_payload"]["messages"][1]["content"])
                prompt["costs_and_rewards"]["inspection_cost_units"] = 9999
                replacement = json.dumps(prompt)
                step["request_payload"]["messages"][1]["content"] = replacement
                unsafe_receipt(rows)["request_payload"]["messages"][1]["content"] = replacement

            tamperers = (
                lambda rows: rows[0].update(lesson_sha256="z" * 64),
                lambda rows: unsafe_episode(rows)["result"]["trace"][0]["observation"].update(refund_count=99),
                lambda rows: unsafe_receipt(rows).update(action="stop"),
                inject_hidden_state,
                modify_initial_cost,
                lambda rows: rows.append({"record_type": "run_error", "error": "failed"}),
            )
            for tamper in tamperers:
                with self.subTest(tamper=tamper):
                    rows = copy.deepcopy(original_events)
                    tamper(rows)
                    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
                    with self.assertRaises(ValueError):
                        load_failure(path, ORIGINAL)

    def test_rejects_uncited_retry_and_unchanged_lesson_after_billing(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "comparison.jsonl"
            make_report(path)
            failure = load_failure(path, ORIGINAL)
        candidate = revised_candidate(failure)
        candidate["supporting_observations"].pop()
        revised = validate_lesson(candidate, failure.trace, "unsafe_duplicate")
        with self.assertRaisesRegex(ValueError, "repair_missing_repeated_retry_evidence"):
            validate_repair(revised, ORIGINAL, failure)
        budget = Budget(max_usd=0.02, max_calls=1)
        events = []
        with self.assertRaisesRegex(PilotError, "repair_validation"):
            run_repair(
                ORIGINAL, failure, reflect=lambda *_: (revised, USAGE),
                budget=budget, emit=events.append,
            )
        self.assertEqual(budget.calls, 1)
        self.assertGreater(budget.spent_usd, 0)
        self.assertNotIn("validated_repair", [row["record_type"] for row in events])
        unchanged = revised_candidate(failure)
        unchanged["summary"] = ORIGINAL.lesson["summary"]
        with self.assertRaisesRegex(ValueError, "repair_unchanged_guidance"):
            validate_repair(validate_lesson(unchanged, failure.trace, "unsafe_duplicate"), ORIGINAL, failure)

    def test_invalid_model_completion_still_counts_against_one_call_limit(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "comparison.jsonl"
            make_report(path)
            failure = load_failure(path, ORIGINAL)
        budget = Budget(max_usd=0.02, max_calls=1)

        def invalid_reflection(*_args):
            raise InvalidModelResponse("invalid", USAGE)

        with self.assertRaises(InvalidModelResponse):
            run_repair(
                ORIGINAL, failure, reflect=invalid_reflection,
                budget=budget, emit=lambda _: None,
            )
        self.assertEqual(budget.calls, 1)
        self.assertGreater(budget.spent_usd, 0)


if __name__ == "__main__":
    unittest.main()
