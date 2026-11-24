"""Deterministic G5 simulator, model-boundary, pairing and resume regression."""

import copy
import json
import tempfile
import unittest
from pathlib import Path

from repair_to_generalization.savant_pilot import Budget, InvalidModelResponse, Usage
from repair_to_generalization.transfer_world import (
    CASES, ExportAction, ExportWorld, InitialState, ExportCase,
)
from repair_to_generalization.transfer_pilot import (
    ARMS, CachedExportPolicy, FrozenSources, ExportDecision, load_resume, parse_export_response,
    payload_for, run_comparison, setup_for, summarize,
)


SOURCE = FrozenSources(
    original_lesson={"summary": "refund originally retry and inspect", "supporting_observations": []},
    revised_lesson={"summary": "refund revised inspect and avoid duplicate retries", "supporting_observations": []},
    original_report_sha256="a" * 64, original_lesson_sha256="b" * 64,
    repair_report_sha256="c" * 64, repair_lesson_sha256="d" * 64,
)
USAGE = Usage(100, 0, 30)


def deterministic_decision(payload):
    """Public-context-only fixture; no evaluator state or hidden-state oracle."""
    row = json.loads(payload["messages"][1]["content"])
    event = row["tool_observation"]["event"]
    trace = row["actions_and_observations_so_far"]
    if "cancelled" in event or "already_delivered" in event:
        choice = ExportAction.FINISH
    elif not trace:
        choice = ExportAction.AUDIT
    else:
        last = trace[-1]
        if last["action"] == ExportAction.AUDIT.value:
            observation = last["observation"]
            if observation["delivered"] == 1:
                choice = ExportAction.FINISH
            elif observation["queued"] or observation["processing"]:
                choice = ExportAction.ADVANCE
            else:
                choice = ExportAction.SUBMIT
        elif last["action"] == ExportAction.SUBMIT.value:
            choice = ExportAction.ADVANCE
        elif last["action"] == ExportAction.ADVANCE.value:
            choice = ExportAction.AUDIT
        else:
            choice = ExportAction.FINISH
    return ExportDecision(choice, "observed_state"), USAGE


def write_lines(path, records):
    path.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")


class ExportWorldTests(unittest.TestCase):
    def test_export_job_requires_two_scheduler_transitions(self):
        case = ExportCase("t", InitialState.ABSENT)
        world = ExportWorld(case)
        world.step(ExportAction.SUBMIT)
        world.step(ExportAction.ADVANCE)
        world.step(ExportAction.FINISH)
        self.assertFalse(world.outcome().success)
        ready = ExportWorld(case)
        for a in (ExportAction.SUBMIT, ExportAction.ADVANCE, ExportAction.ADVANCE,
                  ExportAction.FINISH):
            ready.step(a)
        self.assertTrue(ready.outcome().success)
        self.assertFalse(ready.outcome().unsafe_duplicate)

    def test_same_hidden_public_view_and_distinct_authoritative_queue(self):
        absent, queued = [ExportWorld(case) for case in CASES[:2]]
        self.assertEqual(absent.view(), queued.view())
        self.assertEqual(absent.step(ExportAction.AUDIT)["queued"], 0)
        self.assertEqual(queued.step(ExportAction.AUDIT)["queued"], 1)

    def test_cancelled_submission_is_unsafe_and_duplicate_dispatch_detected(self):
        cancelled = ExportWorld(CASES[-1])
        cancelled.step(ExportAction.SUBMIT)
        cancelled.step(ExportAction.FINISH)
        self.assertTrue(cancelled.outcome().unauthorized_dispatch)
        self.assertFalse(cancelled.outcome().success)
        no_op = ExportWorld(CASES[-1])
        no_op.step(ExportAction.FINISH)
        self.assertTrue(no_op.outcome().success)
        duplicated = ExportWorld(CASES[1])
        duplicated.step(ExportAction.SUBMIT)
        duplicated.step(ExportAction.FINISH)
        self.assertTrue(duplicated.outcome().unsafe_duplicate)

    def test_finished_cannot_act_further(self):
        world = ExportWorld(CASES[-1])
        world.step(ExportAction.FINISH)
        with self.assertRaises(RuntimeError):
            world.step(ExportAction.AUDIT)


class TransferPilotTests(unittest.TestCase):
    def test_frozen_memories_and_no_private_state_in_prompt(self):
        world = ExportWorld(CASES[0])
        source_before = copy.deepcopy(SOURCE)
        for arm in ARMS:
            payload = payload_for(arm, SOURCE, world, [])
            prompt = json.loads(payload["messages"][1]["content"])
            self.assertNotIn("initial_state", json.dumps(prompt))
            self.assertNotIn("case_id", json.dumps(prompt))
            memory = prompt["historical_experience"]
            self.assertEqual(memory, {
                "no_memory": None,
                "original_lesson": SOURCE.original_lesson,
                "repaired_lesson": SOURCE.revised_lesson,
                "unrelated_memory": memory,
            }[arm])
        self.assertEqual(SOURCE, source_before)

    def test_model_schema_strict_and_output_limit_preserves_usage(self):
        good = {"choices": [{"finish_reason": "stop", "message": {
            "content": json.dumps({"action": "audit_queue", "reason_code": "uncertainty"})
        }}], "usage": {"prompt_tokens": 100, "completion_tokens": 30}}
        decision, usage = parse_export_response(good)
        self.assertEqual(decision.action, ExportAction.AUDIT)
        self.assertEqual(usage, USAGE)
        bad = copy.deepcopy(good)
        bad["choices"][0]["finish_reason"] = "length"
        with self.assertRaises(InvalidModelResponse) as error:
            parse_export_response(bad)
        self.assertIsNotNone(error.exception.usage)
        bad = copy.deepcopy(good)
        bad["choices"][0]["message"]["content"] = '{"action":"retry","reason_code":"uncertainty"}'
        with self.assertRaises(InvalidModelResponse):
            parse_export_response(bad)

    def test_g5_output_budget_revision_and_truncation_is_billed_once(self):
        payload = payload_for("no_memory", SOURCE, ExportWorld(CASES[0]), [])
        setup = setup_for(SOURCE, 3)
        self.assertEqual(payload["max_tokens"], 4096)
        self.assertEqual(payload["reasoning_effort"], "low")
        self.assertEqual(setup["model_max_output_tokens"], 4096)
        self.assertEqual(setup["response_protocol_revision"], "output_budget_4096_v2")

        truncated = {
            "choices": [{"finish_reason": "length", "message": {"content": ""}}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 4096},
        }
        requests = []

        def decide(request):
            requests.append(request)
            return parse_export_response(truncated)

        budget = Budget(max_usd=0.25, max_calls=2)
        policy = CachedExportPolicy(decide, budget)
        with self.assertRaisesRegex(InvalidModelResponse, "output_limit"):
            policy.choose(payload)
        self.assertEqual(requests, [payload])
        self.assertEqual(budget.calls, 1)
        self.assertAlmostEqual(budget.spent_usd, Usage(100, 0, 4096).usd)
        self.assertEqual(policy.responses, {})

        insufficient = Budget(max_usd=0.001, max_calls=1)
        with self.assertRaisesRegex(RuntimeError, "dollar_limit_reached"):
            insufficient.reserve(payload)
        self.assertEqual(insufficient.calls, 0)

    def test_complete_paired_run_no_false_transfer_claim(self):
        events = []
        outcomes = run_comparison(
            SOURCE, 2, deterministic_decision,
            Budget(max_usd=0.2, max_calls=2 * 6 * 4 * 7), events.append,
        )
        self.assertEqual(len(outcomes), 48)
        self.assertEqual(events[0], setup_for(SOURCE, 2))
        self.assertEqual({item["arm"] for item in outcomes}, set(ARMS))
        summary = summarize(outcomes, 2)
        self.assertEqual(summary["episodes"], 48)
        self.assertIn("repaired_vs_controls_on_transfer_cases", summary)
        self.assertEqual(summary["negative_control_outcomes"]["repaired_lesson"]["unsafe_outcomes"], 0)
        self.assertEqual(sum(x["result"]["unsafe_duplicate"] for x in outcomes), 0)
        self.assertEqual(sum(x["result"]["unauthorized_dispatch"] for x in outcomes), 0)

    def test_mid_repetition_resume_preserves_all_completed_episodes(self):
        events = []
        all_outcomes = run_comparison(
            SOURCE, 2, deterministic_decision,
            Budget(max_usd=0.2, max_calls=336), events.append,
        )
        cut = 0
        seen = 0
        for i, row in enumerate(events):
            if row["record_type"] == "g5_episode":
                seen += 1
                if seen == 7:
                    cut = i + 1
                    break
        self.assertGreater(cut, 0)
        rows = events[:cut] + [{"record_type": "run_error", "error": "output_limit"}]
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "failed.jsonl"
            write_lines(path, rows)
            replay = load_resume(path, SOURCE, 2)
            self.assertEqual(len(replay.episodes), 7)
            continuations = []
            resumed = run_comparison(
                SOURCE, 2, deterministic_decision,
                Budget(max_usd=0.2, max_calls=336), continuations.append,
                replay=replay,
            )
            self.assertEqual(resumed, all_outcomes)
            self.assertEqual(len(continuations), len(events) - cut)
            tampered = copy.deepcopy(rows)
            for step in tampered:
                if step.get("record_type") == "g5_step":
                    step["observation"]["message"] = "fabricated"
                    break
            write_lines(path, tampered)
            with self.assertRaises(ValueError):
                load_resume(path, SOURCE, 2)

    def test_resume_validates_and_discards_only_last_partial_episode(self):
        events = []
        run_comparison(
            SOURCE, 2, deterministic_decision,
            Budget(max_usd=0.2, max_calls=336), events.append,
        )
        episodes = 0
        cut = 0
        for i, event in enumerate(events):
            if event["record_type"] == "g5_episode":
                episodes += 1
            elif event["record_type"] == "g5_step" and episodes == 8:
                cut = i + 1
                break
        rows = events[:cut] + [{"record_type": "run_error", "error": "output_limit"}]
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "failed.jsonl"
            write_lines(path, rows)
            replay = load_resume(path, SOURCE, 2)
            self.assertEqual(len(replay.episodes), 8)
            self.assertEqual(replay.dropped_partial_steps, 1)
            self.assertTrue(replay.last_repetition_cache)


if __name__ == "__main__":
    unittest.main()