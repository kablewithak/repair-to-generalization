"""Contract and evaluation tests; no network or credits required."""

import json
import unittest

from repair_to_generalization.refund_simulator import Action, RefundCase, RefundSimulator, ToolSemantics
from repair_to_generalization.savant_pilot import (
    Arm,
    Budget,
    Decision,
    PilotError,
    Usage,
    make_payload,
    parse_completion,
    run_episode,
)


class SavantPilotTests(unittest.TestCase):
    def test_initial_prompt_cannot_distinguish_hidden_commit(self):
        committed = RefundSimulator(RefundCase("a", ToolSemantics.NON_IDEMPOTENT, True, False))
        uncommitted = RefundSimulator(RefundCase("b", ToolSemantics.NON_IDEMPOTENT, False, False))
        self.assertEqual(
            make_payload(Arm.RAW_LESSON, committed.agent_view(), [], None),
            make_payload(Arm.RAW_LESSON, uncommitted.agent_view(), [], None),
        )

    def test_simulated_inspection_can_resolve_ambiguous_timeout(self):
        case = RefundCase("uncommitted", ToolSemantics.NON_IDEMPOTENT, False, False)
        decisions = iter([Action.INSPECT, Action.RETRY, Action.STOP])
        observed_prompts = []

        def decide(payload):
            observed_prompts.append(json.loads(payload["messages"][1]["content"]))
            return Decision(next(decisions), "observed_state"), Usage(100, 0, 20)

        result = run_episode(case, Arm.NO_MEMORY, decide=decide, budget=Budget())
        self.assertTrue(result.success)
        self.assertEqual(result.actions_used, 3)
        self.assertEqual(result.model_calls, 3)
        self.assertEqual(observed_prompts[1]["actions_and_observations_so_far"][0]["observation"]["refund_count"], 0)

    def test_blind_reuse_causes_duplicate_refund(self):
        case = RefundCase("committed", ToolSemantics.NON_IDEMPOTENT, True, False)
        decisions = iter([Action.RETRY, Action.STOP])
        result = run_episode(
            case,
            Arm.RAW_LESSON,
            decide=lambda _: (Decision(next(decisions), "prior_lesson"), Usage(100, 0, 20)),
            budget=Budget(),
        )
        self.assertTrue(result.unsafe_duplicate)
        self.assertFalse(result.success)

    def test_schema_contract_and_token_usage_are_enforced(self):
        completion = {
            "choices": [{"finish_reason": "stop", "message": {
                "content": '{"action":"inspect","reason_code":"uncertainty"}'
            }}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 50,
                      "prompt_tokens_details": {"cached_tokens": 10}},
        }
        action, usage = parse_completion(completion)
        self.assertEqual(action.action, Action.INSPECT)
        self.assertGreater(usage.usd, 0)
        completion["choices"][0]["finish_reason"] = "length"
        with self.assertRaises(PilotError):
            parse_completion(completion)

    def test_budget_blocks_calls_before_network(self):
        payload = make_payload(
            Arm.NO_MEMORY,
            RefundSimulator(RefundCase("x", ToolSemantics.IDEMPOTENT, True, False)).agent_view(),
            [],
            None,
        )
        with self.assertRaisesRegex(PilotError, "dollar_limit_reached"):
            Budget(max_usd=0.000001).reserve(payload)
        budget = Budget(max_calls=1)
        budget.reserve(payload)
        with self.assertRaisesRegex(PilotError, "call_limit_reached"):
            budget.reserve(payload)


if __name__ == "__main__":
    unittest.main()
