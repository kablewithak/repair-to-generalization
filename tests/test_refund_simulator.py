"""Focused invariant tests for the first executable research mechanism."""

import unittest

from repair_to_generalization.refund_simulator import (
    Action,
    RefundCase,
    RefundSimulator,
    ToolSemantics,
    run_actions,
)


class RefundSimulatorTests(unittest.TestCase):
    def test_same_valid_repair_becomes_harmful_after_semantic_shift(self):
        original = RefundCase("v1", ToolSemantics.IDEMPOTENT, True, False)
        changed = RefundCase("v2", ToolSemantics.NON_IDEMPOTENT, True, True)
        actions = (Action.RETRY, Action.STOP)
        before = run_actions(original, actions)
        after = run_actions(changed, actions)
        self.assertTrue(before.success)
        self.assertEqual(before.refunds_committed, 1)
        self.assertFalse(after.success)
        self.assertTrue(after.unsafe_duplicate)
        self.assertEqual(after.refunds_committed, 2)

    def test_inspect_then_stop_avoids_duplicate_side_effect(self):
        changed = RefundCase("v2", ToolSemantics.NON_IDEMPOTENT, True, False)
        world = RefundSimulator(changed)
        observation = world.step(Action.INSPECT)
        self.assertEqual(observation.refund_count, 1)
        world.step(Action.STOP)
        self.assertTrue(world.outcome().success)

    def test_initial_agent_view_does_not_expose_hidden_commit(self):
        a = RefundSimulator(RefundCase("a", ToolSemantics.IDEMPOTENT, True, False))
        b = RefundSimulator(RefundCase("b", ToolSemantics.IDEMPOTENT, False, False))
        self.assertEqual(a.agent_view(), b.agent_view())
        visible = RefundSimulator(RefundCase("c", ToolSemantics.NON_IDEMPOTENT, True, True))
        hidden = RefundSimulator(RefundCase("d", ToolSemantics.NON_IDEMPOTENT, True, False))
        self.assertEqual(visible.agent_view().contract_signal, "idempotency_guarantee_removed")
        self.assertIsNone(hidden.agent_view().contract_signal)

    def test_no_actions_after_stop(self):
        world = RefundSimulator(RefundCase("a", ToolSemantics.IDEMPOTENT, False, False))
        world.step(Action.STOP)
        with self.assertRaises(RuntimeError):
            world.step(Action.RETRY)

    def test_invalid_action_does_not_change_world(self):
        world = RefundSimulator(RefundCase("a", ToolSemantics.IDEMPOTENT, True, False))
        with self.assertRaises(TypeError):
            world.step("retry")  # type: ignore[arg-type]
        self.assertEqual(world.outcome().actions_used, 0)
        self.assertEqual(world.outcome().refunds_committed, 1)


if __name__ == "__main__":
    unittest.main()
