"""First deterministic comparison of experience reuse after tool-semantics drift.

This is an engineering control experiment, not an LLM result. The signal-gated
policy checks current task state; it does NOT yet validate a stored lesson with
counterfactual probes.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from enum import Enum

from repair_to_generalization.refund_simulator import (
    Action,
    RefundCase,
    RefundSimulator,
    ToolSemantics,
)


class Strategy(str, Enum):
    REUSE = "reuse"
    INVALIDATE_ALL = "invalidate_all"
    SIGNAL_GATED_RECHECK = "signal_gated_recheck"


@dataclass(frozen=True)
class Trial:
    case_id: str
    strategy: Strategy
    success: bool
    unsafe_duplicate: bool
    actions_used: int
    inspected: bool


@dataclass(frozen=True)
class Summary:
    strategy: Strategy
    cases: int
    successes: int
    unsafe_duplicates: int
    mean_actions: float


def run_trial(case: RefundCase, strategy: Strategy) -> Trial:
    """A strategy only receives public observations, never hidden case state."""
    if not isinstance(strategy, Strategy):
        raise TypeError("strategy must be a Strategy enum value")

    world = RefundSimulator(case)
    view = world.agent_view()
    recheck = (
        strategy is Strategy.INVALIDATE_ALL
        or (
            strategy is Strategy.SIGNAL_GATED_RECHECK
            and view.contract_signal is not None
        )
    )
    if recheck:
        inspected = world.step(Action.INSPECT)
        if inspected.refund_count is None:
            raise RuntimeError("inspect did not return an authoritative count")
        if inspected.refund_count == 0:
            world.step(Action.RETRY)
    else:
        world.step(Action.RETRY)

    world.step(Action.STOP)
    result = world.outcome()
    return Trial(
        case_id=case.case_id,
        strategy=strategy,
        success=result.success,
        unsafe_duplicate=result.unsafe_duplicate,
        actions_used=result.actions_used,
        inspected=recheck,
    )


def benchmark_cases() -> tuple[RefundCase, ...]:
    """Matched initial-commit states, with visible and hidden drift conditions."""
    return (
        RefundCase("original-committed", ToolSemantics.IDEMPOTENT, True, False),
        RefundCase("original-not-committed", ToolSemantics.IDEMPOTENT, False, False),
        RefundCase("visible-drift-committed", ToolSemantics.NON_IDEMPOTENT, True, True),
        RefundCase("visible-drift-not-committed", ToolSemantics.NON_IDEMPOTENT, False, True),
        RefundCase("hidden-drift-committed", ToolSemantics.NON_IDEMPOTENT, True, False),
        RefundCase("hidden-drift-not-committed", ToolSemantics.NON_IDEMPOTENT, False, False),
    )


def run_matrix() -> tuple[Trial, ...]:
    return tuple(
        run_trial(case, strategy)
        for case in benchmark_cases()
        for strategy in Strategy
    )


def summarize(trials: tuple[Trial, ...]) -> tuple[Summary, ...]:
    summaries = []
    for strategy in Strategy:
        selected = [trial for trial in trials if trial.strategy is strategy]
        if not selected:
            continue
        summaries.append(
            Summary(
                strategy=strategy,
                cases=len(selected),
                successes=sum(trial.success for trial in selected),
                unsafe_duplicates=sum(trial.unsafe_duplicate for trial in selected),
                mean_actions=round(
                    sum(trial.actions_used for trial in selected) / len(selected), 3
                ),
            )
        )
    return tuple(summaries)


def main() -> None:
    trials = run_matrix()
    print(
        json.dumps(
            {
                "scope": "deterministic demonstration; no model evaluated",
                "trials": [asdict(trial) for trial in trials],
                "summary": [asdict(row) for row in summarize(trials)],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
