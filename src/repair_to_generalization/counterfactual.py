"""Counterfactual validation of an old retry lesson against a sandbox tool.

The reusable lesson is: "retry after an acknowledgement timeout". A pair of
isolated probes challenges that lesson with both possible prior commit states.
The gate sees probe outcomes and live tool observations, never live hidden state.

Crucial assumption: a probe sandbox is available and accurately mirrors the
current tool semantics. We report stale/missing sandbox counterexamples rather
than hiding that assumption. This is not an LLM result.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Protocol

from repair_to_generalization.comparison import Strategy, benchmark_cases, run_trial
from repair_to_generalization.refund_simulator import (
    Action,
    Outcome,
    RefundCase,
    RefundSimulator,
    ToolSemantics,
    run_actions,
)


class Verdict(str, Enum):
    ACCEPT = "accept"
    REJECT = "reject"
    UNVERIFIABLE = "unverifiable"


class ProbeClient(Protocol):
    def try_retry(self, *, prior_commit: bool) -> Outcome:
        """Run a disposable episode; never use real transaction state."""


class RefundProbeSandbox:
    """Independent test fixture; its semantics may match or lag the live tool."""

    def __init__(self, semantics: ToolSemantics) -> None:
        self._semantics = semantics

    def try_retry(self, *, prior_commit: bool) -> Outcome:
        case = RefundCase(
            case_id=f"probe-prior-commit-{int(prior_commit)}",
            semantics=self._semantics,
            committed_before_timeout=prior_commit,
            contract_change_visible=False,
        )
        return run_actions(case, (Action.RETRY, Action.STOP))


@dataclass(frozen=True)
class Evidence:
    verdict: Verdict
    probe_episodes: int
    probe_actions: int
    failed_conditions: tuple[str, ...]


def validate_retry_lesson(sandbox: ProbeClient | None) -> Evidence:
    """Challenge unconditional retry with both counterfactual prior states."""
    if sandbox is None:
        return Evidence(Verdict.UNVERIFIABLE, 0, 0, ())

    failures: list[str] = []
    actions = 0
    for prior_commit in (True, False):
        result = sandbox.try_retry(prior_commit=prior_commit)
        actions += result.actions_used
        if not result.completed or not result.success or result.unsafe_duplicate:
            failures.append(f"prior_commit={prior_commit}")
    return Evidence(
        verdict=Verdict.REJECT if failures else Verdict.ACCEPT,
        probe_episodes=2,
        probe_actions=actions,
        failed_conditions=tuple(failures),
    )


@dataclass(frozen=True)
class ExperimentTrial:
    case_id: str
    strategy: str
    success: bool
    unsafe_duplicate: bool
    live_actions: int
    probe_actions: int
    verdict: str | None


def run_gated_trial(case: RefundCase, sandbox: ProbeClient | None) -> ExperimentTrial:
    """Probe a lesson before it can affect an independent live episode.

    Only the experiment fixture knows the live case. The gate uses its probe
    client and the live tool's action/observation surface, not case.semantics.
    """
    evidence = validate_retry_lesson(sandbox)
    live = RefundSimulator(case)
    if evidence.verdict is Verdict.ACCEPT:
        live.step(Action.RETRY)
    else:
        observed = live.step(Action.INSPECT)
        if observed.refund_count is None:
            raise RuntimeError("inspection must return an observed refund count")
        if observed.refund_count == 0:
            live.step(Action.RETRY)
    live.step(Action.STOP)
    result = live.outcome()
    return ExperimentTrial(
        case_id=case.case_id,
        strategy="counterfactual_gate",
        success=result.success,
        unsafe_duplicate=result.unsafe_duplicate,
        live_actions=result.actions_used,
        probe_actions=evidence.probe_actions,
        verdict=evidence.verdict.value,
    )


def run_matrix() -> tuple[ExperimentTrial, ...]:
    """Six matched tasks, four strategies; sandbox is perfectly faithful.

    Important: sandbox configuration is supplied by the experiment fixture,
    not inferred by the gate. Fidelity is deliberately varied in separate
    negative-control trials.
    """
    trials: list[ExperimentTrial] = []
    for case in benchmark_cases():
        for strategy in Strategy:
            trial = run_trial(case, strategy)
            trials.append(
                ExperimentTrial(
                    case_id=trial.case_id,
                    strategy=strategy.value,
                    success=trial.success,
                    unsafe_duplicate=trial.unsafe_duplicate,
                    live_actions=trial.actions_used,
                    probe_actions=0,
                    verdict=None,
                )
            )
        sandbox = RefundProbeSandbox(case.semantics)
        trials.append(run_gated_trial(case, sandbox))
    return tuple(trials)


@dataclass(frozen=True)
class Score:
    strategy: str
    cases: int
    successes: int
    unsafe_duplicates: int
    mean_live_actions: float
    mean_probe_actions: float
    mean_total_actions: float


def summarize(trials: tuple[ExperimentTrial, ...]) -> tuple[Score, ...]:
    scores: list[Score] = []
    for strategy in (*[s.value for s in Strategy], "counterfactual_gate"):
        matches = [trial for trial in trials if trial.strategy == strategy]
        if not matches:
            continue
        n = len(matches)
        scores.append(
            Score(
                strategy=strategy,
                cases=n,
                successes=sum(t.success for t in matches),
                unsafe_duplicates=sum(t.unsafe_duplicate for t in matches),
                mean_live_actions=round(sum(t.live_actions for t in matches) / n, 3),
                mean_probe_actions=round(sum(t.probe_actions for t in matches) / n, 3),
                mean_total_actions=round(
                    sum(t.live_actions + t.probe_actions for t in matches) / n, 3
                ),
            )
        )
    return tuple(scores)


def main() -> None:
    trials = run_matrix()
    shifted = RefundCase("stale-sandbox-drift", ToolSemantics.NON_IDEMPOTENT, True, False)
    stale = run_gated_trial(shifted, RefundProbeSandbox(ToolSemantics.IDEMPOTENT))
    missing = run_gated_trial(shifted, None)
    print(
        json.dumps(
            {
                "scope": "deterministic research control; no LLM evaluated",
                "assumption": "sandbox mirrors current semantics (except negative control)",
                "summary": [asdict(row) for row in summarize(trials)],
                "negative_controls": {
                    "stale_sandbox": asdict(stale),
                    "no_sandbox": asdict(missing),
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
