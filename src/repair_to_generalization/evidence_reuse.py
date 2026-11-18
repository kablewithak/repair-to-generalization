"""Amortize counterfactual lesson validation across repeated agent episodes.

The agent-side controller observes only tool signals, probe results and tool
responses. The evaluator alone supplies synthetic world configurations and
sandbox semantics. This is a deterministic control study, not an LLM evaluation.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Callable

from repair_to_generalization.comparison import Strategy, run_trial
from repair_to_generalization.counterfactual import (
    ExperimentTrial,
    ProbeClient,
    RefundProbeSandbox,
    Verdict,
    run_gated_trial,
    validate_retry_lesson,
)
from repair_to_generalization.refund_simulator import (
    Action,
    RefundCase,
    RefundSimulator,
    ToolSemantics,
)


class EvidenceCache:
    """Reuse an accepted/rejected lesson until a signal change or periodic audit.

    Unannounced semantic drift is NOT detected until an audit. If the sandbox
    stays stale, even an audit cannot identify the change.
    """

    def __init__(self, audit_every: int) -> None:
        if isinstance(audit_every, bool) or not isinstance(audit_every, int) or audit_every < 1:
            raise ValueError("audit_every must be a positive integer")
        self._audit_every = audit_every
        self._has_evidence = False
        self._verdict = Verdict.UNVERIFIABLE
        self._signal: str | None = None
        self._since_validation = 0

    def execute(self, world: RefundSimulator, sandbox: ProbeClient | None) -> ExperimentTrial:
        signal = world.agent_view().contract_signal
        should_revalidate = (
            not self._has_evidence
            or signal != self._signal
            or self._since_validation >= self._audit_every
        )
        probe_actions = 0
        if should_revalidate:
            evidence = validate_retry_lesson(sandbox)
            self._verdict = evidence.verdict
            self._signal = signal
            self._has_evidence = evidence.verdict is not Verdict.UNVERIFIABLE
            self._since_validation = 0
            probe_actions = evidence.probe_actions

        # A rejected/unverifiable unconditional retry is replaced by a live
        # inspection, never by a blind retry.
        if self._verdict is Verdict.ACCEPT:
            world.step(Action.RETRY)
        else:
            observed = world.step(Action.INSPECT)
            if observed.refund_count is None:
                raise RuntimeError("inspection returned no refund count")
            if observed.refund_count == 0:
                world.step(Action.RETRY)
        world.step(Action.STOP)
        self._since_validation += 1
        outcome = world.outcome()
        return ExperimentTrial(
            case_id=outcome.case_id,
            strategy=f"evidence_cache_{self._audit_every}",
            success=outcome.success,
            unsafe_duplicate=outcome.unsafe_duplicate,
            live_actions=outcome.actions_used,
            probe_actions=probe_actions,
            verdict=self._verdict.value,
        )


def scenario_cases(*, announced: bool) -> tuple[RefundCase, ...]:
    """Ten unchanged episodes followed by fourteen changed episodes.

    Drift begins between scheduled audits for the cache-4 baseline, so silent
    drift can produce real failures before the next validation.
    """
    return tuple(
        RefundCase(
            case_id=f"{'announced' if announced else 'silent'}-{index:02}",
            semantics=(
                ToolSemantics.IDEMPOTENT if index < 10 else ToolSemantics.NON_IDEMPOTENT
            ),
            committed_before_timeout=index % 2 == 0,
            contract_change_visible=announced and index >= 10,
        )
        for index in range(24)
    )


def run_cached_sequence(
    cases: tuple[RefundCase, ...],
    *,
    audit_every: int,
    sandbox_for_case: Callable[[RefundCase], ProbeClient | None],
) -> tuple[ExperimentTrial, ...]:
    controller = EvidenceCache(audit_every)
    return tuple(
        controller.execute(RefundSimulator(case), sandbox_for_case(case))
        for case in cases
    )


@dataclass(frozen=True)
class SequenceScore:
    scenario: str
    strategy: str
    cases: int
    successes: int
    unsafe_duplicates: int
    validations: int
    mean_live_actions: float
    mean_probe_actions: float
    mean_total_actions: float


def score(scenario: str, trials: tuple[ExperimentTrial, ...]) -> SequenceScore:
    if not trials or len({trial.strategy for trial in trials}) != 1:
        raise ValueError("score requires nonempty single-strategy trials")
    n = len(trials)
    return SequenceScore(
        scenario=scenario,
        strategy=trials[0].strategy,
        cases=n,
        successes=sum(trial.success for trial in trials),
        unsafe_duplicates=sum(trial.unsafe_duplicate for trial in trials),
        validations=sum(trial.probe_actions > 0 for trial in trials),
        mean_live_actions=round(sum(trial.live_actions for trial in trials) / n, 3),
        mean_probe_actions=round(sum(trial.probe_actions for trial in trials) / n, 3),
        mean_total_actions=round(
            sum(trial.live_actions + trial.probe_actions for trial in trials) / n, 3
        ),
    )


def run_experiment() -> tuple[SequenceScore, ...]:
    output: list[SequenceScore] = []
    for announced in (True, False):
        scenario = "announced_drift" if announced else "silent_drift"
        cases = scenario_cases(announced=announced)
        for strategy in Strategy:
            baseline = tuple(
                run_trial(case, strategy) for case in cases
            )
            converted = tuple(
                ExperimentTrial(
                    case_id=item.case_id,
                    strategy=item.strategy.value,
                    success=item.success,
                    unsafe_duplicate=item.unsafe_duplicate,
                    live_actions=item.actions_used,
                    probe_actions=0,
                    verdict=None,
                )
                for item in baseline
            )
            output.append(score(scenario, converted))

        faithful = lambda case: RefundProbeSandbox(case.semantics)
        per_task = tuple(run_gated_trial(case, faithful(case)) for case in cases)
        output.append(score(scenario, per_task))
        for interval in (4, 8, 12):
            cached = run_cached_sequence(
                cases, audit_every=interval, sandbox_for_case=faithful
            )
            output.append(score(scenario, cached))

    # Negative control: a sandbox still exposes the original tool semantics.
    cases = scenario_cases(announced=False)
    stale = run_cached_sequence(
        cases,
        audit_every=4,
        sandbox_for_case=lambda _case: RefundProbeSandbox(ToolSemantics.IDEMPOTENT),
    )
    output.append(score("silent_drift_stale_sandbox", stale))
    return tuple(output)


def main() -> None:
    print(
        json.dumps(
            {
                "scope": "deterministic sequences; no model evaluated",
                "assumption": "faithful sandbox in main comparisons; stale sandbox in negative control",
                "results": [asdict(result) for result in run_experiment()],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
