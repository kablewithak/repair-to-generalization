# From Repair to Generalization — Research Compass

> **Purpose:** A small public checkpoint, **not** the private PRD. Update this file only when evidence changes a research gate or the next decisive experiment. No separate progress bureaucracy.

## North star (locked)

**Can an AI agent acquire experience from its own interactions, repair that experience when environmental assumptions fail, and apply the revised knowledge to unseen environments?**

The contribution must concern **learning and transfer**. Verifier design, memory, benchmarks, and RL are means to answer this question, not independent workstreams.

## Research gates

| Gate | Required evidence | State |
| --- | --- | --- |
| G1 — Environment | Versioned, inspectable tool semantics with hidden state, realized actions, deterministic outcomes | Done for synthetic refund v1 |
| G2 — Failure baseline | Paired trajectories expose safety, completion, and evidence-cost tradeoffs | Exploratory baseline recorded |
| **G3 — Experience acquisition** | **Savant produces lessons from its own observed trajectories; evidence-grounded lessons are evaluated against fixed baselines** | **Mechanism demonstrated; comparative benefit mixed (one unsafe regression)** |
| **G4 — Experience repair** | **After contradicted assumptions, agent revises/invalidates lessons and improves on paired changed-world cases** | **Repair validated; improvement over original not demonstrated** |
| G5 — Held-out transfer | Repaired knowledge is evaluated on unseen environment changes/tasks, with independent repetitions and negative controls | Not started |

Parameter-level RL using a trainable open-weight model is **optional after the learning loop is proven useful**. The current SubQ Savant API does not establish access to model weights or fine-tuning.

## Established baseline evidence (20 November 2026)

- Initial deterministic and SubQ pilots showed that naive reuse can be unsafe; cheap authoritative inspection makes several strategies indistinguishable.
- The 36-episode cost pilot completed without duplicates; paid probes added overhead without improving completion in that pilot.
- The 60-episode verifier-trust pilot found one duplicate with verification unavailable; stochastic first decisions made single-rollout comparisons unreliable.
- A 200-response replicated first-decision pilot measured policy shifts but **not** realized task completion.
- **Paired full-recovery pilot:** 60/60 pairs completed; 100 model calls, 126 reused decisions, $0.037843 billed. In 60 changed-world trajectories: 48 successes, **2 actual duplicate refunds**, **10 unsuccessful immediate stops**, and no action-limit incompletions.
- The two duplicates were: (1) hidden drift with no verifier, `retry → stop`; (2) visible drift with a stale verifier returning `accept` despite the contract warning, `retry → inspect(2) → stop`.
- **Major confound:** success is worth +20 points while inspection costs 25. Safe inspection may score worse than stopping. The ten unsuccessful stops do not establish inability to recover.
- **Dependence:** six outcomes per treatment/drift cell came from only three independently sampled first decisions where hidden commit states shared identical prompts. These pilots do not establish statistical superiority or broad generalization.
- At this stage, the existing experience record was produced by **guided scripted recoveries**, not Savant's own learned lessons. Later G3 runs superseded this acquisition limitation.

Detailed synthetic traces from the 20 November paired run are stored outside Git under `reports/runs/savant_recovery_pilot_20261120_162035.jsonl` on the user's machine, and in the private handover ZIP. Do not commit raw run reports, credentials, or the private PRD by default.

## Subsequent agent-authored experience evidence (20–22 November 2026)

- **G3.1:** Savant generated a validated, evidence-grounded lesson from its own successful V1 recovery; acquisition is demonstrated, not a learning effect.
- **G3.2:** 15 paired simulator comparisons (30 episodes): no memory succeeded 10/15, agent-authored memory 11/15; one memory-induced unsafe duplicate versus none for no memory. This does **not** establish a positive learning effect.
- **G4.1:** After the witnessed unsafe failure, Savant generated a valid revised lesson, tied to the original provenance and failed public trajectory. Repair acquisition was demonstrated, reuse still unmeasured.
- **G4.2:** 15 matched tri-arm comparisons (45 episodes, including a resumed first repetition): no memory 9/15 successes, original lesson 12/15, repaired lesson 12/15. No unsafe duplicates occurred in this comparison. Repaired versus original had **two paired success wins and two losses**; no net repair advantage. Its lower mean live tool cost was partly due to unsuccessful stopping, not clean efficiency.
- These comparisons establish observable **experience-conditioned agent behavior** in one synthetic refund family. They do not establish model-weight updates, statistically reliable improvement, or independent task-family transfer.

Detailed G3/G4 run traces remain in local ignored `reports/runs/` files and user-supplied JSONL, not in Git. They are not recreated by these summaries.

## Next decisive experiment: G4.3 diagnostic repair-effect replication

**Reason:** The repair mechanism passed validation, but G4.2 did not show beneficial repair. The three prior-commit conditions were uninformative for the original-versus-repaired decision: all arms already succeeded in each. Concentrate the next fixed-budget test on the two no-commit V2 conditions where the decision is diagnostic.

1. Hold **both Savant-authored lessons, prompts, tool semantics, and scoring fixed**; only restrict case selection to `v2-no-commit-hidden` (supplied G4.1 repair feedback) and `v2-no-commit-visible` (not supplied in repair feedback).
2. Use 10 fresh repetitions with three paired arms: 60 episodes / 20 matched triads. Cap paid inference at $0.05; preserve partial trace and full-repetition resume checks.
3. Primary evidence is repaired **versus original** paired safe task success and unsafe duplicates, with the two conditions reported separately. No-memory remains a control. Cost is secondary and cannot count a cheap unsuccessful stop as improvement.
4. A **candidate** repair-effect signal needs net paired success wins on the non-exposed visible condition, no safety regressions on either condition, and no net overall loss. If absent, revisit the repair mechanism before G5; do not buy more repetitions just to pursue a positive result.

**Interpretation limit:** G4.3 selects cases after inspecting G4.2, so it is **post-hoc exploratory replication**, not a clean independent confirmatory evaluation. Even a positive G4.3 signal cannot by itself close G5 or establish statistical significance. The north-star transfer claim needs genuinely unseen environment changes or tasks and negative controls.

## Scope guardrails

**In scope now:** one synthetic refund tool family; one real model via configured SubQ API; agent-authored experience and repairs; deterministic validation; fixed baselines; paired tests; traces and cost limits.

**Out of scope until the G4 effect decision:** GPU fine-tuning/RL, multi-agent systems, new tool families, deployment, dashboards, hosted infrastructure, human reviewers, an LLM judge for deterministic state, and publication claims. G5 transfer requires a distinct, explicitly designed follow-up.

**One decision per milestone:** *Which uncertainty will this change resolve, and how will the result advance the active gate?* If there is no clear answer, do not build it.

## Workflow (locked)

- Repo: `https://github.com/kablewithak/repair-to-generalization` (`main`); local Windows root: `C:\Users\kabom\Documents\Machine Learning\Machine Learning Workspace\repair-to-generalization`; existing `.venv`.
- Engineering happens repo-side through a short branch/PR/merge. Then give **PowerShell main sync and merged-branch deletion** instructions. The user runs SubQ experiments locally and supplies results; do not request redundant local tests after our engineering validation.
- Keep the private PRD off GitHub. Never commit API keys or personal data. No GitHub/Kaggle CLI is assumed. Avoid unnecessary SHA tracking, repeated checks, and process ceremony.
- Update **only this compass** at material gate transitions, with observed results and the next decision. A new chat should start from this file and the handover ZIP, not reconstruct the project from memory.
