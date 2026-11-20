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
| **G3 — Experience acquisition** | **Savant produces lessons from its own observed trajectories; evidence-grounded lessons are evaluated against fixed baselines** | **ACTIVE — not implemented** |
| G4 — Experience repair | After contradicted assumptions, agent revises/invalidates lessons and improves on paired changed-world cases | Not started |
| G5 — Held-out transfer | Repaired knowledge is evaluated on unseen environment changes/tasks, with independent repetitions and negative controls | Not started |

Parameter-level RL using a trainable open-weight model is **optional after the learning loop is proven useful**. The current SubQ Savant API does not establish access to model weights or fine-tuning.

## Established evidence (20 November 2026)

- Initial deterministic and SubQ pilots showed that naive reuse can be unsafe; cheap authoritative inspection makes several strategies indistinguishable.
- The 36-episode cost pilot completed without duplicates; paid probes added overhead without improving completion in that pilot.
- The 60-episode verifier-trust pilot found one duplicate with verification unavailable; stochastic first decisions made single-rollout comparisons unreliable.
- A 200-response replicated first-decision pilot measured policy shifts but **not** realized task completion.
- **Paired full-recovery pilot:** 60/60 pairs completed; 100 model calls, 126 reused decisions, $0.037843 billed. In 60 changed-world trajectories: 48 successes, **2 actual duplicate refunds**, **10 unsuccessful immediate stops**, and no action-limit incompletions.
- The two duplicates were: (1) hidden drift with no verifier, `retry → stop`; (2) visible drift with a stale verifier returning `accept` despite the contract warning, `retry → inspect(2) → stop`.
- **Major confound:** success is worth +20 points while inspection costs 25. Safe inspection may score worse than stopping. The ten unsuccessful stops do not establish inability to recover.
- **Dependence:** six outcomes per treatment/drift cell came from only three independently sampled first decisions where hidden commit states shared identical prompts. These pilots do not establish statistical superiority or broad generalization.
- The existing experience record was produced by **guided scripted recoveries**, not Savant's own learned lessons. **No learning or post-training improvement has been demonstrated.**

Detailed synthetic traces from the 20 November paired run are stored outside Git under `reports/runs/savant_recovery_pilot_20261120_162035.jsonl` on the user's machine, and in the private handover ZIP. Do not commit raw run reports, credentials, or the private PRD by default.

## Next milestone: G3, not another verifier pilot

**Smallest useful implementation:**

1. Have Savant **act** on v1 recovery episodes and receive real tool feedback. Record only observations available to the agent; hidden simulator state stays with the scorer.
2. Ask Savant to generate a structured, evidence-grounded lesson from its **own completed** trajectory: claim, supporting step references, assumptions, applicability, uncertainty, and invalidation triggers. Reject unsupported claims deterministically.
3. Compare **no memory / fixed historical lesson / agent-acquired lesson** on paired, held-out cases. Fresh independent repetitions; no reuse of test outcomes during acquisition.
4. Report actual success, unsafe actions, unsuccessful stopping, inspections, probe actions, model calls, and USD separately. Do not let an arbitrary net-point reward make abandonment the preferred outcome.

**G3 exit:** Reproducible agent-generated lessons, valid provenance, a completed held-out comparison, and a clearly reported positive or negative learning effect. A lesson being generated is **not** by itself evidence of improvement. If ineffective, fix acquisition before advancing to G4.

## Scope guardrails

**In scope now:** one synthetic refund tool family; one real model via configured SubQ API; experience acquisition; deterministic validation; fixed baselines; held-out testing; traces and cost limits.

**Out of scope for G3:** GPU fine-tuning/RL, multi-agent systems, new tool families, deployment, dashboards, hosted infrastructure, human reviewers, an LLM judge for deterministic state, and publication claims. Add them only if an observed G3 result creates a specific need.

**One decision per milestone:** *Which uncertainty will this change resolve, and how will the result advance the active gate?* If there is no clear answer, do not build it.

## Workflow (locked)

- Repo: `https://github.com/kablewithak/repair-to-generalization` (`main`); local Windows root: `C:\Users\kabom\Documents\Machine Learning\Machine Learning Workspace\repair-to-generalization`; existing `.venv`.
- Engineering happens repo-side through a short branch/PR/merge. Then give **PowerShell main sync and merged-branch deletion** instructions. The user runs SubQ experiments locally and supplies results; do not request redundant local tests after our engineering validation.
- Keep the private PRD off GitHub. Never commit API keys or personal data. No GitHub/Kaggle CLI is assumed. Avoid unnecessary SHA tracking, repeated checks, and process ceremony.
- Update **only this compass** at material gate transitions, with observed results and the next decision. A new chat should start from this file and the handover ZIP, not reconstruct the project from memory.
