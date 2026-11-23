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
| **G4 — Experience repair** | **After contradicted assumptions, agent revises/invalidates lessons and improves on paired changed-world cases** | **Repair acquisition demonstrated; incentive-confounded effect, non-exposed transfer not established** |
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
- **G4.3:** 10 repetitions / 60 episodes on two no-commit V2 cases: repaired lesson 11/20 successes, original 8/20, no-memory 2/20; no duplicates. On the **repair-exposed hidden** condition repaired 7/10 vs original 4/10, but on the **non-exposed visible** condition both were 4/10. The predeclared non-exposed criterion was **not met**. In the six visible repaired failures, three stopped immediately, and three inspected a committed refund count of 0 then stopped. Tool costs were higher under repair.
- **G4.4:** With the exact same Savant-authored lessons and simulator but inspection cost **5 instead of 25**, the five-repetition, 30-episode evaluation achieved 10/10 successes for no memory, 9/10 for original memory, and 10/10 for repaired memory; **zero duplicates**. The non-exposed visible condition scored 5/5 repaired vs 4/5 original **and 5/5 no-memory**. Repaired visible unsuccessful stops fell from 6/10 at cost 25 to 0/5 at cost 5. The difference is consistent with the reward-design confound, but the cross-price runs used fresh stochastic samples and different repetition counts; **do not claim a statistically paired causal effect**. The new setting has a ceiling effect that cannot distinguish repaired experience from no memory.
- These comparisons establish observable **experience-conditioned agent behavior** in one synthetic refund family. They do not establish model-weight updates, statistically reliable improvement, or independent task-family transfer.

Detailed G3/G4 run traces remain in local ignored `reports/runs/` files and user-supplied JSONL, not in Git. They are not recreated by these summaries.

## Next decisive gate: G5 transfer to a new task setting

**Result:** G4.4 supports incentive misalignment as a major alternative explanation for G4.3's unsuccessful stops. Lower inspection cost led to near-ceiling success for **all** memory arms. The repaired lesson did not demonstrate a distinct advantage over no memory on the non-exposed case. Further identical refund repetitions are not the decisive next investment.

**Research question:** Does Savant's existing *unchanged* agent-authored repair improve safety and successful action selection in a genuinely new tool/task setting, beyond no memory, the original lesson, and irrelevant-memory controls?

**Minimal design contract before any paid inference:**

1. Freeze the G3.1 original and G4.1 repaired lessons, including their source hashes. Do **not** rewrite them for the test task, hand-author a repair, or introduce labeled test feedback into the agent memory.
2. Define at least one new independent workflow domain with different tool names/state transitions from the refund simulator. Include a task where wrong retries can cause duplicate side effects and where authoritative observation can establish completion, plus a negative-control condition where the refund lesson is irrelevant or misleading.
3. Fix cases, prompts, reward/cost alignment, a simple deterministic truth oracle, and the scoring gate **before** calling Savant. Avoid rewards that favor abandonment; compare safe task completion, unsafe side effects, false positive transfers, and cost separately.
4. Use paired repetitions with **no memory, original memory, repaired memory, and unrelated-memory control**, full observable traces, bounded inference and automated validation. No human reviewers or adjudication.
5. **Transfer candidate threshold:** repaired experience must show a net paired safe-completion advantage over original and no memory on new relevant tasks, no extra unsafe outcomes, and no comparable uplift from unrelated memory. Report counterexamples and uncertainty rather than claiming broad generalization from a single new family.

Engineering next: define the smallest executable new-world simulator and fixed cases, then add its paired harness and deterministic tests in one reviewable PR. Do not alter or retest the frozen G4 reports. G5 will remain **unproven** until those new tasks run successfully.

## Scope guardrails

**In scope now:** one synthetic refund tool family; one real model via configured SubQ API; agent-authored experience and repairs; deterministic validation; fixed baselines; paired tests; traces and cost limits.

**Out of scope until the G4 effect decision:** GPU fine-tuning/RL, multi-agent systems, new tool families, deployment, dashboards, hosted infrastructure, human reviewers, an LLM judge for deterministic state, and publication claims. G5 transfer requires a distinct, explicitly designed follow-up.

**One decision per milestone:** *Which uncertainty will this change resolve, and how will the result advance the active gate?* If there is no clear answer, do not build it.

## Workflow (locked)

- Repo: `https://github.com/kablewithak/repair-to-generalization` (`main`); local Windows root: `C:\Users\kabom\Documents\Machine Learning\Machine Learning Workspace\repair-to-generalization`; existing `.venv`.
- Engineering happens repo-side through a short branch/PR/merge. Then give **PowerShell main sync and merged-branch deletion** instructions. The user runs SubQ experiments locally and supplies results; do not request redundant local tests after our engineering validation.
- Keep the private PRD off GitHub. Never commit API keys or personal data. No GitHub/Kaggle CLI is assumed. Avoid unnecessary SHA tracking, repeated checks, and process ceremony.
- Update **only this compass** at material gate transitions, with observed results and the next decision. A new chat should start from this file and the handover ZIP, not reconstruct the project from memory.
