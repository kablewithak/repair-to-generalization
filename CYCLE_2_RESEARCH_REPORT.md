# From Repair to Generalization — Cycle 2 research report
**26 November 2026 · Savant-only synthetic-agent study · Negative and exploratory findings**

## Abstract
Can a fixed-weight AI agent acquire experience, revise invalidated knowledge, and generalize repaired knowledge to unseen environments? We tested original public interaction history, compressed and retrieved experience, and persistent model-authored receipt claims. Savant acquired observations and revised scoped predictions after contract drift. A small closed-loop development comparison was positive, but independent replication violated its prespecified safety criterion before post-drift evaluation; transfer into a distinct fulfillment family failed to outperform raw history and regressed on a cancellation control. Genuine 10K/25K/50K context probes qualified. The matched scaling comparison was interrupted at 87/162 scored episodes by reasoning-token exhaustion. Full history showed no paired safe-completion gain from the added history in available pairs; partial repaired claims showed two exploratory wins. **The north-star hypothesis was not established under the tested conditions.** This does not prove experiential learning impossible.

## Question, background and hypotheses
Cycle 1 established an inspectable simulator and model-authored lesson revision, but did not establish reliable subsequent improvement. Cycle 2 investigated whether original interaction evidence, rather than only textual lessons, could support reconstruction; whether persistent scoped revisions improve decisions; and whether they transfer. Experience is external to fixed model weights. The contribution is an empirical capability boundary and reproducible synthetic instrument, not a novelty, architecture-superiority, or broad model-comparison claim. Prior work on agent memory, experience consolidation and long-context retrieval motivated these hypotheses; no systematic literature review or independent novelty assessment is claimed.

## Experimental method
- Model: `subq-savant-1.4`, low reasoning effort, no fallback, no weight updates; action-only strict JSON after the original two-field contract failed. Completion ceiling 8192, including reasoning. Public observations and tool semantics only; hidden state and evaluator outcomes excluded from policy memory. Deterministic reference policy demonstrates solvability, never acts for the paid agent.
- Linked refund/export simulator: unknown acknowledgement, current permissions and dependencies, idempotent/non-idempotent refund, live versus lagging delivery receipt; unsafe duplicate and unauthorized effects counted even if later corrected. Successful completion requires all feasible authorized effects exactly once. Distinct fulfillment simulator adds inventory capacity, holds consumed by dispatch, releases and cancellation.
- A0 current only; A1 model-compressed lessons; A2 lexical retrieval of complete prior episodes; A3 complete original public history; A4 history plus persistent evidence-linked model-authored predictions/advice. G8 uses own independent streams. Claim transition labels derive from model predictions; the harness schedules reflection and validates evidence/lineage, not semantic truth. Fixed-history comparisons share a replay-verified acquisition source. Checkpoints never enter memory.
- Case manifests, rotation, limits, criteria and stopping rules are versioned in code. All completed scored pairs retained, including counterexamples. Missing episodes are not imputed. Conditional confirmation and ablations were not run after failed screens. Network recovery was explicitly bounded, accounted and documented. Output-contract failures stop; no resampling to obtain a favorable answer.

## Findings by gate
| Gate | Observed evidence | Verdict |
|---|---|---|
| G6 | 10K probes 3/3 complete (2 safe); 25K probes 3/3 (2 safe); 50K probes 3/3 (2 safe). Genuine history, not padded tokens. | Core protocol qualification PASS; not a safety or learning pass. |
| G7 | 59/144 scored; safe A0 8/15, A1 10/15, A2 8/14, A3 8/15. Relevant net paired safe wins A3 vs A0/A1/A2 = 0/-2/0; two added paired unsafe outcomes vs A1. | NOT MET, independently of later truncation. |
| G8 | Fixed-history review stopped before behavioral evaluation (9/10 accepted claims). Closed-loop development 24/24, four correct predictions; safe checkpoints A0 2/4, A3 2/4, A4 3/4. One shared-case win; no extra paired unsafe or control regression. Independent replication 17/108: first stream four correct claims and pre-drift extra A4 duplicate vs A0, then truncation before any post-repair checkpoint. | NOT MET. Mechanism observed, reproducible safe benefit unestablished. |
| G9 | 24/24 development episodes, 145 valid calls. Safe A0 4/6, raw A3 6/6, stale A4 5/6, revised A4 5/6. Active fulfillment A0/stale 3/4 vs A3/A4 4/4. Revised A4 regressed on cancellation, no added paired unsafe. | Development NOT MET; formal five-arm 180-episode main and independent confirmation not run. |
| G10 | Qualified 50K source; 87/162 matched characterization episodes scored before retrieval-arm truncation. See below. | BLOCKED, not complete. |
| G11 | Code, frozen manifests, summary, reproducibility command and private evidence preservation. | Release prepared; local actual-report reproduction and final archival verification remain. |

### G10 observed scaling and economics
Base acquisition: 50 episodes, 366 events; 32 new service-scope episodes extend it to 82 episodes/608 events. Three 50K probes start at 51,057 / 51,056 / 51,060 input tokens. Qualification: 198/198 accepted decisions; $0.73344888. All eight pending-export extension cases duplicated an export. The diagnostic comparison reused the frozen G7 compressed lessons and four accepted G8 reviews, not a regenerated fifth claim. The additional service scopes contain analogous mechanisms and are not a pure token-length manipulation. Same cases were exposed during G7; this is not independent confirmation.

| Condition | Safe / scored | Unsafe | Decision USD / safe | Median call ms | Max input |
|---|---:|---:|---:|---:|---:|
| 25K A1 | 5/10 | 5 | .008303 | 2407 | 2450 |
| 50K A1 | 5/10 | 4 | .011953 | 2703 | 4375 |
| 25K A2 | 3/9 | 6 | .017966 | 2789 | 4499 |
| 50K A2 | 4/10 | 6 | .017434 | 3375 | 4553 |
| 25K A3 | 3/9 | 6 | .056277 | 2867.5 | 31216 |
| 50K A3 | 4/10 | 6 | .059683 | 3195.5 | 51368 |
| 25K A4_partial | 4/9 | 5 | .036717 | 2805 | 32434 |
| 50K A4_partial | 7/10 | 3 | .036532 | 3047 | 52670 |
| 25K A3 reversed blocks | 1/5 | 4 | .093379 | 2766 | 31092 |
| 50K A3 reversed blocks | 1/5 | 4 | .151259 | 2953 | 51401 |

Decision cost includes failed calls within each condition; preparation is separate. The 25K and 50K source acquisition/qualification costs are $.51645676 and $1.24990564 (nested, not additive project costs). Historical compression including failed attempt $.0113206, new compression $.0108127, historical partial reflection including failed continuation $.0171473. For standalone economics add the corresponding preparation once per hypothetical condition, then divide by observed safe successes; do not sum such shared preparation across conditions. Zero successes would make cost per safe success undefined.

| Chronological policy | Matched pairs | 25K safe | 50K safe | Wins / losses (50K) | Added unsafe (50K) |
|---|---:|---:|---:|---:|---:|
| A1 | 10 | 5 | 5 | 2/2 | 1 |
| A2 | 9 | 3 | 3 | 0/0 | 0 |
| A3 | 9 | 3 | 3 | 0/0 | 0 |
| A4_partial | 9 | 4 | 6 | 2/0 | 0 |

Reversed episode blocks versus chronological A3: 5 pairs at each tier, no wins/losses or added unsafe. Only the observed subset supports this descriptive result, not order invariance. The comparison produced 85 finished trajectories, 37 safe successes and 49 unsafe duplicate episodes among 87 scored. Two scored trajectories reached the action limit; one more trajectory was interrupted before scoring. No permission or dependency violations were recorded in the scored set. 801 calls, 800 accepted, $1.29052090, no unknown charge. Provider-call elapsed times sum to 47.06 minutes, not total wall time. The final call was A2 at 25K source, repetition 2, study-main-06, step index 3: 2,901 input tokens, 8,192 completion/reasoning tokens, null content, length finish. This is not evidence of a 50K input ceiling.

## Failure taxonomy and evidence-linked demonstrations
1. **Premature repeated effects:** G7, G8, G10 duplicate exports when stale/empty delivery receipts are treated as proof no job exists. A4_partial at G10 repetition 1, study-main-03, 25K: audit delivered 0 at tick 0; submit; advance twice; audit delivered 2; finish unsafe. Same case at 50K: audit 0, advance, audit 0, advance, audit 1, finish safely without resubmission. This is one observed paired difference, not a causal demonstration of claim repair (claims are identical at both tiers). The other A4_partial win was study-main-04, repetition 1.
2. **Unfinished operational goal:** G9 cancelled order, A4 inspected authority and reconciled stale register but closed without rereading and releasing the existing hold. It was an incomplete goal, not a scored unauthorized mutation. Raw A3 and stale A4 succeeded.
3. **Response contract:** output exhaustion with both small and large input contexts; invalid compression evidence, invalid claim advice and inconsistent retain/revise operation. Structured prediction labels alone do not guarantee execution safety.
4. **Transport:** two unknown-usage network failures in closed-loop development; conservatively charged and recovered under documented amendments, not removed from evidence.

The demo is replay of recorded synthetic behavior, not a new agent success. Reproduction below reconstructs the entire G10 prefix including the failed call, frozen snapshots and paired summaries; the raw reports retain the original observations. The two A4_partial wins are listed in the output's scale contrasts.

## Accounting and limitations
Sixteen unique available Cycle 2 report segments contain 2,535 recorded calls, 2,526 accepted responses, four output-limit errors, two network errors, and three schema/claim errors. Segment charges sum to $3.63367018, of which $.01088230 is conservative unknown reservation and $3.62278788 reported usage. Continuation-chain totals are not added again. Latest conservative project planning spend is $4.52, inclusive of historical prior; actual lifetime invoice/credit balance is not verified (Cycle 1 G5.1 usage incomplete). No further inference authorized. Raw reports, private PRD and reasoning text are not published.

The samples are small, synthetic, repeated and partially interrupted. No p-values, population confidence estimates or general-purpose reliability claims are warranted. One fresh replication stream reached acquisition/claim revision, none reached post-repair scoring; one distinct-family development cohort was completed. The independent main, repair ablation and transfer confirmation were not completed. Reflection timing is harness-controlled. Observed predictions and provenance are not proof of semantic advice validity. G10 adds genuine history with potentially useful analogous evidence, so neither length alone nor repaired memory causality is isolated. Higher context tiers were conditional and not attempted.

## Reproduce without inference
Python 3.11+, standard library, repository root, original ignored reports under `reports/runs/`. The recorded scaling commit is `89ff81cdbc51280350b9939c054960f4f60a4ef5`. Use the release version for the read-only verifier. No API key or network call is needed.
```powershell
$env:PYTHONPATH = Join-Path (Get-Location) 'src'
python -m repair_to_generalization.g10_reproduce `
    --source reports/runs/savant_g7_source_20261125_150614.jsonl `
    --extension reports/runs/savant_g10_50k_20261126_014723.jsonl `
    --compression reports/runs/savant_g7_main_v2_20261125_170804.jsonl `
    --claims reports/runs/savant_g8_diagnostic_20261125_182237.jsonl `
    --report reports/runs/savant_g10_scaling_20261126_100602.jsonl
```
Expected verified result: 801 recorded calls, 87 scored episodes, error output_limit, zero new model calls, G10 false. A successful verifier exit confirms report consistency, not provider attestation or a scientific pass. The private archive contains original report hashes and the historical Cycle 1 package; use exact bytes. Offline regression command: `python -m unittest discover -s tests -q`. Code and frozen manifests: `study_design.py`, `g7_compare.py`, `g8_independent.py`, `fulfillment_world.py`, `g9_fulfillment_dev.py`, `g10_50k_qualification.py`, `g10_scaling.py`. Related PRs 32–45 preserve protocol decisions and failures.

## Conclusions and actionable recommendations
Within these tested settings, the agent can acquire its own interaction evidence and revise scoped predictions after observed changes. We did not establish reliable improvement from repaired knowledge or safe independently replicated transfer. Full-history access alone was insufficient. The repeated duplicate-export behavior suggests future work on explicit reconciliation state and evidence-to-action consistency, and provider work on separating reasoning budget from guaranteed structured decision emission. A new study would need its own frozen protocol, unchanged baselines and independent task families, not a rerun to rescue these outcomes.

**North star achieved: NO — not established.**  
**Research completion: release prepared; final offline reproduction and private archive verification remain.**  
Do not inflate the historical 62/100 qualitative checkpoint without a separate rubric review.
