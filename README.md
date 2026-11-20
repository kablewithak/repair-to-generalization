# From Repair to Generalization

Research into when previously validated AI-agent experience becomes unsafe to reuse after tool or policy behavior changes.

Status: **exploratory research pilot**. No measured model improvements claimed. The approved research PRD stays local and is ignored by Git.

## Local PowerShell setup

Requires Python 3.11+ and a virtual environment (see the project handover instructions). No third-party dependencies are needed for the first G0 check.

```powershell
$env:PYTHONPATH = (Join-Path (Get-Location) 'src')
python -m unittest discover -s tests -v
```

## SubQ access preflight

Set `SUBQ_API_KEY` in your local PowerShell session and run:

```powershell
python -m repair_to_generalization.preflight
```

The script checks `GET https://api.subq.ai/v1/models` for `subq-savant-1.4`, without making an inference request. It cannot confirm token billing or generation quality. It never prints or stores the API key. If the model is not listed, the key does not currently have access to it.

Official reference: https://docs.subq.ai/endpoints/models-api/

## First Savant decision pilot

The small pilot runs six synthetic refund cases under three fixed conditions: no memory, an old retry lesson, and the same lesson accompanied by verified sandbox-probe evidence. It is **exploratory**, not a claim of improvement. It records tool actions, authoritative outcomes, token usage and USD costs. The sandbox condition assumes a faithful replica of current tool behavior.

With `SUBQ_API_KEY` set in your PowerShell session:

```powershell
$env:PYTHONPATH = Join-Path (Get-Location) 'src'
.\.venv\Scripts\python.exe -m repair_to_generalization.savant_pilot --live --max-usd 0.10
```

This explicitly authorizes paid inference with a **per-run** $0.10 limit; the default call limit is 72. No API key is saved. Results go to ignored `reports/runs/` JSONL files. For the deterministic control study, use `python -m repair_to_generalization.evidence_reuse` instead.

## Cost-sensitive decision pilot

The first Savant pilot solved all six cases under every memory condition, always inspecting live state. The follow-up therefore **prices inspection explicitly** while preserving those same six underlying cases. It evaluates cheap (1 point) and expensive (8 points) inspection against three memory conditions. Synthetic scoring awards +20 for exactly one refund, subtracts 40 for a duplicate, and charges live tool and sandbox probe costs. These costs are experimental units, not dollars.

```powershell
$env:PYTHONPATH = Join-Path (Get-Location) 'src'
.\.venv\Scripts\python.exe -m repair_to_generalization.cost_pilot --live --max-usd 0.10
```

The command prints a per-condition summary and writes detailed traces to ignored `reports/runs/`. The 36 episodes are paired cost scenarios, **not** 36 independent task families. It requires `SUBQ_API_KEY`, uses a per-run API spending ceiling, and makes no real refund requests.


## Paired full-recovery pilot

The replicated first-decision study measures **action preferences**, not completed recoveries. This next experiment executes whole Savant-driven tool trajectories in matched original/changed environments. Each repetition reuses a sampled model decision whenever the complete public prompt is identical; trajectories branch only when public evidence or feedback differs. It compares hidden versus visible drift, prior commit states, and the five verifier-trust treatments at an inspection cost of 25 points.

The experience record is built from **two executed, successful guided v1 recoveries**, not a hard-coded lesson or a model-written reflection. The pilot does not claim self-learning or sandbox fidelity. It reports realized task success, duplicate refunds, incomplete/stopped episodes, and live/probe costs separately; cases are synthetic and not independent task families.

With `SUBQ_API_KEY` set in your PowerShell session:

```powershell
$env:PYTHONPATH = Join-Path (Get-Location) 'src'
.\.venv\Scripts\python.exe -m repair_to_generalization.recovery_pilot --live --max-usd 0.50
```

The default is three fresh stochastic repetitions per matched condition (60 pairs, 120 simulated trajectories). Identical prompts are billed once *within* a repetition. No paid requests are made without `--live`; the `--max-usd` budget is a per-run cap. Traces and acquisition evidence are written as JSONL under ignored `reports/runs/`. This is an exploratory completion-and-safety measurement, not evidence of generalization until results are analyzed.

## G3.1: agent-acquired experience

The acquisition pilot lets Savant choose actions on **one original V1 synthetic
refund episode**, observe the resulting tool feedback, and write one structured
lesson. Deterministic validation checks evidence references and witnessed task
feedback against the actual trajectory. The lesson remains *advisory*: its
future usefulness, and any learning effect, require the separate G3.2 held-out
comparison. Older guided-recovery pilots remain unchanged.

Set `SUBQ_API_KEY` only in your PowerShell environment, then run:

```powershell
$env:PYTHONPATH = Join-Path (Get-Location) 'src'
.\.venv\Scripts\python.exe -m repair_to_generalization.acquisition_pilot --live --max-usd 0.10
```

This run uses at most **four agent decisions and one reflection call**, with no
automatic retries. A paid call requires `--live`; budget reservations and
billable invalid responses count toward the ceiling. The ignored JSONL report
retains each observed step even if a later model response fails validation.
Add `--initially-committed` only to select the other V1 hidden-state condition
for a separately authorized experiment; this flag is never passed to Savant.

## G3.2: measured use of self-acquired experience

After inspecting and accepting a G3.1 report, compare two arms: **no memory**
and **the unchanged Savant-authored lesson from that report**. The harness
replays the original acquisition and validates its evidence before use. The
report and lesson are identified by SHA-256 hashes; neither is manually
rewritten into a policy. Both arms receive the same task, model, action budget,
and synthetic inspection cost (25 units). Within each repetition, identical
complete public prompts reuse the same model response.

Five held-out simulator conditions cover the other V1 prior-commit state and
four V2 combinations of prior commit and visible/hidden drift. The exact G3.1
training condition is excluded. These are **five conditions of one simulator**,
not five independent tasks. Behavior changes, safe completions, duplicates,
and cost are reported separately; a changed prompt or decision alone does not
establish learning or statistical significance.

With `SUBQ_API_KEY` set locally, and using your original G3.1 report:

```powershell
$env:PYTHONPATH = Join-Path (Get-Location) 'src'
.\.venv\Scripts\python.exe -m repair_to_generalization.learning_pilot --experience reports\runs\savant_acquisition_20261120_210809.jsonl --live --repetitions 3 --max-usd 0.10
```

The command authorizes paid inference **only when run with `--live`**. It uses
three fresh repetitions, a maximum of 120 model calls (usually fewer because
identical prompts reuse responses), and a per-run $0.10 spending ceiling.
Completed episodes and individual tool steps (including steps preceding a
failed model call) go into ignored `reports/runs/`. Do not claim model-weight learning:
the proposed learning mechanism is **agent-level acquisition, persistence,
and reuse of experience**. A positive learning-effect claim requires comparing
the held-out outcomes and safety against the no-memory baseline.
