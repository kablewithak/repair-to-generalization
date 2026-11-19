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
