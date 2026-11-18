# From Repair to Generalization

Research into when previously validated AI-agent experience becomes unsafe to reuse after tool or policy behavior changes.

Status: **G0 feasibility**. No experimental results yet. The approved research PRD stays local and is ignored by Git.

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
