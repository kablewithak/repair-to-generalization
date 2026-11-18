# From Repair to Generalization

Research into when previously validated AI-agent experience becomes unsafe to reuse after tool or policy behavior changes.

Status: **G0 feasibility**. No experimental results yet. The research PRD is kept locally and is not tracked in Git.

## Prerequisites

Python 3.11+ and an active virtual environment.

```powershell
python -m pip install -e .
python -m unittest discover -s tests -v
```

## SubQ access preflight

Set `SUBQ_API_KEY` in your local PowerShell session, then run:

```powershell
python -m repair_to_generalization.preflight
```

This checks the official `/v1/models` endpoint to determine whether `subq-savant-1.4` is listed. **It makes no text-generation requests and cannot validate token billing or model responses.** API keys are not printed or stored by this tool. A model not listed means the key does not currently have access to it.

Official reference: https://docs.subq.ai/endpoints/models-api/
