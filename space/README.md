---
title: From Repair to Generalization
sdk: static
app_file: index.html
colorFrom: blue
colorTo: red
fullWidth: true
short_description: The gap between knowledge revision and safe agent behavior
---

# From Repair to Generalization

A static, interactive research publication based on Cycle 2 of the [research repository](https://github.com/kablewithak/repair-to-generalization). No model calls, backend, tokens, or external font requests are required.

## Preview / deploy

Open `index.html` from this directory in a browser, or create a private **Static** Hugging Face Space and upload `README.md`, `index.html`, `styles.css`, `app.js`, and `evidence.js` together at the repository root. Review the preview before changing visibility. Do not upload the private evidence archive or raw reports.

## Data and limitations

The five source hashes are pinned in `export_evidence.py`. To regenerate the allowlisted excerpts from the original private JSONL reports, run `python export_evidence.py PATH_TO_REPORTS`. This checks hashes, extracts exact recorded public actions and observations, checks selected simulator-state reconstructions against recorded outcomes, and computes matched denominators. The source reports are not distributed here. The published UI is a replay visualization, not the full offline verifier. No new scientific results are claimed. See the [research report](https://github.com/kablewithak/repair-to-generalization/blob/main/CYCLE_2_RESEARCH_REPORT.md) for methods, limitations and local reproduction.

## Provider usage context

The supplied SubQ dashboard (26 November 2026) displays $4.36 spend, 42.9M tokens and 4.1K requests for all models in the last 30 days. These are account-level figures, not reconciled model-specific or project-only billing. The archived experimental accounting remains unchanged. The screenshot and credit balance are not distributed.

## V5 visual publication

The opening concept diagram, synchronized recorded-action explorer, planned G7 outcome matrix, and G10 metric switch are static client-side displays. The matrix uses exact frozen case/repetition coordinates and explicitly distinguishes missing observations. Job ordinals and evaluator state visualizations are reconstructed from the recorded actions; they are not provider identifiers or additional inference. The typography uses platform-local font fallbacks; no font assets or external font calls are included.

V5 removes the redundant billing explanation and standalone limitations section from the page; the provider usage scope remains visible, and study-specific limitations remain beside findings and in the research report. The response-budget figure uses independently sized and spaced numerator and denominator.
