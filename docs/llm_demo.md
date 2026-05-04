# LLM Demo: Mission Report Assistant, Evidence Copilot, Mission Planner, and Evidence Auditor

This document explains the optional LLM API integration demo for AeroRescue-AI.

The LLM modules are post-processing assistants only:

- Mission Report Assistant drafts an auxiliary structured report from existing mission evidence.
- Mission Evidence Copilot answers questions from the assembled evidence package.
- LLM Mission Planner generates a structured tool plan and the backend executes only white-listed mission evidence tools.
- LLM Evidence Auditor reviews mission outputs for consistency, missing evidence, unsafe phrases, and authenticity-boundary violations.
- These modules do not run detection, segmentation, thermal measurement, flight control, arbitrary code, shell commands, network requests, or field decision-making.

## Configure A Real API

Use environment variables:

```bash
export LLM_ENABLE=true
export LLM_PROVIDER=openai
export OPENAI_API_KEY=your_api_key_here
export OPENAI_MODEL=gpt-5.5
```

If the API key is missing, disabled, or the provider call fails, the system falls back to MockProvider.

## Use MockProvider

MockProvider is the default safe demo path:

```bash
export LLM_ENABLE=false
unset OPENAI_API_KEY
```

This mode does not call external APIs and is suitable for local tests, rehearsals, and competition demos where network access is uncertain.

## Run The One-Click Demo

From the repository root:

```bash
app/venv/bin/python scripts/run_llm_demo.py
```

If your active Python environment already has the project dependencies installed, this equivalent command also works:

```bash
python scripts/run_llm_demo.py
```

The script loads:

```text
demo_missions/urban_rescue_llm_demo/
```

It writes demo outputs under:

```text
outputs/missions/urban_rescue_llm_demo/outputs/reports/
```

Key generated files:

- `llm_mission_report.json`
- `evidence_ledger.json`
- `final_report_v2.md`
- `mission_copilot_answers.json`
- `mission_planner_result.json`
- `llm_evidence_audit.json`
- `llm_demo_summary.json`

## Safety Tests

Run the LLM safety regression suite:

```bash
app/venv/bin/python tests/test_llm_safety.py
```

Run the one-click demo smoke test:

```bash
app/venv/bin/python tests/smoke_test_llm_demo_script.py
app/venv/bin/python tests/test_llm_mission_planner.py
app/venv/bin/python tests/smoke_test_mission_planner_panel.py
app/venv/bin/python tests/test_llm_evidence_auditor.py
app/venv/bin/python tests/smoke_test_evidence_audit_panel.py
```

## Authenticity Boundaries

- `human_candidate` is an unverified suspected human target and requires manual review.
- Simulated thermal is a visualization cue, not radiometric measurement.
- Image-plane planning is visual planning context, not field navigation.
- Demo/uploaded mask is a supplied risk-area mask, not verified automatic segmentation.
- Fast Preview is preview stitching, not ODM mapping output.
- EC-TERP is an auxiliary priority model, not an operational rescue decision.
- The mission planner can only execute white-listed backend tools and cannot access arbitrary files, shell commands, external networks, API keys, or environment variables.
- The evidence auditor only reports issues and suggestions. It does not overwrite mission evidence or generated reports.
- All LLM outputs require human review.
