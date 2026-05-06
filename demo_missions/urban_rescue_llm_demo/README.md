# Urban Rescue LLM Demo Mission

This demo mission is a stable data pack for showing 灾情感知及影响评估 LLM post-processing features during a competition demo.

## What It Contains

- `mission_result.json`: mission metadata and authenticity notice.
- `detection_result.json`: one `human_candidate` target that requires manual verification.
- `segmentation_result.json`: demo/uploaded mask risk areas.
- `thermal_result.json`: simulated thermal hotspot cues.
- `path_planning_result.json`: image-plane planning result.
- `ec_terp_result.json`: auxiliary EC-TERP priority.
- `evidence_ledger.json`: preloaded evidence events.
- `expected_llm_report_mock.json`: expected MockProvider-style report shape.

## Authenticity Boundaries

- `human_candidate` is not a verified civilian or survivor.
- Simulated thermal is not radiometric temperature measurement.
- Image-plane path output is not a field navigation route.
- Demo/uploaded mask is not necessarily automatic segmentation.
- Fast Preview, when used, is not an ODM mapping artifact.
- EC-TERP priority is decision-support only, not a rescue conclusion.
- All outputs require human review.

## Run The One-Click Demo

From the repository root:

```bash
app/venv/bin/python scripts/run_llm_demo.py
```

The script works without an API key. If LLM API access is disabled or unavailable, it uses MockProvider and still runs the full report, evidence, Final Report V2, and Mission Evidence Copilot flow.
