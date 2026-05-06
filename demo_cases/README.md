# Demo Cases

This directory defines 3-5 complete 灾情感知及影响评估 competition demo cases. The generated visual outputs are written to `static/images/showcase/<case_id>/` by the offline script.

## Generate Showcase Outputs

```bash
python scripts/generate_demo_cases.py
```

Optional:

```bash
python scripts/generate_demo_cases.py --case case_01_flood --model yolov11m
```

The generator does not start Gradio, does not download data, and does not call external APIs.

## Standard Generated Artifacts

Each generated case should contain:

```text
static/images/showcase/case_01_flood/
├── input.jpg
├── demo_mask.png
├── detection_overlay.png
├── segmentation_overlay.png
├── risk_aware_path_overlay.png
├── dual_path_overlay.png
├── target_table.csv
├── terp_ranking.csv
├── path_comparison.json
├── rescue_report.txt
└── case_summary.md
```

## Manual Demo Mask Policy

Some cases use a manually prepared demo mask so that the decision layer, TERP ranking, environment risk fusion, and Risk-Aware A* can be demonstrated without requiring a trained segmentation checkpoint.

Required wording:

> This mask is manually prepared for decision-layer demonstration. It is not an automatic segmentation prediction.

Do not describe `demo_mask.png` as an automatic segmentation output unless it was actually produced by a trained local checkpoint.

## Case Purposes

| Case | Purpose | Main Artifacts |
| --- | --- | --- |
| Case 1 Flood Civilian Rescue | Show water risk and TERP priority | segmentation overlay, TERP table, path comparison |
| Case 2 Building Collapse | Show major/destroyed building risk | environment summary, report warning |
| Case 3 Road Blocked | Show road-blocked path cost | baseline vs Risk-Aware A* comparison |
| Case 4 Multi-target Priority | Show TERP ranking across target types | target table, TERP ranking |
| Case 5 No Target / Fallback | Show safe no-target behavior | no-target report, fallback summary |

## Use In README, PPT, And Demo Video

- Use `input.jpg`, `detection_overlay.png`, `segmentation_overlay.png`, and `dual_path_overlay.png` for visual slides.
- Use `terp_ranking.csv` for a compact priority table.
- Use `path_comparison.json` to explain baseline vs Risk-Aware A*.
- Use `rescue_report.txt` as the generated Chinese report example.
- Use `case_summary.md` to explain source image, mask policy, detection status, and current limitations.

## Notes

- Use local images or copied reference images only.
- Avoid committing full raw datasets.
- Do not commit checkpoints.
- Do not fabricate mAP, FPS, or model comparison metrics.
- For class-id masks, PNG is recommended.
- If no mask is available, the case should explicitly show fallback behavior.
