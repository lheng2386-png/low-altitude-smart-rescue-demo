# No Target Or Fallback

Scenario: No clear rescue target or missing automatic segmentation checkpoint

Status: generated

Input: `static/images/logo.jpg`

Target count: 0

Detection status: Loaded YOLO weights: models/yolov11m/best.pt; detected 0 targets.

Mask policy: This mask is manually prepared for decision-layer demonstration. It is not an automatic segmentation prediction.

Mask validation: Mask contains only background; environment risk may be limited.

Auto segmentation checkpoint used: False

Artifacts:

- `input.jpg`
- `demo_mask.png`
- `detection_overlay.png`
- `segmentation_overlay.png`
- `risk_aware_path_overlay.png`
- `dual_path_overlay.png`
- `target_table.csv`
- `terp_ranking.csv`
- `path_comparison.json`
- `rescue_report.txt`
- `case_summary.md`

Current limitations:

- Demo masks are manually prepared and should not be described as automatic model predictions.
- Path planning is an image-plane reference route, not a GPS route.
- No UAV flight control or real road network is connected.
