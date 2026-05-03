# Flood Civilian Rescue

Scenario: Flooded low-altitude UAV scene

Status: generated

Input: `app/examples/230714-india-flooding-mb-0831-d3a66d_jpg.rf.3e607c4f8f121834224f95ab0d44ddd6.jpg`

Target count: 7

Detection status: Loaded YOLO weights: models/yolov11m/best.pt; detected 7 targets.

Mask policy: This mask is manually prepared for decision-layer demonstration. It is not an automatic segmentation prediction.

Mask validation: Segmentation mask is valid.

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
