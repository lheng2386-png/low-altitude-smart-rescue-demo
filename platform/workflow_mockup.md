# Platform Workflow Mockup

```text
Mission Created
  ↓
UAV Image / Video Uploaded
  ↓
Target Detection
  ↓
Segmentation Environment Layer
  ↓
Scene Applicability Gate
  ↓
TERP Priority Dashboard
  ↓
Risk-Aware Access Planning
  ↓
Chinese Rescue Report
  ↓
Case Archive
```

## Scene Applicability Gate

The Scene Applicability Gate prevents the system from over-claiming. It checks whether current inputs are sufficient for decision support:

| Condition | Gate Result |
| --- | --- |
| No target detected | Do not generate target route |
| No segmentation mask | Allow target-only TERP and default-cost route |
| Invalid mask | Fallback to no segmentation |
| Missing segmentation checkpoint | Fallback to uploaded mask or no segmentation |
| Valid target + valid mask | Enable full environment-aware TERP and risk-aware A* |

