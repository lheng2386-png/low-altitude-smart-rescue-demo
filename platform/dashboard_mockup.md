# Platform Dashboard Mockup

This is the 灾情感知及影响评估 platform-inspired dashboard design. It is not a running web platform yet; it defines the mature product direction for the competition prototype.

## Dashboard Layout

```text
┌──────────────────────────────────────────────────────────────────────┐
│ 灾情感知及影响评估 Mission Dashboard                                      │
├───────────────┬──────────────────────────┬───────────────────────────┤
│ Case Archive  │ UAV Image / Video Review │ Rescue Decision Panel      │
│               │                          │                           │
│ Case 01 Flood │ [image viewer]           │ Highest TERP Target: T001  │
│ Case 02 Ruin  │ [detection overlay]      │ Risk Level: High           │
│ Case 03 Road  │ [segmentation overlay]   │ Access Cost: Medium        │
│ Case 04 Multi │ [dual path overlay]      │ Suggested Action: Verify   │
├───────────────┴──────────────────────────┴───────────────────────────┤
│ Report Center: recognition summary, TERP ranking, path comparison     │
└──────────────────────────────────────────────────────────────────────┘
```

## Dashboard Panels

| Panel | Data Source | Role |
| --- | --- | --- |
| Mission list | Case archive | Switch between rescue scenes |
| Image viewer | Uploaded image/video | Inspect raw UAV view |
| Detection overlay | YOLOv11 | Check target boxes |
| Segmentation overlay | Uploaded mask / checkpoint | Understand water, blocked road, damage |
| TERP table | Decision layer | Prioritize rescue targets |
| Path panel | A* planner | Compare ordinary and risk-aware routes |
| Report center | Template generator | Export Chinese rescue assistance report |

