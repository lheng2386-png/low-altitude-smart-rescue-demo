# Platform Design Derived From Mature UAV Rescue Workflow

AeroRescue-AI uses the copied ARGUS frontend/backend structures in `integrated_modules/argus/` to shape a platform-style workflow.

## Migrated Structure

| Copied Structure | AeroRescue-AI Use |
| --- | --- |
| Home / overview pages | Mission dashboard and case overview |
| Report page | Rescue report center |
| API client pattern | Future platform backend integration |
| Image router | Future image/video upload management |
| Report router | Future report persistence |
| Report / image schemas | Future database models |

## Current Prototype Boundary

The current system is still a local Gradio prototype. It does not include database storage, login, WebODM, GIS, GPS, or flight-control integration.

## Future Platform Architecture

```text
Frontend dashboard
  ↕
Backend API
  ↕
Case database / image store
  ↕
Detection worker + segmentation worker + TERP worker + report worker
```

