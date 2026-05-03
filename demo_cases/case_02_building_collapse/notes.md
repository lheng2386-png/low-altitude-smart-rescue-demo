# Case 2 Building Collapse

- Input source: local repository image candidates listed in `case_config.json`.
- Mask policy: manually prepared demo mask with major and destroyed building regions.
- Segmentation note: the demo mask is for decision-layer demonstration. It is not an automatic segmentation prediction.
- Detection: uses real local YOLO detection when weights are available.
- Auto segmentation checkpoint: not used.
- Current limitation: building-damage areas are illustrative mask regions for TERP and path-cost testing.
