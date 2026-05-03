# Case 3 Road Blocked

- Input source: local repository image candidates listed in `case_config.json`.
- Mask policy: manually prepared demo mask with `road_clear` and `road_blocked` regions.
- Segmentation note: the demo mask is for decision-layer demonstration. It is not an automatic segmentation prediction.
- Detection: uses real local YOLO detection when weights are available.
- Current limitation: road accessibility is represented by image-plane costs, not by a real road network.
