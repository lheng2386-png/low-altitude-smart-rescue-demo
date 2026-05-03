# Case 4 Multi-target Priority

- Input source: local repository image candidates listed in `case_config.json`.
- Mask policy: manually prepared mixed environment demo mask.
- Segmentation note: the demo mask is for decision-layer demonstration. It is not an automatic segmentation prediction.
- Detection: uses real local YOLO detection when weights are available.
- Focus: TERP ranking should prioritize likely trapped civilians above rescuers and animals when detected.
