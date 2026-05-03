# Case 5 No Target Or Fallback

- Input source: local repository image candidates listed in `case_config.json`.
- Mask policy: manually prepared background-only demo mask.
- Segmentation note: the demo mask is for decision-layer demonstration. It is not an automatic segmentation prediction.
- Detection: uses a high confidence threshold and a likely non-disaster image to demonstrate safe no-target behavior.
- Auto segmentation checkpoint: intentionally not required. Missing checkpoint fallback should be described in the report and summary.
- Current limitation: this case tests system robustness rather than rescue accuracy.
