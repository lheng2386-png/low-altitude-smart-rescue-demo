# EC-TERP Algorithm

## 1. Problem Background

灾情感知及影响评估 already has target detection, segmentation-based risk cues, image-plane path planning, Mission Evidence Ledger, and Final Report V2. A plain target-environment-route ranking is useful, but it does not explicitly ask whether the evidence is strong, whether the route is only a preview, or whether high-priority areas remain uncovered.

EC-TERP addresses that gap as an assistive rescue-priority ranking algorithm. It helps human operators decide which detected target should be reviewed first. It does not confirm victims and does not replace rescue command decisions.

## 2. Why Plain TERP Is Not Enough

The original TERP idea focuses on target urgency, environmental risk, and route accessibility. In this project, the system also needs to consider:

- whether the result came from strong model output, uploaded/demo input, or simulated/preview data
- whether path evidence is available or degraded
- whether high-priority areas remain unsearched
- whether a target is a Transformer `human_candidate` requiring manual review
- whether uncertainty should reduce the final priority

EC-TERP keeps TERP transparent, then adds coverage, evidence, and uncertainty constraints.

## 3. Formula

```text
EC-TERP = αT + βE + γR + δC + λQ - μU
```

Where:

- `T`: Target urgency
- `E`: Environment risk
- `R`: Route accessibility
- `C`: Coverage gap
- `Q`: Evidence quality
- `U`: Uncertainty penalty

The initial weights are expert-prior weights, not trained parameters. They should be calibrated with real validation cases in future work.

## 4. Component Meanings

`T` measures how urgent the target category appears. For example, a detected `civilian` starts higher than an animal target, while `human_candidate` remains below confirmed project classes because it requires manual review.

`E` measures surrounding risk from available segmentation or damage context, such as major damage, destroyed buildings, water, or blocked roads.

`R` measures image-plane route accessibility. A found low-cost image-plane path increases the score. This is not GPS navigation.

`C` measures coverage gap. If high-priority areas remain unsearched, this term can raise the importance of reviewing affected targets.

`Q` measures evidence quality from the Mission Evidence Ledger. Strong evidence contributes more than medium, weak, or none.

`U` is an uncertainty penalty. Low confidence, Transformer-only `human_candidate`, missing segmentation, missing path evidence, and required human review can increase this penalty.

## 5. Evidence Quality Q

The evidence quality term is linked to Mission Evidence Ledger:

- `strong = 1.00`
- `medium = 0.70`
- `weak = 0.35`
- `none = 0.00`

This prevents simulated, preview, or missing evidence from being treated as if it were a strong model result or real measurement.

## 6. Uncertainty Penalty U

The uncertainty penalty is deliberately subtracted from the score. It keeps the ranking conservative when:

- detector confidence is low
- the target is only a `human_candidate`
- segmentation is missing
- route planning is unavailable
- evidence quality is weak
- human review is required

This is especially important for avoiding overconfident rescue claims.

## 7. Relationship With Evidence Ledger

Mission Evidence Ledger decides whether module outputs are strong, medium, weak, or none. EC-TERP consumes that evidence quality instead of guessing. If the ledger marks a module as simulated, preview-only, failed, or not run, EC-TERP should not upgrade that result into strong evidence.

## 8. Relationship With Path Planning

EC-TERP can use route accessibility from the image-plane path planner. This path is a decision-support preview in pixel coordinates. It is not a GPS route, GIS route, autonomous navigation command, or real rescue route.

## 9. Synthetic Evaluation

The built-in EC-TERP evaluation cases are synthetic demo cases. They are useful for smoke tests, ranking sanity checks, sensitivity analysis, and competition explanation. They are not real rescue data and must not be described as a real benchmark.

## 10. Truthfulness Boundaries

- EC-TERP is an assistive priority ranking algorithm.
- EC-TERP is not an automatic rescue decision system.
- EC-TERP does not confirm real trapped people.
- `human_candidate` is not confirmed civilian.
- Image-plane path planning is not GPS navigation.
- Simulated thermal is not real temperature measurement.
- Uploaded/demo masks are not automatic model predictions.
- Synthetic demo cases are not real rescue data.
- No mAP, precision, recall, SOTA, GPS, GIS, or real rescue benchmark claim should be made without real validation evidence.
