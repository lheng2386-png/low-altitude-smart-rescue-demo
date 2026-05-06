# EC-TERP Algorithm Notes

## Why EC-TERP

The original TERP layer ranks rescue targets from target, environment, and route signals. 灾情感知及影响评估 now also has a Mission Evidence Ledger, lightweight Decision Fusion, and coverage-gap evaluation. EC-TERP adds these evidence-aware terms into a single transparent auxiliary priority score.

EC-TERP is not a trained model and is not an automatic rescue decision. It is a rule-based scoring algorithm for competition-stage decision support.

## Formula

```text
EC-TERP = αT + βE + γR + δC + λQ - μU
```

- `T`: Target urgency
- `E`: Environment risk
- `R`: Route accessibility
- `C`: Coverage gap / unsearched high-priority area
- `Q`: Evidence quality
- `U`: Uncertainty penalty

All component scores are normalized before being combined. `U` is a penalty term and is subtracted.

## Initial Weights

- `α = 0.30`
- `β = 0.25`
- `γ = 0.20`
- `δ = 0.15`
- `λ = 0.10`
- `μ = 0.15`

These are expert-prior weights, not learned parameters. They should be calibrated later with a small validation set, sensitivity analysis, and ablation experiments.

## Evidence Level Mapping

- `strong = 1.00`
- `medium = 0.70`
- `weak = 0.35`
- `none = 0.00`

Evidence quality comes from Mission Evidence Ledger records or explicit structured input. It is not fabricated from code existence.

## Uncertainty Penalty

The uncertainty term increases when:

- detection confidence is low
- the target is Transformer-only or `human_candidate`
- evidence quality is weak or missing
- route planning is missing or failed
- segmentation context is unavailable
- the target requires human review

This prevents low-confidence, preview, or weak-evidence outputs from dominating the rescue priority ranking.

## Difference From TERP

- TERP: Target, Environment, Route.
- EC-TERP: Target, Environment, Route, Coverage, Evidence, Uncertainty.

EC-TERP is better aligned with the current evidence-chain architecture because it can react to weak evidence, missing masks, failed route planning, and uncovered high-priority areas.

## Truthfulness Boundaries

- EC-TERP is an auxiliary priority algorithm.
- EC-TERP does not confirm real trapped victims.
- EC-TERP is not an automatic rescue decision.
- Route accessibility is still based on image-plane reference paths, not GPS navigation.
- `human_candidate` is not a confirmed civilian and requires human review.
- Uploaded or demo masks are not automatic model predictions.
- Simulated thermal results are not real temperature measurements.
- The current weights are not validated learned parameters.

## Future Validation Plan

- Build a small validation set of representative disaster cases.
- Compare Top-1 priority agreement with expert review.
- Measure Top-3 priority recall for manually annotated urgent targets.
- Run weight sensitivity analysis.
- Run grid search or Bayesian tuning on the EC-TERP weights.
- Add ablation tests for `Q` and `U`.

## Competition Demo Wording

Suggested wording:

“灾情感知及影响评估 proposes EC-TERP, an evidence-constrained rescue priority algorithm. It extends target-environment-route priority scoring with coverage gap, mission evidence quality, and uncertainty penalty. The method uses strong/medium/weak/none evidence levels from the Mission Evidence Ledger to prevent simulated, preview, or low-confidence results from over-influencing rescue ranking. EC-TERP remains an auxiliary decision-support score and requires human review.”
