# EC-TERP Evaluation Notes

## Why Evaluation Is Needed

A formula alone is not enough for a competition-stage rescue decision system. EC-TERP needs a repeatable framework that can show:

- how EC-TERP differs from baseline TERP
- whether rankings remain stable under weight perturbation
- which terms contribute to ranking changes
- how to generate a transparent report for defense and review

This framework does not train models and does not claim real rescue validation.

## Evaluation Content

The evaluation service supports:

- TERP vs EC-TERP comparison
- Top-1 Agreement
- Top-3 Recall
- Average Rank Shift
- Single-factor sensitivity analysis
- Random weight stability analysis
- Ablation study

## Synthetic Demo Cases

The built-in cases are synthetic/demo cases. They are designed to test algorithm logic, ranking behavior, uncertainty penalties, evidence quality effects, route accessibility, and coverage gap terms.

They are not real rescue data and must not be reported as field validation.

## Sensitivity Analysis Method

The single-factor sensitivity analysis perturbs these weights independently:

- `α`: target urgency
- `β`: environment risk
- `γ`: route accessibility
- `δ`: coverage gap
- `λ`: evidence quality
- `μ`: uncertainty penalty

Default perturbation factors are `0.8`, `0.9`, `1.1`, and `1.2`. The analysis measures Top-1 stability, Top-3 stability, and average rank shift.

The random weight stability analysis samples multiple perturbed weight sets around the base prior weights and recomputes rankings.

## Ablation Method

The ablation study compares full EC-TERP with:

- w/o Environment Risk
- w/o Route Accessibility
- w/o Coverage Gap
- w/o Evidence Quality
- w/o Uncertainty Penalty

Each ablation sets the corresponding weight to zero without claiming the remaining weights are learned or re-calibrated.

## Output Files

The evaluation script writes:

- `terp_vs_ec_terp_comparison.json`
- `sensitivity_results.json`
- `random_weight_stability.json`
- `ablation_results.json`
- `ec_terp_evaluation_report.md`

By default these files are written under `outputs/ec_terp_evaluation/`, which is runtime output and should not be committed.

## Truthfulness Boundaries

- This is not a public benchmark result.
- This is not large-scale real rescue validation.
- The built-in cases are synthetic/demo cases.
- EC-TERP weights are expert-prior weights, not trained parameters.
- Results must be regenerated from code and should not be hardcoded.
- Future validation requires a real annotated validation set.

## Competition Demo Wording

Suggested wording:

“To validate EC-TERP, 灾情感知及影响评估 provides a reproducible evaluation framework covering TERP comparison, weight sensitivity analysis, random weight stability, and ablation study. The current built-in examples are synthetic demo cases for algorithm verification; future work can replace them with a real annotated validation set for calibration and formal reporting.”
