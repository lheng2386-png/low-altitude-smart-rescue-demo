"""Evaluation and sensitivity analysis helpers for EC-TERP.

The built-in cases are synthetic/demo cases for algorithm validation plumbing.
They are not real rescue benchmark data and should not be reported as such.
"""

import copy
import csv
import json
import random
import statistics
from pathlib import Path

from ec_terp_engine import (
    clamp,
    compute_ec_terp_score,
    compute_environment_risk,
    compute_route_accessibility,
    compute_target_urgency,
    load_ec_terp_weights,
)


class ECTERPEvaluationError(Exception):
    pass


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = ROOT_DIR / "outputs" / "ec_terp_evaluation"


DEFAULT_SYNTHETIC_EVAL_CASES = [
    {
        "case_id": "case_001_flood_road_blocked",
        "case_type": "synthetic_demo",
        "scenario": "Flooded road with one civilian and one animal target.",
        "targets": [
            {
                "id": "T001",
                "class_name": "civilian",
                "confidence": 0.86,
                "bbox": [120, 80, 200, 220],
                "center": [160, 150],
                "area": 11200,
                "human_review_required": True,
            },
            {
                "id": "T002",
                "class_name": "dog",
                "confidence": 0.91,
                "bbox": [300, 180, 360, 250],
                "center": [330, 215],
                "area": 4200,
                "human_review_required": True,
            },
        ],
        "segmentation_summary": {
            "water": 0.18,
            "road_blocked": 0.12,
            "major_damage": 0.08,
            "destroyed_building": 0.04,
            "road_clear": 0.22,
        },
        "path_results": {
            "T001": {"found": True, "path_length": 140, "total_cost": 230},
            "T002": {"found": True, "path_length": 80, "total_cost": 90},
        },
        "coverage_result": {"unsearched_high_priority_ratio": 0.35},
        "target_evidence_levels": {"T001": "strong", "T002": "strong"},
        "human_priority_order": ["T001", "T002"],
    },
    {
        "case_id": "case_002_damaged_building_human_candidate",
        "case_type": "synthetic_demo",
        "scenario": "Damaged building with one YOLO civilian and one Transformer human_candidate.",
        "targets": [
            {
                "id": "T011",
                "class_name": "civilian",
                "confidence": 0.78,
                "bbox": [80, 90, 150, 210],
                "center": [115, 150],
                "area": 8400,
                "source_backend": "yolo_rescue_targets",
                "human_review_required": True,
            },
            {
                "id": "TR012",
                "class_name": "human_candidate",
                "confidence": 0.82,
                "bbox": [260, 120, 340, 240],
                "center": [300, 180],
                "area": 9600,
                "source_backend": "transformer_rescuedet",
                "human_review_required": True,
            },
        ],
        "segmentation_summary": {
            "water": 0.02,
            "road_blocked": 0.08,
            "major_damage": 0.22,
            "destroyed_building": 0.16,
            "road_clear": 0.18,
        },
        "path_results": {
            "T011": {"found": True, "path_length": 90, "total_cost": 120},
            "TR012": {"found": True, "path_length": 70, "total_cost": 85},
        },
        "coverage_result": {"unsearched_high_priority_ratio": 0.25},
        "target_evidence_levels": {"T011": "strong", "TR012": "medium"},
        "human_priority_order": ["T011", "TR012"],
    },
    {
        "case_id": "case_003_low_confidence_target",
        "case_type": "synthetic_demo",
        "scenario": "Low-confidence civilian candidate near moderate risk area.",
        "targets": [
            {
                "id": "T021",
                "class_name": "civilian",
                "confidence": 0.32,
                "bbox": [100, 100, 180, 210],
                "center": [140, 155],
                "area": 8800,
                "human_review_required": True,
            },
            {
                "id": "T022",
                "class_name": "cow",
                "confidence": 0.88,
                "bbox": [300, 120, 390, 230],
                "center": [345, 175],
                "area": 9900,
                "human_review_required": True,
            },
        ],
        "segmentation_summary": {
            "water": 0.04,
            "road_blocked": 0.06,
            "major_damage": 0.1,
            "destroyed_building": 0.02,
            "road_clear": 0.3,
        },
        "path_results": {
            "T021": {"found": True, "path_length": 100, "total_cost": 110},
            "T022": {"found": True, "path_length": 95, "total_cost": 105},
        },
        "coverage_result": {"unsearched_high_priority_ratio": 0.18},
        "target_evidence_levels": {"T021": "weak", "T022": "strong"},
        "human_priority_order": ["T021", "T022"],
    },
    {
        "case_id": "case_004_path_unavailable_high_risk",
        "case_type": "synthetic_demo",
        "scenario": "High-risk damaged area where one target is not path-accessible.",
        "targets": [
            {
                "id": "T031",
                "class_name": "civilian",
                "confidence": 0.9,
                "bbox": [50, 70, 140, 210],
                "center": [95, 140],
                "area": 12600,
                "human_review_required": True,
            },
            {
                "id": "T032",
                "class_name": "civilian",
                "confidence": 0.72,
                "bbox": [280, 150, 350, 250],
                "center": [315, 200],
                "area": 7000,
                "human_review_required": True,
            },
        ],
        "segmentation_summary": {
            "water": 0.12,
            "road_blocked": 0.2,
            "major_damage": 0.25,
            "destroyed_building": 0.2,
            "road_clear": 0.06,
        },
        "path_results": {
            "T031": {"found": False, "path_length": 0, "total_cost": 0},
            "T032": {"found": True, "path_length": 75, "total_cost": 95},
        },
        "coverage_result": {"unsearched_high_priority_ratio": 0.42},
        "target_evidence_levels": {"T031": "strong", "T032": "strong"},
        "human_priority_order": ["T032", "T031"],
    },
    {
        "case_id": "case_005_coverage_gap_case",
        "case_type": "synthetic_demo",
        "scenario": "Multiple targets with a large unsearched high-priority area.",
        "targets": [
            {
                "id": "T041",
                "class_name": "civilian",
                "confidence": 0.74,
                "bbox": [60, 90, 130, 190],
                "center": [95, 140],
                "area": 7000,
                "human_review_required": True,
            },
            {
                "id": "T042",
                "class_name": "horse",
                "confidence": 0.9,
                "bbox": [240, 80, 340, 210],
                "center": [290, 145],
                "area": 13000,
                "human_review_required": True,
            },
            {
                "id": "TR043",
                "class_name": "human_candidate",
                "confidence": 0.62,
                "bbox": [390, 150, 455, 250],
                "center": [422, 200],
                "area": 6500,
                "source_backend": "transformer_rescuedet",
                "human_review_required": True,
            },
        ],
        "segmentation_summary": {
            "water": 0.06,
            "road_blocked": 0.07,
            "major_damage": 0.14,
            "destroyed_building": 0.08,
            "road_clear": 0.24,
        },
        "path_results": {
            "T041": {"found": True, "path_length": 120, "total_cost": 160},
            "T042": {"found": True, "path_length": 70, "total_cost": 80},
            "TR043": {"found": True, "path_length": 85, "total_cost": 120},
        },
        "coverage_result": {"unsearched_high_priority_ratio": 0.68},
        "target_evidence_levels": {"T041": "strong", "T042": "strong", "TR043": "medium"},
        "human_priority_order": ["T041", "TR043", "T042"],
    },
]


def _json_safe(value):
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


def load_eval_cases(case_file=None):
    if case_file is None:
        return copy.deepcopy(DEFAULT_SYNTHETIC_EVAL_CASES)
    path = Path(case_file)
    if not path.exists():
        raise ECTERPEvaluationError(f"Evaluation case file does not exist: {path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ECTERPEvaluationError(f"Failed to parse evaluation case JSON: {exc}") from exc
    if isinstance(data, dict) and "cases" in data:
        data = data["cases"]
    if not isinstance(data, list):
        raise ECTERPEvaluationError("Evaluation cases must be a list or a dict with a 'cases' list.")
    return data


def get_target_order(rankings):
    if not rankings:
        return []
    ordered = list(rankings)
    if any("rank" in item for item in ordered if isinstance(item, dict)):
        ordered.sort(key=lambda item: int(item.get("rank", 10**9)))
    elif any("ec_terp_score" in item for item in ordered if isinstance(item, dict)):
        ordered.sort(key=lambda item: float(item.get("ec_terp_score", 0.0)), reverse=True)
    elif any("baseline_terp_score" in item for item in ordered if isinstance(item, dict)):
        ordered.sort(key=lambda item: float(item.get("baseline_terp_score", 0.0)), reverse=True)
    order = []
    for item in ordered:
        if isinstance(item, str):
            order.append(item)
        elif isinstance(item, dict):
            target_id = item.get("target_id") or item.get("id")
            if target_id is not None:
                order.append(str(target_id))
    return order


def compute_top1_agreement(pred_order, human_order):
    if not pred_order or not human_order:
        return 0.0
    return 1.0 if str(pred_order[0]) == str(human_order[0]) else 0.0


def compute_topk_recall(pred_order, human_order, k=3):
    if not pred_order or not human_order:
        return 0.0
    pred_topk = set(str(item) for item in pred_order[:k])
    human_topk = set(str(item) for item in human_order[:k])
    if not human_topk:
        return 0.0
    return len(pred_topk & human_topk) / len(human_topk)


def compute_average_rank_shift(order_a, order_b):
    if not order_a or not order_b:
        return 0.0
    pos_a = {str(target_id): idx for idx, target_id in enumerate(order_a, start=1)}
    pos_b = {str(target_id): idx for idx, target_id in enumerate(order_b, start=1)}
    common = set(pos_a) & set(pos_b)
    if not common:
        return 0.0
    return sum(abs(pos_a[target_id] - pos_b[target_id]) for target_id in common) / len(common)


def compute_ranking_metrics(pred_order, human_order=None, baseline_order=None):
    return {
        "top1_agreement": None if not human_order else compute_top1_agreement(pred_order, human_order),
        "top3_recall": None if not human_order else compute_topk_recall(pred_order, human_order, k=3),
        "average_rank_shift_vs_baseline": None
        if not baseline_order
        else compute_average_rank_shift(pred_order, baseline_order),
        "pred_order": list(pred_order or []),
        "human_order": list(human_order) if human_order else None,
        "baseline_order": list(baseline_order) if baseline_order else None,
    }


def _target_id(target):
    return target.get("id") or target.get("target_id") or "unknown"


def run_baseline_terp_for_case(case):
    targets = case.get("targets", []) or []
    segmentation_summary = case.get("segmentation_summary") or {}
    path_results = case.get("path_results") or {}
    rankings = []
    for target in targets:
        target_id = _target_id(target)
        target_result = compute_target_urgency(target)
        environment_result = compute_environment_risk(target, segmentation_summary=segmentation_summary)
        route_result = compute_route_accessibility(path_results.get(target_id))
        score = clamp(
            0.5 * target_result["score"]
            + 0.3 * environment_result["score"]
            + 0.2 * route_result["score"]
        )
        rankings.append(
            {
                "target_id": target_id,
                "class_name": target.get("class_name", "unknown"),
                "baseline_terp_score": round(score, 4),
                "components": {
                    "target_urgency": target_result,
                    "environment_risk": environment_result,
                    "route_accessibility": route_result,
                },
                "truthfulness_note": "Lightweight baseline TERP ranking for EC-TERP comparison. This is computed from synthetic/user-provided case data.",
            }
        )
    rankings.sort(key=lambda item: item["baseline_terp_score"], reverse=True)
    for idx, item in enumerate(rankings, start=1):
        item["rank"] = idx
    return rankings


def run_ec_terp_for_case(case, weights_config=None):
    targets = case.get("targets", []) or []
    segmentation_summary = case.get("segmentation_summary") or {}
    path_results = case.get("path_results") or {}
    coverage_result = case.get("coverage_result") or {}
    evidence_levels = case.get("target_evidence_levels") or {}
    rankings = []
    for target in targets:
        target_id = _target_id(target)
        result = compute_ec_terp_score(
            target,
            segmentation_summary=segmentation_summary,
            path_result=path_results.get(target_id),
            coverage_result=coverage_result,
            target_evidence_level=evidence_levels.get(target_id, "none"),
            segmentation_available=bool(segmentation_summary),
            transformer_only=str(target.get("source_backend", "")).startswith("transformer")
            or str(target.get("class_name", "")).lower() == "human_candidate",
            weights_config=weights_config,
        )
        rankings.append(result)
    rankings.sort(key=lambda item: item.get("ec_terp_score", 0.0), reverse=True)
    for idx, item in enumerate(rankings, start=1):
        item["rank"] = idx
    return rankings


def _mean(values):
    values = [float(value) for value in values if value is not None]
    if not values:
        return None
    return round(sum(values) / len(values), 4)


def run_terp_vs_ec_terp_comparison(cases=None, case_file=None, weights_config=None):
    cases = cases if cases is not None else load_eval_cases(case_file)
    comparisons = []
    top1_values = []
    top3_values = []
    rank_shift_values = []
    for case in cases:
        baseline_rankings = run_baseline_terp_for_case(case)
        ec_rankings = run_ec_terp_for_case(case, weights_config=weights_config)
        baseline_order = get_target_order(baseline_rankings)
        ec_order = get_target_order(ec_rankings)
        human_order = case.get("human_priority_order")
        metrics = compute_ranking_metrics(ec_order, human_order=human_order, baseline_order=baseline_order)
        if metrics["top1_agreement"] is not None:
            top1_values.append(metrics["top1_agreement"])
        if metrics["top3_recall"] is not None:
            top3_values.append(metrics["top3_recall"])
        if metrics["average_rank_shift_vs_baseline"] is not None:
            rank_shift_values.append(metrics["average_rank_shift_vs_baseline"])
        explanation = (
            "EC-TERP 在 baseline TERP 基础上加入 coverage gap、evidence quality 和 uncertainty penalty。"
            f"本 synthetic/demo case 的 baseline order={baseline_order}，EC-TERP order={ec_order}。"
        )
        comparisons.append(
            {
                "case_id": case.get("case_id", "unknown"),
                "scenario": case.get("scenario", ""),
                "case_type": case.get("case_type", "synthetic_demo"),
                "baseline_order": baseline_order,
                "ec_terp_order": ec_order,
                "human_order": list(human_order) if human_order else None,
                "metrics": metrics,
                "baseline_rankings": baseline_rankings,
                "ec_terp_rankings": ec_rankings,
                "explanation": explanation,
            }
        )
    return {
        "success": True,
        "case_count": len(cases),
        "comparisons": comparisons,
        "summary": {
            "mean_top1_agreement": _mean(top1_values),
            "mean_top3_recall": _mean(top3_values),
            "mean_rank_shift_vs_baseline": _mean(rank_shift_values),
        },
        "truthfulness_note": "The comparison uses synthetic or user-provided evaluation cases. It is not a large-scale real rescue benchmark.",
    }


def perturb_weights(base_weights, weight_name, factor):
    updated = copy.deepcopy(base_weights or load_ec_terp_weights())
    if weight_name not in updated:
        raise ECTERPEvaluationError(f"Unknown EC-TERP weight: {weight_name}")
    updated[weight_name] = float(updated[weight_name]) * float(factor)
    updated["perturbation_note"] = f"{weight_name} multiplied by {factor}; weights are not re-normalized."
    return updated


WEIGHT_NAMES = [
    "target_urgency_weight",
    "environment_risk_weight",
    "route_accessibility_weight",
    "coverage_gap_weight",
    "evidence_quality_weight",
    "uncertainty_penalty_weight",
]


def _top3_stability(order_a, order_b):
    return compute_topk_recall(order_a, order_b, k=3)


def run_single_factor_sensitivity_analysis(cases=None, base_weights=None, perturbation_factors=None):
    cases = cases if cases is not None else load_eval_cases()
    base_weights = copy.deepcopy(base_weights or load_ec_terp_weights())
    perturbation_factors = perturbation_factors or [0.8, 0.9, 1.1, 1.2]
    base_orders = {case["case_id"]: get_target_order(run_ec_terp_for_case(case, weights_config=base_weights)) for case in cases}
    results = []
    for weight_name in WEIGHT_NAMES:
        for factor in perturbation_factors:
            perturbed = perturb_weights(base_weights, weight_name, factor)
            for case in cases:
                case_id = case.get("case_id", "unknown")
                perturbed_order = get_target_order(run_ec_terp_for_case(case, weights_config=perturbed))
                base_order = base_orders.get(case_id, [])
                results.append(
                    {
                        "weight_name": weight_name,
                        "factor": float(factor),
                        "case_id": case_id,
                        "top1_changed": bool(base_order and perturbed_order and base_order[0] != perturbed_order[0]),
                        "top3_stability": round(_top3_stability(perturbed_order, base_order), 4),
                        "average_rank_shift": round(compute_average_rank_shift(base_order, perturbed_order), 4),
                        "base_order": base_order,
                        "perturbed_order": perturbed_order,
                    }
                )
    grouped = {}
    for item in results:
        grouped.setdefault(item["weight_name"], []).append(item["average_rank_shift"])
    mean_shift_by_weight = {name: _mean(values) or 0.0 for name, values in grouped.items()}
    most_sensitive = max(mean_shift_by_weight, key=mean_shift_by_weight.get) if mean_shift_by_weight else None
    top1_stability_values = [0.0 if item["top1_changed"] else 1.0 for item in results]
    return {
        "success": True,
        "results": results,
        "summary": {
            "mean_top1_stability": _mean(top1_stability_values),
            "mean_top3_stability": _mean([item["top3_stability"] for item in results]),
            "mean_average_rank_shift": _mean([item["average_rank_shift"] for item in results]),
            "most_sensitive_weight": most_sensitive,
            "mean_rank_shift_by_weight": mean_shift_by_weight,
        },
        "truthfulness_note": "Sensitivity analysis measures ranking stability under weight perturbation. Results depend on the provided cases.",
    }


def run_random_weight_stability_analysis(cases=None, base_weights=None, n_trials=100, perturbation_ratio=0.15, random_seed=42):
    cases = cases if cases is not None else load_eval_cases()
    base_weights = copy.deepcopy(base_weights or load_ec_terp_weights())
    rng = random.Random(random_seed)
    base_orders = {case["case_id"]: get_target_order(run_ec_terp_for_case(case, weights_config=base_weights)) for case in cases}
    trial_results = []
    for trial_idx in range(int(n_trials)):
        weights = copy.deepcopy(base_weights)
        for name in WEIGHT_NAMES:
            factor = 1.0 + rng.uniform(-float(perturbation_ratio), float(perturbation_ratio))
            weights[name] = float(weights[name]) * factor
        for case in cases:
            case_id = case.get("case_id", "unknown")
            order = get_target_order(run_ec_terp_for_case(case, weights_config=weights))
            base_order = base_orders.get(case_id, [])
            trial_results.append(
                {
                    "trial": trial_idx + 1,
                    "case_id": case_id,
                    "top1_changed": bool(base_order and order and base_order[0] != order[0]),
                    "top3_stability": round(_top3_stability(order, base_order), 4),
                    "average_rank_shift": round(compute_average_rank_shift(base_order, order), 4),
                    "base_order": base_order,
                    "perturbed_order": order,
                    "weights": {name: round(float(weights[name]), 6) for name in WEIGHT_NAMES},
                }
            )
    return {
        "success": True,
        "n_trials": int(n_trials),
        "perturbation_ratio": float(perturbation_ratio),
        "summary": {
            "top1_stability": _mean([0.0 if item["top1_changed"] else 1.0 for item in trial_results]),
            "top3_stability": _mean([item["top3_stability"] for item in trial_results]),
            "average_rank_shift": _mean([item["average_rank_shift"] for item in trial_results]),
        },
        "trial_results": trial_results,
        "truthfulness_note": "Random weight stability is computed on the provided evaluation cases and should not be interpreted as large-scale validation.",
    }


ABLATIONS = {
    "without_environment_risk": "environment_risk_weight",
    "without_route_accessibility": "route_accessibility_weight",
    "without_coverage_gap": "coverage_gap_weight",
    "without_evidence_quality": "evidence_quality_weight",
    "without_uncertainty_penalty": "uncertainty_penalty_weight",
}


def build_ablation_weights(base_weights, ablation_name):
    if ablation_name not in ABLATIONS:
        raise ECTERPEvaluationError(f"Unknown ablation: {ablation_name}")
    weights = copy.deepcopy(base_weights or load_ec_terp_weights())
    weights[ABLATIONS[ablation_name]] = 0.0
    weights["ablation_note"] = f"{ABLATIONS[ablation_name]} set to 0. Other weights are not re-normalized."
    return weights


def generate_ablation_interpretation(ablation_name, rank_shift, top1_changed):
    change_note = "Top-1 发生变化" if top1_changed else "Top-1 未发生变化"
    if ablation_name == "without_evidence_quality":
        return f"去掉证据质量项后，弱证据目标可能更容易上升，说明 Q 项有助于抑制低可信结果对排序的过度影响。{change_note}，平均排名变化 {round(rank_shift, 4)}。"
    if ablation_name == "without_uncertainty_penalty":
        return f"去掉不确定性惩罚后，低置信度或 human_candidate 目标可能排名上升，说明 U 项有助于保持人工复核约束。{change_note}，平均排名变化 {round(rank_shift, 4)}。"
    if ablation_name == "without_route_accessibility":
        return f"去掉路径可达性后，路径不可达目标可能被排得过高，说明 R 项对救援可执行性有约束作用。{change_note}，平均排名变化 {round(rank_shift, 4)}。"
    if ablation_name == "without_environment_risk":
        return f"去掉环境风险后，建筑损毁、水域和道路阻断对排序影响减弱，说明 E 项用于表达灾害环境压力。{change_note}，平均排名变化 {round(rank_shift, 4)}。"
    if ablation_name == "without_coverage_gap":
        return f"去掉覆盖缺口后，未搜索高优先级区域对排序约束减弱，说明 C 项用于提醒补充巡检需求。{change_note}，平均排名变化 {round(rank_shift, 4)}。"
    return f"{ablation_name} 的消融导致平均排名变化 {round(rank_shift, 4)}。{change_note}。"


def run_ablation_study(cases=None, base_weights=None):
    cases = cases if cases is not None else load_eval_cases()
    base_weights = copy.deepcopy(base_weights or load_ec_terp_weights())
    base_orders = {case["case_id"]: get_target_order(run_ec_terp_for_case(case, weights_config=base_weights)) for case in cases}
    results = []
    for ablation_name in ABLATIONS:
        weights = build_ablation_weights(base_weights, ablation_name)
        for case in cases:
            case_id = case.get("case_id", "unknown")
            ablated_order = get_target_order(run_ec_terp_for_case(case, weights_config=weights))
            base_order = base_orders.get(case_id, [])
            shift = compute_average_rank_shift(base_order, ablated_order)
            top1_changed = bool(base_order and ablated_order and base_order[0] != ablated_order[0])
            results.append(
                {
                    "ablation_name": ablation_name,
                    "case_id": case_id,
                    "base_order": base_order,
                    "ablated_order": ablated_order,
                    "top1_changed": top1_changed,
                    "top3_stability": round(_top3_stability(ablated_order, base_order), 4),
                    "average_rank_shift": round(shift, 4),
                    "interpretation": generate_ablation_interpretation(ablation_name, shift, top1_changed),
                }
            )
    grouped = {}
    for item in results:
        grouped.setdefault(item["ablation_name"], []).append(item["average_rank_shift"])
    mean_shift_by_ablation = {name: _mean(values) or 0.0 for name, values in grouped.items()}
    most_impactful = max(mean_shift_by_ablation, key=mean_shift_by_ablation.get) if mean_shift_by_ablation else None
    return {
        "success": True,
        "ablation_results": results,
        "summary": {
            "most_impactful_ablation": most_impactful,
            "mean_rank_shift_by_ablation": mean_shift_by_ablation,
        },
        "truthfulness_note": "Ablation study shows how EC-TERP ranking changes when individual components are removed.",
    }


def generate_ec_terp_evaluation_report(
    comparison_result,
    sensitivity_result,
    random_stability_result=None,
    ablation_result=None,
):
    lines = [
        "# EC-TERP 算法验证与灵敏度分析报告",
        "",
        "## 一、实验目的",
        "本报告用于验证 EC-TERP 相比 baseline TERP 的排序差异、权重扰动稳定性和各项贡献。报告由代码真实计算生成，不代表真实救援实测或公开数据集 SOTA。",
        "",
        "## 二、实验设置",
        f"- case 数量：{comparison_result.get('case_count', 0)}",
        "- case 类型：synthetic/demo 或用户提供 case",
        "- 默认权重：α=0.30, β=0.25, γ=0.20, δ=0.15, λ=0.10, μ=0.15",
        "- 指标：Top-1 Agreement、Top-3 Recall、Average Rank Shift",
        "- 真实性边界：synthetic/demo cases 不是真实救援数据，权重是专家先验。",
        "",
        "## 三、TERP vs EC-TERP 排名对比",
    ]
    for item in comparison_result.get("comparisons", []):
        lines.extend(
            [
                f"### {item.get('case_id')}",
                f"- 场景：{item.get('scenario', '')}",
                f"- Baseline order：{item.get('baseline_order', [])}",
                f"- EC-TERP order：{item.get('ec_terp_order', [])}",
                f"- Human/demo order：{item.get('human_order')}",
                f"- Metrics：{item.get('metrics', {})}",
                f"- 解释：{item.get('explanation', '')}",
                "",
            ]
        )
    summary = sensitivity_result.get("summary", {})
    lines.extend(
        [
            "## 四、单因素灵敏度分析",
            f"- mean_top1_stability：{summary.get('mean_top1_stability')}",
            f"- mean_top3_stability：{summary.get('mean_top3_stability')}",
            f"- mean_average_rank_shift：{summary.get('mean_average_rank_shift')}",
            f"- most_sensitive_weight：{summary.get('most_sensitive_weight')}",
            "",
            "## 五、随机权重稳定性分析",
        ]
    )
    if random_stability_result:
        random_summary = random_stability_result.get("summary", {})
        lines.extend(
            [
                f"- n_trials：{random_stability_result.get('n_trials')}",
                f"- top1_stability：{random_summary.get('top1_stability')}",
                f"- top3_stability：{random_summary.get('top3_stability')}",
                f"- average_rank_shift：{random_summary.get('average_rank_shift')}",
                "",
            ]
        )
    else:
        lines.extend(["- 未运行随机权重稳定性分析。", ""])
    lines.append("## 六、消融实验")
    if ablation_result:
        lines.append(f"- most_impactful_ablation：{ablation_result.get('summary', {}).get('most_impactful_ablation')}")
        for name, value in (ablation_result.get("summary", {}).get("mean_rank_shift_by_ablation") or {}).items():
            lines.append(f"- {name}: mean_rank_shift={value}")
        lines.append("")
        seen = set()
        for item in ablation_result.get("ablation_results", []):
            name = item.get("ablation_name")
            if name in seen:
                continue
            seen.add(name)
            lines.append(f"- {name}: {item.get('interpretation')}")
    else:
        lines.append("- 未运行消融实验。")
    lines.extend(
        [
            "",
            "## 七、结论",
            "- EC-TERP 能利用证据质量和不确定性约束排序。",
            "- 排序稳定性取决于样例、权重和输入证据。",
            "- 当前只是小型 synthetic/user-provided 验证，不是大规模真实救援评测。",
            "",
            "## 八、真实性边界",
            "- 本报告不代表真实救援实测。",
            "- 本报告不代表公开数据集 SOTA。",
            "- EC-TERP 权重为专家先验，不是训练得到。",
            "- 后续需要真实标注验证集校准和灵敏度分析。",
            "- EC-TERP 是辅助优先级算法，不是自动救援决策。",
            "- image-plane path 不是真实 GPS 导航。",
        ]
    )
    return "\n".join(lines)


def _write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(payload), ensure_ascii=False, indent=2), encoding="utf-8")
    return str(path)


def _write_ec_terp_ranking_csv(path, comparison_result):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "case_id",
                "case_type",
                "rank",
                "target_id",
                "class_name",
                "ec_terp_score",
                "ec_terp_level",
                "human_review_required",
            ],
        )
        writer.writeheader()
        for comparison in (comparison_result or {}).get("comparisons", []):
            for ranking in comparison.get("ec_terp_rankings", []):
                writer.writerow(
                    {
                        "case_id": comparison.get("case_id"),
                        "case_type": comparison.get("case_type", "synthetic_demo"),
                        "rank": ranking.get("rank"),
                        "target_id": ranking.get("target_id"),
                        "class_name": ranking.get("class_name"),
                        "ec_terp_score": ranking.get("ec_terp_score"),
                        "ec_terp_level": ranking.get("ec_terp_level"),
                        "human_review_required": ranking.get("human_review_required"),
                    }
                )
    return str(path)


def save_ec_terp_evaluation_outputs(results, output_dir=None):
    output_dir = Path(output_dir) if output_dir else DEFAULT_OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        files = {
            "comparison": _write_json(output_dir / "terp_vs_ec_terp_comparison.json", results.get("comparison_result")),
            "sensitivity": _write_json(output_dir / "sensitivity_results.json", results.get("sensitivity_result")),
            "evaluation_summary": _write_json(
                output_dir / "ec_terp_evaluation_summary.json",
                {
                    "success": bool(results.get("success")),
                    "case_count": (results.get("comparison_result") or {}).get("case_count"),
                    "comparison_summary": (results.get("comparison_result") or {}).get("summary"),
                    "truthfulness_note": results.get("truthfulness_note"),
                    "case_type_note": "Built-in cases are synthetic demo cases, not real rescue benchmark data.",
                },
            ),
            "ranking_csv": _write_ec_terp_ranking_csv(
                output_dir / "ec_terp_ranking_results.csv",
                results.get("comparison_result"),
            ),
            "sensitivity_summary": _write_json(
                output_dir / "ec_terp_sensitivity_summary.json",
                {
                    "summary": (results.get("sensitivity_result") or {}).get("summary"),
                    "truthfulness_note": (results.get("sensitivity_result") or {}).get("truthfulness_note"),
                    "case_type_note": "Sensitivity results are computed from synthetic or user-provided cases.",
                },
            ),
        }
        if results.get("random_stability_result") is not None:
            files["random_stability"] = _write_json(output_dir / "random_weight_stability.json", results.get("random_stability_result"))
        if results.get("ablation_result") is not None:
            files["ablation"] = _write_json(output_dir / "ablation_results.json", results.get("ablation_result"))
            files["ablation_summary"] = _write_json(
                output_dir / "ec_terp_ablation_summary.json",
                {
                    "summary": (results.get("ablation_result") or {}).get("summary"),
                    "truthfulness_note": (results.get("ablation_result") or {}).get("truthfulness_note"),
                    "case_type_note": "Ablation results are computed from synthetic or user-provided cases.",
                },
            )
        files["warnings_or_limitations"] = _write_json(
            output_dir / "warnings_or_limitations.json",
            {
                "warnings": [
                    "Built-in evaluation cases are synthetic/demo cases, not real rescue data.",
                    "EC-TERP weights are expert-prior weights, not trained parameters.",
                    "Results are algorithm validation outputs, not mAP/precision/recall/SOTA claims.",
                    "Image-plane path evidence is not GPS navigation.",
                    "EC-TERP is an auxiliary priority score and does not replace human rescue decisions.",
                ],
                "truthfulness_note": "All EC-TERP evaluation outputs must be interpreted as synthetic/user-provided algorithm analysis unless a real annotated validation set is supplied.",
            },
        )
        report_path = output_dir / "ec_terp_evaluation_report.md"
        report_path.write_text(results.get("report_markdown", ""), encoding="utf-8")
        files["report"] = str(report_path)
        return {"success": True, "output_dir": str(output_dir), "files": files}
    except Exception as exc:
        return {"success": False, "output_dir": str(output_dir), "files": {}, "error": str(exc)}


def run_full_ec_terp_evaluation(case_file=None, output_dir=None, n_random_trials=100):
    cases = load_eval_cases(case_file)
    weights = load_ec_terp_weights()
    comparison = run_terp_vs_ec_terp_comparison(cases=cases, weights_config=weights)
    sensitivity = run_single_factor_sensitivity_analysis(cases=cases, base_weights=weights)
    random_stability = run_random_weight_stability_analysis(
        cases=cases,
        base_weights=weights,
        n_trials=n_random_trials,
    )
    ablation = run_ablation_study(cases=cases, base_weights=weights)
    report = generate_ec_terp_evaluation_report(comparison, sensitivity, random_stability, ablation)
    bundle = {
        "success": True,
        "comparison_result": comparison,
        "sensitivity_result": sensitivity,
        "random_stability_result": random_stability,
        "ablation_result": ablation,
        "report_markdown": report,
        "truthfulness_note": "EC-TERP evaluation uses synthetic or user-provided cases and does not claim large-scale real rescue validation.",
    }
    saved = save_ec_terp_evaluation_outputs(bundle, output_dir=output_dir)
    bundle["output_files"] = saved.get("files", {})
    bundle["save_result"] = saved
    return bundle
