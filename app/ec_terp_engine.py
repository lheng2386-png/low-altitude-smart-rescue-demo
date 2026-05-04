"""Evidence-Constrained TERP priority scoring for AeroRescue-AI.

EC-TERP is a transparent auxiliary priority algorithm. It combines target
urgency, environment risk, image-plane route accessibility, coverage gap,
evidence quality, and uncertainty. It does not confirm victims and does not
replace human rescue decisions.
"""

import json
from pathlib import Path


class ECTERPError(Exception):
    pass


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = ROOT_DIR / "configs" / "ec_terp_weights.json"

DEFAULT_WEIGHTS = {
    "target_urgency_weight": 0.30,
    "environment_risk_weight": 0.25,
    "route_accessibility_weight": 0.20,
    "coverage_gap_weight": 0.15,
    "evidence_quality_weight": 0.10,
    "uncertainty_penalty_weight": 0.15,
    "normalization": {"score_min": 0.0, "score_max": 100.0},
    "evidence_level_scores": {
        "strong": 1.0,
        "medium": 0.7,
        "weak": 0.35,
        "none": 0.0,
    },
    "truthfulness_note": "These are transparent prior weights for EC-TERP. They are not learned model parameters and should be calibrated with a validation set in future work.",
}


def load_ec_terp_weights(config_path=None):
    """Load EC-TERP prior weights, falling back to built-in defaults."""
    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    if not path.exists():
        weights = dict(DEFAULT_WEIGHTS)
        weights["config_status"] = "default_used_missing_file"
        weights["config_path"] = str(path)
        return weights
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        weights = dict(DEFAULT_WEIGHTS)
        weights["config_status"] = "default_used_json_error"
        weights["config_path"] = str(path)
        weights["config_error"] = str(exc)
        return weights
    weights = dict(DEFAULT_WEIGHTS)
    weights.update(loaded or {})
    weights["config_status"] = "loaded"
    weights["config_path"] = str(path)
    return weights


def clamp(value, min_value=0.0, max_value=100.0):
    try:
        numeric = float(value)
    except Exception:
        numeric = min_value
    return max(float(min_value), min(float(max_value), numeric))


def normalize_score(value, input_min=0.0, input_max=100.0):
    """Normalize scores to 0-1.

    Values already in 0-1 remain compatible, while ordinary 0-100 scores are
    scaled by the configured interval.
    """
    value = clamp(value, input_min, input_max)
    if 0.0 <= float(value) <= 1.0 and float(input_min) == 0.0 and float(input_max) == 100.0:
        return float(value)
    span = max(float(input_max) - float(input_min), 1e-9)
    return clamp((float(value) - float(input_min)) / span, 0.0, 1.0)


def evidence_level_to_score(evidence_level, weights_config=None):
    config = weights_config or load_ec_terp_weights()
    mapping = config.get("evidence_level_scores", DEFAULT_WEIGHTS["evidence_level_scores"])
    key = str(evidence_level or "none").strip().lower()
    return float(mapping.get(key, 0.0))


def _ratio_from_summary(summary, *keys):
    if not summary:
        return 0.0
    for key in keys:
        if key not in summary:
            continue
        value = summary.get(key)
        if isinstance(value, dict):
            for ratio_key in ("ratio", "area_ratio", "class_area_ratio"):
                if ratio_key in value:
                    return clamp(value.get(ratio_key), 0.0, 1.0)
            if "pixels" in value and "total_pixels" in summary:
                return clamp(float(value.get("pixels", 0.0)) / max(float(summary.get("total_pixels", 1.0)), 1.0), 0.0, 1.0)
        try:
            return clamp(float(value), 0.0, 1.0)
        except Exception:
            continue
    return 0.0


def _target_id(target):
    return target.get("id") or target.get("target_id") or target.get("object_id") or "unknown"


def compute_target_urgency(target):
    target = target or {}
    class_name = str(target.get("class_name") or target.get("label") or "unknown").lower()
    base_scores = {
        "civilian": 90.0,
        "human_candidate": 75.0,
        "person": 75.0,
        "rescuer": 20.0,
        "dog": 45.0,
        "cat": 45.0,
        "horse": 55.0,
        "cow": 55.0,
        "vehicle": 10.0,
        "fire": 0.0,
        "unknown": 30.0,
    }
    base = base_scores.get(class_name, 30.0)
    confidence = clamp(target.get("confidence", 0.0), 0.0, 1.0)
    confidence_bonus = confidence * 10.0
    area = max(float(target.get("area", 0.0) or 0.0), 0.0)
    bbox = target.get("bbox") or []
    if area <= 0.0 and len(bbox) >= 4:
        area = max(0.0, float(bbox[2]) - float(bbox[0])) * max(0.0, float(bbox[3]) - float(bbox[1]))
    area_bonus = clamp(area / 5000.0 * 10.0, 0.0, 10.0)
    review_penalty = 8.0 if class_name in {"human_candidate", "person"} else 0.0
    score = clamp(base + confidence_bonus + area_bonus - review_penalty, 0.0, 100.0)
    reason = f"目标类别 {class_name} 的基础紧急度为 {base}，置信度和目标尺度提供补充加权。"
    if class_name in {"human_candidate", "person"}:
        reason += " human_candidate 不是 confirmed civilian，必须人工复核。"
    if class_name in {"vehicle", "fire"}:
        reason += " 该类别主要作为环境风险补充，不作为被困人员目标。"
    return {
        "score": round(score, 4),
        "components": {
            "class_name": class_name,
            "base": round(base, 4),
            "confidence": round(confidence, 4),
            "confidence_bonus": round(confidence_bonus, 4),
            "area": round(area, 4),
            "area_bonus": round(area_bonus, 4),
            "review_penalty": round(review_penalty, 4),
        },
        "reason": reason,
    }


def compute_environment_risk(target=None, environment_context=None, segmentation_summary=None):
    if environment_context and "environment_risk_score" in environment_context:
        raw_score = clamp(environment_context.get("environment_risk_score"), 0.0, 100.0)
        if raw_score <= 20.0:
            raw_score *= 5.0
        return {
            "score": round(clamp(raw_score), 4),
            "components": {"source": "environment_context", "environment_risk_score": environment_context.get("environment_risk_score")},
            "reason": environment_context.get("environment_reason") or "使用目标周边语义环境风险分数。",
        }

    destroyed = _ratio_from_summary(segmentation_summary, "destroyed_building", "building_total_destruction", "total_destruction")
    major = _ratio_from_summary(segmentation_summary, "major_damage", "building_major_damage")
    water = _ratio_from_summary(segmentation_summary, "water")
    road_blocked = _ratio_from_summary(segmentation_summary, "road_blocked")
    road_clear = _ratio_from_summary(segmentation_summary, "road_clear")
    if not any([destroyed, major, water, road_blocked, road_clear]):
        return {
            "score": 0.0,
            "components": {
                "destroyed_building_ratio": 0.0,
                "major_damage_ratio": 0.0,
                "water_ratio": 0.0,
                "road_blocked_ratio": 0.0,
                "road_clear_ratio": 0.0,
            },
            "reason": "缺少语义分割或环境输入，环境风险按 0 处理。",
        }
    score = 100.0 * (0.40 * destroyed + 0.30 * major + 0.15 * water + 0.15 * road_blocked - 0.08 * road_clear)
    score = clamp(score)
    return {
        "score": round(score, 4),
        "components": {
            "destroyed_building_ratio": round(destroyed, 4),
            "major_damage_ratio": round(major, 4),
            "water_ratio": round(water, 4),
            "road_blocked_ratio": round(road_blocked, 4),
            "road_clear_ratio": round(road_clear, 4),
        },
        "reason": "根据 destroyed/major damage、水域、阻断道路和可通行道路比例计算图像平面环境风险。",
    }


def compute_route_accessibility(path_result=None, path_comparison=None):
    if not path_result:
        return {
            "score": 0.0,
            "components": {"path_available": False},
            "reason": "缺少路径规划结果，因此路径可达性证据不足；当前路径如存在也只是 image-plane reference path，不是真实 GPS 导航。",
        }
    if not path_result.get("found"):
        return {
            "score": 0.0,
            "components": {"path_available": True, "found": False},
            "reason": "路径规划未找到可达路径；该路径判断仍属于 image-plane reference path，不是真实 GPS 导航。",
        }
    length = float(path_result.get("path_length") or len(path_result.get("path", [])) or 0.0)
    total_cost = float(path_result.get("total_cost", 0.0) or 0.0)
    risk_cost = float(path_result.get("risk_cost", path_result.get("risk_exposure", 0.0)) or 0.0)
    base = 70.0
    if length > 0:
        base += clamp((120.0 - length) / 120.0 * 15.0, -10.0, 15.0)
    if total_cost > 0:
        base -= clamp(total_cost / 200.0 * 20.0, 0.0, 20.0)
    if risk_cost > 0:
        base -= clamp(risk_cost / 100.0 * 15.0, 0.0, 15.0)
    risk_reduction = 0.0
    if path_comparison:
        risk_reduction = float(
            path_comparison.get("risk_reduction")
            or path_comparison.get("path_risk_reduction")
            or path_comparison.get("risk_reduction_ratio")
            or 0.0
        )
        if risk_reduction > 1.0:
            risk_reduction = risk_reduction / 100.0
        base += clamp(risk_reduction * 15.0, 0.0, 15.0)
    score = clamp(base)
    return {
        "score": round(score, 4),
        "components": {
            "found": True,
            "path_length": round(length, 4),
            "total_cost": round(total_cost, 4),
            "risk_cost": round(risk_cost, 4),
            "risk_reduction": round(risk_reduction, 4),
        },
        "reason": "路径可达性来自图像平面路径结果和风险对比；不是 GPS 导航或真实无人机航线。",
    }


def compute_coverage_gap_score(coverage_result=None, decision_fusion_result=None):
    coverage_result = coverage_result or {}
    if not coverage_result and decision_fusion_result:
        coverage_result = decision_fusion_result.get("coverage_result") or decision_fusion_result.get("coverage") or {}
    if "unsearched_high_priority_ratio" in coverage_result:
        ratio = clamp(coverage_result.get("unsearched_high_priority_ratio"), 0.0, 1.0)
        return {
            "score": round(ratio * 100.0, 4),
            "components": {"unsearched_high_priority_ratio": round(ratio, 4)},
            "reason": "覆盖缺口来自未搜索高优先级区域比例；这是 image-plane coverage gap，不是真实 UAV 覆盖航线。",
        }
    if "high_priority_covered_ratio" in coverage_result:
        covered = clamp(coverage_result.get("high_priority_covered_ratio"), 0.0, 1.0)
        return {
            "score": round((1.0 - covered) * 100.0, 4),
            "components": {"high_priority_covered_ratio": round(covered, 4)},
            "reason": "覆盖缺口由高优先级区域已覆盖比例反推；这是 image-plane coverage gap，不是真实 UAV 覆盖航线。",
        }
    return {
        "score": 0.0,
        "components": {},
        "reason": "未提供 coverage evidence，覆盖缺口按 0 处理。",
    }


def _iter_evidence_records(module_evidence_records):
    if not module_evidence_records:
        return []
    if isinstance(module_evidence_records, dict):
        if "evidence_records" in module_evidence_records:
            records = module_evidence_records.get("evidence_records") or {}
            return list(records.values()) if isinstance(records, dict) else list(records or [])
        if "evidence_level" in module_evidence_records:
            return [module_evidence_records]
        return list(module_evidence_records.values())
    return list(module_evidence_records)


def compute_evidence_quality(module_evidence_records=None, target_evidence_level=None):
    config = load_ec_terp_weights()
    if target_evidence_level is not None:
        score = evidence_level_to_score(target_evidence_level, config) * 100.0
        return {
            "score": round(score, 4),
            "components": {"target_evidence_level": str(target_evidence_level)},
            "reason": f"使用目标级证据等级 {target_evidence_level} 映射证据质量。",
        }

    relevant = []
    for record in _iter_evidence_records(module_evidence_records):
        module_key = str(record.get("module_key", "")).lower()
        section = str(record.get("recommended_report_section", "")).lower()
        if any(key in module_key for key in ("detection", "segmentation", "path", "decision_fusion", "thermal")) or any(
            key in section for key in ("模型", "辅助", "真实")
        ):
            relevant.append(record)
    if not relevant:
        return {
            "score": 0.0,
            "components": {"evidence_record_count": 0},
            "reason": "缺少 Mission Evidence Ledger 或目标相关 evidence records，证据质量按 0 处理。",
        }
    scores = [evidence_level_to_score(record.get("evidence_level"), config) for record in relevant]
    quality = sum(scores) / max(len(scores), 1) * 100.0
    return {
        "score": round(quality, 4),
        "components": {
            "evidence_record_count": len(relevant),
            "levels": [record.get("evidence_level", "none") for record in relevant],
        },
        "reason": "根据检测、分割、路径和决策融合相关证据等级计算平均证据质量。",
    }


def compute_uncertainty_penalty(
    target=None,
    evidence_quality_result=None,
    path_result=None,
    segmentation_available=False,
    transformer_only=False,
):
    target = target or {}
    confidence = clamp(target.get("confidence", 0.0), 0.0, 1.0)
    score = 0.0
    reasons = []
    if confidence < 0.35:
        score += 30.0
        reasons.append("检测置信度低于 0.35")
    elif confidence < 0.55:
        score += 15.0
        reasons.append("检测置信度处于 0.35-0.55 区间")
    class_name = str(target.get("class_name", "")).lower()
    if transformer_only or class_name == "human_candidate" or str(target.get("source_backend", "")).startswith("transformer"):
        score += 20.0
        reasons.append("Transformer-only 或 human_candidate 需要人工复核")
    evidence_score = float((evidence_quality_result or {}).get("score", 0.0) or 0.0)
    if evidence_score < 35.0:
        score += 25.0
        reasons.append("证据质量低于 weak/medium 阈值")
    if not path_result or not path_result.get("found"):
        score += 15.0
        reasons.append("缺少可达路径或路径未找到")
    if not segmentation_available:
        score += 10.0
        reasons.append("缺少 segmentation 环境输入")
    if target.get("human_review_required"):
        score += 10.0
        reasons.append("目标标记为需要人工复核")
    score = clamp(score)
    return {
        "score": round(score, 4),
        "components": {
            "confidence": round(confidence, 4),
            "transformer_only": bool(transformer_only),
            "evidence_quality_score": round(evidence_score, 4),
            "path_found": bool(path_result and path_result.get("found")),
            "segmentation_available": bool(segmentation_available),
            "human_review_required": bool(target.get("human_review_required")),
        },
        "reason": "；".join(reasons) if reasons else "未检测到明显不确定性惩罚项。",
    }


def _level_from_score(score):
    if score >= 75.0:
        return "critical"
    if score >= 55.0:
        return "high"
    if score >= 35.0:
        return "medium"
    return "low"


def compute_ec_terp_score(
    target,
    environment_context=None,
    segmentation_summary=None,
    path_result=None,
    path_comparison=None,
    coverage_result=None,
    decision_fusion_result=None,
    module_evidence_records=None,
    target_evidence_level=None,
    segmentation_available=False,
    transformer_only=False,
    weights_config=None,
):
    target = target or {}
    weights = weights_config or load_ec_terp_weights()
    target_result = compute_target_urgency(target)
    environment_result = compute_environment_risk(target, environment_context, segmentation_summary)
    route_result = compute_route_accessibility(path_result, path_comparison)
    coverage_result_detail = compute_coverage_gap_score(coverage_result, decision_fusion_result)
    evidence_result = compute_evidence_quality(module_evidence_records, target_evidence_level)
    uncertainty_result = compute_uncertainty_penalty(
        target,
        evidence_quality_result=evidence_result,
        path_result=path_result,
        segmentation_available=segmentation_available,
        transformer_only=transformer_only,
    )

    norm = weights.get("normalization", {})
    score_min = float(norm.get("score_min", 0.0))
    score_max = float(norm.get("score_max", 100.0))
    t = normalize_score(target_result["score"], score_min, score_max)
    e = normalize_score(environment_result["score"], score_min, score_max)
    r = normalize_score(route_result["score"], score_min, score_max)
    c = normalize_score(coverage_result_detail["score"], score_min, score_max)
    q = normalize_score(evidence_result["score"], score_min, score_max)
    u = normalize_score(uncertainty_result["score"], score_min, score_max)

    score_0_1 = (
        float(weights.get("target_urgency_weight", 0.30)) * t
        + float(weights.get("environment_risk_weight", 0.25)) * e
        + float(weights.get("route_accessibility_weight", 0.20)) * r
        + float(weights.get("coverage_gap_weight", 0.15)) * c
        + float(weights.get("evidence_quality_weight", 0.10)) * q
        - float(weights.get("uncertainty_penalty_weight", 0.15)) * u
    )
    score = round(clamp(score_0_1 * 100.0), 4)
    class_name = str(target.get("class_name", "unknown"))
    human_review_required = bool(
        target.get("human_review_required")
        or class_name.lower() in {"human_candidate", "person"}
        or transformer_only
        or uncertainty_result["score"] > 0.0
    )
    explanation = (
        f"目标 {_target_id(target)} 的 EC-TERP 得分为 {score}，主要由目标紧急度 {target_result['score']}、"
        f"环境风险 {environment_result['score']}、路径可达性 {route_result['score']}、覆盖缺口 {coverage_result_detail['score']}、"
        f"证据质量 {evidence_result['score']} 与不确定性惩罚 {uncertainty_result['score']} 共同决定。"
    )
    limitations = [
        "EC-TERP is an assistive priority ranking algorithm.",
        "It is not an automatic rescue decision system.",
        "Image-plane route accessibility is not GPS navigation.",
        "Synthetic/demo evaluation cases are not real rescue data.",
    ]
    if class_name.lower() in {"human_candidate", "person"} or transformer_only:
        limitations.append("human_candidate is not a confirmed civilian or confirmed survivor and requires manual review.")
    if target_evidence_level:
        limitations.append(f"Evidence level is {target_evidence_level}; weak/preview/simulated evidence must not be upgraded to strong.")
    score_components = {
        "target_urgency": target_result["score"],
        "environment_risk": environment_result["score"],
        "route_accessibility": route_result["score"],
        "coverage_gap": coverage_result_detail["score"],
        "evidence_quality": evidence_result["score"],
        "uncertainty_penalty": uncertainty_result["score"],
    }
    return {
        "target_id": _target_id(target),
        "target_type": class_name,
        "class_name": class_name,
        "ec_terp_score": score,
        "ec_terp_level": _level_from_score(score),
        "score_components": score_components,
        "components": {
            "target_urgency": target_result,
            "environment_risk": environment_result,
            "route_accessibility": route_result,
            "coverage_gap": coverage_result_detail,
            "evidence_quality": evidence_result,
            "uncertainty_penalty": uncertainty_result,
        },
        "weights": {
            "target_urgency_weight": weights.get("target_urgency_weight", 0.30),
            "environment_risk_weight": weights.get("environment_risk_weight", 0.25),
            "route_accessibility_weight": weights.get("route_accessibility_weight", 0.20),
            "coverage_gap_weight": weights.get("coverage_gap_weight", 0.15),
            "evidence_quality_weight": weights.get("evidence_quality_weight", 0.10),
            "uncertainty_penalty_weight": weights.get("uncertainty_penalty_weight", 0.15),
        },
        "formula": "EC-TERP = αT + βE + γR + δC + λQ - μU",
        "evidence_level": str(target_evidence_level or "none"),
        "source_modules": [],
        "is_confirmed_rescue_target": False,
        "human_review_required": human_review_required,
        "recommendation_type": "assistive_priority_ranking",
        "truthfulness_note": "EC-TERP is an evidence-constrained auxiliary priority score. It does not confirm victims or replace human rescue decisions.",
        "explanation": explanation,
        "limitations": limitations,
    }


def rank_targets_by_ec_terp(targets, **kwargs):
    if not targets:
        return []
    rankings = [compute_ec_terp_score(target, **kwargs) for target in targets]
    rankings.sort(key=lambda item: item.get("ec_terp_score", 0.0), reverse=True)
    for index, item in enumerate(rankings, start=1):
        item["rank"] = index
    return rankings


def compare_terp_and_ec_terp(original_terp_rankings=None, ec_terp_rankings=None):
    if not original_terp_rankings or not ec_terp_rankings:
        return {
            "success": False,
            "changed_rankings": [],
            "summary": "缺少原 TERP 或 EC-TERP 排名，无法进行对比。",
            "truthfulness_note": "Ranking comparison is explanatory only and does not prove EC-TERP is objectively more correct.",
        }
    original_pos = {str(item.get("target_id")): item.get("rank", idx + 1) for idx, item in enumerate(original_terp_rankings)}
    changed = []
    for idx, item in enumerate(ec_terp_rankings, start=1):
        target_id = str(item.get("target_id"))
        old_rank = original_pos.get(target_id)
        if old_rank is not None and int(old_rank) != idx:
            changed.append(
                {
                    "target_id": target_id,
                    "original_terp_rank": int(old_rank),
                    "ec_terp_rank": idx,
                    "rank_delta": int(old_rank) - idx,
                    "note": "EC-TERP 引入证据质量、覆盖缺口和不确定性后产生排名变化。",
                }
            )
    summary = "EC-TERP 排名与原 TERP 完全一致。" if not changed else f"EC-TERP 与原 TERP 有 {len(changed)} 个目标排名发生变化。"
    return {
        "success": True,
        "changed_rankings": changed,
        "summary": summary,
        "truthfulness_note": "This comparison only explains ranking changes. It does not claim EC-TERP is always more correct without validation data.",
    }


def format_ec_terp_result_markdown(ec_terp_rankings):
    lines = [
        "## EC-TERP 证据约束优先级结果",
        "",
        "公式：`EC-TERP = αT + βE + γR + δC + λQ - μU`",
        "",
        "- T：目标紧急度",
        "- E：环境风险",
        "- R：路径可达性",
        "- C：覆盖缺口",
        "- Q：证据质量",
        "- U：不确定性惩罚",
        "",
    ]
    if not ec_terp_rankings:
        lines.append("当前没有可排序目标。")
    else:
        lines.extend(["| Rank | Target | Class | EC-TERP | Level | Review |", "| --- | --- | --- | ---: | --- | --- |"])
        for item in ec_terp_rankings:
            lines.append(
                f"| {item.get('rank', '')} | {item.get('target_id', '')} | {item.get('class_name', '')} | "
                f"{item.get('ec_terp_score', 0.0)} | {item.get('ec_terp_level', '')} | "
                f"{'是' if item.get('human_review_required') else '否'} |"
            )
        lines.append("")
        lines.append("### Component Scores")
        for item in ec_terp_rankings:
            comp = item.get("components", {})
            lines.append(
                f"- {item.get('target_id')}: T={comp.get('target_urgency', {}).get('score', 0)}, "
                f"E={comp.get('environment_risk', {}).get('score', 0)}, "
                f"R={comp.get('route_accessibility', {}).get('score', 0)}, "
                f"C={comp.get('coverage_gap', {}).get('score', 0)}, "
                f"Q={comp.get('evidence_quality', {}).get('score', 0)}, "
                f"U={comp.get('uncertainty_penalty', {}).get('score', 0)}"
            )
    lines.extend(
        [
            "",
            "### 真实性边界",
            "- EC-TERP 是辅助优先级算法，不确认真实被困人员。",
            "- human_candidate 不等于 confirmed civilian，必须人工复核。",
            "- 路径可达性来自图像平面参考路径，不是真实 GPS 导航。",
            "- 证据质量来自 Mission Evidence Ledger 或结构化输入，不能凭空伪造。",
        ]
    )
    return "\n".join(lines)
