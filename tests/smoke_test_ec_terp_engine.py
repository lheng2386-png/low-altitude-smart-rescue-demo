import sys
import tempfile
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
APP_DIR = ROOT_DIR / "app"
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from ec_terp_engine import (  # noqa: E402
    compare_terp_and_ec_terp,
    compute_coverage_gap_score,
    compute_ec_terp_score,
    compute_environment_risk,
    compute_evidence_quality,
    compute_route_accessibility,
    compute_target_urgency,
    compute_uncertainty_penalty,
    evidence_level_to_score,
    format_ec_terp_result_markdown,
    load_ec_terp_weights,
    rank_targets_by_ec_terp,
)


def main():
    weights = load_ec_terp_weights()
    for key in [
        "target_urgency_weight",
        "environment_risk_weight",
        "route_accessibility_weight",
        "coverage_gap_weight",
        "evidence_quality_weight",
        "uncertainty_penalty_weight",
    ]:
        assert key in weights

    with tempfile.TemporaryDirectory() as tmp:
        missing_weights = load_ec_terp_weights(Path(tmp) / "missing.json")
        assert missing_weights["config_status"].startswith("default_used")

    assert evidence_level_to_score("strong") == 1.0
    assert evidence_level_to_score("medium") == 0.7
    assert evidence_level_to_score("weak") == 0.35
    assert evidence_level_to_score("none") == 0.0
    assert evidence_level_to_score("unknown") == 0.0

    civilian = {"id": "T001", "class_name": "civilian", "confidence": 0.9, "area": 1600, "human_review_required": True}
    dog = {"id": "T002", "class_name": "dog", "confidence": 0.9, "area": 1600}
    human_candidate = {"id": "TR001", "class_name": "human_candidate", "confidence": 0.9, "area": 1600}
    rescuer = {"id": "T003", "class_name": "rescuer", "confidence": 0.9, "area": 1600}
    assert compute_target_urgency(civilian)["score"] > compute_target_urgency(dog)["score"]
    human_result = compute_target_urgency(human_candidate)
    assert human_result["score"] < compute_target_urgency(civilian)["score"] or "人工复核" in human_result["reason"]
    assert compute_target_urgency(rescuer)["score"] < compute_target_urgency(civilian)["score"]

    segmentation_summary = {
        "destroyed_building": 0.1,
        "major_damage": 0.2,
        "water": 0.15,
        "road_blocked": 0.1,
        "road_clear": 0.2,
    }
    env_result = compute_environment_risk(segmentation_summary=segmentation_summary)
    assert env_result["score"] > 0
    no_env = compute_environment_risk()
    assert no_env["score"] == 0
    assert "缺少" in no_env["reason"]

    found_path = {"found": True, "path_length": 30, "total_cost": 10, "risk_cost": 5, "path": [[0, 0], [10, 10]]}
    missing_path = {"found": False}
    assert compute_route_accessibility(found_path)["score"] > compute_route_accessibility(missing_path)["score"]
    assert "GPS" in compute_route_accessibility(found_path)["reason"] or "image-plane" in compute_route_accessibility(found_path)["reason"]

    coverage = compute_coverage_gap_score({"unsearched_high_priority_ratio": 0.6})
    assert abs(coverage["score"] - 60.0) < 1e-6
    assert compute_coverage_gap_score()["score"] == 0

    assert compute_evidence_quality(target_evidence_level="strong")["score"] == 100.0
    assert compute_evidence_quality(target_evidence_level="medium")["score"] == 70.0
    assert compute_evidence_quality(target_evidence_level="weak")["score"] == 35.0

    low_conf_penalty = compute_uncertainty_penalty({"class_name": "civilian", "confidence": 0.2})
    transformer_penalty = compute_uncertainty_penalty({"class_name": "human_candidate", "confidence": 0.9}, transformer_only=True, segmentation_available=False)
    assert low_conf_penalty["score"] >= 30
    assert transformer_penalty["score"] >= 30

    ec_result = compute_ec_terp_score(
        {
            "id": "T001",
            "class_name": "civilian",
            "confidence": 0.9,
            "bbox": [10, 10, 50, 50],
            "center": [30, 30],
            "area": 1600,
            "human_review_required": True,
        },
        segmentation_summary=segmentation_summary,
        path_result=found_path,
        path_comparison={"risk_reduction": 0.2},
        coverage_result={"unsearched_high_priority_ratio": 0.3},
        target_evidence_level="strong",
        segmentation_available=True,
    )
    assert 0 <= ec_result["ec_terp_score"] <= 100
    assert ec_result["ec_terp_level"] in {"low", "medium", "high", "critical"}
    assert set(ec_result["components"]) == {
        "target_urgency",
        "environment_risk",
        "route_accessibility",
        "coverage_gap",
        "evidence_quality",
        "uncertainty_penalty",
    }
    assert "auxiliary" in ec_result["truthfulness_note"] or "human rescue" in ec_result["truthfulness_note"]

    ranked = rank_targets_by_ec_terp(
        [rescuer, civilian],
        segmentation_summary=segmentation_summary,
        path_result=found_path,
        coverage_result={"unsearched_high_priority_ratio": 0.3},
        target_evidence_level="strong",
        segmentation_available=True,
    )
    assert ranked[0]["class_name"] == "civilian"
    assert ranked[0]["rank"] == 1

    comparison = compare_terp_and_ec_terp(
        [{"target_id": "T001", "rank": 2}, {"target_id": "T002", "rank": 1}],
        [{"target_id": "T001", "rank": 1}, {"target_id": "T002", "rank": 2}],
    )
    assert comparison["success"] is True
    assert isinstance(comparison["summary"], str)

    markdown = format_ec_terp_result_markdown(ranked)
    assert "EC-TERP = αT + βE + γR + δC + λQ - μU" in markdown
    assert "人工复核" in markdown or "真实性边界" in markdown

    print("灾情感知及影响评估 EC-TERP smoke test passed.")


if __name__ == "__main__":
    main()
