import json
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
APP_DIR = ROOT_DIR / "app"
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from ec_terp_visualization_service import generate_ec_terp_visuals  # noqa: E402
from final_report_v2_service import build_final_report_v2  # noqa: E402


def _write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _ranking_payload():
    return {
        "success": True,
        "status": "executed_success",
        "module": "ec_terp_ranking",
        "rankings": [
            {
                "target_id": "T001",
                "target_type": "civilian",
                "rank": 1,
                "ec_terp_score": 78.5,
                "score_components": {
                    "target_urgency": 95.0,
                    "environment_risk": 48.0,
                    "route_accessibility": 70.0,
                    "coverage_gap": 30.0,
                    "evidence_quality": 100.0,
                    "uncertainty_penalty": 10.0,
                },
                "evidence_level": "strong",
                "source_modules": ["detection", "path_planning"],
                "is_confirmed_rescue_target": False,
                "human_review_required": True,
                "recommendation_type": "assistive_priority_ranking",
                "explanation": "Synthetic runtime ranking for visualization smoke test.",
                "limitations": ["EC-TERP is an assistive priority ranking algorithm."],
                "truthfulness_note": "EC-TERP provides assistive image-plane priority ranking only and does not replace human rescue decisions.",
            },
            {
                "target_id": "TR002",
                "target_type": "human_candidate",
                "rank": 2,
                "ec_terp_score": 51.0,
                "score_components": {
                    "target_urgency": 75.0,
                    "environment_risk": 48.0,
                    "route_accessibility": 30.0,
                    "coverage_gap": 30.0,
                    "evidence_quality": 70.0,
                    "uncertainty_penalty": 45.0,
                },
                "evidence_level": "medium",
                "source_modules": ["detection"],
                "is_confirmed_rescue_target": False,
                "human_review_required": True,
                "recommendation_type": "assistive_priority_ranking",
                "explanation": "human_candidate remains a candidate and needs manual review.",
                "limitations": ["Transformer human_candidate is not a confirmed civilian."],
                "truthfulness_note": "EC-TERP provides assistive image-plane priority ranking only and does not replace human rescue decisions.",
            },
        ],
    }


def main():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        missing = generate_ec_terp_visuals(
            ranking_path=root / "outputs" / "ec_terp" / "missing.json",
            eval_dir=root / "outputs" / "ec_terp_evaluation",
            output_dir=root / "outputs" / "ec_terp_visuals_missing",
        )
        assert missing["status"] in {"failed", "degraded"}
        assert Path(missing["metadata_path"]).exists()

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        ranking_path = root / "outputs" / "ec_terp" / "ec_terp_rankings.json"
        _write_json(ranking_path, _ranking_payload())
        _write_json(
            root / "outputs" / "ec_terp_evaluation" / "sensitivity_results.json",
            {
                "success": True,
                "results": [
                    {"weight_name": "target_urgency_weight", "top3_stability": 1.0},
                    {"weight_name": "uncertainty_penalty_weight", "top3_stability": 0.8},
                ],
                "truthfulness_note": "Synthetic demo only.",
            },
        )
        metadata = generate_ec_terp_visuals(
            ranking_path=ranking_path,
            eval_dir=root / "outputs" / "ec_terp_evaluation",
            output_dir=root / "outputs" / "ec_terp_visuals",
        )
        assert metadata["status"] in {"executed_success", "degraded"}
        assert Path(metadata["metadata_path"]).exists()
        assert "Synthetic demo cases are not real rescue data." in metadata["truthfulness_notes"]
        if metadata["generated_figures"]:
            for figure in metadata["generated_figures"]:
                assert Path(figure["path"]).exists()

        script_output = root / "outputs" / "ec_terp_visuals_script"
        completed = subprocess.run(
            [
                sys.executable,
                str(ROOT_DIR / "scripts" / "generate_ec_terp_visuals.py"),
                "--ranking-path",
                str(ranking_path),
                "--eval-dir",
                str(root / "outputs" / "ec_terp_evaluation"),
                "--output-dir",
                str(script_output),
            ],
            check=False,
            text=True,
            capture_output=True,
        )
        assert completed.returncode in {0, 1}
        assert (script_output / "ec_terp_visuals_metadata.json").exists()

        report = build_final_report_v2(root_dir=root)
        markdown = report["report_markdown"]
        assert "EC-TERP = αT + βE + γR + δC + λQ - μU" in markdown
        assert "EC-TERP is an assistive priority ranking algorithm" in markdown
        assert "Image-plane path planning is not GPS navigation" in markdown
        assert "Synthetic demo cases are not real rescue data" in markdown

    doc_path = ROOT_DIR / "docs" / "ec_terp_algorithm.md"
    assert doc_path.exists()
    doc_text = doc_path.read_text(encoding="utf-8")
    assert "EC-TERP is not an automatic rescue decision system" in doc_text
    assert "Synthetic demo cases are not real rescue data" in doc_text

    print("灾情感知及影响评估 EC-TERP visualization smoke test passed.")


if __name__ == "__main__":
    main()
