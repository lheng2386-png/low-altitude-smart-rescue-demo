import sys
import tempfile
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
APP_DIR = ROOT_DIR / "app"
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from ec_terp_evaluation_service import (  # noqa: E402
    compute_average_rank_shift,
    compute_top1_agreement,
    compute_topk_recall,
    generate_ec_terp_evaluation_report,
    load_eval_cases,
    run_ablation_study,
    run_full_ec_terp_evaluation,
    run_random_weight_stability_analysis,
    run_single_factor_sensitivity_analysis,
    run_terp_vs_ec_terp_comparison,
)


def main():
    cases = load_eval_cases()
    assert len(cases) >= 5
    for case in cases:
        assert case.get("case_id")
        assert case.get("targets")
        assert case.get("case_type") == "synthetic_demo"

    assert compute_top1_agreement(["T001", "T002"], ["T001", "T003"]) == 1.0
    assert compute_top1_agreement(["T002"], ["T001"]) == 0.0
    assert compute_topk_recall(["A", "B", "C"], ["B", "C", "D"], k=3) > 0
    assert compute_topk_recall([], ["A"], k=3) == 0.0
    assert compute_average_rank_shift(["A", "B", "C"], ["B", "A", "C"]) > 0

    comparison = run_terp_vs_ec_terp_comparison(cases=cases)
    assert comparison["success"] is True
    assert comparison["case_count"] >= 5
    assert comparison["comparisons"]
    for item in comparison["comparisons"]:
        assert item["baseline_order"]
        assert item["ec_terp_order"]

    sensitivity = run_single_factor_sensitivity_analysis(cases=cases)
    assert sensitivity["success"] is True
    assert sensitivity["results"]
    assert "mean_top1_stability" in sensitivity["summary"]
    assert "most_sensitive_weight" in sensitivity["summary"]

    random_stability = run_random_weight_stability_analysis(cases=cases, n_trials=5)
    assert random_stability["success"] is True
    for key in ["top1_stability", "top3_stability", "average_rank_shift"]:
        assert key in random_stability["summary"]

    ablation = run_ablation_study(cases=cases)
    assert ablation["success"] is True
    assert ablation["ablation_results"]
    assert any(item["ablation_name"] == "without_evidence_quality" for item in ablation["ablation_results"])
    assert "most_impactful_ablation" in ablation["summary"]

    report = generate_ec_terp_evaluation_report(comparison, sensitivity, random_stability, ablation)
    assert "EC-TERP 算法验证与灵敏度分析报告" in report
    assert "TERP vs EC-TERP" in report
    assert "灵敏度分析" in report
    assert "消融实验" in report
    assert "真实性边界" in report

    with tempfile.TemporaryDirectory() as tmp:
        full = run_full_ec_terp_evaluation(output_dir=tmp, n_random_trials=5)
        assert full["success"] is True
        assert isinstance(full["report_markdown"], str)
        assert "report" in full["output_files"]
        assert "evaluation_summary" in full["output_files"]
        assert "ranking_csv" in full["output_files"]
        assert "sensitivity_summary" in full["output_files"]
        assert "ablation_summary" in full["output_files"]
        assert "warnings_or_limitations" in full["output_files"]
        for path in full["output_files"].values():
            assert Path(path).exists()
        warnings_text = Path(full["output_files"]["warnings_or_limitations"]).read_text(encoding="utf-8")
        assert "synthetic/demo" in warnings_text or "synthetic demo" in warnings_text

    print("AeroRescue-AI EC-TERP evaluation smoke test passed.")


if __name__ == "__main__":
    main()
