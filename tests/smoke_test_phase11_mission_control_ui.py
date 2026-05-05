import sys
import tempfile
import shutil
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.ui.mission_control_panel import (  # noqa: E402
    format_candidate_summary,
    format_demo_result_summary,
    format_stage_result_table,
    load_final_report_preview,
    resolve_allowed_missions_root,
    run_one_click_demo_from_ui,
)


def _fake_demo_result(report_path="/tmp/final_report.md"):
    return {
        "mission": {"mission_id": "M_TEST", "mission_name": "UI Demo"},
        "mission_dir": "/tmp/M_TEST",
        "demo_dataset_dir": "/tmp/demo_dataset",
        "final_report_markdown_path": report_path,
        "evidence_ledger_path": "/tmp/ledger.json",
        "stage_results": {
            "local_recon": {"status": "completed", "candidate_count": 2},
            "target_verification": {"status": "completed", "verification_summary": {"verification_count": 2}},
            "thermal_check": {"status": "completed", "thermal_summary": {"thermal_check_count": 1}},
            "decision_fusion": {"status": "completed", "decision_summary": {"decision_candidate_count": 2}},
            "rescue_recommendation": {"status": "completed", "recommendation_summary": {"recommendation_count": 1}},
            "evidence_report": {"status": "completed"},
        },
        "truthfulness_note": "Demo data is for workflow demonstration only and is not operational disaster evidence.",
    }


def main():
    summary = format_demo_result_summary(_fake_demo_result())
    assert "M_TEST" in summary
    assert "Demo data is for workflow demonstration only" in summary

    table = format_stage_result_table(_fake_demo_result()["stage_results"])
    assert isinstance(table, list)
    assert len(table) == 9
    assert any("local_recon" in row[0] for row in table)

    candidate_summary = format_candidate_summary(_fake_demo_result()["stage_results"])
    assert "candidate_count" in candidate_summary
    assert "thermal_summary" in candidate_summary

    with tempfile.TemporaryDirectory() as tmp:
        report_path = Path(tmp) / "final_report.md"
        report_path.write_text("# AeroRescue-AI Final Report 2.0\n\nAI 辅助决策报告", encoding="utf-8")
        preview = load_final_report_preview(report_path)
        assert "Final Report 2.0" in preview
        assert "Final Report 尚未生成" in load_final_report_preview(Path(tmp) / "missing.md")
        try:
            resolve_allowed_missions_root(Path(tmp) / "outside_outputs")
            raise AssertionError("outside mission roots must be rejected")
        except ValueError as exc:
            assert "Missions Root must stay under" in str(exc)

        smoke_root = ROOT_DIR / "outputs" / "mission_control_smoke_test"
        shutil.rmtree(smoke_root, ignore_errors=True)
        outputs = run_one_click_demo_from_ui(smoke_root, "Smoke UI Demo")
        summary_markdown, stage_table, candidate_markdown, final_report_preview, truthfulness_markdown = outputs
        assert "Mission" in summary_markdown or "Demo" in summary_markdown
        assert isinstance(stage_table, list)
        assert "Final Report" in final_report_preview or "报告" in final_report_preview
        assert "Demo data" in truthfulness_markdown or "workflow demonstration" in truthfulness_markdown
        assert "candidate_count" in candidate_markdown
        shutil.rmtree(smoke_root, ignore_errors=True)

    print("AeroRescue-AI phase 11 mission control UI smoke test passed.")


if __name__ == "__main__":
    main()
