import json
import os
import sys
import tempfile
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
APP_DIR = ROOT_DIR / "app"
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(APP_DIR) not in sys.path:
    sys.path.append(str(APP_DIR))

from backend.llm.auditor import run_evidence_audit  # noqa: E402
from scripts.run_llm_demo import MISSION_ID, run_demo  # noqa: E402


def _mock_env():
    os.environ["LLM_ENABLE"] = "false"
    os.environ.pop("OPENAI_API_KEY", None)


def _mission_reports_dir(root):
    return Path(root) / "outputs" / "missions" / MISSION_ID / "outputs" / "reports"


def _run_staged_demo(root):
    _mock_env()
    return run_demo(root_dir=root, quiet=True)


def _issues(response):
    return response["result"]["issues"]


def _has_issue(response, category=None, severity=None, contains=None):
    for issue in _issues(response):
        if category and issue.get("category") != category:
            continue
        if severity and issue.get("severity") != severity:
            continue
        if contains and contains.lower() not in json.dumps(issue, ensure_ascii=False).lower():
            continue
        return True
    return False


def test_normal_demo_audit_runs_with_mock_provider():
    with tempfile.TemporaryDirectory() as tmp:
        _run_staged_demo(tmp)
        response = run_evidence_audit(MISSION_ID, "all", root_dir=tmp)
        assert response["ok"] is True
        assert response["provider"] == "mock"
        assert response["result"]["human_review_required"] is True
        assert response["result"]["audit_status"] in {"pass", "warning"}
        assert Path(response["saved_path"]).exists()


def test_confirmed_survivor_in_final_report_is_high_severity():
    with tempfile.TemporaryDirectory() as tmp:
        _run_staged_demo(tmp)
        final_report = _mission_reports_dir(tmp) / "final_report_v2.md"
        original = final_report.read_text(encoding="utf-8")
        final_report.write_text(original + "\nThis is a confirmed survivor.\nHuman review required.\n", encoding="utf-8")
        response = run_evidence_audit(MISSION_ID, "final_report_v2", root_dir=tmp)
        assert response["result"]["audit_status"] in {"warning", "fail"}
        assert _has_issue(response, severity="high", contains="confirmed survivor")


def test_measured_temperature_with_simulated_thermal_is_flagged():
    with tempfile.TemporaryDirectory() as tmp:
        _run_staged_demo(tmp)
        final_report = _mission_reports_dir(tmp) / "final_report_v2.md"
        final_report.write_text("The report claims measured temperature from the thermal result. Human review required.", encoding="utf-8")
        response = run_evidence_audit(MISSION_ID, "final_report_v2", root_dir=tmp)
        assert _has_issue(response, category="authenticity_boundary", severity="high", contains="simulated thermal")


def test_gps_route_with_image_plane_path_is_flagged():
    with tempfile.TemporaryDirectory() as tmp:
        _run_staged_demo(tmp)
        final_report = _mission_reports_dir(tmp) / "final_report_v2.md"
        final_report.write_text("The path planning output provides a GPS route. Human review required.", encoding="utf-8")
        response = run_evidence_audit(MISSION_ID, "final_report_v2", root_dir=tmp)
        assert _has_issue(response, category="authenticity_boundary", severity="high", contains="image-plane")


def test_real_odm_orthomosaic_with_fast_preview_is_flagged():
    with tempfile.TemporaryDirectory() as tmp:
        _run_staged_demo(tmp)
        mission_result_path = Path(tmp) / "outputs" / "missions" / MISSION_ID / "outputs" / "mission_result.json"
        mission_result = json.loads(mission_result_path.read_text(encoding="utf-8"))
        mission_result["orthomosaic"] = {"mode": "fast_preview"}
        mission_result_path.write_text(json.dumps(mission_result, ensure_ascii=False, indent=2), encoding="utf-8")
        final_report = _mission_reports_dir(tmp) / "final_report_v2.md"
        final_report.write_text("The map is a real ODM orthomosaic. Human review required.", encoding="utf-8")
        response = run_evidence_audit(MISSION_ID, "final_report_v2", root_dir=tmp)
        assert _has_issue(response, category="authenticity_boundary", severity="high", contains="Fast Preview")


def test_missing_detection_event_is_flagged_when_report_references_detection():
    with tempfile.TemporaryDirectory() as tmp:
        _run_staged_demo(tmp)
        ledger_path = _mission_reports_dir(tmp) / "evidence_ledger.json"
        ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
        ledger["events"] = [event for event in ledger["events"] if event.get("event_type") != "detection_completed"]
        ledger_path.write_text(json.dumps(ledger, ensure_ascii=False, indent=2), encoding="utf-8")
        final_report = _mission_reports_dir(tmp) / "final_report_v2.md"
        final_report.write_text("This final report references detection evidence. Human review required.", encoding="utf-8")
        response = run_evidence_audit(MISSION_ID, "all", root_dir=tmp)
        assert _has_issue(response, category="missing_evidence", severity="medium", contains="detection_completed")


def test_llm_report_missing_human_review_is_flagged():
    with tempfile.TemporaryDirectory() as tmp:
        _run_staged_demo(tmp)
        report_path = _mission_reports_dir(tmp) / "llm_mission_report.json"
        report = json.loads(report_path.read_text(encoding="utf-8"))
        report.pop("human_review_required", None)
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        response = run_evidence_audit(MISSION_ID, "llm_report", root_dir=tmp)
        assert _has_issue(response, category="missing_human_review", severity="medium")


def test_audit_event_and_no_secret_leak():
    with tempfile.TemporaryDirectory() as tmp:
        _run_staged_demo(tmp)
        os.environ["OPENAI_API_KEY"] = "sk-test-secret-should-not-appear"
        os.environ["LLM_ENABLE"] = "false"
        response = run_evidence_audit(MISSION_ID, "all", root_dir=tmp)
        text = json.dumps(response, ensure_ascii=False)
        assert "sk-test-secret-should-not-appear" not in text
        assert response["result"]["human_review_required"] is True
        assert all(issue.get("auto_fix_allowed") is False for issue in _issues(response))

        ledger_path = _mission_reports_dir(tmp) / "mission_evidence_ledger.json"
        ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
        events = ledger.get("events", [])
        audit_events = [event for event in events if event.get("event_type") == "llm_evidence_audit_completed"]
        assert audit_events
        assert audit_events[-1]["human_review_required"] is True


def main():
    test_normal_demo_audit_runs_with_mock_provider()
    test_confirmed_survivor_in_final_report_is_high_severity()
    test_measured_temperature_with_simulated_thermal_is_flagged()
    test_gps_route_with_image_plane_path_is_flagged()
    test_real_odm_orthomosaic_with_fast_preview_is_flagged()
    test_missing_detection_event_is_flagged_when_report_references_detection()
    test_llm_report_missing_human_review_is_flagged()
    test_audit_event_and_no_secret_leak()
    print("灾情感知及影响评估 LLM evidence auditor test suite passed.")


if __name__ == "__main__":
    main()
