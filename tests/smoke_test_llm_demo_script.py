import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
MISSION_ID = "urban_rescue_llm_demo"
UNSAFE_OUTPUT_PHRASES = [
    "confirmed survivor",
    "confirmed civilian",
    "measured temperature",
    "real GPS route",
    "real rescue conclusion",
]


def _assert_no_unsafe(text):
    lower = text.lower()
    for phrase in UNSAFE_OUTPUT_PHRASES:
        assert phrase.lower() not in lower, phrase


def main():
    demo_dir = ROOT_DIR / "demo_missions" / MISSION_ID
    for name in [
        "mission_result.json",
        "detection_result.json",
        "segmentation_result.json",
        "thermal_result.json",
        "path_planning_result.json",
        "ec_terp_result.json",
        "evidence_ledger.json",
        "expected_llm_report_mock.json",
        "README.md",
    ]:
        assert (demo_dir / name).exists(), name

    mission_result = json.loads((demo_dir / "mission_result.json").read_text(encoding="utf-8"))
    assert mission_result["mission_id"] == MISSION_ID

    with tempfile.TemporaryDirectory() as tmp:
        env = dict(os.environ)
        env["LLM_ENABLE"] = "false"
        env.pop("OPENAI_API_KEY", None)
        result = subprocess.run(
            [
                sys.executable,
                str(ROOT_DIR / "scripts" / "run_llm_demo.py"),
                "--root-dir",
                tmp,
            ],
            cwd=str(ROOT_DIR),
            env=env,
            text=True,
            capture_output=True,
            check=True,
        )
        stdout = result.stdout
        assert "Using MockProvider because LLM API is disabled or unavailable." in stdout
        assert "Safety checks passed." in stdout
        assert "Demo completed." in stdout
        _assert_no_unsafe(stdout)

        reports_dir = Path(tmp) / "outputs" / "missions" / MISSION_ID / "outputs" / "reports"
        llm_report_path = reports_dir / "llm_mission_report.json"
        final_report_path = reports_dir / "final_report_v2.md"
        evidence_ledger_path = reports_dir / "evidence_ledger.json"
        copilot_path = reports_dir / "mission_copilot_answers.json"
        planner_path = reports_dir / "mission_planner_result.json"
        audit_path = reports_dir / "llm_evidence_audit.json"
        summary_path = reports_dir / "llm_demo_summary.json"
        for path in [llm_report_path, final_report_path, evidence_ledger_path, copilot_path, planner_path, audit_path, summary_path]:
            assert path.exists(), path

        llm_report = json.loads(llm_report_path.read_text(encoding="utf-8"))
        assert llm_report["human_review_required"] is True
        assert llm_report["provider"] == "mock"

        evidence_ledger = json.loads(evidence_ledger_path.read_text(encoding="utf-8"))
        events = evidence_ledger["events"]
        assert any(event.get("event_type") == "llm_report_generated" for event in events)
        report_event = [event for event in events if event.get("event_type") == "llm_report_generated"][-1]
        assert report_event["human_review_required"] is True
        planner_events = [event for event in events if event.get("event_type") == "llm_mission_planner_executed"]
        assert planner_events
        assert planner_events[-1]["human_review_required"] is True
        assert planner_events[-1]["source_module"] == "LLM Tool-Orchestrated Mission Planner"
        audit_events = [event for event in events if event.get("event_type") == "llm_evidence_audit_completed"]
        assert audit_events
        assert audit_events[-1]["human_review_required"] is True
        assert audit_events[-1]["source_module"] == "LLM Evidence Auditor"

        final_report = final_report_path.read_text(encoding="utf-8")
        assert "LLM Mission Report Assistant" in final_report
        assert "人工复核" in final_report or "human review" in final_report.lower()

        copilot_answers = json.loads(copilot_path.read_text(encoding="utf-8"))
        assert len(copilot_answers) >= 5
        copilot_text = json.dumps(copilot_answers, ensure_ascii=False)
        assert "evidence is insufficient" in copilot_text.lower()
        assert "image-plane" in copilot_text.lower()

        planner_result = json.loads(planner_path.read_text(encoding="utf-8"))
        assert planner_result["ok"] is True
        assert planner_result["provider"] == "mock"
        assert planner_result["result"]["human_review_required"] is True
        assert planner_result["result"]["tool_plan"]
        assert planner_result["result"]["executed_tools"]

        audit_result = json.loads(audit_path.read_text(encoding="utf-8"))
        assert audit_result["provider"] == "mock"
        assert audit_result["human_review_required"] is True
        assert audit_result["audit_status"] in {"pass", "warning"}

        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        assert summary["mission_id"] == MISSION_ID
        assert summary["llm_provider"] == "mock"
        assert summary["fallback_used"] is True
        assert summary["evidence_ledger_updated"] is True
        assert summary["safety_checks_passed"] is True
        assert Path(summary["mission_planner_result_path"]).exists()
        assert Path(summary["evidence_audit_result_path"]).exists()
        assert summary["evidence_audit_status"] in {"pass", "warning"}

        _assert_no_unsafe(json.dumps(llm_report, ensure_ascii=False))
        _assert_no_unsafe(final_report)
        _assert_no_unsafe(copilot_text)
        _assert_no_unsafe(json.dumps(planner_result, ensure_ascii=False))
        _assert_no_unsafe(json.dumps(audit_result, ensure_ascii=False))
        _assert_no_unsafe(json.dumps(summary, ensure_ascii=False))

    print("AeroRescue-AI one-click LLM demo script smoke test passed.")


if __name__ == "__main__":
    main()
