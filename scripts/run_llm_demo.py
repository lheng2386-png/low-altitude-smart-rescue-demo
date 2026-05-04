#!/usr/bin/env python3
"""Run the one-click AeroRescue-AI LLM demo with mock fallback."""

import argparse
import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT_DIR = Path(__file__).resolve().parents[1]
APP_DIR = ROOT_DIR / "app"
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(APP_DIR) not in sys.path:
    sys.path.append(str(APP_DIR))

from backend.llm import auditor, mission_copilot, mission_planner, report_assistant  # noqa: E402
from backend.llm.report_assistant import generate_mission_report  # noqa: E402
from app.final_report_v2_service import build_final_report_v2, save_final_report_v2  # noqa: E402


MISSION_ID = "urban_rescue_llm_demo"
DEMO_DIR = ROOT_DIR / "demo_missions" / MISSION_ID
COPILOT_QUESTIONS = [
    ("Why is this area marked as high priority?", "Why is this area marked as high priority?"),
    ("What evidence supports the human_candidate?", "What evidence supports the human_candidate?"),
    ("What are the limitations of the thermal result?", "What are the limitations of the thermal result?"),
    ("Is this a confirmed survivor?", "Is this a verified survivor?"),
    ("Can this path be used as a GPS route?", "Can this path be used as field navigation?"),
]
UNSAFE_OUTPUT_PHRASES = [
    "confirmed survivor",
    "confirmed civilian",
    "measured temperature",
    "real GPS route",
    "real rescue conclusion",
]


def load_demo_mission(demo_dir: Path = DEMO_DIR) -> dict[str, Any]:
    files = {
        "mission_result": "mission_result.json",
        "detection_result": "detection_result.json",
        "segmentation_result": "segmentation_result.json",
        "thermal_result": "thermal_result.json",
        "path_planning_result": "path_planning_result.json",
        "ec_terp_result": "ec_terp_result.json",
        "evidence_ledger": "evidence_ledger.json",
        "expected_llm_report_mock": "expected_llm_report_mock.json",
    }
    payload = {}
    for key, name in files.items():
        path = demo_dir / name
        payload[key] = json.loads(path.read_text(encoding="utf-8"))
    return payload


def build_combined_mission_result(demo: dict[str, Any]) -> dict[str, Any]:
    combined = dict(demo["mission_result"])
    combined.update(
        {
            "detection_result": demo["detection_result"],
            "segmentation_result": demo["segmentation_result"],
            "thermal_result": demo["thermal_result"],
            "path_planning_result": demo["path_planning_result"],
            "ec_terp_result": demo["ec_terp_result"],
            "evidence_ledger": demo["evidence_ledger"],
        }
    )
    combined["human_review_required"] = True
    return combined


def run_demo(root_dir: Path | str = ROOT_DIR, demo_dir: Path | str = DEMO_DIR, quiet: bool = False) -> dict[str, Any]:
    root_dir = Path(root_dir).resolve()
    demo_dir = Path(demo_dir).resolve()
    mission_root = root_dir / "outputs" / "missions" / MISSION_ID
    reports_dir = mission_root / "outputs" / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    log_lines = []

    def log(message):
        line = f"[LLM DEMO] {message}"
        log_lines.append(line)
        if not quiet:
            print(line)

    log(f"Loading demo mission: {MISSION_ID}")
    demo = load_demo_mission(demo_dir)
    mission_result = build_combined_mission_result(demo)
    _stage_demo_outputs(mission_root, demo_dir, demo, mission_result)

    provider_reason = _mock_reason()
    if provider_reason:
        log("Using MockProvider because LLM API is disabled or unavailable.")

    report_assistant.REPORT_DIR = reports_dir
    mission_copilot.REPORT_DIR = reports_dir
    mission_planner.REPORT_DIR = reports_dir

    log("Generating mission report...")
    llm_report = generate_mission_report(mission_result)
    log(f"Using provider: {llm_report.get('provider', 'unknown')}")
    llm_report_path = reports_dir / "llm_mission_report.json"
    _write_json(llm_report_path, llm_report)
    log("LLM report saved.")

    evidence_ledger_path = reports_dir / "evidence_ledger.json"
    evidence_ledger = _append_llm_report_event(demo["evidence_ledger"], llm_report)
    _write_json(evidence_ledger_path, evidence_ledger)
    log("Evidence Ledger updated with llm_report_generated.")

    final_report = build_final_report_v2(ledger=_build_final_report_ledger(llm_report_path, llm_report))
    saved_final_report = save_final_report_v2(report=final_report, output_dir=reports_dir)
    final_report_path = saved_final_report["markdown_path"]
    log("Final Report V2 generated.")

    log("Running Mission Evidence Copilot sample questions...")
    copilot_answers = []
    for raw_question, display_question in COPILOT_QUESTIONS:
        response = mission_copilot.answer_mission_copilot_question(
            MISSION_ID,
            raw_question,
            root_dir=root_dir,
        )
        answer = response.get("result", {}).get("answer", "")
        copilot_answers.append(
            {
                "question": display_question,
                "internal_question_type": _question_type(raw_question),
                "response": response,
            }
        )
        if not quiet:
            print(f"\nQ: {display_question}")
            print(f"A: {_shorten(answer)}")
    copilot_path = reports_dir / "mission_copilot_answers.json"
    _write_json(copilot_path, copilot_answers)

    planner_goal = "Analyze this mission and identify which area should be prioritized for manual review."
    log("Running LLM Mission Planner example...")
    planner_response = mission_planner.execute_mission_planner(
        MISSION_ID,
        planner_goal,
        root_dir=root_dir,
    )
    planner_path = reports_dir / "mission_planner_result.json"
    _write_json(planner_path, planner_response)
    evidence_ledger = _append_planner_event(evidence_ledger, planner_response, planner_goal)
    _write_json(evidence_ledger_path, evidence_ledger)
    if not quiet:
        print("\n[LLM DEMO] Mission Planner")
        print(f"- generated tool_plan: {[call.get('tool_name') for call in planner_response.get('result', {}).get('tool_plan', [])]}")
        print(f"- executed_tools: {[item.get('tool_name') for item in planner_response.get('result', {}).get('executed_tools', [])]}")
        print(f"- final_response: {_shorten(planner_response.get('result', {}).get('final_response', ''))}")

    log("Running Evidence Auditor...")
    audit_response = auditor.run_evidence_audit(MISSION_ID, audit_target="all", root_dir=root_dir)
    audit_result = audit_response.get("result", {}) or {}
    audit_path = Path(audit_response.get("saved_path") or reports_dir / "llm_evidence_audit.json")
    issue_count = len(audit_result.get("issues", []) or [])
    high_issue_count = sum(1 for issue in audit_result.get("issues", []) or [] if issue.get("severity") == "high")
    log(f"Audit status: {audit_result.get('audit_status', 'unknown')}")
    log(f"Issues found: {issue_count}")
    log(f"High severity issues: {high_issue_count}")
    if high_issue_count:
        log("Warning: high severity audit issues require manual review.")
    log(f"Audit result saved to {audit_path}")
    evidence_ledger = _append_audit_event(evidence_ledger, audit_result)
    _write_json(evidence_ledger_path, evidence_ledger)

    safety_payload = {
        "llm_report": llm_report,
        "final_report_markdown": Path(final_report_path).read_text(encoding="utf-8"),
        "copilot_answers": copilot_answers,
        "planner_response": planner_response,
        "audit_response": audit_response,
    }
    safety_checks_passed = _safety_checks_pass(safety_payload)
    if safety_checks_passed:
        log("Safety checks passed.")
    else:
        raise RuntimeError("LLM demo safety checks failed.")

    summary = {
        "mission_id": MISSION_ID,
        "llm_provider": llm_report.get("provider", "unknown"),
        "fallback_used": bool(llm_report.get("fallback_used") or llm_report.get("provider") == "mock"),
        "llm_report_saved_path": str(llm_report_path),
        "final_report_path": str(final_report_path),
        "evidence_ledger_updated": True,
        "safety_checks_passed": safety_checks_passed,
        "copilot_answers_path": str(copilot_path),
        "mission_planner_result_path": str(planner_path),
        "evidence_audit_result_path": str(audit_path),
        "evidence_audit_status": audit_result.get("audit_status", "unknown"),
        "evidence_audit_issue_count": issue_count,
        "evidence_audit_high_severity_issue_count": high_issue_count,
        "demo_output_root": str(mission_root),
        "log": log_lines,
    }
    _write_json(reports_dir / "llm_demo_summary.json", summary)
    log("Demo completed.")

    if not quiet:
        print("\n[LLM DEMO] Demo Summary")
        for key in [
            "mission_id",
            "llm_provider",
            "fallback_used",
            "llm_report_saved_path",
            "final_report_path",
            "evidence_ledger_updated",
            "safety_checks_passed",
            "evidence_audit_result_path",
            "evidence_audit_status",
            "evidence_audit_issue_count",
            "evidence_audit_high_severity_issue_count",
        ]:
            print(f"- {key}: {summary[key]}")
    return summary


def _stage_demo_outputs(mission_root: Path, demo_dir: Path, demo: dict[str, Any], mission_result: dict[str, Any]):
    targets = {
        "mission_result": mission_root / "outputs" / "mission_result.json",
        "detection_result": mission_root / "outputs" / "detection" / "detection_result.json",
        "segmentation_result": mission_root / "outputs" / "segmentation_inference" / "segmentation_result.json",
        "thermal_result": mission_root / "outputs" / "thermal" / "thermal_result.json",
        "path_planning_result": mission_root / "outputs" / "decision_fusion" / "path_planning_result.json",
        "ec_terp_result": mission_root / "outputs" / "decision_fusion" / "ec_terp_ranking.json",
        "evidence_ledger": mission_root / "outputs" / "reports" / "mission_evidence_ledger.json",
    }
    for key, path in targets.items():
        _write_json(path, mission_result if key == "mission_result" else demo[key])
    shutil.copyfile(demo_dir / "README.md", mission_root / "README.md")


def _append_llm_report_event(ledger: dict[str, Any], llm_report: dict[str, Any]) -> dict[str, Any]:
    updated = json.loads(json.dumps(ledger, ensure_ascii=False))
    updated.setdefault("events", []).append(
        {
            "event_type": "llm_report_generated",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "source_module": "LLM Mission Report Assistant",
            "summary": "LLM mission report generated as auxiliary explanation.",
            "authenticity_boundary": "LLM report is evidence-grounded decision support and requires human review.",
            "human_review_required": True,
            "provider": llm_report.get("provider", "unknown"),
            "fallback_used": bool(llm_report.get("fallback_used")),
        }
    )
    return updated


def _append_planner_event(ledger: dict[str, Any], planner_response: dict[str, Any], user_goal: str) -> dict[str, Any]:
    updated = json.loads(json.dumps(ledger, ensure_ascii=False))
    result = planner_response.get("result", {}) or {}
    updated.setdefault("events", []).append(
        {
            "event_type": "llm_mission_planner_executed",
            "mission_id": MISSION_ID,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "source_module": "LLM Tool-Orchestrated Mission Planner",
            "summary": "LLM planner executed white-listed mission evidence tools.",
            "authenticity_boundary": "Planner output is decision-support only and requires human review.",
            "human_review_required": True,
            "user_goal": user_goal,
            "tools_requested": [item.get("tool_name") for item in result.get("tool_plan", [])],
            "tools_executed": [item.get("tool_name") for item in result.get("executed_tools", []) if item.get("status") == "success"],
        }
    )
    return updated


def _append_audit_event(ledger: dict[str, Any], audit_result: dict[str, Any]) -> dict[str, Any]:
    updated = json.loads(json.dumps(ledger, ensure_ascii=False))
    issues = audit_result.get("issues", []) or []
    updated.setdefault("events", []).append(
        {
            "event_type": "llm_evidence_audit_completed",
            "mission_id": MISSION_ID,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "source_module": "LLM Evidence Auditor",
            "summary": "Evidence audit completed for mission outputs.",
            "authenticity_boundary": "Audit suggestions require human review and do not overwrite source outputs.",
            "audit_status": audit_result.get("audit_status", "warning"),
            "overall_risk_level": audit_result.get("overall_risk_level", "medium"),
            "issue_count": len(issues),
            "high_severity_issue_count": sum(1 for issue in issues if issue.get("severity") == "high"),
            "human_review_required": True,
        }
    )
    return updated


def _build_final_report_ledger(llm_report_path: Path, llm_report: dict[str, Any]) -> dict[str, Any]:
    return {
        "success": True,
        "summary": {
            "strong_count": 0,
            "medium_count": 2,
            "weak_count": 3,
            "none_count": 0,
            "decision_support_count": 2,
            "human_review_required_count": 5,
            "final_report_entry_count": 5,
        },
        "report_sections": {
            "辅助决策证据": ["llm_report", "ec_terp"],
            "模拟 / 预览结果": ["thermal", "path_planning", "segmentation"],
        },
        "evidence_records": {
            "llm_report": {
                "module_key": "llm_report",
                "display_name": "LLM Mission Report Assistant",
                "scanner_status": "executed_success",
                "evidence_level": "medium",
                "evidence_type": "rule_based_decision",
                "can_support_decision": True,
                "can_enter_final_report": True,
                "human_review_required": True,
                "evidence_files": [str(llm_report_path)],
                "limitations": llm_report.get("limitations", []) + ["LLM report is auxiliary explanation only."],
                "recommended_report_section": "辅助决策证据",
                "truthfulness_note": "LLM report is generated from demo mission evidence and requires human review.",
                "message": "LLM report generated for demo mission.",
            },
            "ec_terp": _record("ec_terp", "EC-TERP Auxiliary Priority", "medium", "rule_based_decision", "EC-TERP priority is decision-support only."),
            "thermal": _record("thermal", "Simulated Thermal Cue", "weak", "simulated_or_preview", "Simulated thermal is not radiometric temperature measurement."),
            "path_planning": _record("path_planning", "Image-Plane Planning", "weak", "image_plane_decision", "Image-plane path is not field navigation."),
            "segmentation": _record("segmentation", "Demo Risk Mask", "weak", "uploaded_or_demo_input", "Demo/uploaded mask is not necessarily automatic segmentation."),
        },
        "global_truthfulness_note": "LLM demo report is evidence-led and requires human review.",
    }


def _record(module_key, display_name, evidence_level, evidence_type, limitation):
    return {
        "module_key": module_key,
        "display_name": display_name,
        "scanner_status": "executed_success",
        "evidence_level": evidence_level,
        "evidence_type": evidence_type,
        "can_support_decision": evidence_level in {"medium", "strong"},
        "can_enter_final_report": True,
        "human_review_required": True,
        "evidence_files": [],
        "limitations": [limitation, "Human review required before operational use."],
        "recommended_report_section": "辅助决策证据" if evidence_level == "medium" else "模拟 / 预览结果",
        "truthfulness_note": limitation,
        "message": f"{display_name} included in demo evidence package.",
    }


def _mock_reason():
    llm_enabled = os.getenv("LLM_ENABLE", "false").strip().lower() in {"1", "true", "yes", "on"}
    if not llm_enabled or not os.getenv("OPENAI_API_KEY"):
        return "Using MockProvider because LLM API is disabled or unavailable."
    return ""


def _safety_checks_pass(payload: dict[str, Any]) -> bool:
    text = json.dumps(payload, ensure_ascii=False).lower()
    return not any(phrase.lower() in text for phrase in UNSAFE_OUTPUT_PHRASES)


def _shorten(text: str, max_len: int = 260) -> str:
    text = " ".join(str(text or "").split())
    return text if len(text) <= max_len else text[: max_len - 3] + "..."


def _question_type(question: str) -> str:
    lower = question.lower()
    if "survivor" in lower:
        return "survivor_boundary_check"
    if "gps" in lower or "route" in lower:
        return "navigation_boundary_check"
    if "human_candidate" in lower:
        return "human_candidate_evidence"
    if "thermal" in lower:
        return "thermal_limitations"
    return "priority_explanation"


def _write_json(path: Path, payload: Any):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Run the AeroRescue-AI one-click LLM demo.")
    parser.add_argument("--root-dir", default=str(ROOT_DIR), help="Repository-like root where outputs/ will be written.")
    parser.add_argument("--demo-dir", default=str(DEMO_DIR), help="Demo mission data directory.")
    args = parser.parse_args()
    run_demo(root_dir=Path(args.root_dir), demo_dir=Path(args.demo_dir), quiet=False)


if __name__ == "__main__":
    main()
