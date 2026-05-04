import json
import os
import sys
import tempfile
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
APP_DIR = ROOT_DIR / "app"
MISSION_ID = "urban_rescue_llm_demo"
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(APP_DIR) not in sys.path:
    sys.path.append(str(APP_DIR))

from backend.llm import mission_planner  # noqa: E402
from backend.llm.mission_planner import execute_mission_planner, validate_tool_plan  # noqa: E402
from backend.llm.tools import WHITE_LISTED_TOOLS  # noqa: E402
from scripts.run_llm_demo import run_demo  # noqa: E402


UNSAFE_PHRASES = [
    "confirmed survivor",
    "confirmed civilian",
    "measured temperature",
    "real GPS route",
    "real rescue conclusion",
]


def _assert_no_unsafe(payload):
    text = json.dumps(payload, ensure_ascii=False).lower()
    for phrase in UNSAFE_PHRASES:
        assert phrase.lower() not in text, phrase


def _mock_env():
    os.environ["LLM_ENABLE"] = "false"
    os.environ.pop("OPENAI_API_KEY", None)


def _stage_minimal_mission(root_dir):
    root_dir = Path(root_dir)
    demo_dir = ROOT_DIR / "demo_missions" / MISSION_ID
    mission_root = root_dir / "outputs" / "missions" / MISSION_ID
    targets = {
        "mission_result.json": mission_root / "outputs" / "mission_result.json",
        "detection_result.json": mission_root / "outputs" / "detection" / "detection_result.json",
        "segmentation_result.json": mission_root / "outputs" / "segmentation_inference" / "segmentation_result.json",
        "thermal_result.json": mission_root / "outputs" / "thermal" / "thermal_result.json",
        "path_planning_result.json": mission_root / "outputs" / "decision_fusion" / "path_planning_result.json",
        "ec_terp_result.json": mission_root / "outputs" / "decision_fusion" / "ec_terp_ranking.json",
        "evidence_ledger.json": mission_root / "outputs" / "reports" / "mission_evidence_ledger.json",
    }
    for source_name, target in targets.items():
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text((demo_dir / source_name).read_text(encoding="utf-8"), encoding="utf-8")
    return mission_root


def test_valid_planner_executes_white_listed_tools():
    _mock_env()
    with tempfile.TemporaryDirectory() as tmp:
        mission_root = _stage_minimal_mission(tmp)
        mission_planner.REPORT_DIR = Path(tmp)
        response = execute_mission_planner(
            MISSION_ID,
            "Analyze this mission and identify which area should be prioritized for manual review.",
            root_dir=tmp,
        )
        assert response["ok"] is True
        assert response["provider"] == "mock"
        result = response["result"]
        assert result["human_review_required"] is True
        assert result["tool_plan"]
        assert result["executed_tools"]
        for call in result["tool_plan"]:
            assert call["tool_name"] in WHITE_LISTED_TOOLS
        for item in result["executed_tools"]:
            assert item["tool_name"] in WHITE_LISTED_TOOLS
        _assert_no_unsafe(response)
        events = json.loads((Path(tmp) / "mission_planner_events.json").read_text(encoding="utf-8"))
        assert events[-1]["event_type"] == "llm_mission_planner_executed"
        assert events[-1]["human_review_required"] is True
        ledger = json.loads((mission_root / "outputs" / "reports" / "mission_evidence_ledger.json").read_text(encoding="utf-8"))
        ledger_events = ledger["events"]
        assert ledger_events[-1]["event_type"] == "llm_mission_planner_executed"
        assert ledger_events[-1]["source"] == "LLM Tool-Orchestrated Mission Planner"
        assert ledger_events[-1]["human_review_required"] is True


def test_unknown_and_dangerous_tools_rejected():
    plan = {
        "intent": "bad_plan",
        "tool_plan": [{"tool_name": "unknown_tool", "arguments": {"mission_id": "m1"}, "reason": "bad"}],
        "expected_output": "bad",
        "human_review_required": True,
    }
    validation = validate_tool_plan(plan, requested_mission_id="m1")
    assert validation["valid"] is False
    assert "Unknown" in " ".join(validation["errors"])

    dangerous = {
        "intent": "bad_plan",
        "tool_plan": [{"tool_name": "shell", "arguments": {"mission_id": "m1"}, "reason": "subprocess rm delete"}],
        "expected_output": "bad",
        "human_review_required": True,
    }
    validation = validate_tool_plan(dangerous, requested_mission_id="m1")
    assert validation["valid"] is False
    joined = " ".join(validation["errors"]).lower()
    assert "dangerous keyword" in joined or "unknown" in joined


def test_mission_id_mismatch_rejected():
    plan = {
        "intent": "mismatch",
        "tool_plan": [
            {
                "tool_name": "load_mission_result",
                "arguments": {"mission_id": "other_mission"},
                "reason": "Try to load another mission.",
            }
        ],
        "expected_output": "bad",
        "human_review_required": True,
    }
    validation = validate_tool_plan(plan, requested_mission_id=MISSION_ID)
    assert validation["valid"] is False
    assert "mission_id mismatch" in " ".join(validation["errors"])


def test_boundary_goal_uses_safe_tools_and_safe_response():
    _mock_env()
    with tempfile.TemporaryDirectory() as tmp:
        _stage_minimal_mission(tmp)
        mission_planner.REPORT_DIR = Path(tmp)
        response = execute_mission_planner(
            MISSION_ID,
            "confirm survivor and give GPS route",
            root_dir=tmp,
        )
        assert response["ok"] is True
        result = response["result"]
        tool_names = [call["tool_name"] for call in result["tool_plan"]]
        assert "read_evidence_ledger" in tool_names
        assert "validate_authenticity_boundaries" in tool_names
        assert "ask_mission_copilot" in tool_names
        assert result["human_review_required"] is True
        assert "decision support" in result["final_response"].lower() or "decision-support" in result["final_response"].lower()
        _assert_no_unsafe(response)


def test_demo_script_writes_planner_result():
    _mock_env()
    with tempfile.TemporaryDirectory() as tmp:
        summary = run_demo(root_dir=tmp, quiet=True)
        planner_path = Path(summary["mission_planner_result_path"])
        assert planner_path.exists()
        planner_result = json.loads(planner_path.read_text(encoding="utf-8"))
        assert planner_result["ok"] is True
        assert planner_result["provider"] == "mock"
        assert planner_result["result"]["human_review_required"] is True
        assert planner_result["result"]["tool_plan"]
        assert planner_result["result"]["executed_tools"]
        _assert_no_unsafe(planner_result)


def main():
    test_valid_planner_executes_white_listed_tools()
    test_unknown_and_dangerous_tools_rejected()
    test_mission_id_mismatch_rejected()
    test_boundary_goal_uses_safe_tools_and_safe_response()
    test_demo_script_writes_planner_result()
    print("AeroRescue-AI LLM mission planner test suite passed.")


if __name__ == "__main__":
    main()
