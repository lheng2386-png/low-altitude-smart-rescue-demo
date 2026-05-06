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

from backend.llm.evidence_context import build_mission_evidence_context  # noqa: E402
from backend.llm import mission_copilot  # noqa: E402
from backend.llm.guardrails import apply_copilot_guardrails  # noqa: E402


FORBIDDEN = [
    "confirmed civilian",
    "confirmed survivor",
    "measured temperature",
    "real GPS route",
]


def _write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _assert_safe(response):
    assert response["ok"] is True
    result = response["result"]
    assert result["human_review_required"] is True
    assert result["evidence_used"]
    text = json.dumps(response, ensure_ascii=False).lower()
    for term in FORBIDDEN:
        assert term.lower() not in text


def main():
    old_env = {key: os.environ.get(key) for key in ["LLM_ENABLE", "OPENAI_API_KEY", "LLM_PROVIDER"]}
    try:
        os.environ["LLM_ENABLE"] = "false"
        os.environ.pop("OPENAI_API_KEY", None)
        with tempfile.TemporaryDirectory() as events_tmp:
            mission_copilot.REPORT_DIR = Path(events_tmp)

            response = mission_copilot.answer_mission_copilot_question("missing_mission_id", "Why is this area high priority?")
            _assert_safe(response)
            assert "unavailable" in json.dumps(response, ensure_ascii=False).lower()

            survivor = mission_copilot.answer_mission_copilot_question("missing_mission_id", "Is this a confirmed survivor?")
            _assert_safe(survivor)
            assert "insufficient" in survivor["result"]["answer"].lower()

            temperature = mission_copilot.answer_mission_copilot_question("missing_mission_id", "What is the real temperature?")
            _assert_safe(temperature)
            assert "temperature" in temperature["result"]["answer"].lower()

            gps = mission_copilot.answer_mission_copilot_question("missing_mission_id", "Give me GPS route")
            _assert_safe(gps)
            assert "image-plane" in gps["result"]["answer"].lower()

        guarded = apply_copilot_guardrails(
            {
                "answer": "This is a confirmed civilian with measured temperature and real GPS route.",
                "evidence_used": [],
                "limitations": [],
                "human_review_required": False,
                "confidence_note": "",
            },
            evidence_context={"limitations": []},
        )
        assert guarded["human_review_required"] is True
        assert guarded["evidence_used"]
        guarded_text = json.dumps(guarded, ensure_ascii=False).lower()
        for term in FORBIDDEN:
            assert term.lower() not in guarded_text

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_json(
                root / "outputs" / "detection" / "detection_result.json",
                {"success": True, "targets": [{"class_name": "human_candidate"}]},
            )
            context = build_mission_evidence_context("current_mission", root_dir=root)
            assert context["evidence"]["detection_result"]["status"] == "available"
            assert context["evidence"]["thermal_result"]["status"] == "unavailable"
    finally:
        for key, value in old_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    print("灾情感知及影响评估 mission copilot smoke test passed.")


if __name__ == "__main__":
    main()
