import os
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
APP_DIR = ROOT_DIR / "app"
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(APP_DIR) not in sys.path:
    sys.path.append(str(APP_DIR))

from backend.llm import report_assistant  # noqa: E402
from backend.llm.report_assistant import generate_mission_report  # noqa: E402


def main():
    old_env = {key: os.environ.get(key) for key in ["LLM_ENABLE", "OPENAI_API_KEY", "LLM_PROVIDER"]}
    try:
        os.environ["LLM_ENABLE"] = "false"
        os.environ.pop("OPENAI_API_KEY", None)
        report = generate_mission_report(
            {
                "target_count": 1,
                "thermal_mode": "simulated",
                "truthfulness_note": "Simulated thermal, not real temperature.",
                "targets": [{"id": "TR001", "class_name": "human_candidate", "confidence": 0.82}],
                "path": {"mode": "image-plane reference path"},
                "segmentation_source": {"source_type": "uploaded_mask"},
            }
        )
        assert report["provider"] == "mock"
        assert report["fallback_used"] is False
        assert report["mission_summary"]
        assert report["human_review_required"] is True
        assert "simulated thermal" in " ".join(report["limitations"]).lower()
        assert "not a real temperature measurement" in " ".join(report["limitations"]).lower()
        assert "not a gps navigation route" in " ".join(report["limitations"]).lower()
        joined = " ".join(
            [
                report["mission_summary"],
                report["risk_interpretation"],
                report["report_paragraph"],
                " ".join(report["limitations"]),
                " ".join(report["recommended_next_actions"]),
            ]
        ).lower()
        assert "confirmed civilian" not in joined

        os.environ["LLM_ENABLE"] = "true"
        os.environ["LLM_PROVIDER"] = "openai"
        os.environ["OPENAI_API_KEY"] = "test-key"
        original_generate = report_assistant.OpenAIProvider.generate_mission_report

        def _raise_api_error(self, mission_result):
            raise RuntimeError("simulated API failure")

        report_assistant.OpenAIProvider.generate_mission_report = _raise_api_error
        fallback_report = generate_mission_report({"target_count": 0})
        assert fallback_report["provider"] == "mock"
        assert fallback_report["fallback_used"] is True
        assert "simulated API failure" in fallback_report["fallback_reason"]
        report_assistant.OpenAIProvider.generate_mission_report = original_generate
    finally:
        if "original_generate" in locals():
            report_assistant.OpenAIProvider.generate_mission_report = original_generate
        for key, value in old_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    print("AeroRescue-AI LLM mission report assistant smoke test passed.")


if __name__ == "__main__":
    main()
