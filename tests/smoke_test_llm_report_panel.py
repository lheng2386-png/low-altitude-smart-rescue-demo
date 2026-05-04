import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
APP_DIR = ROOT_DIR / "app"
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

import llm_report_panel  # noqa: E402


class _FakeResponse:
    def __init__(self, payload, status_code=200):
        self.payload = payload
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError("HTTP failure")

    def json(self):
        return self.payload


def main():
    original_post = llm_report_panel.requests.post
    try:
        captured = {}

        def fake_post(url, json, timeout):
            captured["url"] = url
            captured["json"] = json
            return _FakeResponse(
                {
                    "provider": "mock",
                    "fallback_used": True,
                    "fallback_reason": "No API key.",
                    "mission_summary": "Mission summary draft.",
                    "risk_interpretation": "Risk interpretation draft.",
                    "human_review_required": True,
                    "limitations": [
                        "human_candidate is an auxiliary candidate and must not be treated as confirmed civilian without human review.",
                        "Simulated thermal output is not a real temperature measurement.",
                    ],
                    "recommended_next_actions": ["Review original imagery."],
                    "report_paragraph": "Auxiliary report paragraph.",
                }
            )

        llm_report_panel.requests.post = fake_post
        outputs = llm_report_panel.request_llm_mission_report(
            "Rescue report text with human_candidate and image-plane path.",
            "Transformer summary",
            "Uploaded Mask status",
            "Scene gate",
            "Damage summary",
            "Scene mode",
            "Rescue entry",
            "Path gate",
            "Path reliability",
            [["T001", "human_candidate", 0.8]],
            [],
            [],
            [],
            "image-plane reference path",
            "Path comparison",
        )
        assert captured["url"].endswith("/api/llm/mission-report")
        assert "mission_result" in captured["json"]
        assert outputs[0] == "AI mission report generated."
        assert "Provider: mock" in outputs[1]
        assert outputs[2] == "Human Review Required"
        assert "Mock/fallback report is displayed" in outputs[5]
        assert "not a real temperature measurement" in outputs[5]
        joined = " ".join(str(item).lower() for item in outputs)
        assert "confirmed survivor" not in joined
        assert "real rescue conclusion" not in joined
        assert "gps route" not in joined

        def failing_post(url, json, timeout):
            raise RuntimeError("backend unavailable")

        llm_report_panel.requests.post = failing_post
        failed_outputs = llm_report_panel.request_llm_mission_report("Rescue report text", "", "", "", "", "", "", "", "", [], [], [], [], "", "")
        assert failed_outputs[0] == "LLM report unavailable. Please check backend or API key."
    finally:
        llm_report_panel.requests.post = original_post

    print("AeroRescue-AI LLM report panel smoke test passed.")


if __name__ == "__main__":
    main()
