import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
APP_DIR = ROOT_DIR / "app"
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

import mission_planner_panel  # noqa: E402


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
    original_post = mission_planner_panel.requests.post
    try:
        captured = {}

        def fake_post(url, json, timeout):
            captured["url"] = url
            captured["json"] = json
            return _FakeResponse(
                {
                    "ok": True,
                    "provider": "mock",
                    "model": "mock-template",
                    "fallback_used": True,
                    "result": {
                        "intent": "manual_review_prioritization",
                        "tool_plan": [
                            {
                                "tool_name": "load_mission_result",
                                "arguments": {"mission_id": "urban_rescue_llm_demo"},
                                "reason": "Load mission structured results.",
                            }
                        ],
                        "executed_tools": [
                            {
                                "tool_name": "load_mission_result",
                                "status": "success",
                                "result_summary": "Loaded mission evidence summary.",
                                "error": None,
                            }
                        ],
                        "final_response": "Decision-support only; human review is required.",
                        "evidence_used": [{"source": "mission_result", "summary": "Mission context loaded."}],
                        "limitations": ["All outputs require human review."],
                        "human_review_required": True,
                    },
                }
            )

        mission_planner_panel.requests.post = fake_post
        outputs = mission_planner_panel.run_mission_planner(
            "urban_rescue_llm_demo",
            "Analyze this mission and identify priority for manual review.",
        )
        assert captured["url"].endswith("/api/llm/mission-planner")
        assert captured["json"]["mission_id"] == "urban_rescue_llm_demo"
        assert "Mission planner completed" in outputs[0]
        assert "Provider: mock" in outputs[1]
        assert "load_mission_result" in outputs[2]
        assert "Decision-support only" in outputs[3]

        def failing_post(url, json, timeout):
            raise RuntimeError("backend unavailable")

        mission_planner_panel.requests.post = failing_post
        failed_outputs = mission_planner_panel.run_mission_planner("demo", "question")
        assert failed_outputs[0] == "Mission planner unavailable. Please check backend or API key."
    finally:
        mission_planner_panel.requests.post = original_post

    print("AeroRescue-AI mission planner panel smoke test passed.")


if __name__ == "__main__":
    main()
