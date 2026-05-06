import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
APP_DIR = ROOT_DIR / "app"
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

import mission_copilot_panel  # noqa: E402


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
    original_post = mission_copilot_panel.requests.post
    try:
        captured = {}

        def fake_post(url, json, timeout):
            captured["url"] = url
            captured["json"] = json
            return _FakeResponse(
                {
                    "ok": True,
                    "provider": "mock",
                    "model": "",
                    "fallback_used": True,
                    "result": {
                        "answer": "Evidence is insufficient for a stronger conclusion.",
                        "evidence_used": [{"source": "detection", "summary": "human_candidate requires review"}],
                        "limitations": ["All outputs require human review."],
                        "human_review_required": True,
                        "confidence_note": "Answer is based only on available mission evidence.",
                    },
                }
            )

        mission_copilot_panel.requests.post = fake_post
        outputs = mission_copilot_panel.ask_mission_copilot("demo_mission_001", "Why is this area high priority?")
        assert captured["url"].endswith("/api/llm/mission-copilot")
        assert captured["json"]["mission_id"] == "demo_mission_001"
        assert outputs[2] == "Human Review Required"
        assert "Evidence is insufficient" in outputs[3]
        assert "detection" in outputs[4]

        def failing_post(url, json, timeout):
            raise RuntimeError("backend unavailable")

        mission_copilot_panel.requests.post = failing_post
        failed_outputs = mission_copilot_panel.ask_mission_copilot("demo", "question")
        assert failed_outputs[0] == "Mission copilot unavailable. Please check backend or API key."
    finally:
        mission_copilot_panel.requests.post = original_post

    print("灾情感知及影响评估 mission copilot panel smoke test passed.")


if __name__ == "__main__":
    main()
