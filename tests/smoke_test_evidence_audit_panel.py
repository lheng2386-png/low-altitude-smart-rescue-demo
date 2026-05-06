import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
APP_DIR = ROOT_DIR / "app"
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

import evidence_audit_panel  # noqa: E402


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
    original_post = evidence_audit_panel.requests.post
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
                    "fallback_used": False,
                    "result": {
                        "audit_status": "pass",
                        "overall_risk_level": "low",
                        "issues": [],
                        "safe_rewrite_suggestions": [],
                        "missing_evidence": [],
                        "positive_checks": ["No configured unsafe phrase was found."],
                        "human_review_required": True,
                    },
                }
            )

        evidence_audit_panel.requests.post = fake_post
        outputs = evidence_audit_panel.run_evidence_audit_panel("urban_rescue_llm_demo", "all")
        assert captured["url"].endswith("/api/llm/evidence-audit")
        assert captured["json"]["mission_id"] == "urban_rescue_llm_demo"
        assert captured["json"]["audit_target"] == "all"
        assert outputs[0] == "Evidence audit completed."
        assert "Audit status: pass" in outputs[1]
        assert "Provider: mock" in outputs[2]
        assert "No issues found" in outputs[3]

        def failing_post(url, json, timeout):
            raise RuntimeError("backend unavailable")

        evidence_audit_panel.requests.post = failing_post
        failed_outputs = evidence_audit_panel.run_evidence_audit_panel("demo", "all")
        assert failed_outputs[0] == "Evidence audit unavailable. Please check backend or API key."
    finally:
        evidence_audit_panel.requests.post = original_post

    print("灾情感知及影响评估 evidence audit panel smoke test passed.")


if __name__ == "__main__":
    main()
