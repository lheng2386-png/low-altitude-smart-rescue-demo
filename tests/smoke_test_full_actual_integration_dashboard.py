import json
import sys
import tempfile
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
APP_DIR = ROOT_DIR / "app"
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from full_actual_integration_dashboard import (  # noqa: E402
    build_status_dashboard,
    classify_integration_state,
    save_status_dashboard,
    scan_full_actual_integration_status,
    write_external_integration_scaffold,
)


EXPECTED_KEYS = {
    "yolo_rescue_targets",
    "transformer_rescuedet_family",
    "qazi_disaster_management",
    "air_sar_detection",
    "vtsar_dataset",
    "python_robotics",
    "fields2cover",
    "sarenv",
    "skai",
    "inasafe",
}


def main():
    dashboard = build_status_dashboard()
    assert dashboard["module"] == "full_actual_integration_dashboard"
    assert dashboard["truthfulness_policy"]["human_review_required"] is True
    assert dashboard["truthfulness_policy"]["no_fake_gps_routes"] is True

    repositories = dashboard["repositories"]
    keys = {item["key"] for item in repositories}
    assert EXPECTED_KEYS.issubset(keys)
    assert len(repositories) >= 10

    for repo in repositories:
        assert repo.get("current_state")
        assert repo.get("target_final_state")
        assert repo.get("truthfulness_limitations")
        assert repo["current_state"] != "evaluated" or repo["is_evaluated"] is True
        if repo["current_state"] == "planned":
            assert repo["is_evaluated"] is False
            assert repo["is_executable_success"] is False
        if repo["current_state"] == "blocked_by_checkpoint":
            assert repo["is_executable_success"] is False
        joined = " ".join(repo.get("truthfulness_limitations", []))
        if repo["family"] == "planning":
            assert "GPS" in joined or "navigation" in joined or repo["key"] in {"fields2cover", "sarenv"}

    planned = classify_integration_state({"current_state": "planned"})
    assert planned["is_evaluated"] is False
    assert planned["is_executable_success"] is False

    blocked = classify_integration_state({"current_state": "blocked_by_checkpoint"})
    assert blocked["blocked"] is True
    assert blocked["is_executable_success"] is False

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        scaffold = write_external_integration_scaffold(root_dir=root)
        assert scaffold["success"] is True
        assert len(scaffold["directories"]) >= 10
        for relative in [
            "external_integrations/detection/yolo_rescue_targets/README.md",
            "external_integrations/detection/transformer_rescuedet/status.json",
            "external_integrations/planning/python_robotics/expected_io_schema.json",
            "external_integrations/damage_impact/inasafe/limitations.md",
        ]:
            assert (root / relative).exists()

        saved = save_status_dashboard(output_path=root / "outputs" / "full_actual_integration" / "status_dashboard.json")
        assert saved["success"] is True
        saved_path = Path(saved["dashboard_path"])
        assert saved_path.exists()
        payload = json.loads(saved_path.read_text(encoding="utf-8"))
        assert payload["truthfulness_policy"]["no_fake_full_integration_claims"] is True
        assert {item["key"] for item in payload["repositories"]}.issuperset(EXPECTED_KEYS)

    scan = scan_full_actual_integration_status()
    assert scan["success"] is True
    assert "planned" in scan["summary"]
    assert "does not treat" in scan["truthfulness_note"]
    assert "evaluated executable integrations" in scan["truthfulness_note"]

    print("灾情感知及影响评估 full actual integration dashboard smoke test passed.")


if __name__ == "__main__":
    main()
