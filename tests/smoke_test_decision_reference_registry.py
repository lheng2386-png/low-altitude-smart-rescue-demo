import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
APP_DIR = ROOT_DIR / "app"
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from decision_reference_registry import (  # noqa: E402
    get_decision_reference_config,
    list_decision_reference_projects,
    summarize_decision_reference_capabilities,
)


def main():
    items = list_decision_reference_projects()
    assert isinstance(items, list)
    keys = {item["reference_key"] for item in items}
    for key in [
        "sarenv_search_planning",
        "skai_building_damage",
        "inasafe_impact_modeling",
        "fields2cover_coverage_planning",
        "pythonrobotics_path_algorithms",
    ]:
        assert key in keys
    assert all(item.get("truthfulness_note") for item in items)
    assert all("status" in item for item in items)
    assert all("active_runtime" in item for item in items)

    cfg = get_decision_reference_config("sarenv_search_planning")
    assert cfg["display_name"].startswith("SAREnv")
    try:
        get_decision_reference_config("missing_reference")
        raise AssertionError("Expected DecisionReferenceRegistryError")
    except Exception:
        pass

    summary = summarize_decision_reference_capabilities()
    assert isinstance(summary, str)
    assert "SAREnv" in summary
    assert "SKAI" in summary
    assert "InaSAFE" in summary
    assert "Fields2Cover" in summary
    assert "PythonRobotics" in summary
    assert "不是 GPS 导航" in summary or "不是完整 GIS" in summary or "辅助决策" in summary

    print("灾情感知及影响评估 decision reference registry smoke test passed.")


if __name__ == "__main__":
    main()
