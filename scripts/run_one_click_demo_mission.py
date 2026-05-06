#!/usr/bin/env python3
"""Run the 灾情感知及影响评估 one-click demo mission without launching Gradio."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.demo.one_click_mission_orchestrator import run_one_click_demo_mission  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="Run 灾情感知及影响评估 one-click demo mission.")
    parser.add_argument("--missions-root", default="outputs/demo_missions")
    parser.add_argument("--mission-name", default="灾情感知及影响评估 Demo")
    parser.add_argument("--demo-output-root", default="outputs/demo_dataset")
    parser.add_argument("--mode", default="full_demo")
    return parser.parse_args()


def main():
    args = parse_args()
    result = run_one_click_demo_mission(
        missions_root=ROOT_DIR / args.missions_root,
        mission_name=args.mission_name,
        demo_output_root=ROOT_DIR / args.demo_output_root,
        workflow_mode=args.mode,
    )
    mission = result.get("mission", {})
    workflow_summary = result.get("workflow_summary", {})
    print(f"mission_id: {mission.get('mission_id', '')}")
    print(f"mission_dir: {result.get('mission_dir', '')}")
    print(f"demo_dataset_dir: {result.get('demo_dataset_dir', '')}")
    print(f"final_report_markdown_path: {result.get('final_report_markdown_path', '')}")
    print(f"evidence_ledger_path: {result.get('evidence_ledger_path', '')}")
    print(f"completed_stage_count: {workflow_summary.get('completed_stage_count', 0)}")
    print(f"failed_stage_count: {workflow_summary.get('failed_stage_count', 0)}")
    print(f"truthfulness_warning: {result.get('truthfulness_note', '')}")


if __name__ == "__main__":
    main()
