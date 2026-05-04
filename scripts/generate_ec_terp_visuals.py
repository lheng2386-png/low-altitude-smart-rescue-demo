#!/usr/bin/env python3
"""Generate EC-TERP visualization artifacts."""

import argparse
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
APP_DIR = ROOT_DIR / "app"
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from ec_terp_visualization_service import generate_ec_terp_visuals  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="Generate EC-TERP visualization figures.")
    parser.add_argument("--ranking-path", default=str(ROOT_DIR / "outputs" / "ec_terp" / "ec_terp_rankings.json"))
    parser.add_argument("--eval-dir", default=str(ROOT_DIR / "outputs" / "ec_terp_evaluation"))
    parser.add_argument("--output-dir", default=str(ROOT_DIR / "outputs" / "ec_terp_visuals"))
    args = parser.parse_args()

    result = generate_ec_terp_visuals(
        ranking_path=args.ranking_path,
        eval_dir=args.eval_dir,
        output_dir=args.output_dir,
    )
    print(f"EC-TERP visuals status: {result.get('status')}")
    print(f"Metadata: {result.get('metadata_path')}")
    for figure in result.get("generated_figures", []):
        print(f"- {figure.get('name')}: {figure.get('path')}")
    if result.get("limitations"):
        print("Limitations:")
        for item in result.get("limitations", []):
            print(f"- {item}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
