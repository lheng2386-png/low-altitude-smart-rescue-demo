#!/usr/bin/env python3
"""Run EC-TERP evaluation and sensitivity analysis."""

import argparse
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
APP_DIR = ROOT_DIR / "app"
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from ec_terp_evaluation_service import run_full_ec_terp_evaluation  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="Run EC-TERP evaluation and sensitivity analysis.")
    parser.add_argument("--case-file", default=None, help="Optional JSON case file. Defaults to built-in synthetic demo cases.")
    parser.add_argument("--output-dir", default=str(ROOT_DIR / "outputs" / "ec_terp_evaluation"), help="Output directory.")
    parser.add_argument("--n-random-trials", type=int, default=100, help="Number of random weight perturbation trials.")
    args = parser.parse_args()

    result = run_full_ec_terp_evaluation(
        case_file=args.case_file,
        output_dir=args.output_dir,
        n_random_trials=args.n_random_trials,
    )
    print("EC-TERP evaluation completed.")
    print(f"Success: {result.get('success')}")
    print(f"Report: {result.get('output_files', {}).get('report')}")
    print(f"Comparison summary: {result.get('comparison_result', {}).get('summary')}")
    print(f"Sensitivity summary: {result.get('sensitivity_result', {}).get('summary')}")
    print(f"Random stability summary: {result.get('random_stability_result', {}).get('summary')}")
    print(result.get("truthfulness_note", ""))


if __name__ == "__main__":
    main()
