"""Adapter runners for source-level SKAI and InaSAFE integration.

The runners are deliberately conservative: they validate source code,
dependencies, config, inputs, and expected output files. They never synthesize
SKAI or InaSAFE results from internal segmentation statistics.
"""

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from external_impact_assessment import (
    OUTPUT_ROOT,
    assess_inasafe_source_level_status,
    assess_skai_source_level_status,
    build_external_impact_assessment_status,
    save_external_impact_assessment_status,
)


ROOT_DIR = Path(__file__).resolve().parents[1]


class ExternalImpactRunnerError(Exception):
    pass


def _utc_timestamp():
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _load_config(config_path):
    path = Path(config_path) if config_path else None
    if path is None:
        return None, "Config path is required."
    if not path.is_absolute():
        path = ROOT_DIR / path
    if not path.exists():
        return None, f"Config file does not exist: {path}"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return None, f"Failed to parse config JSON: {exc}"
    data["_config_path"] = str(path)
    return data, ""


def _as_list(value):
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _resolve_path(value, base_dir=ROOT_DIR):
    if not value:
        return None
    path = Path(value)
    if not path.is_absolute():
        path = Path(base_dir) / path
    return path


def _missing_paths(paths, base_dir=ROOT_DIR):
    missing = []
    resolved = []
    for value in _as_list(paths):
        path = _resolve_path(value, base_dir=base_dir)
        if path is None:
            continue
        resolved.append(str(path))
        if not path.exists():
            missing.append(str(path))
    return resolved, missing


def _nonempty_files(paths, base_dir=ROOT_DIR):
    files = []
    missing = []
    for value in _as_list(paths):
        path = _resolve_path(value, base_dir=base_dir)
        if path is None:
            continue
        if path.is_file() and path.stat().st_size > 0:
            files.append(str(path))
        else:
            missing.append(str(path))
    return files, missing


def _write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(path)


def _runner_status(
    module_key,
    success,
    status,
    config=None,
    output_dir=None,
    command=None,
    stdout="",
    stderr="",
    unavailable_reasons=None,
    verified_output_files=None,
):
    return {
        "module_key": module_key,
        "success": bool(success),
        "status": status,
        "executed": bool(command) and status in {"executed_success", "real_output_verified", "executed_failed"},
        "timestamp": _utc_timestamp(),
        "config_path": (config or {}).get("_config_path", ""),
        "output_dir": str(output_dir or ""),
        "command": command or [],
        "stdout_tail": (stdout or "")[-4000:],
        "stderr_tail": (stderr or "")[-4000:],
        "verified_output_files": verified_output_files or [],
        "unavailable_reasons": unavailable_reasons or [],
        "truthfulness_note": (
            "This runner only records source-level execution and verified output files. "
            "It does not fabricate SKAI or InaSAFE results."
        ),
    }


def _run_command(command, env=None, cwd=None, timeout_seconds=None):
    if not command:
        return 0, "", ""
    result = subprocess.run(
        [str(item) for item in command],
        check=False,
        capture_output=True,
        text=True,
        cwd=str(cwd) if cwd else None,
        env=env,
        timeout=int(timeout_seconds) if timeout_seconds else None,
    )
    return result.returncode, result.stdout or "", result.stderr or ""


def run_skai_adapter(config_path=None, output_dir=None, execute=False):
    """Run or validate a SKAI adapter config."""
    output_dir = Path(output_dir or OUTPUT_ROOT / "skai")
    config, config_error = _load_config(config_path)
    status = assess_skai_source_level_status(output_dir=output_dir)
    reasons = list(status.get("unavailable_reasons", []) or [])
    if config_error:
        reasons.append(config_error)
        result = _runner_status("skai", False, "unavailable", output_dir=output_dir, unavailable_reasons=reasons)
        _write_json(output_dir / "runner_status.json", result)
        save_external_impact_assessment_status(build_external_impact_assessment_status())
        return result

    required_inputs, missing_inputs = _missing_paths(config.get("required_inputs", []))
    expected_outputs = config.get("expected_outputs", [])
    command = config.get("command", [])
    if missing_inputs:
        reasons.append(f"Missing required SKAI inputs: {', '.join(missing_inputs)}.")
    if not command:
        reasons.append("SKAI adapter command is missing.")

    if reasons and not execute:
        result = _runner_status(
            "skai",
            False,
            "unavailable",
            config=config,
            output_dir=output_dir,
            command=command,
            unavailable_reasons=reasons,
        )
        _write_json(output_dir / "runner_status.json", result)
        save_external_impact_assessment_status(build_external_impact_assessment_status())
        return result

    if not execute:
        result = _runner_status(
            "skai",
            True,
            "ready_dry_run",
            config=config,
            output_dir=output_dir,
            command=command,
            verified_output_files=[],
        )
        _write_json(output_dir / "runner_status.json", result)
        save_external_impact_assessment_status(build_external_impact_assessment_status())
        return result

    env = os.environ.copy()
    if config.get("pythonpath"):
        env["PYTHONPATH"] = os.pathsep.join([str(_resolve_path(p)) for p in _as_list(config["pythonpath"]) if p])
    returncode, stdout, stderr = _run_command(
        command,
        env=env,
        cwd=_resolve_path(config.get("working_dir")) if config.get("working_dir") else ROOT_DIR,
        timeout_seconds=config.get("timeout_seconds"),
    )
    verified_outputs, missing_outputs = _nonempty_files(expected_outputs)
    if returncode != 0:
        reasons.append(f"SKAI adapter command failed with exit code {returncode}.")
    if missing_outputs:
        reasons.append(f"Expected SKAI output files were not produced: {', '.join(missing_outputs)}.")
    success = returncode == 0 and bool(verified_outputs) and not missing_outputs
    result = _runner_status(
        "skai",
        success,
        "real_output_verified" if success else "executed_failed",
        config=config,
        output_dir=output_dir,
        command=command,
        stdout=stdout,
        stderr=stderr,
        unavailable_reasons=[] if success else reasons,
        verified_output_files=verified_outputs,
    )
    _write_json(output_dir / "runner_status.json", result)
    save_external_impact_assessment_status(build_external_impact_assessment_status())
    return result


def run_inasafe_adapter(config_path=None, output_dir=None, execute=False):
    """Run or validate an InaSAFE/QGIS adapter config."""
    output_dir = Path(output_dir or OUTPUT_ROOT / "inasafe")
    config, config_error = _load_config(config_path)
    status = assess_inasafe_source_level_status(output_dir=output_dir)
    reasons = list(status.get("unavailable_reasons", []) or [])
    if config_error:
        reasons.append(config_error)
        result = _runner_status("inasafe", False, "unavailable", output_dir=output_dir, unavailable_reasons=reasons)
        _write_json(output_dir / "runner_status.json", result)
        save_external_impact_assessment_status(build_external_impact_assessment_status())
        return result

    required_inputs, missing_inputs = _missing_paths(config.get("required_inputs", []))
    expected_outputs = config.get("expected_outputs", [])
    command = config.get("command", [])
    if missing_inputs:
        reasons.append(f"Missing required InaSAFE GIS inputs: {', '.join(missing_inputs)}.")
    if not command:
        reasons.append("InaSAFE adapter command is missing.")

    if reasons and not execute:
        result = _runner_status(
            "inasafe",
            False,
            "unavailable",
            config=config,
            output_dir=output_dir,
            command=command,
            unavailable_reasons=reasons,
        )
        _write_json(output_dir / "runner_status.json", result)
        save_external_impact_assessment_status(build_external_impact_assessment_status())
        return result

    if not execute:
        result = _runner_status(
            "inasafe",
            True,
            "ready_dry_run",
            config=config,
            output_dir=output_dir,
            command=command,
        )
        _write_json(output_dir / "runner_status.json", result)
        save_external_impact_assessment_status(build_external_impact_assessment_status())
        return result

    env = os.environ.copy()
    if config.get("pythonpath"):
        env["PYTHONPATH"] = os.pathsep.join([str(_resolve_path(p)) for p in _as_list(config["pythonpath"]) if p])
    returncode, stdout, stderr = _run_command(
        command,
        env=env,
        cwd=_resolve_path(config.get("working_dir")) if config.get("working_dir") else ROOT_DIR,
        timeout_seconds=config.get("timeout_seconds"),
    )
    verified_outputs, missing_outputs = _nonempty_files(expected_outputs)
    if returncode != 0:
        reasons.append(f"InaSAFE adapter command failed with exit code {returncode}.")
    if missing_outputs:
        reasons.append(f"Expected InaSAFE output files were not produced: {', '.join(missing_outputs)}.")
    success = returncode == 0 and bool(verified_outputs) and not missing_outputs
    result = _runner_status(
        "inasafe",
        success,
        "real_output_verified" if success else "executed_failed",
        config=config,
        output_dir=output_dir,
        command=command,
        stdout=stdout,
        stderr=stderr,
        unavailable_reasons=[] if success else reasons,
        verified_output_files=verified_outputs,
    )
    _write_json(output_dir / "runner_status.json", result)
    save_external_impact_assessment_status(build_external_impact_assessment_status())
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run external SKAI / InaSAFE impact adapters.")
    parser.add_argument("module", choices=["skai", "inasafe"])
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    if args.module == "skai":
        result = run_skai_adapter(args.config, output_dir=args.output_dir, execute=args.execute)
    else:
        result = run_inasafe_adapter(args.config, output_dir=args.output_dir, execute=args.execute)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result.get("success") or result.get("status") in {"ready_dry_run", "unavailable"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
