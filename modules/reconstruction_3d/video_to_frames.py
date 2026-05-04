"""Extract sampled video frames for downstream reconstruction tools."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

try:
    import cv2
except Exception:  # pragma: no cover - exercised only when OpenCV is absent
    cv2 = None


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def extract_frames(video_path: str | Path, output_dir: str | Path, fps: float = 1.0) -> dict[str, Any]:
    """Extract video frames at a configurable sampling FPS.

    The function reports only extraction metadata. It does not imply that
    reconstruction has succeeded.
    """
    video = Path(video_path)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    metadata_path = output / "frames_metadata.json"

    metadata: dict[str, Any] = {
        "original_video": str(video),
        "fps": float(fps),
        "frame_count": 0,
        "extracted_count": 0,
        "output_dir": str(output),
    }
    if not video.exists():
        metadata.update({"success": False, "status": "input_missing", "error": f"Video not found: {video}"})
        write_json(metadata_path, metadata)
        return metadata
    if cv2 is None:
        metadata.update({"success": False, "status": "dependency_missing", "error": "OpenCV is not installed."})
        write_json(metadata_path, metadata)
        return metadata
    if fps <= 0:
        metadata.update({"success": False, "status": "invalid_fps", "error": "fps must be greater than 0."})
        write_json(metadata_path, metadata)
        return metadata

    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        metadata.update({"success": False, "status": "video_open_failed", "error": f"Could not open {video}"})
        write_json(metadata_path, metadata)
        return metadata

    source_fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    sample_interval = max(1, round(source_fps / float(fps))) if source_fps > 0 else max(1, round(30 / float(fps)))
    extracted_count = 0
    frame_index = 0

    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if frame_index % sample_interval == 0:
            extracted_count += 1
            frame_path = output / f"frame_{extracted_count:06d}.jpg"
            cv2.imwrite(str(frame_path), frame)
        frame_index += 1

    capture.release()
    metadata.update(
        {
            "source_fps": source_fps,
            "frame_count": frame_count or frame_index,
            "extracted_count": extracted_count,
            "sample_interval_frames": sample_interval,
            "success": extracted_count > 0,
            "status": "completed" if extracted_count > 0 else "no_frames_extracted",
            "metadata_path": str(metadata_path),
        }
    )
    write_json(metadata_path, metadata)
    return metadata


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Extract sampled frames from a video.")
    parser.add_argument("--video", required=True, help="Input video path.")
    parser.add_argument("--output", required=True, help="Output frames directory.")
    parser.add_argument("--fps", type=float, default=1.0, help="Sampling FPS.")
    args = parser.parse_args(argv)
    metadata = extract_frames(args.video, args.output, args.fps)
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    return 0 if metadata.get("success") else 1


if __name__ == "__main__":
    raise SystemExit(main())
