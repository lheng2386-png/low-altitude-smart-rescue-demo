"""Filter extracted frames before reconstruction."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}


def _image_paths(frames_dir: Path) -> list[Path]:
    return sorted(path for path in frames_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)


def _average_hash(gray, size: int = 8) -> tuple[int, ...]:
    resized = cv2.resize(gray, (size, size))
    return tuple((resized > resized.mean()).astype("uint8").flatten().tolist())


def _hamming_distance(left: tuple[int, ...], right: tuple[int, ...]) -> int:
    return sum(a != b for a, b in zip(left, right))


def filter_frames(
    frames_dir: str | Path,
    output_dir: str | Path,
    blur_threshold: float = 100.0,
    brightness_threshold: float = 30.0,
    reduce_duplicates: bool = True,
    duplicate_hash_distance: int = 4,
) -> dict[str, Any]:
    """Select frames with enough sharpness, brightness, and visual novelty."""
    source = Path(frames_dir)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    metadata_path = output / "frame_quality_metadata.json"
    metadata: dict[str, Any] = {
        "input_dir": str(source),
        "output_dir": str(output),
        "total_frames": 0,
        "selected_frames": 0,
        "rejected_frames": 0,
        "rejection_reasons": {"blurry": 0, "too_dark": 0, "near_duplicate": 0, "read_failed": 0},
        "selected_files": [],
        "rejected_files": [],
    }
    if not source.exists():
        metadata.update({"success": False, "status": "input_missing"})
        metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        return metadata
    if cv2 is None:
        metadata.update({"success": False, "status": "dependency_missing", "error": "OpenCV is not installed."})
        metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        return metadata

    selected_hashes: list[tuple[int, ...]] = []
    for frame_path in _image_paths(source):
        metadata["total_frames"] += 1
        image = cv2.imread(str(frame_path))
        if image is None:
            reason = "read_failed"
        else:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
            brightness = float(gray.mean())
            reason = ""
            if blur_score < blur_threshold:
                reason = "blurry"
            elif brightness < brightness_threshold:
                reason = "too_dark"
            elif reduce_duplicates:
                image_hash = _average_hash(gray)
                if any(_hamming_distance(image_hash, old_hash) <= duplicate_hash_distance for old_hash in selected_hashes):
                    reason = "near_duplicate"
                else:
                    selected_hashes.append(image_hash)
            else:
                image_hash = _average_hash(gray)
                selected_hashes.append(image_hash)

        if reason:
            metadata["rejection_reasons"][reason] += 1
            metadata["rejected_files"].append({"file": str(frame_path), "reason": reason})
            continue

        destination = output / frame_path.name
        shutil.copy2(frame_path, destination)
        metadata["selected_files"].append(str(destination))

    metadata["selected_frames"] = len(metadata["selected_files"])
    metadata["rejected_frames"] = int(metadata["total_frames"]) - int(metadata["selected_frames"])
    metadata.update({"success": True, "status": "completed", "metadata_path": str(metadata_path)})
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return metadata


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Filter frames for reconstruction quality.")
    parser.add_argument("--frames", required=True, help="Input frames directory.")
    parser.add_argument("--output", required=True, help="Selected frames output directory.")
    parser.add_argument("--blur-threshold", type=float, default=100.0)
    parser.add_argument("--brightness-threshold", type=float, default=30.0)
    parser.add_argument("--no-duplicate-filter", action="store_true")
    args = parser.parse_args(argv)
    metadata = filter_frames(
        args.frames,
        args.output,
        blur_threshold=args.blur_threshold,
        brightness_threshold=args.brightness_threshold,
        reduce_duplicates=not args.no_duplicate_filter,
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    return 0 if metadata.get("success") else 1


if __name__ == "__main__":
    raise SystemExit(main())
