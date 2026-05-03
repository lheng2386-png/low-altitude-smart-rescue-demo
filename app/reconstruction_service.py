import json
from pathlib import Path

import cv2
import numpy as np


ROOT_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT_DIR / "outputs" / "reconstruction"
KEYFRAME_DIR = OUTPUT_DIR / "keyframes"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
KEYFRAME_DIR.mkdir(parents=True, exist_ok=True)


def _as_path(file_obj):
    if file_obj is None:
        return None
    if isinstance(file_obj, str):
        return file_obj
    if isinstance(file_obj, dict):
        return file_obj.get("path") or file_obj.get("name")
    if hasattr(file_obj, "name"):
        return file_obj.name
    return str(file_obj)


def _contact_sheet(images, thumb_size=(220, 124), cols=5):
    if not images:
        return np.zeros((thumb_size[1], thumb_size[0], 3), dtype=np.uint8)
    thumbs = [cv2.resize(img, thumb_size) for img in images]
    rows = []
    for i in range(0, len(thumbs), cols):
        row = thumbs[i:i + cols]
        while len(row) < cols:
            row.append(np.zeros_like(thumbs[0]))
        rows.append(cv2.hconcat(row))
    return cv2.vconcat(rows)


def _write_ply(points, path):
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("ply\nformat ascii 1.0\n")
        handle.write(f"element vertex {len(points)}\n")
        handle.write("property float x\nproperty float y\nproperty float z\n")
        handle.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        handle.write("end_header\n")
        for x, y, z, r, g, b in points:
            handle.write(f"{x:.3f} {y:.3f} {z:.3f} {int(r)} {int(g)} {int(b)}\n")


def process_reconstruction(video_file, max_keyframes=20):
    """Extract keyframes, ORB features, matches, a simple trajectory, and a PLY preview."""
    path = _as_path(video_file)
    if not path:
        return None, None, None, None, None, "未上传视频。", "{}"
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return None, None, None, None, None, f"无法打开视频：{path}", "{}"

    for old in KEYFRAME_DIR.glob("*.jpg"):
        old.unlink()

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    max_keyframes = max(1, min(int(max_keyframes or 20), 20))
    step = max(1, frame_count // max_keyframes) if frame_count else 30
    keyframes = []
    idx = 0
    while len(keyframes) < max_keyframes:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % step == 0:
            keyframes.append(frame.copy())
            cv2.imwrite(str(KEYFRAME_DIR / f"keyframe_{len(keyframes):03d}.jpg"), frame)
        idx += 1
    cap.release()

    orb = cv2.ORB_create(nfeatures=1800)
    feature_count = 0
    match_count = 0
    all_points = []
    features_preview = None
    matches_preview = None
    trajectory = [(0.0, 0.0)]
    current_xy = np.array([0.0, 0.0], dtype=np.float32)

    descriptors = []
    keypoints = []
    for frame in keyframes:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        kp, des = orb.detectAndCompute(gray, None)
        kp = kp or []
        feature_count += len(kp)
        descriptors.append(des)
        keypoints.append(kp)
        if features_preview is None:
            features_preview = cv2.drawKeypoints(frame, kp[:300], None, color=(0, 255, 0))
        for point in kp[:120]:
            x, y = point.pt
            all_points.append((x / 20.0, y / 20.0, float(len(all_points) % 40) / 8.0, 80, 180, 255))

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    for i in range(len(keyframes) - 1):
        if descriptors[i] is None or descriptors[i + 1] is None:
            trajectory.append(tuple(current_xy))
            continue
        matches = sorted(matcher.match(descriptors[i], descriptors[i + 1]), key=lambda m: m.distance)[:120]
        match_count += len(matches)
        if matches and matches_preview is None:
            matches_preview = cv2.drawMatches(
                keyframes[i],
                keypoints[i],
                keyframes[i + 1],
                keypoints[i + 1],
                matches[:40],
                None,
                flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
            )
        if matches:
            shifts = []
            for match in matches[:80]:
                p1 = np.array(keypoints[i][match.queryIdx].pt)
                p2 = np.array(keypoints[i + 1][match.trainIdx].pt)
                shifts.append(p2 - p1)
            current_xy += np.mean(shifts, axis=0) / 30.0
        trajectory.append(tuple(current_xy))

    keyframe_preview = _contact_sheet(keyframes)
    if features_preview is None:
        features_preview = keyframe_preview.copy()
    if matches_preview is None:
        matches_preview = keyframe_preview.copy()

    traj_img = np.zeros((500, 700, 3), dtype=np.uint8) + 245
    pts = np.array(trajectory, dtype=np.float32)
    if len(pts) > 1:
        pts -= pts.min(axis=0)
        denom = np.maximum(pts.max(axis=0), 1)
        pts = pts / denom * np.array([620, 420]) + np.array([40, 40])
    for i in range(len(pts) - 1):
        cv2.line(traj_img, tuple(pts[i].astype(int)), tuple(pts[i + 1].astype(int)), (30, 90, 220), 2)
    for i, pt in enumerate(pts):
        cv2.circle(traj_img, tuple(pt.astype(int)), 4, (0, 130, 0), -1)
        cv2.putText(traj_img, str(i + 1), tuple((pt + 5).astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)

    keyframe_preview_path = OUTPUT_DIR / "keyframes_preview.jpg"
    features_path = OUTPUT_DIR / "features_preview.jpg"
    matches_path = OUTPUT_DIR / "matches_preview.jpg"
    trajectory_path = OUTPUT_DIR / "camera_trajectory.jpg"
    ply_path = OUTPUT_DIR / "point_cloud.ply"
    result_path = OUTPUT_DIR / "reconstruction_result.json"
    cv2.imwrite(str(keyframe_preview_path), keyframe_preview)
    cv2.imwrite(str(features_path), features_preview)
    cv2.imwrite(str(matches_path), matches_preview)
    cv2.imwrite(str(trajectory_path), traj_img)
    _write_ply(all_points[:5000], ply_path)

    result = {
        "video_name": Path(path).name,
        "frame_count": frame_count,
        "keyframe_count": len(keyframes),
        "feature_count": feature_count,
        "match_count": match_count,
        "point_cloud_path": str(ply_path),
        "trajectory_path": str(trajectory_path),
        "status": "completed_lightweight_reconstruction_preview",
    }
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    status = "三维重建预处理完成：已抽取关键帧、ORB 特征、相邻帧匹配、简化相机轨迹和 PLY 点云预览。"
    return (
        str(keyframe_preview_path),
        str(features_path),
        str(matches_path),
        str(trajectory_path),
        str(ply_path),
        status,
        json.dumps(result, ensure_ascii=False, indent=2),
    )

