"""A* rescue path planning on RescueNet-style cost maps."""

from heapq import heappop, heappush

import cv2
import numpy as np
from PIL import Image


PATH_COSTS = {
    0: 3.0,    # background
    1: 100.0,  # water
    2: 8.0,    # no_damage_building
    3: 25.0,   # minor_damage
    4: 60.0,   # major_damage
    5: 100.0,  # destroyed_building
    6: 20.0,   # vehicle
    7: 1.0,    # road_clear
    8: 80.0,   # road_blocked
    9: 15.0,   # tree
    10: 100.0, # pool
}


def _as_rgb_array(image):
    if isinstance(image, Image.Image):
        return np.array(image.convert("RGB"))
    array = np.asarray(image)
    if array.ndim == 2:
        return np.stack([array, array, array], axis=-1)
    if array.shape[-1] == 4:
        array = array[:, :, :3]
    return array.astype(np.uint8)


def build_cost_map(segmentation_mask, image_width, image_height):
    """Build a traversal-cost map from an optional RescueNet-style mask."""
    default_cost = PATH_COSTS[0]
    cost_map = np.full((int(image_height), int(image_width)), default_cost, dtype=np.float32)

    if segmentation_mask is None:
        return cost_map

    mask = np.asarray(segmentation_mask)
    if mask.shape[:2] != (int(image_height), int(image_width)):
        mask = cv2.resize(
            mask.astype(np.uint8),
            (int(image_width), int(image_height)),
            interpolation=cv2.INTER_NEAREST,
        )

    for class_id, cost in PATH_COSTS.items():
        cost_map[mask == class_id] = cost

    return cost_map


def clamp_point(point, width, height):
    """Clamp a point into the image boundary and return integer coordinates."""
    x, y = point
    x = int(max(0, min(int(width) - 1, round(float(x)))))
    y = int(max(0, min(int(height) - 1, round(float(y)))))
    return x, y


def get_target_point(target):
    """Get the center point of a structured target."""
    center = target.get("center")
    if center and len(center) >= 2:
        return int(round(float(center[0]))), int(round(float(center[1])))

    bbox = target.get("bbox", [0, 0, 0, 0])
    x1, y1, x2, y2 = [float(value) for value in bbox[:4]]
    return int(round((x1 + x2) / 2)), int(round((y1 + y2) / 2))


def _heuristic(a, b):
    return float(np.hypot(a[0] - b[0], a[1] - b[1]))


def astar(cost_map, start, goal):
    """Run A* on an image-space cost map using 8-neighborhood movement."""
    height, width = cost_map.shape[:2]
    start = clamp_point(start, width, height)
    goal = clamp_point(goal, width, height)

    if start == goal:
        return {
            "found": True,
            "path": [start],
            "total_cost": 0.0,
            "path_length": 1,
            "message": "起点与目标点重合，返回单点路径。",
        }

    moves = [
        (-1, 0, 1.0),
        (1, 0, 1.0),
        (0, -1, 1.0),
        (0, 1, 1.0),
        (-1, -1, 1.41421356237),
        (-1, 1, 1.41421356237),
        (1, -1, 1.41421356237),
        (1, 1, 1.41421356237),
    ]

    open_heap = []
    heappush(open_heap, (0.0, start))
    came_from = {}
    g_score = {start: 0.0}
    closed = set()

    while open_heap:
        _, current = heappop(open_heap)
        if current in closed:
            continue
        if current == goal:
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            path.reverse()
            return {
                "found": True,
                "path": path,
                "total_cost": round(float(g_score[goal]), 2),
                "path_length": len(path),
                "message": "A* 路径规划成功。",
            }

        closed.add(current)
        cx, cy = current

        for dx, dy, step_distance in moves:
            nx, ny = cx + dx, cy + dy
            if nx < 0 or ny < 0 or nx >= width or ny >= height:
                continue

            step_cost = step_distance * float(cost_map[ny, nx])
            tentative_g = g_score[current] + step_cost
            neighbor = (nx, ny)

            if tentative_g < g_score.get(neighbor, float("inf")):
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                priority = tentative_g + _heuristic(neighbor, goal)
                heappush(open_heap, (priority, neighbor))

    return {
        "found": False,
        "path": [],
        "total_cost": 0.0,
        "path_length": 0,
        "message": "未能在当前代价地图中找到可用路径。",
    }


def plan_rescue_path(ranked_targets, segmentation_mask, image_width, image_height, start_point=None):
    """Plan a rescue route from the start point to the highest-risk target."""
    if not ranked_targets:
        return {
            "found": False,
            "start": None,
            "goal": None,
            "target_id": None,
            "target_class": None,
            "path": [],
            "total_cost": 0.0,
            "path_length": 0,
            "message": "当前未检测到明确救援目标，无法规划路径。",
        }

    cost_map = build_cost_map(segmentation_mask, image_width, image_height)
    target = ranked_targets[0]
    goal = get_target_point(target)

    if start_point is None:
        start_point = (20, int(image_height) - 20)

    start = clamp_point(start_point, image_width, image_height)
    goal = clamp_point(goal, image_width, image_height)

    result = astar(cost_map, start, goal)
    result.update(
        {
            "start": [int(start[0]), int(start[1])],
            "goal": [int(goal[0]), int(goal[1])],
            "target_id": target.get("target_id"),
            "target_class": target.get("class_name"),
        }
    )
    return result


def create_path_overlay(image, path_result):
    """Draw the planned route on top of an image."""
    if image is None:
        return None

    image_rgb = _as_rgb_array(image).copy()
    if not path_result or not path_result.get("found"):
        return image_rgb

    path = path_result.get("path") or []
    if not path:
        return image_rgb

    overlay = image_rgb.copy()
    for index in range(1, len(path)):
        pt1 = tuple(int(value) for value in path[index - 1])
        pt2 = tuple(int(value) for value in path[index])
        cv2.line(overlay, pt1, pt2, (255, 215, 0), 3, cv2.LINE_AA)

    start = tuple(int(value) for value in path_result.get("start", path[0]))
    goal = tuple(int(value) for value in path_result.get("goal", path[-1]))

    cv2.circle(overlay, start, 7, (0, 200, 0), -1)
    cv2.circle(overlay, goal, 7, (220, 0, 0), -1)
    cv2.putText(overlay, "S", (start[0] + 8, start[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 120, 0), 2, cv2.LINE_AA)
    cv2.putText(overlay, "T", (goal[0] + 8, goal[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (170, 0, 0), 2, cv2.LINE_AA)

    return overlay
