# Fourth Step: A* Path Planning Layer

This step adds a lightweight rescue path planning layer on top of the existing detection, risk ranking, and RescueNet-style mask fusion workflow.

## What Was Added

- `app/path_planner.py`
- A* route planning from a rescue start point to the highest-risk target
- Path overlay rendering on the image
- Path planning summary in the Gradio interface
- Report text describing the planned route and cost

## `app/path_planner.py`

This module keeps all path logic away from the UI layer.

Implemented functions:

- `build_cost_map(segmentation_mask, image_width, image_height)`
- `clamp_point(point, width, height)`
- `get_target_point(target)`
- `astar(cost_map, start, goal)`
- `plan_rescue_path(ranked_targets, segmentation_mask, image_width, image_height, start_point=None)`
- `create_path_overlay(image, path_result)`

## A* Inputs And Outputs

Inputs:

- rescue start point
- highest-risk target as goal
- image size
- optional RescueNet-style segmentation mask

Outputs:

- `found`
- `path`
- `total_cost`
- `path_length`
- `message`
- `start`
- `goal`
- `target_id`
- `target_class`

## RescueNet Class Cost Map

| class_id | class_name | cost |
| --- | --- | ---: |
| 0 | background | 3 |
| 1 | water | 100 |
| 2 | no_damage_building | 8 |
| 3 | minor_damage | 25 |
| 4 | major_damage | 60 |
| 5 | destroyed_building | 100 |
| 6 | vehicle | 20 |
| 7 | road_clear | 1 |
| 8 | road_blocked | 80 |
| 9 | tree | 15 |
| 10 | pool | 100 |

High-risk areas are represented as high-cost regions, not as fully blocked cells in this version.

## Start Point Handling

- `start_x` and `start_y` are provided from the Image Tab.
- If `start_y < 0`, the system falls back to `image_height - 20`.
- If no start is provided, the default start is `(20, image_height - 20)`.
- Points are clamped into image bounds before planning.

## Without A Segmentation Mask

If no mask is uploaded, the planner builds a uniform default cost map with cost `3` everywhere.

That means the path is only a planar demonstration path and does not consider water, blocked roads, or damaged buildings.

## With A Segmentation Mask

If a RescueNet-style mask is uploaded, the planner converts semantic classes into traversal costs.

This lets the A* route prefer clear roads and avoid water, blocked roads, and heavily damaged areas where possible.

## Current Limits

- This is an image-plane path, not a real GPS route.
- There is no real road graph or map tile system.
- High-risk regions are high cost, not fully forbidden.
- The Video Tab does not support path planning.

## Next Steps

- Improve path planning with more realistic passability constraints.
- Add multi-target rescue routing.
- Add Detection-Models comparison experiments.
- Use ARGUS-style platform ideas later for task management and report workflows.
