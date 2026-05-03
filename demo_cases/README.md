# Demo Cases

This directory describes how to organize 3-5 complete AeroRescue-AI competition demo cases.

The repository does not need to include large raw datasets. Each case can be prepared locally with a small image, optional segmentation mask, expected outputs, and notes.

## Recommended Case Structure

```text
demo_cases/case_01_flood/
├── input.jpg
├── mask.png
├── expected_outputs.md
└── notes.md
```

## Case 1 Flood Civilian Rescue

- Scene: flood or waterlogged area.
- Target: `civilian`.
- Main demonstration: water risk, TERP priority, Risk-Aware A* route.
- Expected output: civilian ranked high, path planner prefers avoiding water when mask is available.

## Case 2 Building Collapse

- Scene: damaged or collapsed buildings.
- Target: `civilian` / `rescuer`.
- Main demonstration: `major_damage` and `destroyed_building` environmental risk.
- Expected output: environment score increases around severe damage zones.

## Case 3 Road Blocked

- Scene: blocked road or obstructed access.
- Target: animal or civilian.
- Main demonstration: `road_blocked` cost map and path detour.
- Expected output: baseline path may cross blocked road, Risk-Aware A* should prefer a lower-risk alternative when possible.

## Case 4 Multi-target Priority

- Scene: multiple civilians, animals, and rescuers.
- Target: multiple target classes.
- Main demonstration: TERP ranking across target class, environment, and route accessibility.
- Expected output: target ranking reflects both target importance and accessibility.

## Case 5 No Target / Low Confidence

- Scene: no clear rescue target or very low-confidence detections.
- Target: none.
- Main demonstration: system avoids overconfident rescue advice.
- Expected output: report explains that no clear rescue target was detected and no path is planned.

## Notes

- Use local images or project-generated outputs.
- Avoid committing large datasets.
- Do not commit model checkpoints.
- For class-id masks, PNG is recommended.
- If no mask is available, the case should explicitly show fallback behavior.
