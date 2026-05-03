# RescueNet-Style Class Palette

The active AeroRescue-AI segmentation layer uses 11 class ids.

| ID | Class | Chinese Name | Environment Risk | Path Cost |
| --- | --- | --- | --- | --- |
| 0 | background | 背景 | Low | 3 |
| 1 | water | 水域 | High | 100 |
| 2 | no_damage_building | 无损建筑 | Low | 8 |
| 3 | minor_damage | 轻微损毁建筑 | Medium | 25 |
| 4 | major_damage | 严重损毁建筑 | High | 60 |
| 5 | destroyed_building | 完全毁坏建筑 | High | 100 |
| 6 | vehicle | 车辆 | Medium | 20 |
| 7 | road_clear | 可通行道路 | Low | 1 |
| 8 | road_blocked | 道路阻断 | High | 80 |
| 9 | tree | 树木 | Medium | 15 |
| 10 | pool | 水池/积水 | High | 100 |

## Mask Format

- Class-id masks should be PNG.
- JPG is not recommended for class-id masks because compression can corrupt pixel labels.
- RGB masks are converted to class ids using the project palette.

