# 第三步说明：RescueNet 风格语义分割 Mask 接入与环境风险融合

## 本步骤新增内容

第三步在现有 Gradio + YOLO 目标检测 + 风险排序 Demo 基础上，新增灾区语义分割层的接入能力：

- `app/segmentation_engine.py`
  - 读取 RescueNet 风格 segmentation mask。
  - 支持单通道 class id mask。
  - 支持 RGB 彩色 mask，并按 RescueNet 颜色映射转换为 class id。
  - 生成 segmentation overlay。
  - 统计不同灾害区域的面积占比。
  - 计算单个检测目标周边的环境风险上下文。

- `app/environment_risk.py`
  - 定义环境风险类别。
  - 定义环境风险分数。
  - 提供环境风险中文描述。

- 修改 `app/risk_engine.py`
  - 没有 segmentation mask 时，自动回退第二步评分逻辑。
  - 有 segmentation mask 时，将环境风险分数融入最终风险评分。

- 修改 `app/priority_ranker.py`
  - 新增可选参数 `segmentation_mask`。
  - ranking table 中加入环境风险分数和目标周边主导环境。

- 修改 `app/report_generator.py`
  - 报告加入语义分割摘要。
  - 如果未上传 segmentation mask，报告明确提示当前未接入语义分割结果。

- 修改 `app/app.py`
  - 图像检测页新增可选 `Optional Segmentation Mask Upload`。
  - 页面新增 segmentation overlay 和 segmentation summary 输出。

## RescueNet 在本项目中的定位

RescueNet 用于补充 YOLO 目标检测无法判断的环境信息，例如：

- `water` / 水域
- `road_blocked` / 道路阻断
- `major_damage` / 严重损毁建筑
- `destroyed_building` / 完全毁坏建筑
- `vehicle` / 车辆
- `tree` / 树木
- `road_clear` / 可通行道路

YOLO 负责回答“图像里有哪些救援目标”，RescueNet 风格 mask 负责回答“目标周边环境是否危险”。

## 是否使用预训练分割模型

当前没有接入可直接推理的 RescueNet 预训练分割模型。

对 RescueNet 仓库的检查结果：

- RescueNet 提供高分辨率 UAV 灾后语义分割数据集说明。
- 仓库包含 PSPNet、UNet、Segmenter 等训练和测试代码。
- 仓库中没有提交可直接推理的 `.pth`、`.pt`、`.ckpt` 或 `.onnx` 分割权重。
- 因此本步骤实现的是“RescueNet 风格 segmentation mask 接入与环境风险融合”，不是完整自动语义分割推理。

后续如果完成 RescueNet 分割模型训练，可以将模型推理输出的 mask 直接接入当前接口。

## RescueNet 类别映射

当前按 RescueNet 实验代码中的 11 类设置：

| Class ID | Class Name | 中文说明 | RGB Color |
| ---: | --- | --- | --- |
| 0 | `background` | 背景 | `(0, 0, 0)` |
| 1 | `water` | 水域 | `(61, 230, 250)` |
| 2 | `no_damage_building` | 无损建筑 | `(180, 120, 120)` |
| 3 | `minor_damage` | 轻微损毁建筑 | `(235, 255, 7)` |
| 4 | `major_damage` | 严重损毁建筑 | `(255, 184, 6)` |
| 5 | `destroyed_building` | 完全毁坏建筑 | `(255, 0, 0)` |
| 6 | `vehicle` | 车辆 | `(255, 0, 245)` |
| 7 | `road_clear` | 可通行道路 | `(140, 140, 140)` |
| 8 | `road_blocked` | 道路阻断 | `(160, 150, 20)` |
| 9 | `tree` | 树木 | `(4, 250, 7)` |
| 10 | `pool` | 水池/积水 | `(255, 235, 0)` |

## Mask 文件准备方式

当前支持两种 mask：

1. 单通道 class id mask
   - 文件格式：`.png` / `.jpg`
   - 每个像素值为类别 id：0 到 10。

2. RGB 彩色 mask
   - 文件格式：`.png` / `.jpg`
   - 每个像素颜色使用上表 RescueNet 颜色映射。
   - 系统会自动将颜色转换为 class id。

如果 mask 尺寸和原图尺寸不同，系统会使用最近邻插值自动 resize 到图像尺寸，保证类别 id 不被插值破坏。

## 环境风险评分规则

环境风险分数范围为 0 到 30。

高风险环境：

| Class | Score |
| --- | ---: |
| `destroyed_building` | 30 |
| `water` | 28 |
| `major_damage` | 26 |
| `road_blocked` | 24 |
| `pool` | 22 |

中风险环境：

| Class | Score |
| --- | ---: |
| `minor_damage` | 14 |
| `tree` | 12 |
| `vehicle` | 10 |

低风险环境：

| Class | Score |
| --- | ---: |
| `no_damage_building` | 4 |
| `road_clear` | 3 |
| `background` | 0 |

## 风险评分融合方式

没有 segmentation mask 时，使用第二步旧公式：

```text
risk_score = class_weight * 70 + confidence * 20 + area_weight * 10
```

有 segmentation mask 时，使用环境增强公式：

```text
base_target_score = class_weight * 55 + confidence * 15 + area_weight * 10
final_risk_score = base_target_score + environment_score
```

最终分数限制在 0 到 100。

风险等级保持不变：

| Score Range | Level |
| --- | --- |
| 0-40 | Low |
| 40-70 | Medium |
| 70-100 | High |

## 当前局限

- 当前不会自动从图像生成 segmentation mask。
- 当前 mask 需要用户上传，或者由后续训练好的 RescueNet 分割模型生成。
- 环境风险只基于目标 bbox 附近的局部 mask 区域，尚未考虑整图路径连通性。
- 当前还没有融合无人机高度、GPS、时间、灾害类型、道路网络等信息。
- 当前环境风险评分是规则模型，适合竞赛 Demo 和可解释原型，后续可根据真实救援数据校准。

## 下一步计划

第四步建议加入 A* 路径规划：

- 将 segmentation mask 转换为通行代价地图。
- 水域、道路阻断、完全毁坏建筑设置为高代价或不可通行。
- 可通行道路设置为低代价。
- 根据救援优先级目标，生成从起点到目标的建议救援路径。
- 在 Gradio 页面显示路径 overlay 和路径风险说明。
