# AeroRescue-AI 综合救援报告

生成时间：2026-05-03 15:57:34

## 正射影像结果

```json
{
  "image_count": 1,
  "has_gps": false,
  "stitcher_attempted": false,
  "feature_matching_attempted": false,
  "stitch_success": false,
  "fallback_reason": "单张图像仅生成预览。",
  "output_path": "/Users/Apple/Documents/New project 2/urban-disaster-monitor/outputs/orthomosaic/orthomosaic_result.jpg"
}
```

## 热红外分析结果

```json
{
  "max_temperature": 80.0,
  "mean_temperature": 65.74,
  "hotspot_count": 158,
  "hotspot_area_ratio": 0.0509,
  "risk_level": "High",
  "risk_explanation": "疑似高温热点范围较大，建议优先复核火源、被困人员或危险热源。",
  "is_simulated_temperature": true
}
```

## 目标检测与综合决策摘要

目标检测与综合决策模块在 Gradio 页面内生成检测图、风险排序、TERP 排名、路径规划摘要与中文救援报告。若需要纳入最终报告，请先在“AI 灾情描述”Tab 中粘贴该报告文本。

## 三维重建结果

```json
该模块尚未执行。
```

## AI 灾情描述

# AI 灾情描述

生成时间：2026-05-03 15:57:34

## 场景概述

任务名称：测试任务

人工场景说明：测试场景

## 航测结果

正射影像生成模块尚未执行。

## 热红外风险

热红外分析模块尚未执行。

## 目标检测与救援优先级

检测报告测试

## 路径可达性

目标检测与综合决策报告中的 TERP、普通 A* 与风险感知 A* 对比结果作为路径可达性依据。

## 三维重建观察

三维重建模块尚未执行。

## 综合风险等级

当前综合风险等级：低-中

## 建议行动

- 优先复核 TERP 排名靠前目标。
- 若热红外热点明显，优先确认是否存在火源、人员或危险热源。
- 若正射/重建信息不足，应补充更多航测图像或视频。
- 路径建议仅为图像平面辅助参考，现场行动需结合真实道路、地形和指挥要求。


## 输出文件索引

- outputs/orthomosaic/orthomosaic_result.jpg
- outputs/orthomosaic/processing_log.json
- outputs/thermal/hotspot_mask.jpg
- outputs/thermal/thermal_heatmap.jpg
- outputs/thermal/thermal_overlay.jpg
- outputs/thermal/thermal_result.json
- outputs/detection/：暂无输出文件
- outputs/reconstruction/：暂无输出文件
- outputs/reports/scene_description.md
