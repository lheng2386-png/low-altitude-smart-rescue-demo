# Mission Evidence Ledger Notes

## 为什么需要 Mission Evidence Ledger

随着 灾情感知及影响评估 的模块数量不断增加，只看“代码是否存在”已经不够。
Scanner 负责判断模块是否执行、是否成功、是否模拟、是否预览、是否真实模型输出或真实测量。
Ledger 进一步判断这些结果能否作为任务证据、能否支持决策、是否需要人工复核。

## Scanner 与 Ledger 的区别

- Scanner：判断模块运行状态。
- Ledger：判断证据价值。

Scanner 回答的是“发生了什么”。
Ledger 回答的是“这些结果能作为多强的证据”。

## 证据等级

- strong：真模型输出、真实测量、真实产物。
- medium：规则计算、image-plane 决策、uploaded mask 辅助判断、辅助模型输出。
- weak：模拟、预览、demo 结果。
- none：未执行、失败、reference only。

## 各模块典型判定

- YOLO detection：strong，但仍需人工复核。
- Transformer auxiliary detection：medium，human_candidate 不等于 confirmed civilian。
- Uploaded mask：medium，不是模型预测。
- Auto segmentation：strong，前提是真 checkpoint 输出。
- Simulated thermal：weak，不是真测温。
- Radiometric thermal：strong，前提是成功解析出 temperature_matrix。
- Fast preview orthomosaic：weak，不是真 ODM。
- Real ODM orthophoto：strong，前提是 odm_orthophoto.tif 存在。
- Risk-Aware A*：medium，图像平面参考路径，不是 GPS。
- Decision Fusion：medium，轻量 image-plane adaptation，不是完整 GIS。
- Registry：none/reference，不是运行证据。

## 真实性边界

- 不伪造模块执行结果。
- 不伪造模型指标。
- 不把模拟当真实。
- 不把预览当真实产物。
- 不把 reference 当当前能力。
- 不替代人工救援判断。

## 后续用途

- Final Report 2.0
- 证据链总览 UI
- 答辩解释
- 自动生成“已执行 / 未执行 / 模拟 / 真实输出”说明

## 比赛展示话术

“系统通过 Mission Evidence Ledger 将各模块运行状态进一步转化为证据等级。真模型输出和真实测量被标记为强证据，规则和图像平面辅助决策被标记为中等证据，模拟/预览结果被标记为弱证据，未执行或参考模块不作为当前任务证据。综合报告由证据链驱动生成，避免夸大系统能力。”
