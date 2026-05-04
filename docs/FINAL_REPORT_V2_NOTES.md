# Final Report 2.0 Notes

## 为什么需要 Final Report 2.0

项目模块越来越多，不能再靠手写方式拼接综合报告。
报告必须由 Mission Evidence Ledger 驱动生成，避免仅凭代码存在夸大能力。

## 数据来源

- Module Execution Status Scanner
- Mission Evidence Ledger
- outputs/ 本地运行产物和 JSON 元数据

## 报告章节结构

- 任务报告说明
- 证据链总览
- 主要模型输出证据
- 真实测量 / 真实产物证据
- 辅助决策证据
- 模拟 / 预览结果
- 执行失败 / 依赖缺失模块
- 未执行模块
- 参考 / 注册表模块
- 人工复核清单
- 综合救援辅助建议
- 真实性边界说明

## 证据驱动原则

- strong / medium / weak / none
- can_support_decision
- can_enter_final_report
- human_review_required

## 真实性边界

- 不根据代码存在判断模块成功。
- 不把模拟当真实。
- 不把预览当真实产物。
- 不把 reference 当当前能力。
- 不把 human_candidate 当 confirmed civilian。
- 不把 image-plane path 当 GPS。
- 不把 Decision Fusion 当完整 GIS / SAREnv / SKAI / InaSAFE / Fields2Cover。

## 输出文件

- `outputs/reports/final_report_v2.md`
- `outputs/reports/final_report_v2.html`
- `outputs/reports/final_report_v2.json`

`outputs/` 不提交 GitHub。

## 比赛展示话术

“Final Report 2.0 由 Mission Evidence Ledger 自动驱动生成。系统先扫描各模块真实运行产物，再为每个模块判定证据等级，最后按主要模型输出、真实测量、辅助决策、模拟预览、失败/未执行模块和人工复核清单组织报告。该机制避免仅凭代码存在夸大能力，使救援辅助报告更加可信。”
