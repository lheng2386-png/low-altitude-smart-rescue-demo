# EC-TERP 算法验证与灵敏度分析报告

## 一、实验目的
本报告用于验证 EC-TERP 相比 baseline TERP 的排序差异、权重扰动稳定性和各项贡献。报告由代码真实计算生成，不代表真实救援实测或公开数据集 SOTA。

## 二、实验设置
- case 数量：5
- case 类型：synthetic/demo 或用户提供 case
- 默认权重：α=0.30, β=0.25, γ=0.20, δ=0.15, λ=0.10, μ=0.15
- 指标：Top-1 Agreement、Top-3 Recall、Average Rank Shift
- 真实性边界：synthetic/demo cases 不是真实救援数据，权重是专家先验。

## 三、TERP vs EC-TERP 排名对比
### case_001_flood_road_blocked
- 场景：Flooded road with one civilian and one animal target.
- Baseline order：['T001', 'T002']
- EC-TERP order：['T001', 'T002']
- Human/demo order：['T001', 'T002']
- Metrics：{'top1_agreement': 1.0, 'top3_recall': 1.0, 'average_rank_shift_vs_baseline': 0.0, 'pred_order': ['T001', 'T002'], 'human_order': ['T001', 'T002'], 'baseline_order': ['T001', 'T002']}
- 解释：EC-TERP 在 baseline TERP 基础上加入 coverage gap、evidence quality 和 uncertainty penalty。本 synthetic/demo case 的 baseline order=['T001', 'T002']，EC-TERP order=['T001', 'T002']。

### case_002_damaged_building_human_candidate
- 场景：Damaged building with one YOLO civilian and one Transformer human_candidate.
- Baseline order：['T011', 'TR012']
- EC-TERP order：['T011', 'TR012']
- Human/demo order：['T011', 'TR012']
- Metrics：{'top1_agreement': 1.0, 'top3_recall': 1.0, 'average_rank_shift_vs_baseline': 0.0, 'pred_order': ['T011', 'TR012'], 'human_order': ['T011', 'TR012'], 'baseline_order': ['T011', 'TR012']}
- 解释：EC-TERP 在 baseline TERP 基础上加入 coverage gap、evidence quality 和 uncertainty penalty。本 synthetic/demo case 的 baseline order=['T011', 'TR012']，EC-TERP order=['T011', 'TR012']。

### case_003_low_confidence_target
- 场景：Low-confidence civilian candidate near moderate risk area.
- Baseline order：['T021', 'T022']
- EC-TERP order：['T022', 'T021']
- Human/demo order：['T021', 'T022']
- Metrics：{'top1_agreement': 0.0, 'top3_recall': 1.0, 'average_rank_shift_vs_baseline': 1.0, 'pred_order': ['T022', 'T021'], 'human_order': ['T021', 'T022'], 'baseline_order': ['T021', 'T022']}
- 解释：EC-TERP 在 baseline TERP 基础上加入 coverage gap、evidence quality 和 uncertainty penalty。本 synthetic/demo case 的 baseline order=['T021', 'T022']，EC-TERP order=['T022', 'T021']。

### case_004_path_unavailable_high_risk
- 场景：High-risk damaged area where one target is not path-accessible.
- Baseline order：['T032', 'T031']
- EC-TERP order：['T032', 'T031']
- Human/demo order：['T032', 'T031']
- Metrics：{'top1_agreement': 1.0, 'top3_recall': 1.0, 'average_rank_shift_vs_baseline': 0.0, 'pred_order': ['T032', 'T031'], 'human_order': ['T032', 'T031'], 'baseline_order': ['T032', 'T031']}
- 解释：EC-TERP 在 baseline TERP 基础上加入 coverage gap、evidence quality 和 uncertainty penalty。本 synthetic/demo case 的 baseline order=['T032', 'T031']，EC-TERP order=['T032', 'T031']。

### case_005_coverage_gap_case
- 场景：Multiple targets with a large unsearched high-priority area.
- Baseline order：['T041', 'TR043', 'T042']
- EC-TERP order：['T041', 'T042', 'TR043']
- Human/demo order：['T041', 'TR043', 'T042']
- Metrics：{'top1_agreement': 1.0, 'top3_recall': 1.0, 'average_rank_shift_vs_baseline': 0.6666666666666666, 'pred_order': ['T041', 'T042', 'TR043'], 'human_order': ['T041', 'TR043', 'T042'], 'baseline_order': ['T041', 'TR043', 'T042']}
- 解释：EC-TERP 在 baseline TERP 基础上加入 coverage gap、evidence quality 和 uncertainty penalty。本 synthetic/demo case 的 baseline order=['T041', 'TR043', 'T042']，EC-TERP order=['T041', 'T042', 'TR043']。

## 四、单因素灵敏度分析
- mean_top1_stability：1.0
- mean_top3_stability：1.0
- mean_average_rank_shift：0.0
- most_sensitive_weight：target_urgency_weight

## 五、随机权重稳定性分析
- n_trials：5
- top1_stability：1.0
- top3_stability：1.0
- average_rank_shift：0.0

## 六、消融实验
- most_impactful_ablation：without_evidence_quality
- without_environment_risk: mean_rank_shift=0.0
- without_route_accessibility: mean_rank_shift=0.0
- without_coverage_gap: mean_rank_shift=0.0
- without_evidence_quality: mean_rank_shift=0.2
- without_uncertainty_penalty: mean_rank_shift=0.2

- without_environment_risk: 去掉环境风险后，建筑损毁、水域和道路阻断对排序影响减弱，说明 E 项用于表达灾害环境压力。Top-1 未发生变化，平均排名变化 0.0。
- without_route_accessibility: 去掉路径可达性后，路径不可达目标可能被排得过高，说明 R 项对救援可执行性有约束作用。Top-1 未发生变化，平均排名变化 0.0。
- without_coverage_gap: 去掉覆盖缺口后，未搜索高优先级区域对排序约束减弱，说明 C 项用于提醒补充巡检需求。Top-1 未发生变化，平均排名变化 0.0。
- without_evidence_quality: 去掉证据质量项后，弱证据目标可能更容易上升，说明 Q 项有助于抑制低可信结果对排序的过度影响。Top-1 未发生变化，平均排名变化 0.0。
- without_uncertainty_penalty: 去掉不确定性惩罚后，低置信度或 human_candidate 目标可能排名上升，说明 U 项有助于保持人工复核约束。Top-1 未发生变化，平均排名变化 0.0。

## 七、结论
- EC-TERP 能利用证据质量和不确定性约束排序。
- 排序稳定性取决于样例、权重和输入证据。
- 当前只是小型 synthetic/user-provided 验证，不是大规模真实救援评测。

## 八、真实性边界
- 本报告不代表真实救援实测。
- 本报告不代表公开数据集 SOTA。
- EC-TERP 权重为专家先验，不是训练得到。
- 后续需要真实标注验证集校准和灵敏度分析。
- EC-TERP 是辅助优先级算法，不是自动救援决策。
- image-plane path 不是真实 GPS 导航。