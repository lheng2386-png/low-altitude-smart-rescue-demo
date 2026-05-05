"""Static validation roadmap for AeroRescue-AI.

This module records what is already an engineering workflow, what remains
demo/mock/lightweight, and which real model or experimental validations should
be completed next. It intentionally does not train models, download data, run
ODM, run YOLO, parse real thermal files, or call LLM APIs.
"""

from __future__ import annotations


CAPABILITY_LAYERS = [
    {
        "layer_key": "workflow_layer",
        "title_zh": "工程工作流闭环",
        "status": "completed",
        "items": [
            "S1-S9 任务工作流",
            "灵活入口：可从完整流程、外部地图或 S4 局部精查开始",
            "一键演示任务",
            "任务控制中心界面",
            "证据链台账",
            "Final Report 2.0 辅助决策报告",
        ],
        "note": "项目已经具备完整救援任务流程闭环，但这不等于所有 AI 模型都已经完成真实验证。",
    },
    {
        "layer_key": "lightweight_layer",
        "title_zh": "轻量演示 / Demo 能力",
        "status": "in_progress",
        "items": [
            "模拟或导入的检测结果",
            "演示掩码 / 用户上传掩码",
            "模拟热红外",
            "快速建图预览",
            "图像平面路径",
            "EC-TERP 个案评分",
        ],
        "note": "这些能力可用于流程演示，但不能被表述为真实模型推理、真实测温、真实 ODM 正射或 GPS 导航。",
    },
    {
        "layer_key": "real_validation_layer",
        "title_zh": "真实模型与实验验证",
        "status": "pending",
        "items": [
            "目标检测权重验证",
            "语义分割模型或掩码评估",
            "EC-TERP 消融实验",
            "路径规划对比实验",
            "真实 ODM 建图验证",
            "真实热红外文件解析测试",
        ],
        "note": "下一阶段需要用数据、权重、指标和实验报告证明关键 AI 能力。",
    },
]


VALIDATION_TASKS = [
    {
        "task_id": "mini_benchmark_dataset",
        "title_zh": "小型验证数据集骨架",
        "category": "dataset",
        "priority": "Must",
        "current_status": "pending",
        "current_state_note": "尚未建立统一 mini benchmark 目录和 manifest。",
        "target_deliverables": [
            "datasets/mini_benchmark/",
            "detection/images + labels + data.yaml",
            "segmentation/images + masks + class_map.json",
            "mapping/rgb_overlap_images",
            "thermal/simulated + radiometric_optional",
            "mission_cases/case_flood_001",
        ],
        "suggested_files": ["datasets/mini_benchmark/manifest.json", "datasets/mini_benchmark/README.md"],
        "success_criteria": [
            "至少包含 detection / segmentation / mapping / thermal / mission_cases 的目录结构",
            "有 manifest.json",
            "明确 demo / real / imported 数据来源",
        ],
        "truthfulness_boundary": "小型验证数据集只用于原型验证，不能表述为大规模真实灾害业务数据集。",
    },
    {
        "task_id": "detection_weight_validation",
        "title_zh": "目标检测权重验证",
        "category": "model_validation",
        "priority": "Must",
        "current_status": "pending",
        "current_state_note": "S4 当前支持 mock/imported candidates；真实权重指标仍待验证。",
        "target_deliverables": [
            "models/yolov11n/best.pt 或其他真实权重",
            "outputs/evaluation/detection_eval_report.json",
            "detection example visualizations",
        ],
        "suggested_files": ["app/services/yolo_detection_adapter.py", "outputs/evaluation/detection_eval_report.json"],
        "success_criteria": [
            "能加载权重",
            "能单图推理",
            "能批量验证",
            "输出 Precision / Recall / mAP50 / mAP50-95",
        ],
        "truthfulness_boundary": "模拟或导入的检测结果不是真实模型推理结果。AI 检测到的人只能称为候选目标，不能称为已确认平民。",
    },
    {
        "task_id": "segmentation_validation",
        "title_zh": "语义分割模型或 Mask 验证",
        "category": "model_validation",
        "priority": "Must",
        "current_status": "pending",
        "current_state_note": "S2/S3 可使用 uploaded/demo mask；自动分割 checkpoint 仍需指标验证。",
        "target_deliverables": [
            "segmentation_eval_report.json",
            "mIoU / class IoU / pixel accuracy",
            "RescueNet class mapping validation",
        ],
        "suggested_files": ["outputs/evaluation/segmentation_eval_report.json", "configs/rescuenet_class_map.json"],
        "success_criteria": [
            "uploaded/demo mask 来源被明确标注",
            "如果有 checkpoint，能输出自动分割结果",
            "如果没有 checkpoint，不能声称自动分割",
        ],
        "truthfulness_boundary": "上传掩码或演示掩码不等于自动语义分割模型结果。",
    },
    {
        "task_id": "ec_terp_ablation",
        "title_zh": "EC-TERP 消融实验",
        "category": "algorithm_validation",
        "priority": "Must",
        "current_status": "pending",
        "current_state_note": "S7 已能生成辅助排序；不同证据项影响仍需消融证明。",
        "target_deliverables": ["ec_terp_ablation_report.json", "ec_terp_ablation_table.md"],
        "suggested_files": ["outputs/evaluation/ec_terp_ablation_report.json", "docs/ec_terp_ablation_table.md"],
        "success_criteria": [
            "对比仅目标证据",
            "目标证据 + 环境风险",
            "目标证据 + 热红外辅助",
            "完整 EC-TERP",
            "展示不同证据如何影响优先级",
        ],
        "truthfulness_boundary": "EC-TERP 排序是辅助决策优先级建议，不是最终救援命令。",
    },
    {
        "task_id": "path_planning_comparison",
        "title_zh": "路径规划对比实验",
        "category": "algorithm_validation",
        "priority": "Must",
        "current_status": "pending",
        "current_state_note": "S8 可输出 image-plane route suggestion；baseline/risk-aware 对比仍需统一报告。",
        "target_deliverables": ["path_planning_comparison.json", "path_planning_comparison.md"],
        "suggested_files": ["outputs/evaluation/path_planning_comparison.json", "docs/path_planning_comparison.md"],
        "success_criteria": [
            "对比普通 A* 和风险感知 A*",
            "输出 path_length / total_cost / high_risk_exposure_ratio",
        ],
        "truthfulness_boundary": "图像平面路径不是 GPS 导航路线。",
    },
    {
        "task_id": "real_odm_validation",
        "title_zh": "真实 ODM 建图验证",
        "category": "mapping_validation",
        "priority": "High",
        "current_status": "pending",
        "current_state_note": "S1 支持 Fast Preview 和可选 ODM 调用；真实 ODM 样例包待补。",
        "target_deliverables": [
            "outputs/odm_real_test/",
            "odm_orthophoto.tif",
            "orthophoto_preview.jpg",
            "odm_run.log",
            "REAL_ODM_VALIDATION.md",
        ],
        "suggested_files": ["outputs/odm_real_test/odm_run.log", "REAL_ODM_VALIDATION.md"],
        "success_criteria": [
            "使用真实重叠航拍图",
            "生成真实 ODM 输出",
            "不把 Fast Preview 伪装成真实正射",
        ],
        "truthfulness_boundary": "快速预览不是真实 ODM 地理配准正射影像。",
    },
    {
        "task_id": "thermal_reality_check",
        "title_zh": "热红外真实能力验证",
        "category": "thermal_validation",
        "priority": "High",
        "current_status": "pending",
        "current_state_note": "S6 可绑定 simulated/imported thermal support；真实 radiometric 文件解析仍待验证。",
        "target_deliverables": [
            "THERMAL_REALITY_BOUNDARY.md",
            "thermal_parse_report.json",
            "optional radiometric sample parsing",
        ],
        "suggested_files": ["THERMAL_REALITY_BOUNDARY.md", "outputs/evaluation/thermal_parse_report.json"],
        "success_criteria": [
            "明确模拟热红外和真实辐射热红外文件的区别",
            "如果没有真实辐射热红外文件，不生成真实 temperature_matrix",
        ],
        "truthfulness_boundary": "模拟热红外不是真实测温；普通 RGB/JPG/PNG 图像不能提供真实 temperature_matrix。",
    },
    {
        "task_id": "yolo_adapter_real_inference",
        "title_zh": "S4 真实目标检测适配器接入",
        "category": "integration",
        "priority": "High",
        "current_status": "pending",
        "current_state_note": "当前 S4 service 只标准化 imported/mock detections；真实推理适配器待补。",
        "target_deliverables": ["app/services/yolo_detection_adapter.py", "tests/smoke_test_yolo_adapter_no_weights.py"],
        "suggested_files": ["app/services/yolo_detection_adapter.py", "tests/smoke_test_yolo_adapter_no_weights.py"],
        "success_criteria": [
            "有权重时能真实推理",
            "无权重时返回不可用，不崩溃，不伪造检测",
        ],
        "truthfulness_boundary": "模型权重缺失时，系统不能伪造目标检测结果。",
    },
    {
        "task_id": "human_review_center",
        "title_zh": "人工复核中心 UI",
        "category": "product_ui",
        "priority": "High",
        "current_status": "pending",
        "current_state_note": "S5 已有 review_status 数据结构；集中复核 UI 待落地。",
        "target_deliverables": ["app/ui/human_review_panel.py"],
        "suggested_files": ["app/ui/human_review_panel.py"],
        "success_criteria": [
            "支持保留候选目标",
            "支持驳回误检",
            "支持标记需要二次巡查",
            "支持标记紧急复核",
            "支持添加复核备注",
        ],
        "truthfulness_boundary": "人工复核动作只辅助决策，不等于已经确认救援结果。",
    },
    {
        "task_id": "evidence_drilldown",
        "title_zh": "候选目标证据追溯",
        "category": "product_ui",
        "priority": "High",
        "current_status": "pending",
        "current_state_note": "Evidence Ledger 和 stage results 已有候选证据；按 candidate_id drill-down UI 待补。",
        "target_deliverables": ["app/ui/candidate_detail_panel.py"],
        "suggested_files": ["app/ui/candidate_detail_panel.py"],
        "success_criteria": [
            "可以按候选目标编号查看 S4-S8 证据链",
            "显示检测、裁剪图、热红外、EC-TERP、路径建议和证据编号",
        ],
        "truthfulness_boundary": "证据追溯只能汇总已有证据，不能编造缺失阶段输出。",
    },
    {
        "task_id": "model_registry",
        "title_zh": "模型注册表",
        "category": "infrastructure",
        "priority": "Medium",
        "current_status": "pending",
        "current_state_note": "模型、解析器和 ODM 环境状态仍分散在不同模块。",
        "target_deliverables": ["model_registry.json", "UI model status table"],
        "suggested_files": ["configs/model_registry.json", "app/ui/model_registry_panel.py"],
        "success_criteria": [
            "显示检测模型、分割模型、热红外解析能力、ODM 环境状态",
            "区分可用、缺失、已验证、未验证",
        ],
        "truthfulness_boundary": "模型被登记不代表已经验证；必须提供评估指标才能称为已验证。",
    },
    {
        "task_id": "evaluation_dashboard",
        "title_zh": "实验结果面板",
        "category": "product_ui",
        "priority": "Medium",
        "current_status": "pending",
        "current_state_note": "各类评估产物尚未统一汇总到 UI。",
        "target_deliverables": ["app/ui/evaluation_dashboard_panel.py"],
        "suggested_files": ["app/ui/evaluation_dashboard_panel.py"],
        "success_criteria": [
            "展示 detection mAP",
            "segmentation mIoU",
            "EC-TERP ablation",
            "path planning comparison",
            "ODM validation",
            "thermal parser status",
        ],
        "truthfulness_boundary": "实验结果面板必须区分原型小数据集结果和真实业务验证结果。",
    },
]


LIGHTWEIGHT_CAPABILITY_NOTES = [
    {
        "capability": "一键演示",
        "note": "可继续用 demo/mock 数据，但必须标注 workflow demonstration only。",
        "truthfulness_boundary": "演示数据只用于流程展示，不是实际灾害现场证据。",
    },
    {
        "capability": "模拟热红外",
        "note": "可用于流程演示，但不是测温。",
        "truthfulness_boundary": "模拟热红外不是真实测温。",
    },
    {
        "capability": "演示掩码",
        "note": "可用于 S2/S3 验证，但不是自动分割。",
        "truthfulness_boundary": "上传掩码或演示掩码不等于自动语义分割模型结果。",
    },
    {
        "capability": "图像平面路径",
        "note": "可用于路径风险演示，但不是 GPS 导航。",
        "truthfulness_boundary": "图像平面路径不是 GPS 导航路线。",
    },
    {
        "capability": "EC-TERP 个案分析",
        "note": "早期可用 case study，后续必须补 ablation。",
        "truthfulness_boundary": "EC-TERP 排序是辅助决策优先级建议，不是最终救援命令。",
    },
    {
        "capability": "LLM 报告助手",
        "note": "可作为报告润色，不是核心救援判断来源。",
        "truthfulness_boundary": "最终报告是 AI 辅助决策报告，不是最终救援结论。",
    },
    {
        "capability": "三维预览",
        "note": "可作为展示增强，不是测绘级三维成果。",
        "truthfulness_boundary": "三维预览不是测绘级三维重建成果。",
    },
]


NEXT_PHASE_ORDER = [
    "小型验证数据集骨架",
    "目标检测权重验证",
    "语义分割模型或掩码评估",
    "EC-TERP 消融实验",
    "路径规划对比实验",
    "真实 ODM 建图验证包",
    "热红外真实能力验证",
    "人工复核中心 + 候选目标证据追溯界面",
    "模型注册表",
    "实验结果面板",
]


ROADMAP_TRUTHFULNESS_REMINDERS = [
    "演示数据只用于流程展示，不是实际灾害现场证据。",
    "模拟或导入的检测结果不是真实模型推理结果。",
    "模拟热红外不是真实测温。",
    "快速预览不是真实 ODM 地理配准正射影像。",
    "上传掩码或演示掩码不等于自动语义分割模型结果。",
    "图像平面路径不是 GPS 导航路线。",
    "AI 检测到的人只能称为候选目标，不能称为已确认平民。",
    "最终报告是 AI 辅助决策报告，不是最终救援结论。",
]
