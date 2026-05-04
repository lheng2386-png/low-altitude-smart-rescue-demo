# Module Execution Status Scanner Notes

## Why This Scanner Exists

Code existing in the repository does not mean the module has executed successfully.

Likewise, a module that executed once does not automatically mean it produced a real model output, a real measurement, or a fully validated result.

The scanner inspects `outputs/` artifacts and JSON metadata so later reports can state what actually ran.

## Status Definitions

- `not_run`
- `implemented_but_not_run`
- `executed_success`
- `executed_failed`
- `dependency_missing`
- `reference_only`
- `simulated_result`
- `real_model_output`
- `real_measurement`
- `preview_only`
- `unknown`

## Module-by-Module Evidence

### Detection

- `outputs/detection/detection_result.json`
- `outputs/detection/detection_metadata.json`
- `outputs/detection/dual_detection_result.json`
- `outputs/detection/dual_detection_consensus.json`

### Transformer Detection

- `outputs/detection/transformer_detection_result.json`
- `outputs/detection/transformer_detection_metadata.json`

### Segmentation

- `outputs/segmentation_inference/...`
- segmentation source metadata
- uploaded mask vs auto model output must be distinguished

### Thermal

- `outputs/thermal/thermal_result.json`
- `thermal_mode = simulated` means simulated_result
- `thermal_mode = radiometric` and `is_real_temperature_measurement = true` means real_measurement

### Orthomosaic

- `outputs/orthomosaic/processing_log.json`
- `outputs/odm/**/odm_orthophoto.tif`
- fast preview must not be treated as real ODM orthomosaic

### ODM

- `outputs/odm/**/odm_orthophoto.tif`
- `outputs/odm/**/odm_run.log`

### Decision Fusion

- `outputs/decision_fusion/decision_fusion_summary.json`
- optional lightweight artifacts such as damage or coverage score JSON

### Reconstruction

- `outputs/reconstruction/reconstruction_result.json`
- keyframe preview / feature preview / point cloud preview

### Scene Description

- `outputs/reports/scene_description.md`

### Report Export

- `outputs/reports/final_report.md`
- `outputs/reports/final_report.html`

### Registries

- `app/detection_backend_registry.py`
- `app/decision_reference_registry.py`

These are registry/management modules, not runtime results.

## Truthfulness Boundary

- Do not infer success from file existence alone.
- Do not treat simulated thermal as real temperature.
- Do not treat fast preview as real ODM orthomosaic.
- Do not treat uploaded mask as model prediction.
- Do not treat registry modules as runtime outputs.
- Do not treat image-plane paths as GPS navigation.

## Future Use

- Mission Evidence Ledger
- Comprehensive Report 2.0
- Presentation / defense evidence summary

## Competition Wording

系统通过 Module Execution Status Scanner 动态扫描 `outputs/` 运行产物和 JSON 元数据，判断各模块是否执行、是否成功、是否为真实模型输出或模拟/预览结果。该机制避免仅凭代码存在夸大系统能力，为后续证据链报告提供可信依据。
