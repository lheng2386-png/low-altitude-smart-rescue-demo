"""S4 source-level fusion for external SAR detection references.

This adapter fuses external repository/literature provenance into S4 detection
records without executing unreproduced code or fabricating detections.
"""

from __future__ import annotations

import json
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
REFERENCE_ADAPTER_VERSION = "s4-reference-fusion-v1"


REFERENCE_SOURCES = [
    {
        "source_key": "qazi_disaster_management",
        "kind": "repository",
        "repository_name": "qazi0/real-time-disaster-management",
        "repository_url": "https://github.com/qazi0/real-time-disaster-management",
        "external_path": "external_integrations/detection/qazi_disaster_management",
        "fusion_role": "disaster-management detection workflow reference",
        "label_policy": {
            "canonical_labels": ["disaster_context_reference"],
            "can_create_s4_targets": False,
            "candidate_class_mapping": {},
        },
    },
    {
        "source_key": "air_sar_detection",
        "kind": "repository",
        "repository_name": "Accenture/AIR",
        "repository_url": "https://github.com/Accenture/AIR",
        "external_path": "external_integrations/detection/air_sar_detection",
        "fusion_role": "search-and-rescue person detection reference",
        "label_policy": {
            "canonical_labels": ["person"],
            "can_create_s4_targets": False,
            "candidate_class_mapping": {"person": "human_candidate"},
        },
    },
    {
        "source_key": "bahmanyar_merkle_2023",
        "kind": "literature",
        "paper_title": "Saving Lives from Above: Person Detection in Disaster Response Using Deep Neural Networks",
        "authors": ["R. Bahmanyar", "N. Merkle"],
        "venue": "ISPRS Annals of the Photogrammetry, Remote Sensing and Spatial Information Sciences",
        "year": 2023,
        "doi": "10.5194/isprs-annals-X-1-W1-2023-343-2023",
        "reference_url": "https://doi.org/10.5194/isprs-annals-X-1-W1-2023-343-2023",
        "fusion_role": "aerial and UAV person-detection literature reference",
        "label_policy": {
            "canonical_labels": ["person", "human_candidate"],
            "can_create_s4_targets": False,
            "candidate_class_mapping": {"person": "human_candidate"},
        },
    },
]


def _read_json(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return {}


def _source_with_local_status(source, root_dir=None):
    root = Path(root_dir or ROOT_DIR)
    enriched = dict(source)
    external_path = source.get("external_path")
    if external_path:
        source_dir = root / external_path
        status = _read_json(source_dir / "status.json")
        expected_io = _read_json(source_dir / "expected_io_schema.json")
        enriched.update(
            {
                "local_status_path": str(source_dir / "status.json"),
                "expected_io_schema_path": str(source_dir / "expected_io_schema.json"),
                "current_state": status.get("current_state", "unknown"),
                "target_final_state": status.get("target_final_state", ""),
                "dependency_status": status.get("dependency_status", "unknown"),
                "checkpoint_status": status.get("checkpoint_status", "unknown"),
                "dataset_status": status.get("dataset_status", "unknown"),
                "truthfulness_limitations": status.get("truthfulness_limitations")
                or expected_io.get("truthfulness_boundary")
                or [],
                "expected_input_schema": expected_io.get("expected_input_schema", []),
                "expected_output_schema": expected_io.get("expected_output_schema", []),
            }
        )
    else:
        enriched.update(
            {
                "current_state": "literature_reference",
                "target_final_state": "future_training_or_adapter_reference",
                "dependency_status": "not_applicable",
                "checkpoint_status": "not_applicable",
                "dataset_status": "not_applicable",
                "truthfulness_limitations": [
                    "Literature references are not runtime model outputs.",
                    "Person detections from future adapters must remain human_candidate until reviewed.",
                    "No paper metrics are claimed unless reproduced on a local validation setup.",
                ],
                "expected_input_schema": ["aerial or UAV disaster response imagery after real adapter implementation"],
                "expected_output_schema": ["person/human_candidate detections with source metadata"],
            }
        )
    enriched["adapter_version"] = REFERENCE_ADAPTER_VERSION
    enriched["executable_now"] = False
    enriched["is_model_output"] = False
    enriched["human_review_required"] = True
    return enriched


def build_s4_reference_fusion_context(root_dir=None):
    """Build testable S4 provenance, label policy, and boundary metadata."""
    sources = [_source_with_local_status(source, root_dir=root_dir) for source in REFERENCE_SOURCES]
    person_reference_count = sum(
        1
        for source in sources
        if "person" in (source.get("label_policy", {}).get("canonical_labels") or [])
        or "human_candidate" in (source.get("label_policy", {}).get("canonical_labels") or [])
    )
    return {
        "adapter_version": REFERENCE_ADAPTER_VERSION,
        "fusion_level": "source_level_reference_fusion",
        "stage_key": "local_recon",
        "reference_count": len(sources),
        "person_detection_reference_count": person_reference_count,
        "sources": sources,
        "runtime_policy": {
            "active_model_outputs_only": ["yolo_rescue_targets", "transformer_rescuedet_argus", "dual_backend_compare"],
            "reference_sources_do_not_create_targets": True,
            "person_reference_maps_to": "human_candidate",
            "human_review_required": True,
            "can_enter_terp": "only current executable model targets can enter TERP",
            "can_enter_path_planning": "only current executable YOLO targets can enter path planning",
        },
        "truthfulness_note": (
            "S4 fuses qazi0, AIR, and Bahmanyar-Merkle 2023 as source-level reference metadata. "
            "They do not execute in runtime and do not create detections until reproduced adapters produce real local outputs."
        ),
    }


def annotate_targets_with_reference_policy(targets, reference_context=None):
    """Attach S4 reference-fusion policy to existing real/imported targets."""
    context = reference_context or build_s4_reference_fusion_context()
    source_keys = [source.get("source_key") for source in context.get("sources", [])]
    annotated = []
    for target in targets or []:
        item = dict(target)
        class_name = str(item.get("class_name") or "").lower()
        policy = {
            "reference_fusion_adapter": context.get("adapter_version"),
            "source_keys": source_keys,
            "did_reference_sources_create_target": False,
            "human_review_required": True,
        }
        if class_name in {"person", "people", "human_candidate"}:
            policy["person_reference_alignment"] = "aligned_with_air_and_bahmanyar_merkle_person_detection_scope"
            policy["candidate_class_policy"] = "treat as human_candidate until manually reviewed"
        item["s4_reference_policy"] = policy
        item["human_review_required"] = True
        annotated.append(item)
    return annotated


def enrich_detection_result_with_reference_fusion(result, root_dir=None):
    """Add source-level reference fusion metadata to a detection runtime result."""
    if not isinstance(result, dict):
        return result
    context = build_s4_reference_fusion_context(root_dir=root_dir)
    enriched = dict(result)
    enriched["s4_reference_fusion"] = context
    enriched["targets"] = annotate_targets_with_reference_policy(enriched.get("targets", []), context)
    if enriched.get("primary_result") and isinstance(enriched["primary_result"], dict):
        primary = dict(enriched["primary_result"])
        primary["targets"] = annotate_targets_with_reference_policy(primary.get("targets", []), context)
        primary["s4_reference_fusion"] = context
        enriched["primary_result"] = primary
    enriched["truthfulness_note"] = " ".join(
        part
        for part in [enriched.get("truthfulness_note", ""), context.get("truthfulness_note", "")]
        if part
    )
    return enriched


def format_s4_reference_fusion_summary(context):
    """Return compact Chinese Markdown for UI/report surfaces."""
    if not context:
        return "S4 源码级参考融合：unavailable"
    source_names = []
    for source in context.get("sources", []):
        source_names.append(source.get("repository_name") or source.get("paper_title") or source.get("source_key"))
    return (
        "S4 源码级参考融合：已接入 reference fusion adapter\n"
        f"- 融合层级：{context.get('fusion_level')}\n"
        f"- 参考来源：{'; '.join(source_names)}\n"
        f"- 人员检测参考数：{context.get('person_detection_reference_count', 0)}\n"
        "- 运行边界：这些参考源不生成当前检测框；只有本地可执行 YOLO / Transformer 结果会进入候选目标流程。\n"
        f"- 真实性说明：{context.get('truthfulness_note', '')}"
    )
