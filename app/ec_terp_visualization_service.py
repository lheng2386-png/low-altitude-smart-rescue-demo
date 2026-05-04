"""Visualization helpers for EC-TERP outputs.

The figures generated here are presentation aids for an assistive image-plane
priority algorithm. They are not rescue decisions, GPS routes, or real rescue
benchmarks.
"""

import json
from collections import Counter
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_RANKING_PATH = ROOT_DIR / "outputs" / "ec_terp" / "ec_terp_rankings.json"
DEFAULT_EVAL_DIR = ROOT_DIR / "outputs" / "ec_terp_evaluation"
DEFAULT_OUTPUT_DIR = ROOT_DIR / "outputs" / "ec_terp_visuals"

TRUTHFULNESS_NOTES = [
    "EC-TERP provides assistive image-plane priority ranking only.",
    "It does not replace human rescue command decisions.",
    "Image-plane path planning is not GPS navigation.",
    "Synthetic demo cases are not real rescue data.",
]

COMPONENT_LABELS = {
    "target_urgency": "T: Target urgency",
    "environment_risk": "E: Environment risk",
    "route_accessibility": "R: Route accessibility",
    "coverage_gap": "C: Coverage gap",
    "evidence_quality": "Q: Evidence quality",
    "uncertainty_penalty": "U: Uncertainty penalty",
}


class ECTERPVisualizationError(Exception):
    pass


def _read_json(path):
    path = Path(path)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"_json_error": True, "path": str(path), "error": str(exc)}


def _write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(path)


def _load_matplotlib():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        return plt, None
    except Exception as exc:
        return None, str(exc)


def load_ec_terp_ranking_payload(ranking_path=None):
    """Load EC-TERP runtime ranking payload without fabricating missing data."""
    path = Path(ranking_path) if ranking_path else DEFAULT_RANKING_PATH
    payload = _read_json(path)
    if payload is None:
        return {
            "success": False,
            "ranking_path": str(path),
            "rankings": [],
            "error_code": "RANKING_INPUT_MISSING",
            "message": "EC-TERP ranking input was not found.",
        }
    if isinstance(payload, dict) and payload.get("_json_error"):
        return {
            "success": False,
            "ranking_path": str(path),
            "rankings": [],
            "error_code": "RANKING_JSON_ERROR",
            "message": payload.get("error", "EC-TERP ranking JSON could not be parsed."),
        }
    rankings = payload.get("rankings", []) if isinstance(payload, dict) else []
    return {
        "success": bool(rankings),
        "ranking_path": str(path),
        "rankings": rankings if isinstance(rankings, list) else [],
        "payload": payload,
        "error_code": None if rankings else "RANKING_EMPTY",
        "message": f"Loaded {len(rankings) if isinstance(rankings, list) else 0} EC-TERP ranking items.",
    }


def _target_label(item):
    target_id = str(item.get("target_id") or item.get("id") or "unknown")
    target_type = str(item.get("target_type") or item.get("class_name") or "")
    return f"{target_id}\n{target_type}" if target_type else target_id


def _component_value(item, component):
    components = item.get("score_components") or {}
    if component in components:
        try:
            return float(components.get(component, 0.0))
        except Exception:
            return 0.0
    raw_components = item.get("components") or {}
    raw = raw_components.get(component) or {}
    try:
        return float(raw.get("score", 0.0))
    except Exception:
        return 0.0


def plot_topk_ranking(rankings, output_path, top_k=10, plt=None):
    """Generate an EC-TERP Top-K ranking bar chart."""
    if not rankings:
        raise ECTERPVisualizationError("No EC-TERP ranking items are available for Top-K plotting.")
    plt = plt or _load_matplotlib()[0]
    if plt is None:
        raise ECTERPVisualizationError("matplotlib is not available.")

    items = sorted(rankings, key=lambda item: int(item.get("rank", 9999)))[:top_k]
    labels = [_target_label(item) for item in items]
    scores = [float(item.get("ec_terp_score", 0.0)) for item in items]
    ranks = [int(item.get("rank", idx + 1)) for idx, item in enumerate(items)]

    fig, ax = plt.subplots(figsize=(max(8, len(items) * 1.3), 5))
    bars = ax.bar(labels, scores, color="#4C78A8")
    ax.set_ylim(0, 100)
    ax.set_ylabel("EC-TERP Score")
    ax.set_title("EC-TERP Top-K Assistive Priority Ranking")
    ax.text(
        0.5,
        -0.24,
        "Assistive priority ranking only; not an automatic rescue decision.",
        transform=ax.transAxes,
        ha="center",
        fontsize=9,
    )
    for bar, rank in zip(bars, ranks):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1, f"#{rank}", ha="center", fontsize=9)
    fig.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return str(output_path)


def plot_component_breakdown(rankings, output_path, top_k=8, plt=None):
    """Generate a component breakdown chart, showing U as a negative penalty."""
    if not rankings:
        raise ECTERPVisualizationError("No EC-TERP ranking items are available for component plotting.")
    plt = plt or _load_matplotlib()[0]
    if plt is None:
        raise ECTERPVisualizationError("matplotlib is not available.")

    items = sorted(rankings, key=lambda item: int(item.get("rank", 9999)))[:top_k]
    labels = [_target_label(item) for item in items]
    components = [
        "target_urgency",
        "environment_risk",
        "route_accessibility",
        "coverage_gap",
        "evidence_quality",
        "uncertainty_penalty",
    ]
    values = {component: [_component_value(item, component) for item in items] for component in components}

    fig, ax = plt.subplots(figsize=(max(9, len(items) * 1.4), 5.5))
    bottoms = [0.0] * len(items)
    colors = {
        "target_urgency": "#4C78A8",
        "environment_risk": "#F58518",
        "route_accessibility": "#54A24B",
        "coverage_gap": "#B279A2",
        "evidence_quality": "#72B7B2",
        "uncertainty_penalty": "#E45756",
    }
    for component in components[:-1]:
        ax.bar(labels, values[component], bottom=bottoms, label=COMPONENT_LABELS[component], color=colors[component], alpha=0.82)
        bottoms = [base + value for base, value in zip(bottoms, values[component])]
    penalty = [-value for value in values["uncertainty_penalty"]]
    ax.bar(labels, penalty, label="U: Uncertainty penalty (negative)", color=colors["uncertainty_penalty"], alpha=0.86)
    ax.axhline(0, color="#333333", linewidth=0.8)
    ax.set_ylabel("Component score (0-100; U shown negative)")
    ax.set_title("EC-TERP Score Component Breakdown")
    ax.legend(loc="upper right", fontsize=8)
    ax.text(
        0.5,
        -0.24,
        "U is a penalty term in EC-TERP and is subtracted from the final score.",
        transform=ax.transAxes,
        ha="center",
        fontsize=9,
    )
    fig.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return str(output_path)


def plot_evidence_quality_distribution(rankings, output_path, plt=None):
    """Generate evidence-level distribution chart for EC-TERP ranking items."""
    if not rankings:
        raise ECTERPVisualizationError("No EC-TERP ranking items are available for evidence plotting.")
    plt = plt or _load_matplotlib()[0]
    if plt is None:
        raise ECTERPVisualizationError("matplotlib is not available.")

    order = ["strong", "medium", "weak", "none"]
    counts = Counter(str(item.get("evidence_level", "none")) for item in rankings)
    values = [counts.get(level, 0) for level in order]

    fig, ax = plt.subplots(figsize=(7, 4.8))
    ax.bar(order, values, color=["#2F855A", "#4C78A8", "#F2C94C", "#8A8A8A"])
    ax.set_ylabel("Target count")
    ax.set_title("EC-TERP Evidence Quality Distribution")
    ax.text(
        0.5,
        -0.22,
        "Weak / simulated / preview evidence cannot support automatic rescue conclusions.",
        transform=ax.transAxes,
        ha="center",
        fontsize=9,
    )
    fig.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return str(output_path)


def _load_sensitivity_results(eval_dir=None):
    eval_dir = Path(eval_dir) if eval_dir else DEFAULT_EVAL_DIR
    full = _read_json(eval_dir / "sensitivity_results.json")
    if isinstance(full, dict) and isinstance(full.get("results"), list):
        return full
    summary = _read_json(eval_dir / "ec_terp_sensitivity_summary.json")
    if isinstance(summary, dict):
        return summary
    return None


def plot_sensitivity_summary(eval_dir, output_path, plt=None):
    """Generate a sensitivity summary chart if EC-TERP evaluation outputs exist."""
    payload = _load_sensitivity_results(eval_dir)
    if not payload:
        raise ECTERPVisualizationError("EC-TERP sensitivity input was not found.")
    plt = plt or _load_matplotlib()[0]
    if plt is None:
        raise ECTERPVisualizationError("matplotlib is not available.")

    if isinstance(payload.get("results"), list):
        by_weight = {}
        for item in payload.get("results", []):
            name = item.get("weight_name", "unknown")
            by_weight.setdefault(name, []).append(float(item.get("top3_stability", 0.0)))
        labels = list(by_weight.keys())
        values = [sum(v) / max(1, len(v)) for v in by_weight.values()]
    else:
        summary = payload.get("summary", {}) if isinstance(payload, dict) else {}
        labels = ["top1_stability", "top3_stability"]
        values = [
            float(summary.get("mean_top1_stability", 0.0)),
            float(summary.get("mean_top3_stability", 0.0)),
        ]

    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 1.2), 4.8))
    ax.bar(labels, values, color="#72B7B2")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Ranking stability")
    ax.set_title("EC-TERP Sensitivity Summary")
    ax.tick_params(axis="x", rotation=30)
    ax.text(
        0.5,
        -0.34,
        "Synthetic demo only unless replaced with a real annotated validation set.",
        transform=ax.transAxes,
        ha="center",
        fontsize=9,
    )
    fig.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return str(output_path)


def generate_ec_terp_visuals(ranking_path=None, eval_dir=None, output_dir=None):
    """Generate EC-TERP visualization figures and metadata.

    Missing ranking/evaluation inputs or missing matplotlib produce structured
    degraded/failed metadata instead of crashing the mission workflow.
    """
    ranking_path = Path(ranking_path) if ranking_path else DEFAULT_RANKING_PATH
    eval_dir = Path(eval_dir) if eval_dir else DEFAULT_EVAL_DIR
    output_dir = Path(output_dir) if output_dir else DEFAULT_OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    input_files = [str(ranking_path), str(eval_dir)]
    generated = []
    limitations = []
    status = "executed_success"
    ranking_payload = load_ec_terp_ranking_payload(ranking_path)
    rankings = ranking_payload.get("rankings", [])
    if not ranking_payload.get("success"):
        status = "degraded"
        limitations.append(ranking_payload.get("message", "EC-TERP ranking input is missing or empty."))

    plt, matplotlib_error = _load_matplotlib()
    if plt is None:
        status = "degraded"
        limitations.append(f"matplotlib is unavailable; PNG figures were not generated: {matplotlib_error}")
    else:
        figure_jobs = [
            ("topk_ranking_chart", plot_topk_ranking, output_dir / "ec_terp_topk_ranking.png"),
            ("component_breakdown_chart", plot_component_breakdown, output_dir / "ec_terp_component_breakdown.png"),
            ("evidence_quality_distribution_chart", plot_evidence_quality_distribution, output_dir / "ec_terp_evidence_quality_distribution.png"),
        ]
        for name, func, path in figure_jobs:
            try:
                generated.append({"name": name, "path": func(rankings, path, plt=plt)})
            except Exception as exc:
                status = "degraded"
                limitations.append(f"{name} was not generated: {exc}")
        try:
            generated.append(
                {
                    "name": "sensitivity_summary_chart",
                    "path": plot_sensitivity_summary(eval_dir, output_dir / "ec_terp_sensitivity_summary.png", plt=plt),
                }
            )
        except Exception as exc:
            status = "degraded" if generated else status
            limitations.append(f"sensitivity_summary_chart was not generated: {exc}")

    metadata = {
        "module": "ec_terp_visualization",
        "status": status,
        "input_files": input_files,
        "generated_figures": generated,
        "truthfulness_notes": TRUTHFULNESS_NOTES,
        "limitations": limitations,
        "human_review_required": True,
    }
    metadata_path = _write_json(output_dir / "ec_terp_visuals_metadata.json", metadata)
    metadata["metadata_path"] = metadata_path
    metadata["success"] = status in {"executed_success", "degraded"}
    return metadata


def build_visuals_map(metadata):
    """Return the UI-friendly visuals map from visualization metadata."""
    figures = {item.get("name"): item.get("path") for item in (metadata or {}).get("generated_figures", []) if isinstance(item, dict)}
    return {
        "topk_ranking_chart": figures.get("topk_ranking_chart"),
        "component_breakdown_chart": figures.get("component_breakdown_chart"),
        "evidence_quality_distribution_chart": figures.get("evidence_quality_distribution_chart"),
        "sensitivity_summary_chart": figures.get("sensitivity_summary_chart"),
    }
