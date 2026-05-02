from risk_engine import calculate_risk


def rank_targets(targets, image_width, image_height):
    ranked_targets = []

    for target in targets:
        risk = calculate_risk(target, image_width, image_height)
        ranked_targets.append(
            {
                "target_id": target["id"],
                "class_name": target["class_name"],
                "confidence": target["confidence"],
                "bbox": target["bbox"],
                "risk_score": risk["risk_score"],
                "risk_level": risk["risk_level"],
                "reason": risk["risk_reason"],
            }
        )

    ranked_targets.sort(key=lambda item: item["risk_score"], reverse=True)

    for index, target in enumerate(ranked_targets, start=1):
        target["rank"] = index

    return ranked_targets
