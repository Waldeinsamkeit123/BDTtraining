from __future__ import annotations

import numpy as np


def build_training_weights(
    physics_weight: np.ndarray,
    labels: np.ndarray,
    mask: np.ndarray,
    class_weights: list[float],
) -> tuple[np.ndarray, dict[str, object]]:
    """Deterministically match the V4 one-bin flat class targets."""
    weights = np.zeros_like(physics_weight, dtype=np.float64)
    target = np.asarray(class_weights, dtype=np.float64)
    before: dict[str, float] = {}
    counts: dict[str, int] = {}
    for class_index, desired in enumerate(target):
        class_mask = mask & (labels == class_index)
        counts[str(class_index)] = int(np.sum(class_mask))
        class_sum = float(np.sum(physics_weight[class_mask]))
        before[str(class_index)] = class_sum
        if class_sum <= 0.0 or not np.any(class_mask):
            raise ValueError(f"Class {class_index} has zero events or zero absolute physics weight")
        weights[class_mask] = physics_weight[class_mask] * desired / class_sum
    normalization = float(np.sum(mask)) / float(np.sum(weights[mask]))
    weights[mask] *= normalization
    after = {
        str(index): float(np.sum(weights[mask & (labels == index)]))
        for index in range(len(target))
    }
    return weights, {
        "method": "deterministic one-bin flat importance weighting",
        "raw_event_count": counts,
        "sum_abs_physics_weight": before,
        "sum_training_weight": after,
        "class_weight_targets": target.tolist(),
    }

