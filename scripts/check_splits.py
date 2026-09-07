#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from bdttraining.config import cache_path, load_config
from bdttraining.dataset import load_cache, validate_cache
from bdttraining.reweighting import build_training_weights
from bdttraining.training import split_masks


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate full-cache folds and deterministic reweighting.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    config = load_config(args.config)
    arrays, metadata = load_cache(cache_path(config))
    validate_cache(config, metadata)
    labels = arrays["label"].astype(np.int16)
    physics = arrays["physics_weight"].astype(np.float64)
    class_names = [item["name"] for item in config["classes"]]
    class_weights = [float(item["class_weight"]) for item in config["classes"]]
    result = {"fold_definition": config["folds"], "folds": []}
    for fold in range(5):
        train, early, held = split_masks(arrays, fold)
        train_weight, _ = build_training_weights(physics, labels, train, class_weights)
        fold_result = {"fold": fold, "splits": {}}
        for split_name, mask in (("train", train), ("early_stop", early), ("held_out", held)):
            fold_result["splits"][split_name] = {
                name: {
                    "events": int(np.sum(mask & (labels == index))),
                    "sum_abs_physics_weight": float(np.sum(physics[mask & (labels == index)])),
                    "sum_training_weight": (
                        float(np.sum(train_weight[mask & (labels == index)]))
                        if split_name == "train" else None
                    ),
                }
                for index, name in enumerate(class_names)
            }
        if not np.all(np.isfinite(train_weight[train])) or np.any(train_weight[train] <= 0):
            raise ValueError(f"Fold {fold} produced invalid training weights")
        result["folds"].append(fold_result)
    text = json.dumps(result, indent=2, sort_keys=True)
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

