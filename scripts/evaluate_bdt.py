#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score


def main() -> None:
    parser = argparse.ArgumentParser(description="Recompute binary OOF metrics from saved predictions.")
    parser.add_argument("predictions", type=Path)
    args = parser.parse_args()
    with np.load(args.predictions, allow_pickle=False) as source:
        label = source["label"]
        target = (label == 0).astype(np.int8)
        score = source["oof_score"]
        weight = source["physics_weight"]
    finite = np.isfinite(score)
    result = {
        "events": int(np.sum(finite)),
        "coverage": float(np.mean(finite)),
        "oof_auc_unweighted": float(roc_auc_score(target[finite], score[finite])),
        "oof_auc_physics_weighted": float(
            roc_auc_score(target[finite], score[finite], sample_weight=weight[finite])
        ),
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

