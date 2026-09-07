#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from bdttraining.config import load_config
from bdttraining.training import train


def main() -> None:
    parser = argparse.ArgumentParser(description="Train strict held-out-fold XGBoost models.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--fold", type=int, action="append", choices=range(5))
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--max-events", type=int, default=0)
    parser.add_argument("--num-boost-round", type=int)
    parser.add_argument("--early-stopping-rounds", type=int)
    parser.add_argument("--device", choices=("cpu", "cuda"))
    args = parser.parse_args()
    config = load_config(args.config)
    summary = train(
        config,
        output_dir=args.output_dir.resolve() if args.output_dir else None,
        folds=args.fold,
        max_events=args.max_events,
        num_boost_round=args.num_boost_round,
        early_stopping_rounds=args.early_stopping_rounds,
        device=args.device,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
