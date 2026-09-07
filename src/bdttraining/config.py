from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CACHE_PIPELINE_FILES = ("config.py", "dataset.py", "features.py")


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).expanduser().resolve()
    with config_path.open(encoding="utf-8") as handle:
        config = json.load(handle)
    config["_config_path"] = str(config_path)
    validate_config(config)
    return config


def validate_config(config: dict[str, Any]) -> None:
    required = {
        "name",
        "channel",
        "data_dir",
        "output_dir",
        "selection",
        "classes",
        "samples",
        "features",
        "reweighting",
        "xgboost",
    }
    missing = sorted(required - set(config))
    if missing:
        raise ValueError(f"Config is missing required fields: {', '.join(missing)}")
    if config["channel"] not in {"1L", "0L"}:
        raise ValueError("channel must be 1L or 0L")
    if int(config["features"]["max_cleaned_jets"]) != 8:
        raise ValueError("The V4 same-information baseline requires 8 CleanedJet slots")
    if int(config["features"]["max_target_fatjets"]) != 3:
        raise ValueError("The V4 same-information baseline requires 3 TargetFatJet slots")
    class_names = [item["name"] for item in config["classes"]]
    if len(class_names) != len(set(class_names)):
        raise ValueError("Class names must be unique")


def resolve_path(config: dict[str, Any], key: str) -> Path:
    path = Path(config[key]).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def canonical_config(config: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in config.items() if not key.startswith("_")}


def source_signature(paths: list[Path]) -> list[dict[str, Any]]:
    return [
        {
            "path": str(path.resolve()),
            "size": path.stat().st_size,
            "mtime_ns": path.stat().st_mtime_ns,
        }
        for path in sorted(paths)
    ]


def compute_config_hash(config: dict[str, Any], paths: list[Path]) -> str:
    module_dir = Path(__file__).resolve().parent
    pipeline_hashes = {
        name: hashlib.sha256((module_dir / name).read_bytes()).hexdigest()
        for name in CACHE_PIPELINE_FILES
    }
    payload = {
        "config": canonical_config(config),
        "sources": source_signature(paths),
        "cache_pipeline_sha256": pipeline_hashes,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def cache_path(config: dict[str, Any]) -> Path:
    return resolve_path(config, "output_dir") / "prepared" / "events.npz"
