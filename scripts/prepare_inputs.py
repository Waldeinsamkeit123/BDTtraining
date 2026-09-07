#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from bdttraining.config import load_config, resolve_path
from bdttraining.dataset import feature_audit, load_cache, prepare_cache, topology_audit


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the selected same-information BDT NPZ cache.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    path, metadata = prepare_cache(config, force=args.force)
    arrays, metadata = load_cache(path)
    audit = feature_audit(arrays["X"], metadata["feature_names"])
    outdir = resolve_path(config, "output_dir") / "prepared"
    with (outdir / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)
    with (outdir / "feature_audit.json").open("w", encoding="utf-8") as handle:
        json.dump(audit, handle, indent=2, sort_keys=True)
    with (outdir / "feature_audit.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(audit[0]))
        writer.writeheader()
        writer.writerows(audit)
    audit_by_name = {item["feature"]: item for item in audit}
    feature_table = [{**spec, **audit_by_name[spec["name"]]} for spec in metadata["feature_specs"]]
    with (outdir / "feature_table.json").open("w", encoding="utf-8") as handle:
        json.dump(feature_table, handle, indent=2, sort_keys=True)
    with (outdir / "feature_table.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(feature_table[0]))
        writer.writeheader()
        writer.writerows(feature_table)
    with (outdir / "topology_audit.json").open("w", encoding="utf-8") as handle:
        json.dump(topology_audit(arrays["X"], metadata["feature_names"]), handle, indent=2, sort_keys=True)
    print(json.dumps(metadata, indent=2, sort_keys=True))
    print(f"Prepared cache: {path}")


if __name__ == "__main__":
    main()
