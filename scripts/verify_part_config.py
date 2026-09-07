#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from bdttraining.config import load_config


def _samples_from_helper(path: Path, channel: str):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "SAMPLES" for target in node.targets
        ):
            return ast.literal_eval(node.value)[channel]
    raise ValueError(f"SAMPLES assignment not found in {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Statically compare BDT config with current ParT V4 sources.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    config = load_config(args.config)
    yaml_path = Path(config["part_yaml"])
    train_dir = yaml_path.parents[1]
    train_script = train_dir / f"train_{config['channel']}_boosted_v4.sh"
    helper = train_dir / "prepare_merged_inputs.py"
    with yaml_path.open(encoding="utf-8") as handle:
        part = yaml.safe_load(handle)

    expected_selection = {
        "1L": "(nCleanedJet>=2) & (CleanedJet_nbtag_loose>=1) & (CleanedJet_nhftag_loose>=1) & (nTargetFatJet>=1)",
        "0L": "(nCleanedJet>=4) & (CleanedJet_nbtag_loose>=1) & (CleanedJet_nhftag_loose>=2) & (HT>=250) & (nTargetFatJet>=1)",
    }[config["channel"]]
    expected_labels = [item["name"] for item in config["classes"]]
    expected_samples = [(item["name"], item["file"]) for item in config["samples"]]
    helper_samples = list(_samples_from_helper(helper, config["channel"]))
    inputs = part["inputs"]
    ak4_vars = [item[0] for item in inputs["jet_features"]["vars"]]
    ak8_vars = [item[0] for item in inputs["ak8_features"]["vars"]]
    result = {
        "yaml": str(yaml_path),
        "train_script": str(train_script),
        "helper": str(helper),
        "selection_exact": part["selection"].strip() == expected_selection,
        "labels_exact": part["labels"]["value"] == expected_labels,
        "sample_mapping_exact": helper_samples == expected_samples,
        "base_weight_exact": part["weights"]["reweight_basewgt"] == "np.abs(weight)",
        "reweight_method": part["weights"]["reweight_method"],
        "reweight_variables": part["weights"]["reweight_vars"],
        "class_weights": part["weights"]["class_weights"],
        "ak4_length": inputs["jet_features"]["length"],
        "ak4_feature_vars": ak4_vars,
        "ak8_length": inputs["ak8_features"]["length"],
        "ak8_feature_vars": ak8_vars,
        "ak8_mass_formula": part["new_variables"]["ak8_mass"],
        "lepton_collection_present": "lep_features" in inputs,
    }
    script_text = train_script.read_text(encoding="utf-8")
    result["fold_train_expression_present"] = bool(re.search(r'event%5!=\$\{fold\}', script_text))
    result["fold_held_out_expression_present"] = bool(re.search(r'event%5==\$\{fold\}', script_text))
    required_true = (
        "selection_exact", "labels_exact", "sample_mapping_exact", "base_weight_exact",
        "fold_train_expression_present", "fold_held_out_expression_present",
    )
    failures = [name for name in required_true if not result[name]]
    if failures:
        raise ValueError("ParT source comparison failed: " + ", ".join(failures))
    text = json.dumps(result, indent=2, sort_keys=True)
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
