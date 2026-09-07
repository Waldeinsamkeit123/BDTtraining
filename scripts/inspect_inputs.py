#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import awkward as ak
import numpy as np
import uproot

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from bdttraining.config import cache_path, load_config
from bdttraining.dataset import (
    count_duplicate_keys,
    discover_sources,
    inspect_schema,
    load_cache,
    selection_mask,
    validate_cache,
)
from bdttraining.features import required_branches


def _part_selected_identifiers(config, sources):
    branches = sorted(required_branches(config["channel"]))
    selected = {"run": [], "luminosityBlock": [], "event": [], "sample_id": []}
    selected_by_sample = Counter()
    topology_values = {
        name: []
        for name in (
            "minDR_b", "DR_Tarb1", "DR_Tarb2", "minDR_TarClean",
            "minDEta_TarClean", "minDPhi_TarClean",
        )
    }
    ordering = Counter()
    lepton_pdgid = Counter()
    raw_fatjet_multiplicity = Counter()
    selected_fatjet_multiplicity = Counter()
    raw_fatjet_max = 0
    selected_fatjet_max = 0
    raw = 0
    for sample_id, sample, path in sources:
        for arrays in uproot.iterate(
            f"{path}:Events", expressions=branches, step_size="100 MB", library="ak", how=dict
        ):
            raw += len(arrays["event"])
            raw_nfat = np.asarray(arrays["nTargetFatJet"])
            if raw_nfat.size:
                raw_fatjet_max = max(raw_fatjet_max, int(np.max(raw_nfat)))
            for value in raw_nfat:
                raw_fatjet_multiplicity[">3" if value > 3 else str(int(value))] += 1
            keep = selection_mask(arrays, config["channel"])
            selected_by_sample[sample] += int(np.sum(keep))
            selected["run"].append(np.asarray(arrays["run"])[keep].astype(np.uint32))
            selected["luminosityBlock"].append(
                np.asarray(arrays["luminosityBlock"])[keep].astype(np.uint32)
            )
            selected["event"].append(np.asarray(arrays["event"])[keep].astype(np.uint64))
            selected["sample_id"].append(np.full(int(np.sum(keep)), sample_id, dtype=np.int16))
            selected_nfat = raw_nfat[keep]
            if selected_nfat.size:
                selected_fatjet_max = max(selected_fatjet_max, int(np.max(selected_nfat)))
            for value in selected_nfat:
                selected_fatjet_multiplicity[">3" if value > 3 else str(int(value))] += 1
            for name in topology_values:
                topology_values[name].append(np.asarray(arrays[name])[keep].astype(np.float64))
            for collection in ("CleanedJet_pt", "TargetFatJet_pt"):
                values = arrays[collection][keep]
                ordered = np.asarray(ak.all(values[:, 1:] <= values[:, :-1], axis=1))
                ordering[f"{collection}_events"] += len(ordered)
                ordering[f"{collection}_nonincreasing"] += int(np.sum(ordered))
            if config["channel"] == "1L":
                for value in ak.to_list(ak.flatten(arrays["Lepton_pdgId"][keep])):
                    lepton_pdgid[str(int(value))] += 1
    raw_topology = {}
    for name, chunks in topology_values.items():
        values = np.concatenate(chunks)
        finite = np.isfinite(values)
        sentinel = finite & (values >= 900)
        usable = finite & ~sentinel
        raw_topology[name] = {
            "count": int(values.size),
            "finite_fraction_raw": float(np.mean(finite)),
            "nan_count_raw": int(np.sum(~finite)),
            "sentinel_ge_900_count": int(np.sum(sentinel)),
            "raw_finite_min": float(np.min(values[finite])) if np.any(finite) else None,
            "raw_finite_max": float(np.max(values[finite])) if np.any(finite) else None,
            "usable_min": float(np.min(values[usable])) if np.any(usable) else None,
            "usable_max": float(np.max(values[usable])) if np.any(usable) else None,
        }
    ordering_report = {
        collection: {
            "events": ordering[f"{collection}_events"],
            "nonincreasing_pt_events": ordering[f"{collection}_nonincreasing"],
            "nonincreasing_fraction": (
                ordering[f"{collection}_nonincreasing"] / ordering[f"{collection}_events"]
                if ordering[f"{collection}_events"] else None
            ),
        }
        for collection in ("CleanedJet_pt", "TargetFatJet_pt")
    }
    audit = {
        "topology_raw_values": raw_topology,
        "collection_order_observation": ordering_report,
        "lepton_pdgid_values": dict(lepton_pdgid),
        "targetfatjet_multiplicity_raw": {
            "counts": {key: int(raw_fatjet_multiplicity.get(key, 0)) for key in ("0", "1", "2", "3", ">3")},
            "max": raw_fatjet_max,
        },
        "targetfatjet_multiplicity_selected": {
            "counts": {key: int(selected_fatjet_multiplicity.get(key, 0)) for key in ("0", "1", "2", "3", ">3")},
            "max": selected_fatjet_max,
        },
    }
    return {name: np.concatenate(values) for name, values in selected.items()}, raw, selected_by_sample, audit


def _structured(run, lumi, event):
    return np.rec.fromarrays([run, lumi, event], names="run,lumi,event")


def _collision_sample_pairs(arrays, sample_names):
    keys = _structured(arrays["run"], arrays["luminosityBlock"], arrays["event"])
    order = np.argsort(keys, order=["run", "lumi", "event"])
    ordered = keys[order]
    duplicate = np.zeros(len(ordered), dtype=bool)
    duplicate[1:] |= ordered[1:] == ordered[:-1]
    duplicate[:-1] |= ordered[1:] == ordered[:-1]
    groups = defaultdict(set)
    for index in order[duplicate]:
        key = (
            int(arrays["run"][index]),
            int(arrays["luminosityBlock"][index]),
            int(arrays["event"][index]),
        )
        groups[key].add(sample_names[int(arrays["sample_id"][index])])
    pairs = Counter(tuple(sorted(value)) for value in groups.values())
    return {
        "collision_key_count": len(groups),
        "all_collisions_cross_sample": all(len(value) > 1 for value in groups.values()),
        "sample_combinations": [
            {"samples": list(samples), "keys": count}
            for samples, count in pairs.most_common()
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect ROOT schema and verify exact ParT/BDT event identity.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--schema-only", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    config = load_config(args.config)
    sources = discover_sources(config)
    schema = inspect_schema(config, sources)
    result = {
        "channel": config["channel"],
        "files": schema["files"],
        "required_topology_branches": {
            name: all(name in item for item in [schema["reference_types"]])
            for name in (
                "minDR_b", "DR_Tarb1", "DR_Tarb2", "minDR_TarClean",
                "minDEta_TarClean", "minDPhi_TarClean",
            )
        },
    }
    if not args.schema_only:
        arrays, metadata = load_cache(cache_path(config))
        validate_cache(config, metadata)
        part, raw, selected_by_sample, root_audit = _part_selected_identifiers(config, sources)
        part_keys = _structured(part["run"], part["luminosityBlock"], part["event"])
        bdt_keys = _structured(arrays["run"], arrays["luminosityBlock"], arrays["event"])
        part_unique = np.unique(part_keys)
        bdt_unique = np.unique(bdt_keys)
        missing = np.setdiff1d(part_unique, bdt_unique)
        extra = np.setdiff1d(bdt_unique, part_unique)
        fold_mismatch = int(np.sum(arrays["fold_id"] != (arrays["event"] % 5)))
        result.update(
            {
                "raw_events": raw,
                "part_selected_events": int(len(part_keys)),
                "bdt_selected_events": int(len(bdt_keys)),
                "selected_by_sample_rescan": dict(selected_by_sample),
                "missing_in_bdt": int(len(missing)),
                "extra_in_bdt": int(len(extra)),
                "part_duplicate_keys": count_duplicate_keys(part["run"], part["luminosityBlock"], part["event"]),
                "bdt_duplicate_keys": count_duplicate_keys(arrays["run"], arrays["luminosityBlock"], arrays["event"]),
                "bdt_sample_qualified_duplicate_keys": count_duplicate_keys(
                    arrays["run"], arrays["luminosityBlock"], arrays["event"], arrays["sample_id"]
                ),
                "unqualified_key_collision_details": _collision_sample_pairs(
                    arrays, metadata["sample_names"]
                ),
                "fold_mismatch": fold_mismatch,
                "exact_key_match": len(missing) == 0 and len(extra) == 0 and len(part_keys) == len(bdt_keys),
                "root_value_audit": root_audit,
            }
        )
    text = json.dumps(result, indent=2, sort_keys=True)
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
