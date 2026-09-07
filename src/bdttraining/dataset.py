from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterator

import awkward as ak
import numpy as np
import uproot

from .config import cache_path, compute_config_hash, resolve_path, source_signature
from .features import PEPPER_TOPOLOGY_FEATURES, build_feature_matrix, feature_specs, required_branches


def discover_sources(config: dict[str, Any]) -> list[tuple[int, str, Path]]:
    data_dir = resolve_path(config, "data_dir")
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Input directory does not exist: {data_dir}")
    sources: list[tuple[int, str, Path]] = []
    matched: dict[Path, str] = {}
    for sample_id, sample in enumerate(config["samples"]):
        template = str(sample["file"])
        exact = data_dir / template
        pattern = template.replace("_merged.root", "_merged_*.root")
        paths = sorted(set(([exact] if exact.is_file() else []) + list(data_dir.glob(pattern))))
        if not paths:
            raise FileNotFoundError(
                f"Expected sample '{sample['name']}' is missing: {data_dir / pattern}"
            )
        for path in paths:
            resolved = path.resolve()
            if resolved in matched:
                raise ValueError(
                    f"Input file matched two sample groups: {resolved} "
                    f"({matched[resolved]} and {sample['name']})"
                )
            matched[resolved] = sample["name"]
            sources.append((sample_id, sample["name"], resolved))
    return sources


def inspect_schema(config: dict[str, Any], sources: list[tuple[int, str, Path]]) -> dict[str, Any]:
    required = required_branches(config["channel"])
    files: list[dict[str, Any]] = []
    reference_types: dict[str, str] | None = None
    for _, sample, path in sources:
        try:
            with uproot.open(path) as root_file:
                if "Events" not in root_file:
                    raise ValueError("missing Events tree")
                tree = root_file["Events"]
                if tree.num_entries <= 0:
                    raise ValueError("Events tree has zero entries")
                missing = sorted(required - set(tree.keys()))
                if missing:
                    raise ValueError("missing required branches: " + ", ".join(missing))
                types = {name: tree[name].typename for name in sorted(required)}
                if reference_types is None:
                    reference_types = types
                else:
                    incompatible = [
                        name
                        for name in required
                        if _coarse_type(types[name]) != _coarse_type(reference_types[name])
                    ]
                    if incompatible:
                        raise ValueError("incompatible required branch types: " + ", ".join(sorted(incompatible)))
                files.append(
                    {
                        "sample": sample,
                        "path": str(path),
                        "events": int(tree.num_entries),
                        "branches": len(tree.keys()),
                        "size_bytes": path.stat().st_size,
                    }
                )
        except Exception as exc:
            raise RuntimeError(f"Cannot use {sample} input {path}: {exc}") from exc
    return {"files": files, "required_branches": sorted(required), "reference_types": reference_types}


def _coarse_type(typename: str) -> tuple[str, str]:
    shape = "jagged" if typename.endswith("[]") else "scalar"
    base = typename.removesuffix("[]").lower()
    if "float" in base or base == "double":
        kind = "float"
    elif "int" in base or base in {"char", "short", "long"}:
        kind = "integer"
    elif "bool" in base:
        kind = "bool"
    else:
        kind = base
    return shape, kind


def selection_mask(arrays: dict[str, ak.Array], channel: str) -> np.ndarray:
    njet = np.asarray(arrays["nCleanedJet"])
    nfat = np.asarray(arrays["nTargetFatJet"])
    tags = arrays["CleanedJet_tag"]
    nb = np.asarray(ak.sum(tags >= 50, axis=1))
    nhf = np.asarray(ak.sum(tags >= 40, axis=1))
    if channel == "1L":
        return (njet >= 2) & (nb >= 1) & (nhf >= 1) & (nfat >= 1)
    return (
        (njet >= 4)
        & (nb >= 1)
        & (nhf >= 2)
        & (np.asarray(arrays["HT"]) >= 250)
        & (nfat >= 1)
    )


def label_masks(arrays: dict[str, ak.Array], channel: str) -> tuple[list[np.ndarray], dict[str, np.ndarray]]:
    higgs = np.asarray(arrays["higgs_decay"])
    zdecay = np.asarray(arrays["z_decay"])
    flavor = np.asarray(arrays["tt_hf_flavor"])
    count = np.asarray(arrays["tt_hf_count"])
    if channel == "0L":
        ngentop = np.asarray(arrays["n_gentop"])
        top = ngentop > 0
    else:
        top = np.ones_like(higgs, dtype=bool)

    z_higgs_requirement = (higgs == 0) if channel == "0L" else np.ones_like(higgs, dtype=bool)
    fine = {
        "ttHcc": (higgs == 4) & top,
        "ttHbb": (higgs == 5) & top,
        "ttZqq": (zdecay >= 1) & (zdecay <= 3) & z_higgs_requirement & top,
        "ttZcc": (zdecay == 4) & z_higgs_requirement & top,
        "ttZbb": (zdecay == 5) & z_higgs_requirement & top,
        "ttLF": (flavor == 0) & (higgs == 0) & (zdecay == 0) & top,
        "ttcj": (flavor == 4) & (count <= 1) & (higgs == 0) & (zdecay == 0) & top,
        "ttcc": (flavor == 4) & (count > 2) & (higgs == 0) & (zdecay == 0) & top,
        "tt2c": (flavor == 4) & (count == 2) & (higgs == 0) & (zdecay == 0) & top,
        "ttbj": (flavor == 5) & (count <= 1) & (higgs == 0) & (zdecay == 0) & top,
        "ttbb": (flavor == 5) & (count > 2) & (higgs == 0) & (zdecay == 0) & top,
        "tt2b": (flavor == 5) & (count == 2) & (higgs == 0) & (zdecay == 0) & top,
    }
    tthz = np.logical_or.reduce([fine[name] for name in ("ttHcc", "ttHbb", "ttZqq", "ttZcc", "ttZbb")])
    ttjets = np.logical_or.reduce([fine[name] for name in ("ttLF", "ttcj", "ttcc", "tt2c", "ttbj", "ttbb", "tt2b")])
    if channel == "1L":
        return [tthz, ttjets], fine

    wdecay = np.asarray(arrays["w_decay"])
    ngentop = np.asarray(arrays["n_gentop"])
    notop = ngentop == 0
    wcq = notop & (wdecay == 4) & (higgs == 0)
    wqq = notop & (wdecay > 0) & (wdecay != 4) & (higgs == 0)
    zbb = notop & (zdecay == 5) & (higgs == 0) & (wdecay == 0)
    zcc = notop & (zdecay == 4) & (higgs == 0) & (wdecay == 0)
    zqq = notop & (zdecay >= 1) & (zdecay <= 3) & (higgs == 0) & (wdecay == 0)
    qcdraw = notop & (zdecay == 0) & (higgs == 0) & (wdecay == 0)
    fine.update({"wcq": wcq, "wqq": wqq, "zbb": zbb, "zcc": zcc, "zqq": zqq, "qcdraw": qcdraw})
    qcd = np.logical_or.reduce([qcdraw, wcq, wqq, zbb, zcc, zqq])
    return [tthz, ttjets, qcd], fine


def _iter_file(path: Path, branches: list[str]) -> Iterator[dict[str, ak.Array]]:
    yield from uproot.iterate(
        f"{path}:Events",
        expressions=branches,
        step_size="100 MB",
        library="ak",
        how=dict,
    )


def prepare_cache(config: dict[str, Any], force: bool = False) -> tuple[Path, dict[str, Any]]:
    sources = discover_sources(config)
    source_paths = [item[2] for item in sources]
    expected_hash = compute_config_hash(config, source_paths)
    destination = cache_path(config)
    if destination.exists() and not force:
        arrays, metadata = load_cache(destination)
        del arrays
        if metadata.get("config_hash") != expected_hash:
            raise RuntimeError(
                f"Prepared cache is stale: {destination}. Re-run prepare_inputs.py --force."
            )
        return destination, metadata

    schema = inspect_schema(config, sources)
    branches = sorted(required_branches(config["channel"]))
    include_phi = bool(config["features"].get("include_absolute_phi", True))
    stores: dict[str, list[np.ndarray]] = defaultdict(list)
    selected_by_sample: Counter[str] = Counter()
    raw_by_sample: Counter[str] = Counter()
    fine_counts: Counter[str] = Counter()
    multiplicity: Counter[str] = Counter()
    overlap = 0
    unlabeled = 0
    cleanedht_max_abs_diff = 0.0
    lepton_counts: Counter[str] = Counter()
    specs = feature_specs(config["channel"], include_phi)

    for sample_id, sample, path in sources:
        for arrays in _iter_file(path, branches):
            nrows = len(arrays["event"])
            raw_by_sample[sample] += nrows
            selected = selection_mask(arrays, config["channel"])
            masks, fine = label_masks(arrays, config["channel"])
            membership = np.sum(np.column_stack(masks), axis=1)
            overlap += int(np.sum(selected & (membership > 1)))
            unlabeled += int(np.sum(selected & (membership == 0)))
            if np.any(selected & (membership != 1)):
                raise ValueError(
                    f"Selected events in {sample}/{path.name} do not map to exactly one coarse class"
                )
            if not np.any(selected):
                continue
            selected_by_sample[sample] += int(np.sum(selected))
            selected_arrays = {name: values[selected] for name, values in arrays.items()}
            matrix, built_specs = build_feature_matrix(selected_arrays, config["channel"], include_phi)
            if [item.name for item in built_specs] != [item.name for item in specs]:
                raise AssertionError("Feature order changed while preparing the cache")
            labels = np.full(int(np.sum(selected)), -1, dtype=np.int8)
            for class_index, class_mask in enumerate(masks):
                labels[class_mask[selected]] = class_index
            for name, fine_mask in fine.items():
                fine_counts[name] += int(np.sum(selected & fine_mask))
            nfat = np.asarray(arrays["nTargetFatJet"])[selected]
            for value in nfat:
                multiplicity[">3" if value > 3 else str(int(value))] += 1
            if config["channel"] == "1L":
                for value in np.asarray(arrays["nLepton"])[selected]:
                    lepton_counts[str(int(value))] += 1
            derived_ht = np.asarray(ak.sum(selected_arrays["CleanedJet_pt"], axis=1), dtype=np.float64)
            root_ht = np.asarray(selected_arrays["CleanedHT"], dtype=np.float64)
            if derived_ht.size:
                cleanedht_max_abs_diff = max(cleanedht_max_abs_diff, float(np.max(np.abs(derived_ht - root_ht))))

            stores["X"].append(matrix)
            stores["run"].append(np.asarray(selected_arrays["run"], dtype=np.uint32))
            stores["luminosityBlock"].append(np.asarray(selected_arrays["luminosityBlock"], dtype=np.uint32))
            stores["event"].append(np.asarray(selected_arrays["event"], dtype=np.uint64))
            stores["sample_id"].append(np.full(len(labels), sample_id, dtype=np.int16))
            stores["label"].append(labels)
            stores["fold_id"].append((np.asarray(selected_arrays["event"], dtype=np.uint64) % 5).astype(np.int8))
            stores["physics_weight"].append(np.abs(np.asarray(selected_arrays["weight"], dtype=np.float64)))

    if not stores["X"]:
        raise ValueError("No selected events were found")
    payload = {name: np.concatenate(values, axis=0) for name, values in stores.items()}
    class_names = [item["name"] for item in config["classes"]]
    class_counts = {name: int(np.sum(payload["label"] == index)) for index, name in enumerate(class_names)}
    zero_classes = [name for name, count in class_counts.items() if count == 0]
    if zero_classes:
        raise ValueError("Configured classes have zero selected events: " + ", ".join(zero_classes))
    duplicate_keys = count_duplicate_keys(payload["run"], payload["luminosityBlock"], payload["event"])
    duplicate_sample_keys = count_duplicate_keys(
        payload["run"], payload["luminosityBlock"], payload["event"], payload["sample_id"]
    )
    metadata = {
        "config_hash": expected_hash,
        "config_path": config["_config_path"],
        "channel": config["channel"],
        "selection": config["selection"],
        "source_files": source_signature(source_paths),
        "schema": schema,
        "feature_names": [item.name for item in specs],
        "feature_specs": [item.__dict__ for item in specs],
        "sample_names": [item["name"] for item in config["samples"]],
        "raw_events": int(sum(raw_by_sample.values())),
        "selected_events": int(payload["X"].shape[0]),
        "raw_by_sample": dict(raw_by_sample),
        "selected_by_sample": dict(selected_by_sample),
        "class_counts": class_counts,
        "fine_class_counts": dict(fine_counts),
        "label_overlap": overlap,
        "unlabeled": unlabeled,
        "duplicate_event_keys": duplicate_keys,
        "duplicate_sample_qualified_event_keys": duplicate_sample_keys,
        "targetfatjet_multiplicity": dict(multiplicity),
        "lepton_multiplicity": dict(lepton_counts),
        "cleanedht_max_abs_difference_root_vs_sumpt": cleanedht_max_abs_diff,
        "fold_counts": {str(i): int(np.sum(payload["fold_id"] == i)) for i in range(5)},
    }
    payload["metadata_json"] = np.asarray(json.dumps(metadata, sort_keys=True))
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination, **payload)
    return destination, metadata


def load_cache(path: str | Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    with np.load(Path(path), allow_pickle=False) as source:
        arrays = {name: source[name] for name in source.files if name != "metadata_json"}
        metadata = json.loads(str(source["metadata_json"].item()))
    return arrays, metadata


def validate_cache(config: dict[str, Any], metadata: dict[str, Any]) -> None:
    sources = discover_sources(config)
    expected = compute_config_hash(config, [item[2] for item in sources])
    if metadata.get("config_hash") != expected:
        raise RuntimeError("Prepared cache does not match the current config or ROOT source signatures")


def count_duplicate_keys(
    run: np.ndarray,
    lumi: np.ndarray,
    event: np.ndarray,
    sample_id: np.ndarray | None = None,
) -> int:
    if sample_id is None:
        keys = np.rec.fromarrays([run, lumi, event], names="run,lumi,event")
    else:
        keys = np.rec.fromarrays(
            [sample_id, run, lumi, event], names="sample_id,run,lumi,event"
        )
    return int(len(keys) - len(np.unique(keys)))


def feature_audit(matrix: np.ndarray, names: list[str]) -> list[dict[str, Any]]:
    report = []
    for index, name in enumerate(names):
        values = np.asarray(matrix[:, index], dtype=np.float64)
        finite = np.isfinite(values)
        clean = values[finite]
        quantiles = np.quantile(clean, [0.01, 0.5, 0.99]) if clean.size else [np.nan] * 3
        report.append(
            {
                "feature": name,
                "count": int(values.size),
                "finite_fraction": float(np.mean(finite)),
                "nan_fraction": float(np.mean(~finite)),
                "min": float(np.min(clean)) if clean.size else None,
                "p01": float(quantiles[0]) if clean.size else None,
                "median": float(quantiles[1]) if clean.size else None,
                "p99": float(quantiles[2]) if clean.size else None,
                "max": float(np.max(clean)) if clean.size else None,
            }
        )
    return report


def topology_audit(matrix: np.ndarray, names: list[str]) -> dict[str, dict[str, Any]]:
    full = {item["feature"]: item for item in feature_audit(matrix, names)}
    return {name: full[name] for name in PEPPER_TOPOLOGY_FEATURES}
