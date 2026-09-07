from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import xgboost as xgb
from sklearn.metrics import roc_auc_score

from .config import cache_path, resolve_path
from .dataset import load_cache, validate_cache
from .plotting import plot_binary_roc, plot_feature_importance, plot_training_metrics
from .reweighting import build_training_weights


def stable_event_hash(
    run: np.ndarray, lumi: np.ndarray, event: np.ndarray, sample_id: np.ndarray
) -> np.ndarray:
    with np.errstate(over="ignore"):
        value = (
            run.astype(np.uint64) * np.uint64(73856093)
            + lumi.astype(np.uint64) * np.uint64(19349663)
            + event.astype(np.uint64) * np.uint64(83492791)
            + sample_id.astype(np.uint64) * np.uint64(2654435761)
        )
        value ^= value >> np.uint64(29)
        value *= np.uint64(0x9E3779B185EBCA87)
        value ^= value >> np.uint64(32)
    return value


def split_masks(arrays: dict[str, np.ndarray], fold: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    held_out = arrays["fold_id"] == fold
    pool = ~held_out
    bucket = stable_event_hash(
        arrays["run"], arrays["luminosityBlock"], arrays["event"], arrays["sample_id"]
    ) % np.uint64(10)
    early_stop = pool & (bucket == 0)
    train = pool & ~early_stop
    if np.any(train & early_stop) or np.any(train & held_out) or np.any(early_stop & held_out):
        raise AssertionError("Fold masks overlap")
    if not np.all(train | early_stop | held_out):
        raise AssertionError("Fold masks do not cover all selected events")
    return train, early_stop, held_out


def deterministic_subsample(arrays: dict[str, np.ndarray], maximum: int) -> dict[str, np.ndarray]:
    if maximum <= 0 or len(arrays["event"]) <= maximum:
        return arrays
    hashes = stable_event_hash(
        arrays["run"], arrays["luminosityBlock"], arrays["event"], arrays["sample_id"]
    )
    selected = np.argpartition(hashes, maximum - 1)[:maximum]
    selected.sort()
    return {name: values[selected] for name, values in arrays.items()}


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)


def _class_weight_report(
    labels: np.ndarray,
    physics_weight: np.ndarray,
    training_weight: np.ndarray,
    mask: np.ndarray,
    class_names: list[str],
) -> dict[str, dict[str, float | int]]:
    return {
        name: {
            "events": int(np.sum(mask & (labels == index))),
            "sum_abs_physics_weight": float(np.sum(physics_weight[mask & (labels == index)])),
            "sum_training_weight": float(np.sum(training_weight[mask & (labels == index)])),
        }
        for index, name in enumerate(class_names)
    }


def train(
    config: dict[str, Any],
    *,
    output_dir: Path | None = None,
    folds: list[int] | None = None,
    max_events: int = 0,
    num_boost_round: int | None = None,
    early_stopping_rounds: int | None = None,
    device: str | None = None,
) -> dict[str, Any]:
    arrays, metadata = load_cache(cache_path(config))
    validate_cache(config, metadata)
    arrays = deterministic_subsample(arrays, max_events)
    outdir = output_dir or resolve_path(config, "output_dir")
    models_dir = outdir / "models"
    predictions_dir = outdir / "predictions"
    plots_dir = outdir / "plots"
    models_dir.mkdir(parents=True, exist_ok=True)
    predictions_dir.mkdir(parents=True, exist_ok=True)
    folds = list(range(5)) if folds is None else folds
    class_names = [item["name"] for item in config["classes"]]
    class_weights = [float(item["class_weight"]) for item in config["classes"]]
    X = arrays["X"]
    labels = arrays["label"].astype(np.int16)
    physics_weight = arrays["physics_weight"].astype(np.float64)
    if config["channel"] == "1L":
        targets = (labels == 0).astype(np.int8)
    else:
        targets = labels
    params = dict(config["xgboost"]["params"])
    if device:
        params["device"] = device
    rounds = int(num_boost_round or config["xgboost"]["num_boost_round"])
    patience = int(early_stopping_rounds or config["xgboost"]["early_stopping_rounds"])
    feature_names = list(metadata["feature_names"])
    if len(feature_names) != X.shape[1]:
        raise ValueError("Cache feature list and feature matrix width disagree")

    oof = np.full((len(labels), len(class_names)) if config["channel"] == "0L" else len(labels), np.nan)
    fold_summaries: list[dict[str, Any]] = []
    gain_total = {name: 0.0 for name in feature_names}
    for fold in folds:
        if fold not in range(5):
            raise ValueError(f"Invalid fold {fold}")
        train_mask, early_mask, held_mask = split_masks(arrays, fold)
        for split_name, split_mask in (("train", train_mask), ("early-stop", early_mask), ("held-out", held_mask)):
            for class_index, class_name in enumerate(class_names):
                if not np.any(split_mask & (labels == class_index)):
                    raise ValueError(f"Fold {fold} {split_name} has no {class_name} events")
        train_weight, train_reweight = build_training_weights(physics_weight, labels, train_mask, class_weights)
        early_weight, early_reweight = build_training_weights(physics_weight, labels, early_mask, class_weights)
        dtrain = xgb.DMatrix(
            X[train_mask], label=targets[train_mask], weight=train_weight[train_mask],
            feature_names=feature_names, missing=np.nan,
        )
        dearly = xgb.DMatrix(
            X[early_mask], label=targets[early_mask], weight=early_weight[early_mask],
            feature_names=feature_names, missing=np.nan,
        )
        dheld = xgb.DMatrix(
            X[held_mask], label=targets[held_mask], weight=physics_weight[held_mask],
            feature_names=feature_names, missing=np.nan,
        )
        history: dict[str, dict[str, list[float]]] = {}
        booster = xgb.train(
            params,
            dtrain,
            num_boost_round=rounds,
            evals=[(dtrain, "train"), (dearly, "early_stop")],
            early_stopping_rounds=patience,
            evals_result=history,
            verbose_eval=False,
        )
        best_iteration = int(booster.best_iteration)
        prediction = np.asarray(booster.predict(dheld, iteration_range=(0, best_iteration + 1)))
        oof[held_mask] = prediction
        model_path = models_dir / f"fold{fold}.json"
        booster.save_model(model_path)
        for feature, gain in booster.get_score(importance_type="gain").items():
            gain_total[feature] = gain_total.get(feature, 0.0) + float(gain)
        fold_summary: dict[str, Any] = {
            "fold": fold,
            "n_train": int(np.sum(train_mask)),
            "n_early_stop": int(np.sum(early_mask)),
            "n_held_out": int(np.sum(held_mask)),
            "best_iteration": best_iteration,
            "best_score": float(booster.best_score),
            "eval_metric": params["eval_metric"],
            "history": history,
            "train_reweighting": train_reweight,
            "early_stop_reweighting": early_reweight,
            "train_class_sums": _class_weight_report(labels, physics_weight, train_weight, train_mask, class_names),
        }
        if config["channel"] == "1L":
            fold_summary["held_out_auc_unweighted"] = float(roc_auc_score(targets[held_mask], prediction))
            fold_summary["held_out_auc_physics_weighted"] = float(
                roc_auc_score(targets[held_mask], prediction, sample_weight=physics_weight[held_mask])
            )
        fold_summaries.append(fold_summary)

    covered = np.isfinite(oof).all(axis=1) if oof.ndim == 2 else np.isfinite(oof)
    if set(folds) == set(range(5)) and not np.all(covered):
        raise AssertionError("Five-fold OOF prediction did not cover every selected event")
    gain_mean = {name: value / len(folds) for name, value in gain_total.items()}
    gain_sum = sum(gain_mean.values())
    importance = sorted(
        [
            {"feature": name, "gain": gain, "normalized_gain": gain / gain_sum if gain_sum else 0.0}
            for name, gain in gain_mean.items()
        ],
        key=lambda item: item["gain"],
        reverse=True,
    )
    _write_json(outdir / "feature_importance.json", importance)
    pepper_topology = {
        "minDR_b", "DR_Tarb1", "DR_Tarb2", "minDR_TarClean",
        "minDEta_TarClean", "minDPhi_TarClean",
    }
    relative_angles = {
        "dPhi_lepton_MET", "dPhi_TargetFatJet_MET", "dR_TargetFatJet_lepton"
    }
    grouped_gain: dict[str, float] = {
        "AK4 calibrated tag categories": 0.0,
        "absolute phi": 0.0,
        "Pepper topology": 0.0,
        "computed relative angles": 0.0,
        "kinematics and multiplicities": 0.0,
    }
    for item in importance:
        name = item["feature"]
        if name.startswith("CleanedJet_tag_"):
            group = "AK4 calibrated tag categories"
        elif name in pepper_topology:
            group = "Pepper topology"
        elif name in relative_angles:
            group = "computed relative angles"
        elif name == "MET_phi" or name == "Lepton_phi" or "_phi_" in name:
            group = "absolute phi"
        else:
            group = "kinematics and multiplicities"
        grouped_gain[group] += item["normalized_gain"]
    _write_json(outdir / "feature_importance_groups.json", grouped_gain)
    plot_feature_importance(importance, plots_dir / "feature_importance.png")
    plot_training_metrics(fold_summaries, plots_dir / "training_metrics.png")

    prediction_payload = {
        "run": arrays["run"],
        "luminosityBlock": arrays["luminosityBlock"],
        "event": arrays["event"],
        "sample_id": arrays["sample_id"],
        "label": labels,
        "fold_id": arrays["fold_id"],
        "physics_weight": physics_weight,
        "oof_score": oof,
    }
    np.savez_compressed(predictions_dir / "oof_predictions.npz", **prediction_payload)
    summary: dict[str, Any] = {
        "config": config["name"],
        "config_hash": metadata["config_hash"],
        "selection": config["selection"],
        "class_definitions": config["classes"],
        "samples": config["samples"],
        "features": feature_names,
        "fold_definition": config["folds"],
        "reweighting": config["reweighting"],
        "hyperparameters": params,
        "num_boost_round": rounds,
        "early_stopping_rounds": patience,
        "max_events": int(max_events),
        "folds": fold_summaries,
        "oof_coverage": float(np.mean(covered)),
        "note_feature_importance": "XGBoost gain is a split-improvement statistic, not a physics sensitivity percentage.",
    }
    if config["channel"] == "1L" and np.any(covered):
        auc_unweighted = float(roc_auc_score(targets[covered], oof[covered]))
        auc_weighted = float(roc_auc_score(targets[covered], oof[covered], sample_weight=physics_weight[covered]))
        summary["oof_auc_unweighted"] = auc_unweighted
        summary["oof_auc_physics_weighted"] = auc_weighted
        plot_binary_roc(targets[covered], oof[covered], physics_weight[covered], auc_unweighted, auc_weighted, plots_dir)
    _write_json(outdir / "summary.json", summary)
    return summary
