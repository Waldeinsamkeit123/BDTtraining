from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve


def plot_binary_roc(
    labels: np.ndarray,
    scores: np.ndarray,
    weights: np.ndarray,
    unweighted_auc: float,
    weighted_auc: float,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, sample_weight, auc_value in (
        ("roc_unweighted.png", None, unweighted_auc),
        ("roc_weighted.png", weights, weighted_auc),
    ):
        fpr, tpr, _ = roc_curve(labels, scores, sample_weight=sample_weight)
        fig, ax = plt.subplots(figsize=(6.4, 5.2))
        ax.plot(fpr, tpr, lw=2, label=f"OOF AUC = {auc_value:.5f}")
        ax.plot([0, 1], [0, 1], color="0.6", linestyle="--")
        ax.set(xlabel="False positive rate", ylabel="True positive rate", xlim=(0, 1), ylim=(0, 1))
        ax.grid(alpha=0.25)
        ax.legend(loc="lower right")
        fig.tight_layout()
        fig.savefig(output_dir / name, dpi=160)
        plt.close(fig)


def plot_training_metrics(folds: list[dict[str, Any]], output: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    for fold in folds:
        history = fold["history"]
        metric = fold["eval_metric"]
        ax.plot(history["train"][metric], alpha=0.75, label=f"fold{fold['fold']} train")
        ax.plot(history["early_stop"][metric], linestyle="--", alpha=0.9, label=f"fold{fold['fold']} early")
    ax.set(xlabel="Boosting round", ylabel="Log loss")
    ax.grid(alpha=0.25)
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160)
    plt.close(fig)


def plot_feature_importance(importance: list[dict[str, Any]], output: Path, top_n: int = 35) -> None:
    shown = [item for item in importance if item["gain"] > 0][:top_n]
    shown.reverse()
    fig_height = max(5.0, 0.24 * len(shown) + 1.4)
    fig, ax = plt.subplots(figsize=(9.0, fig_height))
    ax.barh([item["feature"] for item in shown], [item["normalized_gain"] for item in shown])
    ax.set_xlabel("Normalized mean XGBoost gain")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160)
    plt.close(fig)

