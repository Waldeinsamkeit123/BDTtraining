from __future__ import annotations

from pathlib import Path

import numpy as np
import xgboost as xgb


def load_booster(path: str | Path) -> xgb.Booster:
    booster = xgb.Booster()
    booster.load_model(Path(path))
    return booster


def predict(
    booster: xgb.Booster,
    matrix: np.ndarray,
    feature_names: list[str],
    best_iteration: int | None = None,
) -> np.ndarray:
    data = xgb.DMatrix(matrix, feature_names=feature_names, missing=np.nan)
    kwargs = {} if best_iteration is None else {"iteration_range": (0, best_iteration + 1)}
    return np.asarray(booster.predict(data, **kwargs))

