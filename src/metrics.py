from __future__ import annotations

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def calc_regression_metrics(y_true, y_pred) -> dict:
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)

    try:
        r2 = float(r2_score(y_true_arr, y_pred_arr))
    except ValueError:
        r2 = float("nan")

    rmse = float(np.sqrt(mean_squared_error(y_true_arr, y_pred_arr)))
    mae = float(mean_absolute_error(y_true_arr, y_pred_arr))
    bias = float(np.mean(y_pred_arr - y_true_arr))
    return {
        "r2": r2,
        "rmse": rmse,
        "mae": mae,
        "bias": bias,
        "n_samples": int(len(y_true_arr)),
    }

