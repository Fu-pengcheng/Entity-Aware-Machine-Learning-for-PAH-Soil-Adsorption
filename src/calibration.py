from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd

from src.metrics import calc_regression_metrics


def _stable_hash_mod(value: object, modulo: int) -> int:
    digest = hashlib.sha256(str(value).encode("utf-8")).hexdigest()
    return int(digest[:16], 16) % int(modulo)


def few_shot_additive_calibration(
    pred_df: pd.DataFrame,
    group_col: str,
    target_col: str = "y_true",
    pred_col: str = "y_pred",
    m_values: list[int] | None = None,
    repeats: int = 30,
    seed: int = 42,
    min_test_after_cal: int = 5,
) -> dict[str, pd.DataFrame]:
    if m_values is None:
        m_values = [1, 3, 5, 10]

    out_rows: list[dict] = []
    working = pred_df.copy()
    working[group_col] = working[group_col].astype(str)
    if 0 not in m_values:
        all_m = [0] + list(m_values)
    else:
        all_m = sorted(m_values)

    for soil_id, soil_df in working.groupby(group_col, sort=True):
        n_soil = len(soil_df)
        y_true_s = soil_df[target_col].astype(float)
        y_pred_s = soil_df[pred_col].astype(float)

        for idx in soil_df.index:
            out_rows.append(
                {
                    group_col: soil_id,
                    "m": 0,
                    "repeat": 0,
                    "row_index": int(idx),
                    "role": "test",
                    "prediction_type": "zero_shot_all",
                    "is_calibration": False,
                    "is_test": True,
                    "bias": 0.0,
                    "y_true": float(y_true_s.loc[idx]),
                    "y_pred_source": float(y_pred_s.loc[idx]),
                    "y_pred": float(y_pred_s.loc[idx]),
                }
            )

        for m in all_m:
            if m == 0:
                continue
            if n_soil < (m + min_test_after_cal):
                continue
            idx_all = soil_df.index.to_numpy()
            for rep in range(int(repeats)):
                soil_seed = _stable_hash_mod(soil_id, modulo=997)
                rng = np.random.default_rng(int(seed + rep + 1000 * m + soil_seed))
                cal_idx = np.sort(rng.choice(idx_all, size=int(m), replace=False))
                cal_set = set(int(x) for x in cal_idx)
                test_idx = np.array([int(x) for x in idx_all if int(x) not in cal_set], dtype=int)

                residual_cal = y_true_s.loc[cal_idx].to_numpy(float) - y_pred_s.loc[cal_idx].to_numpy(float)
                bias = float(np.mean(residual_cal))

                for idx in cal_idx:
                    out_rows.append(
                        {
                            group_col: soil_id,
                            "m": int(m),
                            "repeat": int(rep),
                            "row_index": int(idx),
                            "role": "calibration",
                            "prediction_type": "calibration_sample",
                            "is_calibration": True,
                            "is_test": False,
                            "bias": bias,
                            "y_true": float(y_true_s.loc[idx]),
                            "y_pred_source": float(y_pred_s.loc[idx]),
                            "y_pred": float(y_pred_s.loc[idx] + bias),
                        }
                    )
                for idx in test_idx:
                    y_src = float(y_pred_s.loc[idx])
                    out_rows.append(
                        {
                            group_col: soil_id,
                            "m": int(m),
                            "repeat": int(rep),
                            "row_index": int(idx),
                            "role": "test",
                            "prediction_type": "matched_zero_shot",
                            "is_calibration": False,
                            "is_test": True,
                            "bias": 0.0,
                            "y_true": float(y_true_s.loc[idx]),
                            "y_pred_source": y_src,
                            "y_pred": y_src,
                        }
                    )
                    out_rows.append(
                        {
                            group_col: soil_id,
                            "m": int(m),
                            "repeat": int(rep),
                            "row_index": int(idx),
                            "role": "test",
                            "prediction_type": "calibrated",
                            "is_calibration": False,
                            "is_test": True,
                            "bias": bias,
                            "y_true": float(y_true_s.loc[idx]),
                            "y_pred_source": y_src,
                            "y_pred": float(y_src + bias),
                        }
                    )

    long_df = pd.DataFrame(out_rows)

    metric_rows = []
    test_df = long_df[(long_df["role"] == "test") & (long_df["is_test"] == True)].copy()
    for (m, repeat, ptype), sub in test_df.groupby(["m", "repeat", "prediction_type"], sort=True):
        mm = calc_regression_metrics(sub["y_true"], sub["y_pred"])
        metric_rows.append(
            {
                "m": int(m),
                "repeat": int(repeat),
                "prediction_type": str(ptype),
                "eligible_soils": int(sub[group_col].nunique()),
                **mm,
            }
        )
    metrics_by_repeat = pd.DataFrame(metric_rows).sort_values(["m", "prediction_type", "repeat"]).reset_index(drop=True)

    summary_rows = []
    for (m, ptype), sub in metrics_by_repeat.groupby(["m", "prediction_type"], sort=True):
        summary_rows.append(
            {
                "m": int(m),
                "prediction_type": str(ptype),
                "n_repeats": int(sub["repeat"].nunique()),
                "eligible_soils_mean": float(sub["eligible_soils"].mean()),
                "r2_mean": float(sub["r2"].mean()),
                "r2_std": float(sub["r2"].std(ddof=1)) if len(sub) > 1 else 0.0,
                "rmse_mean": float(sub["rmse"].mean()),
                "rmse_std": float(sub["rmse"].std(ddof=1)) if len(sub) > 1 else 0.0,
                "mae_mean": float(sub["mae"].mean()),
                "mae_std": float(sub["mae"].std(ddof=1)) if len(sub) > 1 else 0.0,
                "bias_mean": float(sub["bias"].mean()),
            }
        )
    summary_df = pd.DataFrame(summary_rows).sort_values(["m", "prediction_type"]).reset_index(drop=True)

    z = metrics_by_repeat[metrics_by_repeat["prediction_type"] == "matched_zero_shot"].copy()
    c = metrics_by_repeat[metrics_by_repeat["prediction_type"] == "calibrated"].copy()
    comparable = z.merge(c, on=["m", "repeat"], suffixes=("_zero", "_cal"))
    comp_rows = []
    for m, sub in comparable.groupby("m", sort=True):
        comp_rows.append(
            {
                "m": int(m),
                "n_repeats": int(sub["repeat"].nunique()),
                "zero_r2_mean": float(sub["r2_zero"].mean()),
                "calibrated_r2_mean": float(sub["r2_cal"].mean()),
                "delta_r2_mean": float((sub["r2_cal"] - sub["r2_zero"]).mean()),
                "zero_rmse_mean": float(sub["rmse_zero"].mean()),
                "calibrated_rmse_mean": float(sub["rmse_cal"].mean()),
                "delta_rmse_mean": float((sub["rmse_cal"] - sub["rmse_zero"]).mean()),
                "zero_mae_mean": float(sub["mae_zero"].mean()),
                "calibrated_mae_mean": float(sub["mae_cal"].mean()),
                "delta_mae_mean": float((sub["mae_cal"] - sub["mae_zero"]).mean()),
            }
        )
    comparable_summary = pd.DataFrame(comp_rows).sort_values("m").reset_index(drop=True)

    return {
        "predictions_long": long_df,
        "metrics_by_repeat": metrics_by_repeat,
        "metrics_summary": summary_df,
        "comparable_summary": comparable_summary,
    }
