"""08 — Few-shot Freundlich isotherm baseline (Reviewer 2, Comment 6).

For every eligible target soil, draw the SAME m calibration samples used by the
few-shot additive ML calibration, fit a simple Freundlich isotherm in log-log
form (lnQe = a + b * lnCe) by ordinary least squares on those m samples, and
predict the remaining held-out samples. Metrics are pooled over all test
samples per (m, repeat), exactly mirroring the matched zero-shot and calibrated
tracks in src/calibration.py, so the three tracks are directly comparable.

Note: with m = 1 the two-parameter Freundlich fit is not identifiable, so the
Freundlich track is only reported for m >= 2 (this is itself a relevant
practical limitation of the isotherm baseline).
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

WORKFLOW_ROOT = Path(__file__).resolve().parents[1]
if str(WORKFLOW_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKFLOW_ROOT))

from src.column_mapping import apply_primary_mapping  # noqa: E402
from src.io_utils import configure_output_roots, load_yaml, make_synthetic_primary, read_table, write_df, write_json  # noqa: E402
from src.metrics import calc_regression_metrics  # noqa: E402
from src.models import build_model  # noqa: E402
from src.validation import evaluate_strict_loso  # noqa: E402


def _stable_hash_mod(value: object, modulo: int) -> int:
    digest = hashlib.sha256(str(value).encode("utf-8")).hexdigest()
    return int(digest[:16], 16) % int(modulo)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Few-shot Freundlich baseline vs ML additive calibration.")
    p.add_argument("--config", default="configs/primary_pah.yaml")
    p.add_argument("--smoke", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)
    out_cfg = cfg.get("output_paths", {})
    configure_output_roots(
        primary_root=out_cfg.get("primary_root"),
        external_root=out_cfg.get("external_root"),
    )
    if args.smoke:
        df = make_synthetic_primary(
            n_rows=int(cfg.get("smoke", {}).get("synthetic_rows", 120)),
            n_soils=int(cfg.get("smoke", {}).get("synthetic_soils", 8)),
            seed=46,
        )
    else:
        df = apply_primary_mapping(
            read_table(cfg["input_data_path"], sheet_name=cfg.get("sheet_name", 0))
        ).dataframe

    target_col = cfg["target_col"]
    group_col = cfg["group_col"]
    feature_cols = list(cfg["layer3"])
    params = dict(cfg.get("model", {}).get("params", {}))
    model_name = cfg.get("model", {}).get("name", "XGBRegressor")
    model_factory = lambda seed: build_model(model_name, params=params, random_state=int(seed))

    loso_summary, _, pred_df = evaluate_strict_loso(
        df=df,
        feature_cols=feature_cols,
        target_col=target_col,
        group_col=group_col,
        model_factory=model_factory,
        impute_strategy=cfg.get("preprocessing", {}).get("impute_strategy", "median"),
        scale=False,
        base_seed=42,
        max_folds=cfg.get("smoke", {}).get("max_folds") if args.smoke else None,
    )
    print(f"LOSO summary: {loso_summary}")

    # Attach lnCe via the original row index.
    lnce_map = df["lnCe"].astype(float)
    pred_df = pred_df.copy()
    pred_df["lnCe"] = pred_df["row_index"].map(lnce_map)
    pred_df[group_col] = pred_df[group_col].astype(str)

    m_values = list(cfg.get("m_values", [1, 3, 5, 10]))
    repeats = int(cfg.get("smoke", {}).get("repeats", 3) if args.smoke else 30)
    min_test_after_cal = 5
    seed = 42

    long_rows = []
    eligible_soils = {int(m): 0 for m in m_values}
    for soil_id, soil_df in pred_df.groupby(group_col, sort=True):
        n_soil = len(soil_df)
        y_true_s = soil_df["y_true"].astype(float)
        y_pred_s = soil_df["y_pred"].astype(float)
        lnce_s = soil_df["lnCe"].astype(float)
        idx_all = soil_df.index.to_numpy()

        for m in m_values:
            if n_soil < (m + min_test_after_cal):
                continue
            eligible_soils[int(m)] += 1
            for rep in range(repeats):
                soil_seed = _stable_hash_mod(soil_id, modulo=997)
                rng = np.random.default_rng(int(seed + rep + 1000 * m + soil_seed))
                cal_idx = np.sort(rng.choice(idx_all, size=int(m), replace=False))
                cal_set = set(int(x) for x in cal_idx)
                test_idx = np.array([int(x) for x in idx_all if int(x) not in cal_set], dtype=int)

                yt = y_true_s.loc[test_idx].to_numpy()
                # Track 1: matched zero-shot
                for t, p in zip(yt, y_pred_s.loc[test_idx].to_numpy()):
                    long_rows.append((m, rep, "matched_zero_shot", t, p))
                # Track 2: ML additive calibration
                bias = float(np.mean(y_true_s.loc[cal_idx].to_numpy() - y_pred_s.loc[cal_idx].to_numpy()))
                for t, p in zip(yt, y_pred_s.loc[test_idx].to_numpy() + bias):
                    long_rows.append((m, rep, "ml_additive_calibrated", t, p))
                # Track 3: Freundlich isotherm fit on the same m samples (needs m >= 2)
                if m >= 2:
                    x_cal = lnce_s.loc[cal_idx].to_numpy()
                    y_cal = y_true_s.loc[cal_idx].to_numpy()
                    b, a = np.polyfit(x_cal, y_cal, 1)  # lnQe = b*lnCe + a
                    y_hat = a + b * lnce_s.loc[test_idx].to_numpy()
                    for t, p in zip(yt, y_hat):
                        long_rows.append((m, rep, "freundlich_baseline", t, p))

    long_df = pd.DataFrame(long_rows, columns=["m", "repeat", "prediction_type", "y_true", "y_pred"])
    metric_rows = []
    for (m, rep, ptype), sub in long_df.groupby(["m", "repeat", "prediction_type"], sort=True):
        mm = calc_regression_metrics(sub["y_true"], sub["y_pred"])
        metric_rows.append({"m": int(m), "repeat": int(rep), "prediction_type": ptype,
                            "n_test": int(len(sub)), **mm})
    metrics_by_repeat = pd.DataFrame(metric_rows).sort_values(["m", "prediction_type", "repeat"]).reset_index(drop=True)

    summary = (
        metrics_by_repeat.groupby(["m", "prediction_type"], sort=True)
        .agg(
            n_repeats=("repeat", "nunique"),
            r2_mean=("r2", "mean"), r2_std=("r2", "std"), r2_median=("r2", "median"),
            rmse_mean=("rmse", "mean"), rmse_std=("rmse", "std"),
            mae_mean=("mae", "mean"), mae_std=("mae", "std"),
        )
        .reset_index()
    )
    summary["eligible_soils"] = summary["m"].map(eligible_soils)

    write_df(metrics_by_repeat, "outputs/primary/tables/08_fewshot_freundlich_metrics_by_repeat.csv")
    write_df(summary, "outputs/primary/tables/08_fewshot_freundlich_summary.csv")
    write_json(
        {"script": "08_fewshot_freundlich_baseline.py", "smoke": bool(args.smoke),
         "repeats": repeats, "m_values": m_values, "loso_summary": loso_summary},
        "outputs/primary/logs/08_fewshot_freundlich_runmeta.json",
    )
    print(summary.to_string(index=False))
    print("08 done.")


if __name__ == "__main__":
    main()
