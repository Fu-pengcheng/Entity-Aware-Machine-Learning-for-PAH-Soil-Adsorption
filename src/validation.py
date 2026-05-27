from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.metrics import calc_regression_metrics
from src.preprocessing import fit_imputer, fit_scaler, transform_imputer, transform_scaler


def generate_loso_splits(df: pd.DataFrame, group_col: str, max_folds: int | None = None):
    if group_col not in df.columns:
        raise ValueError(f"group column not found: {group_col}")
    groups = sorted(df[group_col].astype(str).unique().tolist())
    for fold_id, held_out_group in enumerate(groups):
        test_mask = df[group_col].astype(str) == held_out_group
        train_idx = np.where(~test_mask.values)[0]
        test_idx = np.where(test_mask.values)[0]
        if len(train_idx) == 0 or len(test_idx) == 0:
            continue
        yield fold_id, train_idx, test_idx, held_out_group
        if max_folds is not None and (fold_id + 1) >= max_folds:
            break


def _check_feature_leakage(feature_cols: list[str], group_col: str) -> None:
    if group_col in feature_cols:
        raise ValueError(f"Leakage detected: group column {group_col} is in features.")


def evaluate_random_split(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    group_col: str,
    random_seeds: list[int],
    model_factory,
    impute_strategy: str = "median",
    scale: bool = False,
) -> tuple[dict, pd.DataFrame]:
    _check_feature_leakage(feature_cols, group_col)
    rows = []
    for seed in random_seeds:
        train_df, test_df = train_test_split(df, test_size=0.2, random_state=int(seed), shuffle=True)

        imputer = fit_imputer(train_df, feature_cols, strategy=impute_strategy)
        train_proc = transform_imputer(imputer, train_df, feature_cols)
        test_proc = transform_imputer(imputer, test_df, feature_cols)

        if scale:
            scaler = fit_scaler(train_proc, feature_cols)
            train_proc = transform_scaler(scaler, train_proc, feature_cols)
            test_proc = transform_scaler(scaler, test_proc, feature_cols)

        model = model_factory(int(seed))
        model.fit(train_proc[feature_cols].astype(float), train_proc[target_col].astype(float))
        pred = model.predict(test_proc[feature_cols].astype(float))

        m = calc_regression_metrics(test_proc[target_col].astype(float), pred)
        m.update({"seed": int(seed), "n_train": int(len(train_df)), "n_test": int(len(test_df))})
        rows.append(m)

    details = pd.DataFrame(rows)
    summary = {
        "R2": float(details["r2"].mean()),
        "R2_std": float(details["r2"].std(ddof=1)) if len(details) > 1 else float("nan"),
        "RMSE": float(details["rmse"].mean()),
        "RMSE_std": float(details["rmse"].std(ddof=1)) if len(details) > 1 else float("nan"),
        "MAE": float(details["mae"].mean()),
        "MAE_std": float(details["mae"].std(ddof=1)) if len(details) > 1 else float("nan"),
        "n_seeds": int(len(details)),
    }
    return summary, details


def evaluate_strict_loso(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    group_col: str,
    model_factory,
    impute_strategy: str = "median",
    scale: bool = False,
    base_seed: int = 42,
    max_folds: int | None = None,
) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    _check_feature_leakage(feature_cols, group_col)
    fold_rows: list[dict] = []
    pred_rows: list[pd.DataFrame] = []
    y_true_all = []
    y_pred_all = []

    for fold_id, train_idx, test_idx, held_out_group in generate_loso_splits(
        df=df,
        group_col=group_col,
        max_folds=max_folds,
    ):
        train_df = df.iloc[train_idx].copy()
        test_df = df.iloc[test_idx].copy()

        imputer = fit_imputer(train_df, feature_cols, strategy=impute_strategy)
        train_proc = transform_imputer(imputer, train_df, feature_cols)
        test_proc = transform_imputer(imputer, test_df, feature_cols)

        if scale:
            scaler = fit_scaler(train_proc, feature_cols)
            train_proc = transform_scaler(scaler, train_proc, feature_cols)
            test_proc = transform_scaler(scaler, test_proc, feature_cols)

        model_seed = int(base_seed + fold_id)
        model = model_factory(model_seed)
        model.fit(train_proc[feature_cols].astype(float), train_proc[target_col].astype(float))
        pred = np.asarray(model.predict(test_proc[feature_cols].astype(float)), dtype=float)
        y_true = test_proc[target_col].astype(float).to_numpy()

        m = calc_regression_metrics(y_true, pred)
        m.update(
            {
                "fold_id": int(fold_id),
                "held_out_group": held_out_group,
                "n_train": int(len(train_df)),
                "n_test": int(len(test_df)),
            }
        )
        fold_rows.append(m)

        fold_pred = pd.DataFrame(
            {
                "row_index": test_proc.index.values,
                group_col: test_proc[group_col].astype(str).values if group_col in test_proc.columns else held_out_group,
                "fold_id": int(fold_id),
                "held_out_group": held_out_group,
                "y_true": y_true,
                "y_pred": pred,
            },
            index=test_proc.index,
        )
        pred_rows.append(fold_pred)

        y_true_all.append(y_true)
        y_pred_all.append(pred)

    if not y_true_all:
        raise RuntimeError("No LOSO folds evaluated.")

    y_true_cat = np.concatenate(y_true_all)
    y_pred_cat = np.concatenate(y_pred_all)
    overall = calc_regression_metrics(y_true_cat, y_pred_cat)
    summary = {
        "R2": float(overall["r2"]),
        "RMSE": float(overall["rmse"]),
        "MAE": float(overall["mae"]),
        "n_folds": int(len(fold_rows)),
    }
    return summary, pd.DataFrame(fold_rows), pd.concat(pred_rows, axis=0).reset_index(drop=True)
