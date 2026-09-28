"""09 — Mahalanobis vs Euclidean support-distance AD diagnosis (Reviewer 2, Comment 4).

Recomputes the applicability-domain support distance d_k under strict LOSO using
(a) Euclidean distance (main analysis) and
(b) Mahalanobis distance with a fold-internal inverse covariance matrix estimated
    on the training soils only (standardized feature space, Ledoit-Wolf shrinkage),
and compares the error stratification (quartile MAE/R2, Q4/Q1 ratio, correlation
with absolute error) between the two metrics. If the error gradient persists under
Mahalanobis, the weak-support risk pattern cannot be attributed to Euclidean
distance artifacts (scale sensitivity, outlier leverage, amplified narrow-range
features such as HOMO/LUMO).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

WORKFLOW_ROOT = Path(__file__).resolve().parents[1]
if str(WORKFLOW_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKFLOW_ROOT))

from src.ad_utils import attach_quartile_and_weak_support, correlation_distance_error  # noqa: E402
from src.column_mapping import apply_primary_mapping  # noqa: E402
from src.io_utils import configure_output_roots, load_yaml, make_synthetic_primary, read_table, write_df, write_json  # noqa: E402
from src.metrics import calc_regression_metrics  # noqa: E402
from src.models import build_model  # noqa: E402
from src.preprocessing import fit_imputer, transform_imputer  # noqa: E402
from src.validation import evaluate_strict_loso, generate_loso_splits  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Mahalanobis vs Euclidean support-distance AD diagnosis.")
    p.add_argument("--config", default="configs/primary_pah.yaml")
    p.add_argument("--smoke", action="store_true")
    return p.parse_args()


def foldwise_dk(df, feature_cols, group_col, impute_strategy, k_neighbors, method, max_folds=None):
    rows = []
    for fold_id, train_idx, test_idx, held_out_group in generate_loso_splits(
        df=df, group_col=group_col, max_folds=max_folds
    ):
        train_df = df.iloc[train_idx].copy()
        test_df = df.iloc[test_idx].copy()
        imputer = fit_imputer(train_df, feature_cols, strategy=impute_strategy)
        train_proc = transform_imputer(imputer, train_df, feature_cols)
        test_proc = transform_imputer(imputer, test_df, feature_cols)

        scaler = StandardScaler()
        x_train = scaler.fit_transform(train_proc[feature_cols].astype(float))
        x_test = scaler.transform(test_proc[feature_cols].astype(float))

        k = min(max(1, int(k_neighbors)), x_train.shape[0])
        if method == "euclidean":
            nn = NearestNeighbors(n_neighbors=k, metric="euclidean")
        elif method == "mahalanobis":
            lw = LedoitWolf().fit(x_train)  # train-fold only, shrinkage-regularized
            vi = np.linalg.pinv(lw.covariance_)
            nn = NearestNeighbors(n_neighbors=k, metric="mahalanobis", metric_params={"VI": vi})
        else:
            raise ValueError(method)
        nn.fit(x_train)
        distances, _ = nn.kneighbors(x_test, return_distance=True)
        dk = distances.mean(axis=1)
        rows.append(pd.DataFrame({
            "row_index": test_df.index.values,
            group_col: test_df[group_col].astype(str).values,
            "fold_id": int(fold_id),
            "d_k": dk,
        }))
    return pd.concat(rows, axis=0, ignore_index=True)


def quartile_metrics(merged: pd.DataFrame) -> pd.DataFrame:
    out = []
    for q, sub in merged.groupby("quartile", observed=True):
        mm = calc_regression_metrics(sub["y_true"], sub["y_pred"])
        out.append({"quartile": str(q), "n": int(len(sub)), **mm})
    return pd.DataFrame(out)


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
            seed=47,
        )
        max_folds = cfg.get("smoke", {}).get("max_folds")
    else:
        df = apply_primary_mapping(
            read_table(cfg["input_data_path"], sheet_name=cfg.get("sheet_name", 0))
        ).dataframe
        max_folds = None

    target_col = cfg["target_col"]
    group_col = cfg["group_col"]
    feature_cols = list(cfg["layer3"])
    impute_strategy = cfg.get("preprocessing", {}).get("impute_strategy", "median")
    params = dict(cfg.get("model", {}).get("params", {}))
    model_name = cfg.get("model", {}).get("name", "XGBRegressor")
    model_factory = lambda seed: build_model(model_name, params=params, random_state=int(seed))

    loso_summary, _, pred_df = evaluate_strict_loso(
        df=df, feature_cols=feature_cols, target_col=target_col, group_col=group_col,
        model_factory=model_factory, impute_strategy=impute_strategy, scale=False,
        base_seed=42, max_folds=max_folds,
    )
    print(f"LOSO summary: {loso_summary}")
    pred_df = pred_df.copy()
    pred_df["abs_error"] = (pred_df["y_true"] - pred_df["y_pred"]).abs()

    all_quartiles = []
    corr_rows = []
    for method in ["euclidean", "mahalanobis"]:
        dk = foldwise_dk(df, feature_cols, group_col, impute_strategy,
                         k_neighbors=10, method=method, max_folds=max_folds)
        dk = attach_quartile_and_weak_support(dk, "d_k")
        merged = dk.merge(pred_df[["row_index", "y_true", "y_pred", "abs_error"]], on="row_index", how="left")
        qm = quartile_metrics(merged)
        qm["method"] = method
        all_quartiles.append(qm)
        corr = correlation_distance_error(merged, "d_k", "abs_error")
        q1_mae = float(qm.loc[qm["quartile"] == "Q1", "mae"].iloc[0])
        q4_mae = float(qm.loc[qm["quartile"] == "Q4", "mae"].iloc[0])
        corr_rows.append({"method": method, **corr,
                          "q1_mae": q1_mae, "q4_mae": q4_mae,
                          "q4_q1_mae_ratio": q4_mae / q1_mae,
                          "q75_threshold": float(dk["weak_support_threshold_q75"].iloc[0])})
        write_df(dk, f"outputs/primary/tables/09_ad_dk_{method}.csv")

    quartiles_df = pd.concat(all_quartiles, ignore_index=True)
    corr_df = pd.DataFrame(corr_rows)
    write_df(quartiles_df, "outputs/primary/tables/09_ad_quartiles_by_method.csv")
    write_df(corr_df, "outputs/primary/tables/09_ad_method_comparison.csv")
    write_json({"script": "09_ad_mahalanobis_vs_euclidean.py", "smoke": bool(args.smoke),
                "loso_summary": loso_summary},
               "outputs/primary/logs/09_ad_runmeta.json")
    print(quartiles_df.to_string(index=False))
    print(corr_df.to_string(index=False))
    print("09 done.")


if __name__ == "__main__":
    main()
