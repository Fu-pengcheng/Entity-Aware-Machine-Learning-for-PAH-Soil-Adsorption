from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from src.preprocessing import fit_imputer, transform_imputer
from src.validation import generate_loso_splits


def compute_foldwise_dk(
    df: pd.DataFrame,
    feature_cols: list[str],
    group_col: str,
    impute_strategy: str = "median",
    k_neighbors: int = 10,
    metric: str = "euclidean",
    max_folds: int | None = None,
) -> pd.DataFrame:
    rows = []
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

        scaler = StandardScaler()
        x_train = scaler.fit_transform(train_proc[feature_cols].astype(float))
        x_test = scaler.transform(test_proc[feature_cols].astype(float))

        k = min(max(1, int(k_neighbors)), x_train.shape[0])
        nn = NearestNeighbors(n_neighbors=k, metric=metric)
        nn.fit(x_train)
        distances, _ = nn.kneighbors(x_test, return_distance=True)
        dk = distances.mean(axis=1)

        rows.append(
            pd.DataFrame(
                {
                    "row_index": test_df.index.values,
                    group_col: test_df[group_col].astype(str).values,
                    "fold_id": int(fold_id),
                    "held_out_group": held_out_group,
                    "d_k": dk,
                }
            )
        )
    return pd.concat(rows, axis=0, ignore_index=True)


def attach_quartile_and_weak_support(df: pd.DataFrame, distance_col: str = "d_k") -> pd.DataFrame:
    out = df.copy()
    out["quartile"] = pd.qcut(out[distance_col], q=4, labels=["Q1", "Q2", "Q3", "Q4"], duplicates="drop")
    threshold = float(out[distance_col].quantile(0.75))
    out["weak_support"] = out[distance_col] >= threshold
    out["weak_support_threshold_q75"] = threshold
    return out


def summarize_error_by_quartile(df: pd.DataFrame, abs_error_col: str = "abs_error") -> pd.DataFrame:
    out = (
        df.groupby("quartile", as_index=False)
        .agg(
            n=("quartile", "size"),
            mean_abs_error=(abs_error_col, "mean"),
            median_abs_error=(abs_error_col, "median"),
            max_abs_error=(abs_error_col, "max"),
        )
        .sort_values("quartile")
        .reset_index(drop=True)
    )
    return out


def correlation_distance_error(df: pd.DataFrame, distance_col: str = "d_k", abs_error_col: str = "abs_error") -> dict[str, float]:
    x = np.asarray(df[distance_col], dtype=float)
    y = np.asarray(df[abs_error_col], dtype=float)
    pear = pearsonr(x, y)
    spear = spearmanr(x, y)
    return {
        "pearson_r": float(pear.statistic),
        "pearson_p": float(pear.pvalue),
        "spearman_rho": float(spear.statistic),
        "spearman_p": float(spear.pvalue),
    }

