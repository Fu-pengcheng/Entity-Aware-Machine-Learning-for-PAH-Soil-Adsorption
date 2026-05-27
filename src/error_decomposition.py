from __future__ import annotations

import numpy as np
import pandas as pd


def decompose_group_error(
    df: pd.DataFrame,
    group_col: str,
    true_col: str = "y_true",
    pred_col: str = "y_pred",
) -> dict[str, float]:
    out = df.copy()
    out[group_col] = out[group_col].astype(str)
    out["error"] = out[true_col].astype(float) - out[pred_col].astype(float)
    out["group_mean_error"] = out.groupby(group_col)["error"].transform("mean")
    out["within_error"] = out["error"] - out["group_mean_error"]

    sse_total = float(np.sum(np.square(out["error"])))
    sse_soil = float(np.sum(np.square(out["group_mean_error"])))
    sse_within = float(np.sum(np.square(out["within_error"])))
    p_soil = float(sse_soil / sse_total) if sse_total > 0 else float("nan")
    p_within = float(sse_within / sse_total) if sse_total > 0 else float("nan")

    return {
        "SSE_total": sse_total,
        "SSE_soil": sse_soil,
        "SSE_within": sse_within,
        "P_soil": p_soil,
        "P_within": p_within,
        "n_samples": int(len(out)),
        "n_groups": int(out[group_col].nunique()),
    }


def group_level_table(
    df: pd.DataFrame,
    group_col: str,
    true_col: str = "y_true",
    pred_col: str = "y_pred",
) -> pd.DataFrame:
    out = df.copy()
    out["error"] = out[true_col].astype(float) - out[pred_col].astype(float)
    out["abs_error"] = out["error"].abs()
    table = (
        out.groupby(group_col, as_index=False)
        .agg(
            n=("error", "size"),
            mean_error=("error", "mean"),
            mae=("abs_error", "mean"),
            rmse=("error", lambda x: float(np.sqrt(np.mean(np.square(x))))),
            sse=("error", lambda x: float(np.sum(np.square(x)))),
        )
        .sort_values(group_col)
        .reset_index(drop=True)
    )
    return table

