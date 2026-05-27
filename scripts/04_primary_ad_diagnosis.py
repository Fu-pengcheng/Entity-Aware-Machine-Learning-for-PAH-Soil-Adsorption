from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

WORKFLOW_ROOT = Path(__file__).resolve().parents[1]
if str(WORKFLOW_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKFLOW_ROOT))

from src.ad_utils import (  # noqa: E402
    attach_quartile_and_weak_support,
    compute_foldwise_dk,
    correlation_distance_error,
    summarize_error_by_quartile,
)
from src.column_mapping import apply_primary_mapping  # noqa: E402
from src.io_utils import configure_output_roots, load_yaml, make_synthetic_primary, read_table, write_df, write_json  # noqa: E402
from src.models import build_model  # noqa: E402
from src.validation import evaluate_strict_loso  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Primary AD diagnosis with support distance d_k.")
    parser.add_argument("--config", default="configs/primary_pah.yaml")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def load_primary_df(cfg: dict, smoke: bool) -> pd.DataFrame:
    if smoke:
        return make_synthetic_primary(
            n_rows=int(cfg.get("smoke", {}).get("synthetic_rows", 120)),
            n_soils=int(cfg.get("smoke", {}).get("synthetic_soils", 8)),
            seed=45,
        )
    return apply_primary_mapping(
        read_table(cfg["input_data_path"], sheet_name=cfg.get("sheet_name", 0))
    ).dataframe


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)
    out_cfg = cfg.get("output_paths", {})
    configure_output_roots(
        primary_root=out_cfg.get("primary_root"),
        external_root=out_cfg.get("external_root"),
    )
    df = load_primary_df(cfg, args.smoke)

    target_col = cfg["target_col"]
    group_col = cfg["group_col"]
    feature_cols = list(cfg["layer3"])
    params = dict(cfg.get("model", {}).get("params", {}))
    model_name = cfg.get("model", {}).get("name", "XGBRegressor")
    model_factory = lambda seed: build_model(model_name, params=params, random_state=int(seed))

    _, _, pred_df = evaluate_strict_loso(
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
    pred_df["abs_error"] = (pred_df["y_true"] - pred_df["y_pred"]).abs()

    dk_df = compute_foldwise_dk(
        df=df,
        feature_cols=feature_cols,
        group_col=group_col,
        impute_strategy=cfg.get("preprocessing", {}).get("impute_strategy", "median"),
        k_neighbors=10,
        metric="euclidean",
        max_folds=cfg.get("smoke", {}).get("max_folds") if args.smoke else None,
    )

    merged = pred_df.merge(
        dk_df[["row_index", "fold_id", "d_k"]],
        on=["row_index", "fold_id"],
        how="inner",
    )
    merged = attach_quartile_and_weak_support(merged, distance_col="d_k")
    quartile_summary = summarize_error_by_quartile(merged, abs_error_col="abs_error")
    corr = correlation_distance_error(merged, distance_col="d_k", abs_error_col="abs_error")

    write_df(merged, "outputs/primary/predictions/04_primary_ad_diagnosis_samples.csv")
    write_df(quartile_summary, "outputs/primary/tables/04_primary_ad_diagnosis_quartiles.csv")
    write_json(corr, "outputs/primary/tables/04_primary_ad_diagnosis_correlations.json")
    write_json(
        {
            "script": "04_primary_ad_diagnosis.py",
            "smoke": bool(args.smoke),
            "n_rows": int(len(df)),
            "n_samples_ad": int(len(merged)),
        },
        "outputs/primary/logs/04_primary_ad_diagnosis_runmeta.json",
    )
    print("04 done.")


if __name__ == "__main__":
    main()
