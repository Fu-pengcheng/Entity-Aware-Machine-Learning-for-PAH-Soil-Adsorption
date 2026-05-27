from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

WORKFLOW_ROOT = Path(__file__).resolve().parents[1]
if str(WORKFLOW_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKFLOW_ROOT))

from src.column_mapping import apply_primary_mapping  # noqa: E402
from src.error_decomposition import decompose_group_error, group_level_table  # noqa: E402
from src.io_utils import configure_output_roots, load_yaml, make_synthetic_primary, read_table, write_df, write_json  # noqa: E402
from src.models import build_model  # noqa: E402
from src.validation import evaluate_strict_loso  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Primary soil-level error decomposition.")
    parser.add_argument("--config", default="configs/primary_pah.yaml")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def load_primary_df(cfg: dict, smoke: bool) -> pd.DataFrame:
    if smoke:
        return make_synthetic_primary(
            n_rows=int(cfg.get("smoke", {}).get("synthetic_rows", 120)),
            n_soils=int(cfg.get("smoke", {}).get("synthetic_soils", 8)),
            seed=47,
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
    pred = pred_df[[group_col, "y_true", "y_pred"]].copy()

    summary = decompose_group_error(pred, group_col=group_col, true_col="y_true", pred_col="y_pred")
    by_soil = group_level_table(pred, group_col=group_col, true_col="y_true", pred_col="y_pred")

    write_json(summary, "outputs/primary/tables/06_primary_error_decomposition_summary.json")
    write_df(by_soil, "outputs/primary/tables/06_primary_error_decomposition_by_soil.csv")
    write_df(pred, "outputs/primary/predictions/06_primary_error_decomposition_input_predictions.csv")
    write_json(
        {
            "script": "06_primary_error_decomposition.py",
            "smoke": bool(args.smoke),
            "n_rows": int(len(df)),
            "n_soils": int(df[group_col].nunique()),
        },
        "outputs/primary/logs/06_primary_error_decomposition_runmeta.json",
    )
    print("06 done.")


if __name__ == "__main__":
    main()
