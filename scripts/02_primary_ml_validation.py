from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

WORKFLOW_ROOT = Path(__file__).resolve().parents[1]
if str(WORKFLOW_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKFLOW_ROOT))

from src.column_mapping import apply_primary_mapping  # noqa: E402
from src.io_utils import configure_output_roots, load_yaml, make_synthetic_primary, read_table, write_df, write_json  # noqa: E402
from src.models import build_model  # noqa: E402
from src.validation import evaluate_random_split, evaluate_strict_loso  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Primary ML validation: random split + strict LOSO + GG.")
    parser.add_argument("--config", default="configs/primary_pah.yaml")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def load_primary_df(cfg: dict, smoke: bool) -> pd.DataFrame:
    if smoke:
        return make_synthetic_primary(
            n_rows=int(cfg.get("smoke", {}).get("synthetic_rows", 120)),
            n_soils=int(cfg.get("smoke", {}).get("synthetic_soils", 8)),
            seed=43,
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
    seeds = list(cfg.get("random_seeds", [42, 43, 44]))
    if args.smoke:
        seeds = seeds[:3]

    model_factory = lambda seed: build_model(model_name, params=params, random_state=int(seed))

    random_summary, random_detail = evaluate_random_split(
        df=df,
        feature_cols=feature_cols,
        target_col=target_col,
        group_col=group_col,
        random_seeds=seeds,
        model_factory=model_factory,
        impute_strategy=cfg.get("preprocessing", {}).get("impute_strategy", "median"),
        scale=False,
    )
    loso_summary, loso_fold, loso_pred = evaluate_strict_loso(
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

    gg = float(random_summary["R2"] - loso_summary["R2"])
    summary_df = pd.DataFrame(
        [
            {"protocol": "random_split", **random_summary},
            {"protocol": "strict_loso", **loso_summary},
            {"protocol": "generalization_gap", "GG": gg},
        ]
    )

    write_df(summary_df, "outputs/primary/tables/02_primary_ml_validation_summary.csv")
    write_df(random_detail, "outputs/primary/tables/02_primary_ml_validation_random_details.csv")
    write_df(loso_fold, "outputs/primary/tables/02_primary_ml_validation_loso_fold_details.csv")
    write_df(loso_pred, "outputs/primary/predictions/02_primary_ml_validation_loso_predictions.csv")
    write_json(
        {
            "script": "02_primary_ml_validation.py",
            "smoke": bool(args.smoke),
            "n_rows": int(len(df)),
            "n_soils": int(df[group_col].nunique()),
            "GG": gg,
        },
        "outputs/primary/logs/02_primary_ml_validation_runmeta.json",
    )
    print("02 done.")


if __name__ == "__main__":
    main()
