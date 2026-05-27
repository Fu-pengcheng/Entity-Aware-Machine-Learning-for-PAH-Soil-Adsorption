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
    parser = argparse.ArgumentParser(description="Primary layer ablation (Layer1/Layer2/Layer3).")
    parser.add_argument("--config", default="configs/primary_pah.yaml")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def load_primary_df(cfg: dict, smoke: bool) -> pd.DataFrame:
    if smoke:
        return make_synthetic_primary(
            n_rows=int(cfg.get("smoke", {}).get("synthetic_rows", 120)),
            n_soils=int(cfg.get("smoke", {}).get("synthetic_soils", 8)),
            seed=44,
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
    seeds = list(cfg.get("random_seeds", [42, 43, 44]))
    if args.smoke:
        seeds = seeds[:3]
    params = dict(cfg.get("model", {}).get("params", {}))
    model_name = cfg.get("model", {}).get("name", "XGBRegressor")
    model_factory = lambda seed: build_model(model_name, params=params, random_state=int(seed))

    layers = [("Layer1", list(cfg["layer1"])), ("Layer2", list(cfg["layer2"])), ("Layer3", list(cfg["layer3"]))]
    summary_rows = []
    random_details = []
    loso_details = []

    for layer_name, feature_cols in layers:
        rnd_summary, rnd_detail = evaluate_random_split(
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
        gg = float(rnd_summary["R2"] - loso_summary["R2"])
        summary_rows.extend(
            [
                {"layer": layer_name, "protocol": "random_split", **rnd_summary},
                {"layer": layer_name, "protocol": "strict_loso", **loso_summary},
                {"layer": layer_name, "protocol": "generalization_gap", "GG": gg},
            ]
        )
        rnd_detail["layer"] = layer_name
        loso_fold["layer"] = layer_name
        loso_pred["layer"] = layer_name
        random_details.append(rnd_detail)
        loso_details.append(loso_fold)
        write_df(loso_pred, f"outputs/primary/predictions/03_layer_ablation_{layer_name}_loso_predictions.csv")

    write_df(pd.DataFrame(summary_rows), "outputs/primary/tables/03_primary_layer_ablation_summary.csv")
    write_df(pd.concat(random_details, axis=0, ignore_index=True), "outputs/primary/tables/03_primary_layer_ablation_random_details.csv")
    write_df(pd.concat(loso_details, axis=0, ignore_index=True), "outputs/primary/tables/03_primary_layer_ablation_loso_fold_details.csv")
    write_json(
        {
            "script": "03_primary_layer_ablation.py",
            "smoke": bool(args.smoke),
            "n_rows": int(len(df)),
            "n_soils": int(df[group_col].nunique()),
        },
        "outputs/primary/logs/03_primary_layer_ablation_runmeta.json",
    )
    print("03 done.")


if __name__ == "__main__":
    main()
