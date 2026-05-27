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
from src.models import build_model, get_empirical_feature_sets  # noqa: E402
from src.validation import evaluate_random_split, evaluate_strict_loso  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Primary empirical baselines: Freundlich / KOC / MLR.")
    parser.add_argument("--config", default="configs/primary_pah.yaml")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def load_primary_df(cfg: dict, smoke: bool) -> pd.DataFrame:
    if smoke:
        n_rows = int(cfg.get("smoke", {}).get("synthetic_rows", 120))
        n_soils = int(cfg.get("smoke", {}).get("synthetic_soils", 8))
        return make_synthetic_primary(n_rows=n_rows, n_soils=n_soils, seed=42)
    df_raw = read_table(cfg["input_data_path"], sheet_name=cfg.get("sheet_name", 0))
    return apply_primary_mapping(df_raw).dataframe


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)
    out_cfg = cfg.get("output_paths", {})
    configure_output_roots(
        primary_root=out_cfg.get("primary_root"),
        external_root=out_cfg.get("external_root"),
    )
    df = load_primary_df(cfg, smoke=args.smoke)

    target_col = cfg["target_col"]
    group_col = cfg["group_col"]
    seeds = list(cfg.get("random_seeds", [42, 43, 44]))
    if args.smoke:
        seeds = seeds[:3]

    feature_sets = get_empirical_feature_sets()
    rows = []
    details_random = []
    details_loso = []

    for name, features in feature_sets.items():
        model_factory = lambda seed: build_model("LinearRegression", {}, random_state=seed)

        rnd_summary, rnd_detail = evaluate_random_split(
            df=df,
            feature_cols=features,
            target_col=target_col,
            group_col=group_col,
            random_seeds=seeds,
            model_factory=model_factory,
            impute_strategy=cfg.get("preprocessing", {}).get("impute_strategy", "median"),
            scale=True,
        )
        rows.append({"baseline": name, "protocol": "random_split", **rnd_summary})
        rnd_detail["baseline"] = name
        details_random.append(rnd_detail)

        loso_summary, loso_fold, loso_pred = evaluate_strict_loso(
            df=df,
            feature_cols=features,
            target_col=target_col,
            group_col=group_col,
            model_factory=model_factory,
            impute_strategy=cfg.get("preprocessing", {}).get("impute_strategy", "median"),
            scale=True,
            base_seed=42,
            max_folds=cfg.get("smoke", {}).get("max_folds") if args.smoke else None,
        )
        rows.append({"baseline": name, "protocol": "strict_loso", **loso_summary})
        loso_fold["baseline"] = name
        loso_pred["baseline"] = name
        details_loso.append(loso_fold)

        write_df(loso_pred, f"outputs/primary/predictions/01_empirical_{name}_loso_predictions.csv")

    summary_df = pd.DataFrame(rows)
    write_df(summary_df, "outputs/primary/tables/01_empirical_baselines_summary.csv")
    write_df(pd.concat(details_random, axis=0, ignore_index=True), "outputs/primary/tables/01_empirical_baselines_random_details.csv")
    write_df(pd.concat(details_loso, axis=0, ignore_index=True), "outputs/primary/tables/01_empirical_baselines_loso_fold_details.csv")
    write_json(
        {
            "script": "01_primary_empirical_baselines.py",
            "smoke": bool(args.smoke),
            "n_rows": int(len(df)),
            "n_soils": int(df[group_col].nunique()),
        },
        "outputs/primary/logs/01_empirical_baselines_runmeta.json",
    )
    print("01 done.")


if __name__ == "__main__":
    main()
