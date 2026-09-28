from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

WORKFLOW_ROOT = Path(__file__).resolve().parents[1]
if str(WORKFLOW_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKFLOW_ROOT))

from src.column_mapping import apply_primary_mapping  # noqa: E402
from src.io_utils import configure_output_roots, load_yaml, make_synthetic_primary, read_table  # noqa: E402
from src.models import build_model  # noqa: E402
from src.preprocessing import fit_imputer, transform_imputer  # noqa: E402
from src.validation import evaluate_random_split, evaluate_strict_loso, generate_loso_splits  # noqa: E402

LAYER_SUMMARY = "outputs/primary/tables/12_layer_ablation5_summary.csv"
LAYER_FOLDS = "outputs/primary/tables/12_layer_ablation5_loso_fold_details.csv"
SHAP_FOLDS = "outputs/primary/tables/12_shap_per_fold.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="A5b: 5-level layer ablation fold details (Fig. 5) + per-fold SHAP (Fig. 6).")
    parser.add_argument("--config", default="configs/primary_pah.yaml")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--skip-shap", action="store_true")
    parser.add_argument("--skip-ablation", action="store_true")
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


def run_ablation(cfg: dict, df: pd.DataFrame, smoke: bool) -> None:
    target_col, group_col = cfg["target_col"], cfg["group_col"]
    seeds = list(cfg.get("random_seeds", [42, 43, 44]))[: 3 if smoke else None]
    params = dict(cfg.get("model", {}).get("params", {}), n_jobs=4)
    model_name = cfg.get("model", {}).get("name", "XGBRegressor")
    factory = lambda seed: build_model(model_name, params=params, random_state=int(seed))
    impute = cfg.get("preprocessing", {}).get("impute_strategy", "median")
    max_folds = cfg.get("smoke", {}).get("max_folds") if smoke else None

    l2 = list(cfg["layer2"])
    layers = [
        ("Layer 1", list(cfg["layer1"])),
        ("Layer 2", l2),
        ("L2+HOMO", l2 + ["HOMO"]),
        ("L2+LUMO", l2 + ["LUMO"]),
        ("Layer 3", list(cfg["layer3"])),
    ]
    summary_rows, fold_rows = [], []
    for lname, fcols in layers:
        t0 = time.time()
        rnd_summary, _ = evaluate_random_split(
            df=df, feature_cols=fcols, target_col=target_col, group_col=group_col,
            random_seeds=seeds, model_factory=factory, impute_strategy=impute, scale=False,
        )
        loso_summary, loso_fold, _ = evaluate_strict_loso(
            df=df, feature_cols=fcols, target_col=target_col, group_col=group_col,
            model_factory=factory, impute_strategy=impute, scale=False, base_seed=42, max_folds=max_folds,
        )
        summary_rows.append({
            "layer": lname, "n_features": len(fcols),
            "random_R2": rnd_summary["R2"], "random_R2_std": rnd_summary["R2_std"],
            "loso_R2": loso_summary["R2"], "loso_RMSE": loso_summary["RMSE"], "loso_MAE": loso_summary["MAE"],
            "GG": float(rnd_summary["R2"] - loso_summary["R2"]),
        })
        loso_fold["layer"] = lname
        fold_rows.append(loso_fold)
        print(f"12: {lname} ablation done in {time.time()-t0:.0f}s  LOSO R2={loso_summary['R2']:.3f}", flush=True)

    out1, out2 = WORKFLOW_ROOT / LAYER_SUMMARY, WORKFLOW_ROOT / LAYER_FOLDS
    out1.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summary_rows).to_csv(out1, index=False, encoding="utf-8-sig")
    pd.concat(fold_rows, ignore_index=True).to_csv(out2, index=False, encoding="utf-8-sig")
    print(f"12: wrote {LAYER_SUMMARY} and {LAYER_FOLDS}")


def run_shap(cfg: dict, df: pd.DataFrame, smoke: bool) -> None:
    import shap

    target_col, group_col = cfg["target_col"], cfg["group_col"]
    params = dict(cfg.get("model", {}).get("params", {}), n_jobs=4)
    impute = cfg.get("preprocessing", {}).get("impute_strategy", "median")
    max_folds = cfg.get("smoke", {}).get("max_folds") if smoke else None

    layers = [("Layer 1", list(cfg["layer1"])), ("Layer 2", list(cfg["layer2"])), ("Layer 3", list(cfg["layer3"]))]
    records = []
    for lname, fcols in layers:
        t0 = time.time()
        for fold_id, train_idx, test_idx, held in generate_loso_splits(df, group_col, max_folds=max_folds):
            train_df = transform_imputer(fit_imputer(df.iloc[train_idx], fcols, strategy=impute), df.iloc[train_idx], fcols)
            test_df = transform_imputer(fit_imputer(df.iloc[train_idx], fcols, strategy=impute), df.iloc[test_idx], fcols)
            model = build_model("XGBRegressor", params=params, random_state=42 + fold_id)
            model.fit(train_df[fcols].astype(float), train_df[target_col].astype(float))
            X_test = test_df[fcols].astype(float)
            sv = shap.TreeExplainer(model).shap_values(X_test)
            mean_abs = np.abs(np.asarray(sv)).mean(axis=0)
            for feat, val in zip(fcols, mean_abs):
                records.append({"layer": lname, "fold_id": fold_id, "held_out_group": held,
                                "feature": feat, "mean_abs_shap": float(val)})
        print(f"12: {lname} per-fold SHAP done in {time.time()-t0:.0f}s", flush=True)

    shap_df = pd.DataFrame(records)
    out = WORKFLOW_ROOT / SHAP_FOLDS
    out.parent.mkdir(parents=True, exist_ok=True)
    shap_df.to_csv(out, index=False, encoding="utf-8-sig")
    # fold-level top-5 frequency summary
    top5 = (
        shap_df.sort_values(["layer", "fold_id", "mean_abs_shap"], ascending=[True, True, False])
        .groupby(["layer", "fold_id"]).head(5)
        .groupby(["layer", "feature"]).size().rename("top5_count").reset_index()
    )
    n_folds = shap_df.groupby("layer")["fold_id"].nunique().rename("n_folds").reset_index()
    top5 = top5.merge(n_folds, on="layer")
    top5.to_csv(WORKFLOW_ROOT / "outputs/primary/tables/12_shap_top5_frequency.csv", index=False, encoding="utf-8-sig")
    print(f"12: wrote {SHAP_FOLDS} (+top5 frequency)")


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)
    out_cfg = cfg.get("output_paths", {})
    configure_output_roots(primary_root=out_cfg.get("primary_root"), external_root=out_cfg.get("external_root"))
    df = load_primary_df(cfg, args.smoke)
    if not args.skip_ablation:
        run_ablation(cfg, df, args.smoke)
    if not args.skip_shap:
        run_shap(cfg, df, args.smoke)
    print("12 done.")


if __name__ == "__main__":
    main()
