from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

WORKFLOW_ROOT = Path(__file__).resolve().parents[1]
if str(WORKFLOW_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKFLOW_ROOT))

from src.column_mapping import apply_external_mapping  # noqa: E402
from src.io_utils import (  # noqa: E402
    configure_output_roots,
    load_yaml,
    make_synthetic_external,
    read_table,
    write_df,
    write_json,
)
from src.models import build_model  # noqa: E402
from src.validation import evaluate_random_split, evaluate_strict_loso  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="External validation framework migration script.")
    parser.add_argument("--config", default="configs/external_neutral_organic.yaml")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def infer_soil_entity(df: pd.DataFrame, inference_cfg: dict | None = None) -> tuple[pd.DataFrame, bool, dict]:
    out = df.copy()
    cfg = inference_cfg or {}
    method = str(cfg.get("method", "exact_groupby"))
    columns = list(cfg.get("columns", ["soil pH", "soil organic carbon", "CEC", "Clay"]))
    dropna_flag = bool(cfg.get("dropna", False))

    if "Soil_ID_inferred" in out.columns:
        out["Soil_ID_inferred"] = out["Soil_ID_inferred"].astype(str)
        return out, True, {"method": "preexisting_column", "columns": ["Soil_ID_inferred"], "dropna": None}
    soil_cols = [c for c in columns if c in out.columns]
    if len(soil_cols) < 2:
        return out, False, {"method": method, "columns": soil_cols, "dropna": dropna_flag}

    tmp = out.copy()
    if dropna_flag:
        tmp = tmp.dropna(subset=soil_cols).copy()

    if method != "exact_groupby":
        raise ValueError(f"Unsupported soil entity inference method: {method}")
    tmp["Soil_ID_inferred"] = tmp.groupby(soil_cols, dropna=False).ngroup().map(lambda x: f"soil_{int(x):04d}")

    if dropna_flag:
        out = tmp
    else:
        out = tmp

    return out, True, {"method": method, "columns": soil_cols, "dropna": dropna_flag}


def maybe_attach_logp_logs(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    out = df.copy()
    if {"logP", "logS"}.issubset(set(out.columns)):
        return out
    lookup = cfg.get("compound_lookup_path")
    if not lookup:
        return out
    try:
        comp = read_table(lookup, sheet_name=0)
    except Exception:
        return out
    comp = apply_external_mapping(comp).dataframe
    if "CAS Number" not in comp.columns or "CAS Number" not in out.columns:
        return out
    keep = [c for c in ["CAS Number", "logP", "logS"] if c in comp.columns]
    if len(keep) <= 1:
        return out
    out = out.merge(comp[keep], on="CAS Number", how="left", suffixes=("", "_lookup"))
    for c in ["logP", "logS"]:
        if c not in out.columns and f"{c}_lookup" in out.columns:
            out[c] = out[f"{c}_lookup"]
    drop_cols = [c for c in out.columns if c.endswith("_lookup")]
    if drop_cols:
        out = out.drop(columns=drop_cols)
    return out


def run_smoke_external(cfg: dict) -> dict:
    df = make_synthetic_external(
        n_rows=int(cfg.get("smoke", {}).get("synthetic_rows", 140)),
        n_soils=int(cfg.get("smoke", {}).get("synthetic_soils", 10)),
        seed=48,
    )
    target_col = cfg.get("target_col", "logKd")
    group_col = "Soil_ID_inferred"
    feature_cols = [c for c in cfg.get("scalar_features", []) if c in df.columns]
    params = dict(cfg.get("model_params", {}))
    model_name = cfg.get("model", "ExtraTreesRegressor")
    seeds = [42, 43, 44]

    model_factory = lambda seed: build_model(model_name, params=params, random_state=int(seed))
    random_summary, random_detail = evaluate_random_split(
        df=df,
        feature_cols=feature_cols,
        target_col=target_col,
        group_col=group_col,
        random_seeds=seeds,
        model_factory=model_factory,
        impute_strategy=cfg.get("preprocessing", {}).get("impute_strategy", "median"),
        scale=bool(cfg.get("preprocessing", {}).get("standardize", True)),
    )
    loso_summary, loso_fold, loso_pred = evaluate_strict_loso(
        df=df,
        feature_cols=feature_cols,
        target_col=target_col,
        group_col=group_col,
        model_factory=model_factory,
        impute_strategy=cfg.get("preprocessing", {}).get("impute_strategy", "median"),
        scale=bool(cfg.get("preprocessing", {}).get("standardize", True)),
        max_folds=6,
    )
    write_df(pd.DataFrame([{"protocol": "random_split", **random_summary}, {"protocol": "strict_loso", **loso_summary}]), "outputs/external/tables/07_external_smoke_summary.csv")
    write_df(random_detail, "outputs/external/tables/07_external_smoke_random_details.csv")
    write_df(loso_fold, "outputs/external/tables/07_external_smoke_loso_folds.csv")
    write_df(loso_pred, "outputs/external/predictions/07_external_smoke_loso_predictions.csv")
    return {
        "mode": "smoke",
        "n_rows": int(len(df)),
        "n_soils": int(df[group_col].nunique()),
        "feature_cols_used": feature_cols,
    }


def run_framework_check(cfg: dict) -> dict:
    df_raw = read_table(cfg["input_data_path"], sheet_name=cfg.get("sheet_name", 0))
    mapped = apply_external_mapping(df_raw).dataframe
    mapped = maybe_attach_logp_logs(mapped, cfg)
    mapped, has_group, infer_meta = infer_soil_entity(mapped, cfg.get("soil_entity_inference", {}))

    target_col = cfg.get("target_col", "logKd")
    requested_features = list(cfg.get("scalar_features", []))
    available_features = [c for c in requested_features if c in mapped.columns]
    missing_features = [c for c in requested_features if c not in mapped.columns]

    report = {
        "mode": "framework_only",
        "n_rows": int(len(mapped)),
        "n_cols": int(mapped.shape[1]),
        "columns": list(mapped.columns),
        "target_col": target_col,
        "target_present": target_col in mapped.columns,
        "soil_entity_inferred": bool(has_group),
        "n_soils_inferred": int(mapped["Soil_ID_inferred"].nunique()) if has_group else 0,
        "soil_entity_inference": infer_meta,
        "available_scalar_features": available_features,
        "missing_scalar_features": missing_features,
        "has_logP": "logP" in mapped.columns,
        "has_logS": "logS" in mapped.columns,
        "full_external_validation_executed": False,
        "note": "Framework migrated. Full external validation intentionally skipped in this stage.",
    }
    write_df(pd.DataFrame({"column": mapped.columns}), "outputs/external/tables/07_external_columns_mapped.csv")
    return report


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)
    out_cfg = cfg.get("output_paths", {})
    configure_output_roots(
        primary_root=out_cfg.get("primary_root"),
        external_root=out_cfg.get("external_root"),
    )
    if args.smoke:
        report = run_smoke_external(cfg)
    else:
        report = run_framework_check(cfg)

    write_json(report, "outputs/external/logs/07_external_validation_report.json")
    print("07 done.")
    print(report["mode"])


if __name__ == "__main__":
    main()
