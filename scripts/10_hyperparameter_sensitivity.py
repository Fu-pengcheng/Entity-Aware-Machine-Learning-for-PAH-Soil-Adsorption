from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

WORKFLOW_ROOT = Path(__file__).resolve().parents[1]
if str(WORKFLOW_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKFLOW_ROOT))

from src.column_mapping import apply_primary_mapping  # noqa: E402
from src.io_utils import configure_output_roots, load_yaml, make_synthetic_primary, read_table  # noqa: E402
from src.models import build_model  # noqa: E402
from src.validation import evaluate_random_split, evaluate_strict_loso  # noqa: E402

OUT_PATH = "outputs/primary/tables/10_hyperparam_sensitivity.csv"

# config_id -> overrides on top of configs/primary_pah.yaml model.params
SENSITIVITY_CONFIGS: dict[str, dict] = {
    "baseline": {},
    "depth3": {"max_depth": 3},
    "depth6": {"max_depth": 6},
    "lr0.03": {"learning_rate": 0.03},
    "lr0.10": {"learning_rate": 0.10},
    "n500": {"n_estimators": 500},
    "n800": {"n_estimators": 800},
    "sub0.7": {"subsample": 0.7},
    "sub0.6": {"subsample": 0.6},
    "sub1.0": {"subsample": 1.0},
    "conservative": {"max_depth": 3, "learning_rate": 0.03, "n_estimators": 300},
    "aggressive": {"max_depth": 6, "learning_rate": 0.10, "n_estimators": 800},
    "deep_slow": {"max_depth": 6, "learning_rate": 0.03, "n_estimators": 800},
    "shallow_fast": {"max_depth": 3, "learning_rate": 0.10, "n_estimators": 300},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="A4: XGBoost hyperparameter sensitivity under LOSO + random split.")
    parser.add_argument("--config", default="configs/primary_pah.yaml")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--max-configs", type=int, default=None,
                        help="Evaluate at most N not-yet-completed configs this run (resumable).")
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
    base_params = dict(cfg.get("model", {}).get("params", {}))
    model_name = cfg.get("model", {}).get("name", "XGBRegressor")
    seeds = list(cfg.get("random_seeds", [42, 43, 44]))
    if args.smoke:
        seeds = seeds[:3]
    impute_strategy = cfg.get("preprocessing", {}).get("impute_strategy", "median")

    out_file = WORKFLOW_ROOT / OUT_PATH
    done_ids: set[str] = set()
    if out_file.exists():
        done_ids = set(pd.read_csv(out_file)["config_id"].astype(str))

    pending = [cid for cid in SENSITIVITY_CONFIGS if cid not in done_ids]
    if args.max_configs is not None:
        pending = pending[: args.max_configs]
    if not pending:
        print("10: all configs already evaluated, nothing to do.")
        return

    rows = []
    for cid in pending:
        overrides = dict(SENSITIVITY_CONFIGS[cid])
        params = dict(base_params)
        params.update(overrides)
        params["n_jobs"] = 4  # avoid full-core saturation on the workstation
        # keep random_state from config (fixed 42), matching scripts 02/09 exactly

        t0 = time.time()
        model_factory = lambda seed: build_model(model_name, params=params, random_state=int(seed))

        random_summary, _ = evaluate_random_split(
            df=df,
            feature_cols=feature_cols,
            target_col=target_col,
            group_col=group_col,
            random_seeds=seeds,
            model_factory=model_factory,
            impute_strategy=impute_strategy,
            scale=False,
        )
        loso_summary, _, _ = evaluate_strict_loso(
            df=df,
            feature_cols=feature_cols,
            target_col=target_col,
            group_col=group_col,
            model_factory=model_factory,
            impute_strategy=impute_strategy,
            scale=False,
            base_seed=42,
            max_folds=cfg.get("smoke", {}).get("max_folds") if args.smoke else None,
        )
        elapsed = time.time() - t0

        row = {
            "config_id": cid,
            "max_depth": params.get("max_depth"),
            "learning_rate": params.get("learning_rate"),
            "n_estimators": params.get("n_estimators"),
            "subsample": params.get("subsample"),
            "colsample_bytree": params.get("colsample_bytree"),
            "random_R2": random_summary["R2"],
            "random_R2_std": random_summary["R2_std"],
            "loso_R2": loso_summary["R2"],
            "loso_RMSE": loso_summary["RMSE"],
            "loso_MAE": loso_summary["MAE"],
            "GG": float(random_summary["R2"] - loso_summary["R2"]),
            "n_folds": loso_summary["n_folds"],
            "elapsed_sec": round(elapsed, 1),
        }
        rows.append(row)
        print(f"10: {cid} done in {elapsed:.0f}s  LOSO R2={row['loso_R2']:.4f}  GG={row['GG']:.4f}", flush=True)

    new_df = pd.DataFrame(rows)
    if out_file.exists():
        new_df = pd.concat([pd.read_csv(out_file), new_df], ignore_index=True)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    new_df.to_csv(out_file, index=False, encoding="utf-8-sig")
    print(f"10: wrote {OUT_PATH} ({len(new_df)} configs total)")


if __name__ == "__main__":
    main()
