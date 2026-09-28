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
from src.validation import evaluate_random_split, evaluate_strict_loso  # noqa: E402

SUMMARY_PATH = "outputs/primary/tables/11_model_comparison_summary.csv"
RANDOM_PATH = "outputs/primary/tables/11_model_comparison_random_details.csv"
LOSO_PATH = "outputs/primary/tables/11_model_comparison_loso_fold_details.csv"

# Model zoo reproducing Fig. 3. XGBoost uses the manuscript config verbatim.
class TorchMLP:
    """Sklearn-style MLP matching the manuscript description: two hidden layers (64, 32),
    ReLU, Dropout(0.2), weight decay, fold-internal target standardization."""

    def __init__(self, seed: int, epochs: int = 500, lr: float = 1e-3, weight_decay: float = 5e-3, dropout: float = 0.1):
        self.seed, self.epochs, self.lr, self.weight_decay, self.dropout = seed, epochs, lr, weight_decay, dropout

    def fit(self, X, y):
        import torch
        import torch.nn as nn

        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32).reshape(-1, 1)
        torch.manual_seed(self.seed)
        p = X.shape[1]
        self.net_ = nn.Sequential(
            nn.Linear(p, 64), nn.ReLU(), nn.Dropout(self.dropout),
            nn.Linear(64, 32), nn.ReLU(), nn.Dropout(self.dropout),
            nn.Linear(32, 1),
        )
        self.y_mean_, self.y_std_ = float(y.mean()), float(y.std() + 1e-8)
        Xt = torch.tensor(X)
        yt = torch.tensor((y - self.y_mean_) / self.y_std_)
        opt = torch.optim.Adam(self.net_.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        lossf = nn.MSELoss()
        self.net_.train()
        for _ in range(self.epochs):
            opt.zero_grad()
            loss = lossf(self.net_(Xt), yt)
            loss.backward()
            opt.step()
        return self

    def predict(self, X):
        import torch

        X = np.asarray(X, dtype=np.float32)
        self.net_.eval()
        with torch.no_grad():
            out = self.net_(torch.tensor(X)).numpy().ravel()
        return out * self.y_std_ + self.y_mean_


def build_zoo(base_params: dict) -> dict:
    from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor

    zoo = {
        "XGBoost": {
            "factory": lambda seed: build_model("XGBRegressor", params=dict(base_params, n_jobs=4), random_state=seed),
            "scale": False,
        },
        "ExtraTrees": {
            "factory": lambda seed: ExtraTreesRegressor(
                n_estimators=500, max_features=1.0, min_samples_leaf=1, n_jobs=4, random_state=seed
            ),
            "scale": False,
        },
        "RandomForest": {
            "factory": lambda seed: RandomForestRegressor(
                n_estimators=500, max_features=1.0, min_samples_leaf=1, n_jobs=4, random_state=seed
            ),
            "scale": False,
        },
        "MLP": {
            "factory": lambda seed: TorchMLP(seed=seed),
            "scale": True,
        },
    }
    try:
        from catboost import CatBoostRegressor

        zoo["CatBoost"] = {
            "factory": lambda seed: CatBoostRegressor(
                iterations=500, depth=6, learning_rate=0.05, loss_function="RMSE",
                random_seed=seed, verbose=False, thread_count=4, allow_writing_files=False
            ),
            "scale": False,
        }
    except Exception:
        pass
    try:
        from lightgbm import LGBMRegressor

        zoo["LightGBM"] = {
            "factory": lambda seed: LGBMRegressor(
                n_estimators=500, learning_rate=0.05, num_leaves=31, subsample=0.8,
                colsample_bytree=0.8, random_state=seed, n_jobs=4, verbose=-1
            ),
            "scale": False,
        }
    except Exception:
        pass
    return zoo


# display order matching current Fig. 3
MODEL_ORDER = ["XGBoost", "CatBoost", "ExtraTrees", "RandomForest", "LightGBM", "MLP"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="A5a: multi-model random split + LOSO with per-fold details (Fig. 3 redraw).")
    parser.add_argument("--config", default="configs/primary_pah.yaml")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--max-models", type=int, default=None, help="Evaluate at most N pending models (resumable).")
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
    configure_output_roots(primary_root=out_cfg.get("primary_root"), external_root=out_cfg.get("external_root"))
    df = load_primary_df(cfg, args.smoke)

    target_col = cfg["target_col"]
    group_col = cfg["group_col"]
    feature_cols = list(cfg["layer3"])
    base_params = dict(cfg.get("model", {}).get("params", {}))
    seeds = list(cfg.get("random_seeds", [42, 43, 44]))
    if args.smoke:
        seeds = seeds[:3]
    impute = cfg.get("preprocessing", {}).get("impute_strategy", "median")
    max_folds = cfg.get("smoke", {}).get("max_folds") if args.smoke else None

    zoo = build_zoo(base_params)
    summary_file = WORKFLOW_ROOT / SUMMARY_PATH
    done = set()
    if summary_file.exists():
        done = set(pd.read_csv(summary_file)["model"].astype(str))

    pending = [m for m in MODEL_ORDER if m in zoo and m not in done]
    if args.max_models is not None:
        pending = pending[: args.max_models]
    missing = [m for m in MODEL_ORDER if m not in zoo]
    if missing:
        print(f"11 WARNING: models unavailable (import failed): {missing}")
    if not pending:
        print("11: all models already evaluated.")
        return

    summary_rows, random_rows, loso_rows = [], [], []
    for name in pending:
        spec = zoo[name]
        t0 = time.time()
        rnd_summary, rnd_detail = evaluate_random_split(
            df=df, feature_cols=feature_cols, target_col=target_col, group_col=group_col,
            random_seeds=seeds, model_factory=spec["factory"], impute_strategy=impute, scale=spec["scale"],
        )
        loso_summary, loso_fold, _ = evaluate_strict_loso(
            df=df, feature_cols=feature_cols, target_col=target_col, group_col=group_col,
            model_factory=spec["factory"], impute_strategy=impute, scale=spec["scale"],
            base_seed=42, max_folds=max_folds,
        )
        elapsed = time.time() - t0
        summary_rows.append({
            "model": name,
            "random_R2": rnd_summary["R2"], "random_R2_std": rnd_summary["R2_std"],
            "random_RMSE": rnd_summary["RMSE"], "random_MAE": rnd_summary["MAE"],
            "loso_R2": loso_summary["R2"], "loso_RMSE": loso_summary["RMSE"], "loso_MAE": loso_summary["MAE"],
            "loso_fold_R2_std": float(loso_fold["r2"].std(ddof=1)),
            "GG": float(rnd_summary["R2"] - loso_summary["R2"]),
            "n_folds": loso_summary["n_folds"], "n_seeds": rnd_summary["n_seeds"],
            "elapsed_sec": round(elapsed, 1),
        })
        rnd_detail["model"] = name
        loso_fold["model"] = name
        random_rows.append(rnd_detail)
        loso_rows.append(loso_fold)
        print(f"11: {name} done in {elapsed:.0f}s  random R2={rnd_summary['R2']:.3f}  LOSO R2={loso_summary['R2']:.3f}", flush=True)

    def append(path: str, new: pd.DataFrame) -> None:
        f = WORKFLOW_ROOT / path
        f.parent.mkdir(parents=True, exist_ok=True)
        if f.exists():
            new = pd.concat([pd.read_csv(f), new], ignore_index=True)
        new.to_csv(f, index=False, encoding="utf-8-sig")

    append(SUMMARY_PATH, pd.DataFrame(summary_rows))
    append(RANDOM_PATH, pd.concat(random_rows, ignore_index=True))
    append(LOSO_PATH, pd.concat(loso_rows, ignore_index=True))
    print("11: wrote summary/random/loso detail tables.")


if __name__ == "__main__":
    main()
