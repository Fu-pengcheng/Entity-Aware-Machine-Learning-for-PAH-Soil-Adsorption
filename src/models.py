from __future__ import annotations

import copy

from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import LinearRegression


def build_model(model_name: str, params: dict | None = None, random_state: int | None = None):
    params = copy.deepcopy(params or {})
    name = str(model_name)

    if name == "LinearRegression":
        return LinearRegression(**params)

    if name == "ExtraTreesRegressor":
        if random_state is not None and "random_state" not in params:
            params["random_state"] = int(random_state)
        return ExtraTreesRegressor(**params)

    if name in {"XGBRegressor", "xgb", "xgboost"}:
        try:
            from xgboost import XGBRegressor
        except Exception:
            fallback = {
                "n_estimators": 200,
                "max_depth": 10,
                "min_samples_split": 2,
            }
            fallback.update(params)
            if random_state is not None:
                fallback["random_state"] = int(random_state)
            return ExtraTreesRegressor(**fallback)

        if random_state is not None and "random_state" not in params:
            params["random_state"] = int(random_state)
        params.setdefault("objective", "reg:squarederror")
        return XGBRegressor(**params)

    raise ValueError(f"Unsupported model name: {model_name}")


def get_empirical_feature_sets() -> dict[str, list[str]]:
    return {
        "Freundlich": ["lnCe"],
        "KOC": ["lnCe", "OC", "logKow"],
        "MLR": ["lnCe", "OC", "logKow", "pH", "Clay", "CEC", "T", "Ratio"],
    }

