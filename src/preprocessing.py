from __future__ import annotations

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


def validate_columns(df: pd.DataFrame, required_cols: list[str]) -> None:
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")


def fit_imputer(train_df: pd.DataFrame, cols: list[str], strategy: str = "median") -> SimpleImputer:
    validate_columns(train_df, cols)
    imp = SimpleImputer(strategy=strategy)
    imp.fit(train_df[cols])
    return imp


def transform_imputer(imputer: SimpleImputer, df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    validate_columns(df, cols)
    out = df.copy()
    out[cols] = out[cols].astype(float)
    out.loc[:, cols] = imputer.transform(out[cols])
    return out


def fit_scaler(train_df: pd.DataFrame, cols: list[str]) -> StandardScaler:
    validate_columns(train_df, cols)
    scaler = StandardScaler()
    scaler.fit(train_df[cols])
    return scaler


def transform_scaler(scaler: StandardScaler, df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    validate_columns(df, cols)
    out = df.copy()
    out[cols] = out[cols].astype(float)
    out.loc[:, cols] = scaler.transform(out[cols])
    return out

