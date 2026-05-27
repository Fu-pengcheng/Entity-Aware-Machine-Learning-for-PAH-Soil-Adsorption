from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


PRIMARY_ALIASES = {
    "ln_Qe": "lnQe",
}

EXTERNAL_ALIASES = {
    "log Kd": "logKd",
    "log Ce": "logCe",
    "SOC(%)": "soil organic carbon",
    "Soil pH": "soil pH",
    "Clay(%)": "Clay",
    "CEC(cmol+/kg)": "CEC",
    "log SS ratio": "log soil-solution ratio",
    "LogP": "logP",
    "LogS": "logS",
}

PRIMARY_REQUIRED = [
    "Soil_ID",
    "OC",
    "pH",
    "Clay",
    "CEC",
    "T",
    "Ratio",
    "logKow",
    "lnCe",
    "HOMO",
    "LUMO",
    "lnQe",
]

EXTERNAL_MIN_REQUIRED = [
    "logKd",
    "logCe",
    "soil organic carbon",
    "Clay",
    "CEC",
    "soil pH",
    "log soil-solution ratio",
]


@dataclass
class MappingResult:
    dataframe: pd.DataFrame
    renamed: dict[str, str]


def _strip_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c).strip() for c in out.columns]
    return out


def apply_primary_mapping(df: pd.DataFrame) -> MappingResult:
    out = _strip_columns(df).rename(columns=PRIMARY_ALIASES)
    renamed = {k: v for k, v in PRIMARY_ALIASES.items() if k in df.columns}
    return MappingResult(dataframe=out, renamed=renamed)


def apply_external_mapping(df: pd.DataFrame) -> MappingResult:
    out = _strip_columns(df).rename(columns=EXTERNAL_ALIASES)
    renamed = {k: v for k, v in EXTERNAL_ALIASES.items() if k in df.columns}
    return MappingResult(dataframe=out, renamed=renamed)


def missing_columns(df: pd.DataFrame, required_cols: list[str]) -> list[str]:
    return [c for c in required_cols if c not in df.columns]


def build_column_report(
    original_columns: list[str],
    mapped_columns: list[str],
    required_columns: list[str],
) -> pd.DataFrame:
    mapped_set = set(mapped_columns)
    rows = []
    for c in required_columns:
        rows.append(
            {
                "required_column": c,
                "present_after_mapping": c in mapped_set,
            }
        )
    return pd.DataFrame(rows)

