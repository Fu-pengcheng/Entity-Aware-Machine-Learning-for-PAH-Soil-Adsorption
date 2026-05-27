from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


WORKFLOW_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = WORKFLOW_ROOT.parent
_OUTPUT_PRIMARY_ROOT = Path("outputs/primary")
_OUTPUT_EXTERNAL_ROOT = Path("outputs/external")


def now_text() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def resolve_path(raw_path: str | Path) -> Path:
    p = Path(str(raw_path))
    if p.is_absolute():
        return p

    candidates = [
        Path.cwd() / p,
        WORKFLOW_ROOT / p,
        PROJECT_ROOT / p,
    ]
    for c in candidates:
        if c.exists():
            return c
    return WORKFLOW_ROOT / p


def resolve_write_path(raw_path: str | Path) -> Path:
    p = Path(str(raw_path))
    if p.is_absolute():
        return p
    p_str = p.as_posix()
    if p_str.startswith("outputs/primary"):
        suffix = p_str[len("outputs/primary"):].lstrip("/")
        mapped = _OUTPUT_PRIMARY_ROOT / suffix if suffix else _OUTPUT_PRIMARY_ROOT
        return WORKFLOW_ROOT / mapped
    if p_str.startswith("outputs/external"):
        suffix = p_str[len("outputs/external"):].lstrip("/")
        mapped = _OUTPUT_EXTERNAL_ROOT / suffix if suffix else _OUTPUT_EXTERNAL_ROOT
        return WORKFLOW_ROOT / mapped
    return WORKFLOW_ROOT / p


def configure_output_roots(primary_root: str | None = None, external_root: str | None = None) -> None:
    global _OUTPUT_PRIMARY_ROOT, _OUTPUT_EXTERNAL_ROOT
    if primary_root:
        _OUTPUT_PRIMARY_ROOT = Path(primary_root)
    if external_root:
        _OUTPUT_EXTERNAL_ROOT = Path(external_root)


def load_yaml(path: str | Path) -> dict[str, Any]:
    cfg_path = resolve_path(path)
    with cfg_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML root must be mapping: {cfg_path}")
    return data


def read_table(path: str | Path, sheet_name: int | str = 0) -> pd.DataFrame:
    file_path = resolve_path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    suffix = file_path.suffix.lower()
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(file_path, sheet_name=sheet_name)
    if suffix == ".csv":
        return pd.read_csv(file_path)
    raise ValueError(f"Unsupported file type: {suffix}")


def ensure_dir(path: str | Path) -> Path:
    out = resolve_write_path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def write_df(df: pd.DataFrame, path: str | Path) -> Path:
    out = resolve_write_path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    suffix = out.suffix.lower()
    if suffix in {".xlsx", ".xls"}:
        df.to_excel(out, index=False)
    else:
        df.to_csv(out, index=False, encoding="utf-8-sig")
    return out


def write_json(data: dict[str, Any], path: str | Path) -> Path:
    out = resolve_write_path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    return out


def append_log(message: str, log_path: str | Path) -> None:
    p = resolve_write_path(log_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(f"[{now_text()}] {message}\n")


def make_synthetic_primary(
    n_rows: int = 120,
    n_soils: int = 8,
    seed: int = 42,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    soil_ids = [f"S{i:03d}" for i in range(1, n_soils + 1)]
    soil = rng.choice(soil_ids, size=n_rows, replace=True)
    oc = rng.uniform(0.4, 6.0, size=n_rows)
    ph = rng.uniform(4.5, 8.2, size=n_rows)
    clay = rng.uniform(2, 55, size=n_rows)
    cec = rng.uniform(2, 40, size=n_rows)
    temp = rng.uniform(15, 35, size=n_rows)
    ratio = rng.uniform(0.2, 4.0, size=n_rows)
    log_kow = rng.uniform(2.5, 6.5, size=n_rows)
    ln_ce = rng.uniform(-2.0, 4.0, size=n_rows)
    homo = rng.uniform(-10.0, -5.0, size=n_rows)
    lumo = rng.uniform(-3.5, 1.0, size=n_rows)

    ln_qe = (
        0.45 * ln_ce
        + 0.18 * oc
        + 0.09 * log_kow
        - 0.03 * ph
        + 0.002 * clay
        + 0.01 * cec
        + 0.005 * ratio
        + 0.03 * (lumo - homo)
        + rng.normal(0, 0.25, size=n_rows)
    )

    return pd.DataFrame(
        {
            "Soil_ID": soil,
            "OC": oc,
            "pH": ph,
            "Clay": clay,
            "CEC": cec,
            "T": temp,
            "Ratio": ratio,
            "logKow": log_kow,
            "lnCe": ln_ce,
            "HOMO": homo,
            "LUMO": lumo,
            "lnQe": ln_qe,
            "SMOKE_ONLY": True,
        }
    )


def make_synthetic_external(
    n_rows: int = 140,
    n_soils: int = 10,
    seed: int = 43,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    soil_ids = [f"soil_{i:03d}" for i in range(1, n_soils + 1)]
    soil = rng.choice(soil_ids, size=n_rows, replace=True)
    log_ce = rng.uniform(-2.0, 3.5, size=n_rows)
    soc = rng.uniform(0.3, 6.0, size=n_rows)
    clay = rng.uniform(1.0, 60.0, size=n_rows)
    cec = rng.uniform(2.0, 45.0, size=n_rows)
    sph = rng.uniform(4.0, 8.5, size=n_rows)
    log_ss = rng.uniform(-1.5, 1.5, size=n_rows)
    logp = rng.uniform(1.0, 6.0, size=n_rows)
    logs = rng.uniform(-8.0, 1.0, size=n_rows)

    log_kd = (
        0.35 * log_ce
        + 0.22 * soc
        + 0.05 * clay / 10.0
        + 0.08 * cec / 10.0
        - 0.04 * sph
        + 0.15 * logp
        - 0.05 * logs
        + rng.normal(0, 0.25, size=n_rows)
    )

    return pd.DataFrame(
        {
            "Soil_ID_inferred": soil,
            "logCe": log_ce,
            "soil organic carbon": soc,
            "Clay": clay,
            "CEC": cec,
            "soil pH": sph,
            "log soil-solution ratio": log_ss,
            "logP": logp,
            "logS": logs,
            "logKd": log_kd,
            "SMOKE_ONLY": True,
        }
    )
