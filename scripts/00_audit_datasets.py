from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

WORKFLOW_ROOT = Path(__file__).resolve().parents[1]
if str(WORKFLOW_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKFLOW_ROOT))

from src.column_mapping import (  # noqa: E402
    EXTERNAL_MIN_REQUIRED,
    PRIMARY_REQUIRED,
    apply_external_mapping,
    apply_primary_mapping,
    build_column_report,
)
from src.io_utils import configure_output_roots, ensure_dir, load_yaml, read_table, write_df, write_json  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit primary and external datasets.")
    parser.add_argument("--config-primary", default="configs/primary_pah.yaml")
    parser.add_argument("--config-external", default="configs/external_neutral_organic.yaml")
    return parser.parse_args()


def _missing_report(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "column": df.columns,
            "missing_count": [int(df[c].isna().sum()) for c in df.columns],
            "missing_rate": [float(df[c].isna().mean()) for c in df.columns],
        }
    ).sort_values("missing_rate", ascending=False)


def audit_primary(cfg: dict) -> dict:
    out_dir = ensure_dir("outputs/primary/tables")
    df_raw = read_table(cfg["input_data_path"], sheet_name=cfg.get("sheet_name", 0))
    mapped = apply_primary_mapping(df_raw)
    df = mapped.dataframe

    write_df(pd.DataFrame({"column": df_raw.columns}), out_dir / "audit_primary_columns_raw.csv")
    write_df(pd.DataFrame({"column": df.columns}), out_dir / "audit_primary_columns_mapped.csv")
    write_df(_missing_report(df), out_dir / "audit_primary_missing_report.csv")
    write_df(df.describe(include="all").transpose().reset_index().rename(columns={"index": "column"}), out_dir / "audit_primary_describe.csv")

    gcol = cfg.get("group_col", "Soil_ID")
    if gcol in df.columns:
        gsize = df.groupby(gcol, as_index=False).size().rename(columns={"size": "n_samples"})
        write_df(gsize.sort_values("n_samples", ascending=False), out_dir / "audit_primary_group_sizes.csv")

    report = build_column_report(list(df_raw.columns), list(df.columns), PRIMARY_REQUIRED)
    write_df(report, out_dir / "audit_primary_required_column_check.csv")

    return {
        "n_rows": int(len(df)),
        "n_cols": int(df.shape[1]),
        "columns": list(df.columns),
        "missing_required_cols": report.loc[~report["present_after_mapping"], "required_column"].tolist(),
        "renamed": mapped.renamed,
    }


def _infer_external_group(df: pd.DataFrame) -> tuple[pd.DataFrame, bool]:
    soil_cols = [c for c in ["soil pH", "soil organic carbon", "CEC", "Clay"] if c in df.columns]
    out = df.copy()
    if len(soil_cols) < 2:
        return out, False
    out["Soil_ID_inferred"] = out.groupby(soil_cols, dropna=False).ngroup().map(lambda x: f"soil_{int(x):04d}")
    return out, True


def audit_external(cfg: dict) -> dict:
    out_dir = ensure_dir("outputs/external/tables")
    df_raw = read_table(cfg["input_data_path"], sheet_name=cfg.get("sheet_name", 0))
    mapped = apply_external_mapping(df_raw)
    df = mapped.dataframe
    df, inferred = _infer_external_group(df)

    write_df(pd.DataFrame({"column": df_raw.columns}), out_dir / "audit_external_columns_raw.csv")
    write_df(pd.DataFrame({"column": df.columns}), out_dir / "audit_external_columns_mapped.csv")
    write_df(_missing_report(df), out_dir / "audit_external_missing_report.csv")
    write_df(df.describe(include="all").transpose().reset_index().rename(columns={"index": "column"}), out_dir / "audit_external_describe.csv")

    if "Soil_ID_inferred" in df.columns:
        gsize = df.groupby("Soil_ID_inferred", as_index=False).size().rename(columns={"size": "n_samples"})
        write_df(gsize.sort_values("n_samples", ascending=False), out_dir / "audit_external_group_sizes_inferred.csv")

    report = build_column_report(list(df_raw.columns), list(df.columns), EXTERNAL_MIN_REQUIRED)
    write_df(report, out_dir / "audit_external_required_column_check.csv")

    return {
        "n_rows": int(len(df)),
        "n_cols": int(df.shape[1]),
        "columns": list(df.columns),
        "missing_min_required_cols": report.loc[~report["present_after_mapping"], "required_column"].tolist(),
        "renamed": mapped.renamed,
        "soil_entity_inferred": bool(inferred),
        "n_soils_inferred": int(df["Soil_ID_inferred"].nunique()) if "Soil_ID_inferred" in df.columns else 0,
    }


def main() -> None:
    args = parse_args()
    p_cfg = load_yaml(args.config_primary)
    e_cfg = load_yaml(args.config_external)
    out_cfg = p_cfg.get("output_paths", {})
    e_out_cfg = e_cfg.get("output_paths", {})
    configure_output_roots(
        primary_root=out_cfg.get("primary_root"),
        external_root=e_out_cfg.get("external_root", out_cfg.get("external_root")),
    )

    primary_info = audit_primary(p_cfg)
    external_info = audit_external(e_cfg)

    summary = {
        "primary": primary_info,
        "external": external_info,
    }
    write_json(summary, "outputs/primary/logs/audit_summary.json")
    write_json(summary, "outputs/external/logs/audit_summary.json")
    print("Audit done.")
    print(f"Primary rows={primary_info['n_rows']}, cols={primary_info['n_cols']}")
    print(f"External rows={external_info['n_rows']}, cols={external_info['n_cols']}")


if __name__ == "__main__":
    main()
