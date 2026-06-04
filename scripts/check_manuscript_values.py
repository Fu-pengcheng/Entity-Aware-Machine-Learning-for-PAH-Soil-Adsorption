from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

import yaml


WORKFLOW_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXPECTED = "configs/expected_manuscript_values.yml"
REPORT_DIR = WORKFLOW_ROOT / "outputs" / "manuscript_check"


class MissingReproducedValue(RuntimeError):
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare reproduced primary outputs against expected manuscript values."
    )
    parser.add_argument("--expected", default=DEFAULT_EXPECTED, help="Expected-value YAML config.")
    return parser.parse_args()


def resolve_path(path: str | Path) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    return WORKFLOW_ROOT / p


def load_yaml(path: str | Path) -> dict[str, Any]:
    cfg_path = resolve_path(path)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Expected-value config not found: {cfg_path}")
    with cfg_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected-value YAML must contain a mapping: {cfg_path}")
    return data


def load_json(path: str | Path) -> Any:
    json_path = resolve_path(path)
    if not json_path.exists():
        raise MissingReproducedValue(f"missing output file: {json_path.relative_to(WORKFLOW_ROOT)}")
    with json_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_csv_rows(path: str | Path) -> list[dict[str, str]]:
    csv_path = resolve_path(path)
    if not csv_path.exists():
        raise MissingReproducedValue(f"missing output file: {csv_path.relative_to(WORKFLOW_ROOT)}")
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def nested_get(data: Any, key: str) -> Any:
    value = data
    for part in key.split("."):
        if isinstance(value, dict) and part in value:
            value = value[part]
        else:
            raise MissingReproducedValue(f"missing JSON key: {key}")
    return value


def csv_value(path: str, where: dict[str, str], column: str) -> str:
    rows = load_csv_rows(path)
    for row in rows:
        if all(str(row.get(k, "")) == str(v) for k, v in where.items()):
            if column not in row:
                raise MissingReproducedValue(f"missing CSV column: {column}")
            return row[column]
    raise MissingReproducedValue(f"missing CSV row in {path}: {where}")


def require_full_run(runmeta_path: str) -> None:
    meta = load_json(runmeta_path)
    if bool(meta.get("smoke", False)):
        raise MissingReproducedValue(f"output is from smoke mode: {runmeta_path}")


def audit_summary() -> dict[str, Any]:
    return load_json("outputs/primary/logs/audit_summary.json")


def reproduced_n_rows() -> float:
    return float(nested_get(audit_summary(), "primary.n_rows"))


def reproduced_n_soils() -> float:
    summary = audit_summary()
    try:
        return float(nested_get(summary, "primary.n_soils"))
    except MissingReproducedValue:
        return float(len(load_csv_rows("outputs/primary/tables/audit_primary_group_sizes.csv")))


def reproduced_n_pahs() -> float:
    summary = audit_summary()
    try:
        value = nested_get(summary, "primary.n_pahs")
        if value is not None:
            return float(value)
    except MissingReproducedValue:
        pass
    return float(len(load_csv_rows("outputs/primary/tables/audit_primary_pah_proxy.csv")))


def reproduced_xgb_random_r2() -> float:
    require_full_run("outputs/primary/logs/02_primary_ml_validation_runmeta.json")
    return float(csv_value("outputs/primary/tables/02_primary_ml_validation_summary.csv", {"protocol": "random_split"}, "R2"))


def reproduced_xgb_loso_r2() -> float:
    require_full_run("outputs/primary/logs/02_primary_ml_validation_runmeta.json")
    return float(csv_value("outputs/primary/tables/02_primary_ml_validation_summary.csv", {"protocol": "strict_loso"}, "R2"))


def reproduced_xgb_generalization_gap() -> float:
    require_full_run("outputs/primary/logs/02_primary_ml_validation_runmeta.json")
    return float(csv_value("outputs/primary/tables/02_primary_ml_validation_summary.csv", {"protocol": "generalization_gap"}, "GG"))


def reproduced_support_q4_q1_mae_ratio() -> float:
    require_full_run("outputs/primary/logs/04_primary_ad_diagnosis_runmeta.json")
    q1 = float(csv_value("outputs/primary/tables/04_primary_ad_diagnosis_quartiles.csv", {"quartile": "Q1"}, "mean_abs_error"))
    q4 = float(csv_value("outputs/primary/tables/04_primary_ad_diagnosis_quartiles.csv", {"quartile": "Q4"}, "mean_abs_error"))
    if q1 == 0:
        raise MissingReproducedValue("Q1 mean_abs_error is zero; ratio undefined")
    return q4 / q1


def reproduced_soil_level_bias_ratio() -> float:
    require_full_run("outputs/primary/logs/06_primary_error_decomposition_runmeta.json")
    return float(nested_get(load_json("outputs/primary/tables/06_primary_error_decomposition_summary.json"), "P_soil"))


def reproduced_fewshot_m3_calibrated_r2() -> float:
    require_full_run("outputs/primary/logs/05_primary_fewshot_runmeta.json")
    return float(csv_value("outputs/primary/tables/05_primary_fewshot_metrics_summary.csv", {"m": "3", "prediction_type": "calibrated"}, "r2_mean"))


def reproduced_fewshot_m5_calibrated_r2() -> float:
    require_full_run("outputs/primary/logs/05_primary_fewshot_runmeta.json")
    return float(csv_value("outputs/primary/tables/05_primary_fewshot_metrics_summary.csv", {"m": "5", "prediction_type": "calibrated"}, "r2_mean"))


REPRODUCERS = {
    "n_rows": reproduced_n_rows,
    "n_soils": reproduced_n_soils,
    "n_pahs": reproduced_n_pahs,
    "xgb_random_r2": reproduced_xgb_random_r2,
    "xgb_loso_r2": reproduced_xgb_loso_r2,
    "xgb_generalization_gap": reproduced_xgb_generalization_gap,
    "support_q4_q1_mae_ratio": reproduced_support_q4_q1_mae_ratio,
    "soil_level_bias_ratio": reproduced_soil_level_bias_ratio,
    "fewshot_m3_calibrated_r2": reproduced_fewshot_m3_calibrated_r2,
    "fewshot_m5_calibrated_r2": reproduced_fewshot_m5_calibrated_r2,
}


def evaluate_item(item: str, spec: dict[str, Any]) -> dict[str, Any]:
    expected = float(spec["expected"])
    tolerance = float(spec.get("tolerance", 0))
    row = {
        "item": item,
        "expected": expected,
        "reproduced": "",
        "tolerance": tolerance,
        "absolute_difference": "",
        "status": "MISSING",
    }

    reproducer = REPRODUCERS.get(item)
    if reproducer is None:
        return row

    try:
        reproduced = float(reproducer())
    except MissingReproducedValue:
        return row

    diff = abs(reproduced - expected)
    row["reproduced"] = reproduced
    row["absolute_difference"] = diff
    row["status"] = "PASS" if math.isclose(reproduced, expected, rel_tol=0.0, abs_tol=tolerance) else "FAIL"
    return row


def write_report(rows: list[dict[str, Any]]) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    fieldnames = ["item", "expected", "reproduced", "tolerance", "absolute_difference", "status"]
    csv_path = REPORT_DIR / "expected_vs_reproduced.csv"
    json_path = REPORT_DIR / "expected_vs_reproduced.json"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    json_path.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> int:
    cfg = load_yaml(parse_args().expected)
    primary = cfg.get("primary", {})
    if not isinstance(primary, dict) or not primary:
        raise ValueError("Expected-value config must contain non-empty 'primary' mapping.")

    rows = [evaluate_item(item, spec) for item, spec in primary.items()]
    write_report(rows)

    for row in rows:
        print(
            f"{row['status']}\t{row['item']}\t"
            f"expected={row['expected']}\treproduced={row['reproduced']}\t"
            f"tolerance={row['tolerance']}\tdiff={row['absolute_difference']}"
        )

    n_pass = sum(1 for row in rows if row["status"] == "PASS")
    n_fail = sum(1 for row in rows if row["status"] == "FAIL")
    n_missing = sum(1 for row in rows if row["status"] == "MISSING")
    print(f"Summary: PASS={n_pass}, FAIL={n_fail}, MISSING={n_missing}")

    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
