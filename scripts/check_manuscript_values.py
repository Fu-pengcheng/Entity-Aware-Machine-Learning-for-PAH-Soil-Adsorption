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

DEFAULT_CHECKS: list[dict[str, Any]] = [
    {
        "name": "primary audit row count",
        "path": "outputs/primary/logs/audit_summary.json",
        "format": "json",
        "key": "primary.n_rows",
        "expected": 1408,
        "tolerance": 0,
    },
    {
        "name": "external audit row count",
        "path": "outputs/primary/logs/audit_summary.json",
        "format": "json",
        "key": "external.n_rows",
        "expected": 20945,
        "tolerance": 0,
    },
    {
        "name": "primary required columns present",
        "path": "outputs/primary/logs/audit_summary.json",
        "format": "json",
        "key": "primary.missing_required_cols",
        "expected": [],
        "comparison": "exact",
    },
    {
        "name": "external minimum required columns present",
        "path": "outputs/primary/logs/audit_summary.json",
        "format": "json",
        "key": "external.missing_min_required_cols",
        "expected": [],
        "comparison": "exact",
    },
    {
        "name": "smoke tests have zero failures",
        "path": "outputs/primary/logs/99_smoke_test_summary.json",
        "format": "json",
        "key": "failed",
        "expected": 0,
        "tolerance": 0,
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare reproduced CSV/JSON values against expected manuscript values."
    )
    parser.add_argument(
        "--expected",
        help="Optional YAML file with a top-level 'checks' list. Defaults to built-in audit/smoke checks.",
    )
    return parser.parse_args()


def resolve_path(path: str | Path) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    return WORKFLOW_ROOT / p


def load_checks(expected_path: str | None) -> list[dict[str, Any]]:
    if expected_path is None:
        return DEFAULT_CHECKS
    path = resolve_path(expected_path)
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    checks = data.get("checks", [])
    if not isinstance(checks, list):
        raise ValueError("Expected YAML must contain a list at key 'checks'.")
    return checks


def load_json_value(path: Path, key: str) -> Any:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    value: Any = data
    for part in key.split("."):
        if isinstance(value, dict) and part in value:
            value = value[part]
        else:
            raise KeyError(f"JSON key not found: {key}")
    return value


def load_csv_value(path: Path, check: dict[str, Any]) -> Any:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    where = check.get("where", {})
    if where:
        rows = [
            row
            for row in rows
            if all(str(row.get(col, "")) == str(expected) for col, expected in where.items())
        ]
    if not rows:
        raise ValueError(f"No CSV rows matched check filter for {path}")
    row_index = int(check.get("row_index", 0))
    if row_index >= len(rows):
        raise IndexError(f"row_index {row_index} out of range for {path}")
    column = check["column"]
    if column not in rows[row_index]:
        raise KeyError(f"CSV column not found: {column}")
    return rows[row_index][column]


def read_observed(check: dict[str, Any]) -> Any:
    path = resolve_path(check["path"])
    if not path.exists():
        raise FileNotFoundError(f"Output file not found: {path}")
    fmt = str(check.get("format", path.suffix.lstrip(".").lower())).lower()
    if fmt == "json":
        return load_json_value(path, str(check["key"]))
    if fmt == "csv":
        return load_csv_value(path, check)
    raise ValueError(f"Unsupported check format: {fmt}")


def compare_values(observed: Any, expected: Any, check: dict[str, Any]) -> tuple[bool, str]:
    comparison = str(check.get("comparison", "")).lower()
    if comparison == "exact":
        ok = observed == expected
        return ok, f"observed={observed!r}, expected={expected!r}"

    tolerance = float(check.get("tolerance", 0))
    try:
        obs_float = float(observed)
        exp_float = float(expected)
    except (TypeError, ValueError):
        ok = observed == expected
        return ok, f"observed={observed!r}, expected={expected!r}"

    ok = math.isclose(obs_float, exp_float, rel_tol=0.0, abs_tol=tolerance)
    return ok, f"observed={obs_float:.12g}, expected={exp_float:.12g}, tolerance={tolerance:.12g}"


def main() -> int:
    args = parse_args()
    checks = load_checks(args.expected)
    if not checks:
        print("SKIP no manuscript value checks configured.")
        return 0

    failed = 0
    for check in checks:
        name = str(check.get("name", "unnamed check"))
        try:
            observed = read_observed(check)
            ok, detail = compare_values(observed, check.get("expected"), check)
        except Exception as exc:
            ok = False
            detail = f"{type(exc).__name__}: {exc}"
        status = "PASS" if ok else "FAIL"
        print(f"{status}\t{name}\t{detail}")
        failed += 0 if ok else 1

    if failed:
        print(f"FAILED {failed} manuscript value check(s).")
        return 1
    print(f"PASSED {len(checks)} manuscript value check(s).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
