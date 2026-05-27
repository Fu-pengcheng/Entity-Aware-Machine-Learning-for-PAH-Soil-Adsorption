from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

WORKFLOW_ROOT = Path(__file__).resolve().parents[1]
if str(WORKFLOW_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKFLOW_ROOT))

from src.io_utils import configure_output_roots, load_yaml, read_table, write_df, write_json  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run smoke tests for analysis workflow.")
    parser.add_argument("--python", default=sys.executable)
    return parser.parse_args()


def run_cmd(cmd: list[str], cwd: Path) -> tuple[int, str, str, float]:
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True)
    dt = time.time() - t0
    return proc.returncode, proc.stdout, proc.stderr, dt


def main() -> None:
    args = parse_args()
    rows = []

    p_cfg = load_yaml("configs/primary_pah.yaml")
    e_cfg = load_yaml("configs/external_neutral_organic.yaml")
    out_cfg = p_cfg.get("output_paths", {})
    e_out_cfg = e_cfg.get("output_paths", {})
    configure_output_roots(
        primary_root=out_cfg.get("primary_root"),
        external_root=e_out_cfg.get("external_root", out_cfg.get("external_root")),
    )

    for tag, cfg in [("primary_excel_read", p_cfg), ("external_excel_read", e_cfg)]:
        try:
            df = read_table(cfg["input_data_path"], sheet_name=cfg.get("sheet_name", 0))
            rows.append(
                {
                    "test_name": tag,
                    "status": "PASS",
                    "n_rows": int(len(df)),
                    "message": "read ok",
                    "seconds": 0.0,
                }
            )
        except Exception as e:
            rows.append(
                {
                    "test_name": tag,
                    "status": "FAIL",
                    "n_rows": 0,
                    "message": str(e),
                    "seconds": 0.0,
                }
            )

    script_calls = [
        ["scripts/00_audit_datasets.py"],
        ["scripts/01_primary_empirical_baselines.py", "--smoke"],
        ["scripts/02_primary_ml_validation.py", "--smoke"],
        ["scripts/03_primary_layer_ablation.py", "--smoke"],
        ["scripts/04_primary_ad_diagnosis.py", "--smoke"],
        ["scripts/05_primary_few_shot_calibration.py", "--smoke"],
        ["scripts/06_primary_error_decomposition.py", "--smoke"],
        ["scripts/07_external_validation.py", "--smoke"],
    ]

    for call in script_calls:
        cmd = [args.python] + call
        code, out, err, sec = run_cmd(cmd, cwd=WORKFLOW_ROOT)
        status = "PASS" if code == 0 else "FAIL"
        rows.append(
            {
                "test_name": " ".join(call),
                "status": status,
                "n_rows": None,
                "message": (out.strip()[-400:] if out.strip() else err.strip()[-400:]),
                "seconds": round(sec, 3),
            }
        )

    result_df = pd.DataFrame(rows)
    write_df(result_df, "outputs/primary/logs/99_smoke_test_summary.csv")
    write_df(result_df, "outputs/external/logs/99_smoke_test_summary.csv")
    write_json(
        {
            "total_tests": int(len(result_df)),
            "passed": int((result_df["status"] == "PASS").sum()),
            "failed": int((result_df["status"] == "FAIL").sum()),
        },
        "outputs/primary/logs/99_smoke_test_summary.json",
    )
    print(result_df.to_string(index=False))


if __name__ == "__main__":
    main()
