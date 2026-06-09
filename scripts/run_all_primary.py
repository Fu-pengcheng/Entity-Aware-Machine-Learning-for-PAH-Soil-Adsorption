from __future__ import annotations

import csv
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path


WORKFLOW_ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = WORKFLOW_ROOT / "outputs" / "run_logs"

PRIMARY_STEPS = [
    "scripts/00_audit_datasets.py",
    "scripts/01_primary_empirical_baselines.py",
    "scripts/02_primary_ml_validation.py",
    "scripts/03_primary_layer_ablation.py",
    "scripts/04_primary_ad_diagnosis.py",
    "scripts/05_primary_few_shot_calibration.py",
    "scripts/06_primary_error_decomposition.py",
]


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def run_step(step_index: int, script_path: str) -> dict[str, object]:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    step_name = Path(script_path).stem
    stdout_path = LOG_DIR / f"{step_index:02d}_{step_name}.stdout.log"
    stderr_path = LOG_DIR / f"{step_index:02d}_{step_name}.stderr.log"

    start_dt = datetime.now()
    cmd = [sys.executable, script_path]
    proc = subprocess.run(
        cmd,
        cwd=WORKFLOW_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    end_dt = datetime.now()

    stdout_path.write_text(proc.stdout, encoding="utf-8")
    stderr_path.write_text(proc.stderr, encoding="utf-8")

    return {
        "step_index": step_index,
        "script": script_path,
        "command": " ".join(cmd),
        "start_time": start_dt.isoformat(timespec="seconds"),
        "end_time": end_dt.isoformat(timespec="seconds"),
        "duration_seconds": round((end_dt - start_dt).total_seconds(), 3),
        "returncode": int(proc.returncode),
        "status": "PASS" if proc.returncode == 0 else "FAIL",
        "stdout_log": str(stdout_path.relative_to(WORKFLOW_ROOT)),
        "stderr_log": str(stderr_path.relative_to(WORKFLOW_ROOT)),
    }


def write_summary(rows: list[dict[str, object]]) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = LOG_DIR / "run_all_primary_summary.csv"
    json_path = LOG_DIR / "run_all_primary_summary.json"
    fieldnames = [
        "step_index",
        "script",
        "command",
        "start_time",
        "end_time",
        "duration_seconds",
        "returncode",
        "status",
        "stdout_log",
        "stderr_log",
    ]
    with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    json_path.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> int:
    rows: list[dict[str, object]] = []
    for idx, script in enumerate(PRIMARY_STEPS, start=1):
        print(f"[{now_iso()}] running {script}")
        row = run_step(idx, script)
        rows.append(row)
        write_summary(rows)
        print(f"[{now_iso()}] {script} -> {row['status']} in {row['duration_seconds']} s")
        if row["status"] != "PASS":
            print(f"Stopping after failed step: {script}", file=sys.stderr)
            return int(row["returncode"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
