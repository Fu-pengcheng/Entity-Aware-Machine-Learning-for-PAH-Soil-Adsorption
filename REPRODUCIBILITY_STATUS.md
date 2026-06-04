# Reproducibility Status

This repository contains a manuscript reproducibility workflow for the PAH soil adsorption study. The current remote run was executed on the target branch `reproducibility/full-primary-run`.

## Full Primary Run Status

Full primary workflow execution status: **completed successfully**.

Primary scripts completed on the remote server:

- `scripts/00_audit_datasets.py`: PASS
- `scripts/01_primary_empirical_baselines.py`: PASS
- `scripts/02_primary_ml_validation.py`: PASS
- `scripts/03_primary_layer_ablation.py`: PASS
- `scripts/04_primary_ad_diagnosis.py`: PASS
- `scripts/05_primary_few_shot_calibration.py`: PASS
- `scripts/06_primary_error_decomposition.py`: PASS

The primary workflow ran without `--smoke` and did not run external validation.

## Manuscript Value Check

The manuscript value checker completed with:

- PASS: 5
- FAIL: 5
- MISSING: 0

Because not all manuscript value checks passed, the primary numerical results should be described as **full primary workflow executed, but manuscript numerical agreement is partial** rather than fully reproduced.

Checked values:

| item | expected | reproduced | tolerance | status |
| --- | ---: | ---: | ---: | --- |
| n_rows | 1408 | 1408 | 0 | PASS |
| n_soils | 142 | 142 | 0 | PASS |
| n_pahs | 5 | 5 | 0 | PASS |
| xgb_random_r2 | 0.939 | 0.9265399101441332 | 0.002 | FAIL |
| xgb_loso_r2 | 0.649 | 0.6422760767615363 | 0.002 | FAIL |
| xgb_generalization_gap | 0.290 | 0.28426383338259686 | 0.003 | FAIL |
| support_q4_q1_mae_ratio | 2.02 | 1.9167351381455093 | 0.03 | FAIL |
| soil_level_bias_ratio | 0.843 | 0.8206575466440662 | 0.003 | FAIL |
| fewshot_m3_calibrated_r2 | 0.901 | 0.8982122728560146 | 0.003 | PASS |
| fewshot_m5_calibrated_r2 | 0.908 | 0.9068154273317883 | 0.003 | PASS |

No manuscript values were manually corrected.

## Smoke-Tested Components

The smoke-test workflow also completed successfully on the remote server before the full primary run:

- Real Excel readability checks: PASS
- Dataset audit: PASS
- Synthetic smoke execution for scripts `01`-`07`: PASS

The smoke outputs are not manuscript results.

## Not Yet Reproducible External Analyses

External validation was intentionally not run. It remains not yet reproducible at manuscript level because the current external raw dataset does not include confirmed compound-level descriptors:

- `logP`
- `logS`

These should be supplied through `compound_lookup_path` in `configs/external_neutral_organic.yaml` using a reliable compound join key such as `CAS Number`.

## Known Blockers

- External validation requires confirmed `logP/logS` lookup data.
- Five primary manuscript target checks currently fail tolerance matching; these differences should be investigated rather than manually edited.
- The remote environment has `python3` but no `python` command.

## Exact Commands Used

Remote repository setup:

```bash
git clone --branch reproducibility/full-primary-run --single-branch https://github.com/Fu-pengcheng/Entity-Aware-Machine-Learning-for-PAH-Soil-Adsorption.git
cd /mnt/c/Users/PS/workspace/Entity-Aware-Machine-Learning-for-PAH-Soil-Adsorption
```

Environment and lightweight checks:

```bash
python3 -m py_compile src/calibration.py scripts/check_manuscript_values.py scripts/run_all_primary.py
python3 scripts/99_run_smoke_tests.py
python3 scripts/check_manuscript_values.py
```

Full primary run and manuscript check:

```bash
python3 scripts/run_all_primary.py
python3 scripts/check_manuscript_values.py
```

## Run Environment Summary

- Date/time: 2026-06-04 CST
- Remote host: `PUERSAI-HPC`
- Working directory: `/mnt/c/Users/PS/workspace/Entity-Aware-Machine-Learning-for-PAH-Soil-Adsorption`
- Full primary run input commit: `91a7d7c Add primary reproduction entrypoint and manuscript value checks`
- Python: `python3 3.10.12`
- pip: `pip 22.0.2`

Full environment details are recorded in `outputs/run_logs/remote_environment.txt`.