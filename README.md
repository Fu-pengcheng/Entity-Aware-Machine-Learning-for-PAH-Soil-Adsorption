# Analysis Workflow (PAH Soil Adsorption)

This repository provides supporting code, raw data, and analysis scripts for conclusion-level verification of the PAH soil adsorption study.

The goal is to make the analysis logic, validation design, and main scientific conclusions inspectable and rerunnable. It is not intended to reproduce every historical exploratory manuscript number digit-for-digit.

## Scope

- Reuse existing logic where possible.
- Standardize structure, entry scripts, and YAML-based paths.
- Keep raw data untouched.
- Provide smoke tests for code-path checks.
- Provide a full primary workflow for the primary PAH dataset.
- Provide conclusion-level checks for core findings.
- Keep external validation separate until the remaining descriptor lookup is resolved.

Machine-learning workflows can vary slightly across machines because of stochastic training, parallel execution, BLAS/OpenMP behavior, and dependency versions. This repository therefore checks whether reproduced outputs support the same analysis logic and conclusions, rather than treating exact historical values as the public pass/fail criterion.

## Raw data availability and placement (read-only)

The two raw Excel datasets used by the workflow are currently tracked in this repository under `Data/raw/` to support manuscript-level reproducibility checks. Treat these files as read-only inputs: do not move, overwrite, or edit them during analysis.

If a downstream copy of the repository omits raw data for policy or storage reasons, place the same files under `Data/raw/` before running audits or full analyses.

Required filenames:

- `Data/raw/pah_soil_adsorption_primary_raw.xlsx`
- `Data/raw/neutral_organic_adsorption_external_raw.xlsx`

## Install

```bash
pip install -r requirements.txt
```

## Smoke Test Run

From `analysis_workflow/`:

```bash
python scripts/00_audit_datasets.py
python scripts/99_run_smoke_tests.py
```

Smoke tests check dataset readability, auditing, and script executability on tiny synthetic inputs. Smoke outputs are not manuscript results.

## Full Primary Workflow

Run full primary analyses on a machine with adequate compute resources:

```bash
python scripts/run_all_primary.py
```

This runs the primary audit, empirical baselines, random-split vs strict-LOSO validation, descriptor-layer ablation, support-distance diagnosis, few-shot calibration, and error decomposition. It does not run external validation.

## Conclusion-Level Checks

After full primary outputs are generated, run:

```bash
python scripts/check_reproducibility_conclusions.py
```

The checker writes:

- `outputs/manuscript_check/conclusion_level_checks.csv`
- `outputs/manuscript_check/conclusion_level_checks.json`

These checks verify dataset identity and conclusion-level thresholds, for example strong random-split performance, stricter LOSO performance, a positive generalization gap, larger weak-support error, substantial soil-level bias contribution, and few-shot calibration remaining high-performing.

## Individual Workflow Commands (manual, optional)

Only run these after explicit confirmation for full experiments.
Before full runs, users should confirm field mapping consistency and available compute resources. External validation should not be run until external features such as `logP/logS` are supplied and confirmed.

```bash
python scripts/01_primary_empirical_baselines.py --config configs/primary_pah.yaml
python scripts/02_primary_ml_validation.py --config configs/primary_pah.yaml
python scripts/03_primary_layer_ablation.py --config configs/primary_pah.yaml
python scripts/04_primary_ad_diagnosis.py --config configs/primary_pah.yaml
python scripts/05_primary_few_shot_calibration.py --config configs/primary_pah.yaml
python scripts/06_primary_error_decomposition.py --config configs/primary_pah.yaml
```

## Project structure

- `configs/`: dataset and run configs
- `scripts/`: stage entrypoints (`00`-`07`, `99`)
- `src/`: reusable modules (I/O, preprocessing, validation, AD, calibration, etc.)
- `outputs/`: generated tables/figures/predictions/logs for primary and external tracks

## Leakage control rules (enforced)

- No global pre-imputation.
- Imputer fits on train fold only.
- Scaler fits on train fold only.
- Test fold does not participate in preprocessing fit/tuning/training.
- Group column (e.g., `Soil_ID`) is never used as model feature.
- Few-shot calibration samples are excluded from post-calibration test metrics.

## Migration provenance

Core logic was migrated/reused from local source folders:

- `code/src/data/*`
- `code/src/metrics/*`
- `code/src/pipelines/run_td_only.py`
- `code/src/pipelines/run_cec_sensitivity.py`
- `code/src/pipelines/run_quantum_ablation.py`
- `code/src/pipelines/run_dk_weighted.py`
- `code/tools/run_fewshot_target_soil_calibration.py`
- `code/tools/analyze_fewshot_deep_dive.py`
- `external_validation/sun_cross_validation_strict_constraints.py` (framework-level reference)

## Notes

- This repository stage is intentionally **not** a full reimplementation.
- It is a migration skeleton for reproducible open-source packaging and smoke validation.
