# Analysis Workflow Skeleton (PAH Soil Adsorption)

This folder is a lightweight, GitHub-ready workflow skeleton migrated from existing local analysis code.

This repository provides an **analysis workflow scaffold** plus **reproducible scripts**, not a claim that all final manuscript results are already regenerated.

## Scope of this stage

- Reuse existing logic where possible.
- Standardize structure, entry scripts, and YAML-based paths.
- Keep raw data untouched.
- Run smoke tests only:
  - Real Excel files: read + audit only.
  - Modeling scripts: run on tiny synthetic data only.
- Do **not** run full LOSO/random/few-shot/external full experiments in this stage unless explicitly approved.

## Raw data placement (read-only)

Place raw data under `Data/raw/` in the project root (outside this folder).
Raw data are **not** distributed by this repository.

Required filenames:

- `Data/raw/pah_soil_adsorption_primary_raw.xlsx`
- `Data/raw/neutral_organic_adsorption_external_raw.xlsx`

## Install

```bash
pip install -r requirements.txt
```

## Smoke test run

From `analysis_workflow/`:

```bash
python scripts/00_audit_datasets.py
python scripts/99_run_smoke_tests.py
```

Current scaffold status: smoke testing has been completed successfully for dataset readability, auditing, and script executability on tiny synthetic inputs.

## Full workflow commands (manual, optional)

Only run these after explicit confirmation for full experiments.
Before full runs, users must confirm field mapping consistency (especially external features such as `logP/logS`) and available compute resources.

```bash
python scripts/01_primary_empirical_baselines.py --config configs/primary_pah.yaml
python scripts/02_primary_ml_validation.py --config configs/primary_pah.yaml
python scripts/03_primary_layer_ablation.py --config configs/primary_pah.yaml
python scripts/04_primary_ad_diagnosis.py --config configs/primary_pah.yaml
python scripts/05_primary_few_shot_calibration.py --config configs/primary_pah.yaml
python scripts/06_primary_error_decomposition.py --config configs/primary_pah.yaml
python scripts/07_external_validation.py --config configs/external_neutral_organic.yaml
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
