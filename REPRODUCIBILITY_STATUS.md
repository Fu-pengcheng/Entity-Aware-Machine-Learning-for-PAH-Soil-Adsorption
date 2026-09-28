# Reproducibility Status

This repository provides the data, code, and workflow needed to verify the analysis logic and conclusion-level findings of the PAH soil adsorption study.

It is not positioned as a digit-for-digit replay of every historical exploratory manuscript value. Small numerical differences are expected in stochastic machine-learning workflows because of random sampling, parallel training, BLAS/OpenMP behavior, package versions, and hardware differences.

## Full Primary Workflow

The full primary workflow was executed successfully on the remote server.

Completed primary scripts:

- `scripts/00_audit_datasets.py`: PASS
- `scripts/01_primary_empirical_baselines.py`: PASS
- `scripts/02_primary_ml_validation.py`: PASS
- `scripts/03_primary_layer_ablation.py`: PASS
- `scripts/04_primary_ad_diagnosis.py`: PASS
- `scripts/05_primary_few_shot_calibration.py`: PASS
- `scripts/06_primary_error_decomposition.py`: PASS

The workflow ran without `--smoke` and did not run external validation.

## Conclusion-Level Checks

Conclusion-level checks passed for the full primary outputs.

The public check standard is:

- dataset identity is preserved (`n_rows`, `n_soils`, `n_pahs`);
- random-split XGBoost performance remains high;
- strict LOSO performance remains meaningfully lower but still strong;
- the generalization gap remains positive and substantial;
- weak-support samples have higher error than low-support samples;
- soil-level bias explains a substantial fraction of residual structure;
- few-shot target-soil calibration remains high-performing.

Results are stored in:

- `outputs/manuscript_check/conclusion_level_checks.csv`
- `outputs/manuscript_check/conclusion_level_checks.json`

## Smoke-Tested Components

The smoke-test workflow completed successfully before the full primary run:

- real Excel readability checks: PASS;
- dataset audit: PASS;
- synthetic smoke execution for scripts `01`-`07`: PASS.

Smoke outputs are code-path checks only and should not be interpreted as manuscript results.

## External Validation

External validation has not been executed at manuscript level in this repository state.

The blocker remains unresolved: the current external raw dataset does not include confirmed compound-level descriptors:

- `logP`
- `logS`

These should be supplied through `compound_lookup_path` in `configs/external_neutral_organic.yaml` using a reliable compound join key such as `CAS Number`.

## Commands Used

Remote setup:

```bash
git clone --branch reproducibility/full-primary-run --single-branch https://github.com/Fu-pengcheng/Entity-Aware-Machine-Learning-for-PAH-Soil-Adsorption.git
cd /mnt/c/Users/PS/workspace/Entity-Aware-Machine-Learning-for-PAH-Soil-Adsorption
```

Lightweight checks:

```bash
python3 -m py_compile src/calibration.py scripts/check_reproducibility_conclusions.py scripts/run_all_primary.py
python3 scripts/99_run_smoke_tests.py
```

Full primary run and conclusion-level check:

```bash
python3 scripts/run_all_primary.py
python3 scripts/check_reproducibility_conclusions.py
```

## Run Environment Summary

- Date/time: 2026-06-04 CST
- Remote host: `PUERSAI-HPC`
- Working directory: `/mnt/c/Users/PS/workspace/Entity-Aware-Machine-Learning-for-PAH-Soil-Adsorption`
- Full primary run input commit: `91a7d7c Add primary reproduction entrypoint and manuscript value checks`
- Python: `python3 3.10.12`
- pip: `pip 22.0.2`

Full environment details are recorded in `outputs/run_logs/remote_environment.txt`.

## Revision Analyses (Scripts 08–13)

The following scripts were added during manuscript revision and reproduce the additional robustness and sensitivity analyses reported in the revised manuscript and Supplementary Materials:

- `scripts/08_fewshot_freundlich_baseline.py`: few-shot Freundlich isotherm baselines fitted per eligible soil on the same calibration samples (main-text Table 3 final column).
- `scripts/09_ad_mahalanobis_vs_euclidean.py`: applicability-domain analysis recomputed with fold-internal Mahalanobis distance (ridge-regularized covariance), reported alongside Euclidean results (Table S8).
- `scripts/10_hyperparameter_sensitivity.py`: one-at-a-time and joint perturbations of key XGBoost hyperparameters under the full LOSO protocol (Table S7).
- `scripts/11_model_comparison_details.py`: per-model comparison details underlying the Layer 2 → Layer 3 gains across tree-based models (Table S6).
- `scripts/12_layer_ablation_shap_folds.py`: per-fold SHAP attribution shares, rank statistics, and Top-5 frequencies across all 142 LOSO folds (revised Fig. 6).
- `scripts/13_redraw_figs.py`: revised main-text figures with explicit variability information (Figs. 3, 5, 6).

## External Validation

`scripts/07_external_validation.py` implements the external-dataset workflow. The external entity grouping used in the revised manuscript is based on the original soil identifiers obtained from the database developers (Sun et al.); the per-record entity assignment is provided as a row-index mapping in the manuscript's Data S1 rather than redistributed in this repository.
