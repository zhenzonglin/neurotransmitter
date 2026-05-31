# NT-CLSM Analysis

This repository runs neurotransmitter-focused NT-CLSM analyses for acute ischemic stroke.
The workflow is lesion-first: MNI 2mm lesion masks are mapped to 4S156 nodes,
LQT-R/DSI Studio estimates lesion-only structural disconnection edges, and
Hansen plus Alves/Functionnectome neurotransmitter maps weight node and edge
damage features.

## GitHub And USB Migration

Use GitHub for code and small text files only. Use USB transfer for private
clinical data and large analysis resources.

GitHub can include:

```text
.gitignore
README.md
environment.yml
config/
docs/
notebooks/
scripts/
src/
atlases/.gitkeep
lesion/.gitkeep
external/sources.lock
```

USB transfer should include:

```text
phenotype.xlsx
lesion/
atlases/raw/
atlases/processed/
atlases/lqt/
external/lqt_data/
```

Optional USB transfer for faster resume:

```text
derivatives/shared/
derivatives/qc/
```

Do not upload `phenotype.xlsx`, lesion masks, atlas NIfTI files, LQT resources,
or `derivatives/` to GitHub.

## WSL entrypoint

```bash
wsl -d Ubuntu-20.04 -- bash -lc 'cd /home/zhenzong2/analysis/neurotransmitter && <command>'
```

## Setup

```bash
source /home/zhenzong/anaconda3/etc/profile.d/conda.sh
conda env create -f environment.yml
conda activate NT_analysis
python -m ipykernel install --user --name NT_analysis --display-name "Python (NT_analysis)"
Rscript scripts/install_lqt_r_deps.R --project-dir "$(pwd)"
```

## Pipeline

First edit and run:

```text
notebooks/00_project_config.ipynb
```

It writes `config/dat_config.yaml`. The scripts below receive all changing
paths, clinical variables, outcomes, and table names from that file.

```bash
config_path=config/dat_config.yaml
python scripts/prepare_inputs.py --config "${config_path}"
Rscript scripts/install_lqt_r_deps.R --project-dir "$(pwd)"
Rscript scripts/run_lqt_edges.R --config "${config_path}"
python scripts/build_edge_tract_matrix.py --config "${config_path}"
python scripts/run_ml_profile_analysis.py --config "${config_path}" --force-screening
python scripts/generate_html_report.py --config "${config_path}"
```

## Configuration

Change working path, input files, outcome, covariates, binary threshold, and
analysis table names in `notebooks/00_project_config.ipynb`, then rerun its
write-config cell. The scripts infer cohort size from
`derivatives/qc/subject_manifest.csv`.

The formal workflow uses one row per real subject. It does not duplicate
subjects for pilot-scale testing. Cross-validation groups default to
`subject_id` through `analysis.cv_group_column`; change that field only when
the same participant has repeated rows that must stay in the same fold.

`run_ml_profile_analysis.py` performs fold-specific neurotransmitter screening,
voxel-level profile integration, node/edge damage recomputation, LSM impact
scoring, and out-of-sample prediction. Current configured systems are `a4b2`,
`m1`, `vacht`, `d1`, `d2`, `dat`, `nat`, `5ht1a`, `5ht1b`, `5ht2a`, `5ht4`,
`5ht6`, and `5htt`. Outputs are written to:

```text
derivatives/ml_profile/
derivatives/ml_profile/models/
derivatives/ml_profile/profiles/
derivatives/ml_profile/lsm_maps/
derivatives/ml_profile/exploratory_profiles/
```

The main NTDC tables are:

```text
selection_summary.csv
selection_folds.csv
profile_scores.csv
models/model_prediction_performance.csv
models/model_prediction_pairwise_bootstrap.csv
```

For each outer fold, the integrated profiles are:

```text
profiles/fold_XX_ntdc_hansen_profile.nii.gz
profiles/fold_XX_ntdc_alves_profile.nii.gz
```

The shared tract mask matrix is stored at:

```text
derivatives/shared/edge_tract_voxels_2mm.npz
```

Prediction models use readable names:

```text
Clinical
Clinical + SDC
Clinical + NTDC
Clinical + SDC + NTDC
```

`SDC` is the lesion-only structural damage contribution. `NTDC` is computed
after training-fold neurotransmitter screening, voxel-profile integration, and
training-fold node/edge LSM weighting. Both scores use training-fold mean and
standard deviation for z-standardization before being applied to the held-out
fold.

The LSM maps are spatial projections of fold-specific high-dimensional weights.
The node map assigns the trained ROI weight back to all voxels in that ROI. The
edge map projects selected edge weights back to voxels crossed by those
ROI-to-ROI streamlines. These files show where the NTDC model placed spatial
weight; they are not whole-cohort voxelwise significance maps.

The fixed D1/D2/DAT exploratory branch uses an equal-weight dopamine profile
and writes separate out-of-sample prediction files under:

```text
derivatives/ml_profile/exploratory_profiles/
dopamine_d1_d2_dat_scores.csv
dopamine_d1_d2_dat_model_prediction_performance.csv
dopamine_d1_d2_dat_model_prediction_pairwise_bootstrap.csv
```

`scripts/compute_impact_scores.py` contains the reusable impact-score and
prediction functions. It is imported by `run_ml_profile_analysis.py`.

Large imaging data, LQT resources, and derivatives are excluded from git.
