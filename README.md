# NT-CLSM Pilot

This repository runs neurotransmitter-focused NT-CLSM pilots for acute ischemic stroke.
The workflow is lesion-first: MNI 2mm lesion masks are mapped to 4S156 nodes,
LQT-R/DSI Studio estimates lesion-only structural disconnection edges, and
Hansen plus Alves/Functionnectome neurotransmitter maps weight node and edge
damage features.

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
Rscript scripts/install_lqt_r_deps.R --project-dir /home/zhenzong2/analysis/neurotransmitter
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
python scripts/fetch_reference_data.py --config "${config_path}" --maps
python scripts/fetch_lqt_data.py --config "${config_path}"
python scripts/prepare_inputs.py --config "${config_path}"
Rscript scripts/run_lqt_edges.R --config "${config_path}"
python scripts/build_edge_tract_matrix.py --config "${config_path}"
python scripts/run_multi_nt_analysis.py --config "${config_path}"
python scripts/run_nt_profile_analysis.py --config "${config_path}"
```

## Configuration

Change working path, input files, outcome, covariates, binary threshold, and
analysis table names in `notebooks/00_project_config.ipynb`, then rerun its
write-config cell. The scripts infer cohort size from
`derivatives/qc/subject_manifest.csv`.

`run_multi_nt_analysis.py` applies the same node, edge, impact-score, and
prediction workflow to all neurotransmitter maps configured under
`neurotransmitters` in `config/dat_config.yaml`. Current configured systems are
`a4b2`, `m1`, `vacht`, `d1`, `d2`, `dat`, `nat`, `5ht1a`, `5ht1b`, `5ht2a`,
`5ht4`, `5ht6`, and `5htt`. Outputs are written to:

```text
derivatives/nt/<nt_id>/
derivatives/nt/summary/
```

Each receptor/transporter folder contains:

```text
atlases/<nt_id>_hansen_gray_2mm.nii.gz
atlases/<nt_id>_alves_wm_2mm.nii.gz
node/nt_roi_156.csv
node/nt_node_damage.csv
edge/nt_edge_lqt.csv
impact/nt_impact_scores.csv
models/model_prediction_performance.csv
models/model_prediction_pairwise_bootstrap.csv
```

`edge/nt_edge_lqt.csv` is computed as:

```text
edge_nt_damage[i,e] = sum_v lesion_i(v) * edge_tract_mask_e(v) * Alves_NT_WM(v)
```

The shared tract mask matrix is stored at:

```text
derivatives/shared/edge_tract_voxels_2mm.npz
```

`scripts/compute_impact_scores.py` contains the reusable impact-score and
prediction functions. It is imported by `run_multi_nt_analysis.py`; it is not a
separate DAT-only pipeline entrypoint.

## Integrated 13-NT Profile

`run_nt_profile_analysis.py` keeps the single-NT outputs unchanged and creates a
separate integrated profile branch:

```text
derivatives/nt_profile/integrated_13nt/
```

It robustly scales each of the 13 Hansen gray maps and 13 Alves WM maps to
0-1, then averages them voxel by voxel. The node feature is computed inside
each ROI as:

```text
node_profile_damage[i,r] =
sum_v lesion_i(v) * ROI_r(v) * integrated_Hansen_profile(v) / sum_v ROI_r(v)
```

The edge feature is computed as:

```text
edge_profile_damage[i,e] =
sum_v lesion_i(v) * edge_tract_mask_e(v) * integrated_Alves_WM_profile(v)
```

The branch writes node/edge LSM statistics, 10-fold impact scores, and the same
prediction model comparison tables as the single-NT workflow.

Large imaging data, LQT resources, and derivatives are excluded from git.
