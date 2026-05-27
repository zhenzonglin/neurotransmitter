# NT-CLSM DAT Pilot

This repository runs a DAT-focused NT-CLSM pilot for acute ischemic stroke.
The workflow is lesion-first and uses existing projects for the main analyses:
NiiStat for node and voxelwise CLSM, LQT-R plus DSI Studio for edge
disconnectivity, and bctpy/BCT-style correction for network-level edge
statistics.

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
Rscript scripts/install_lqt_r_deps.R
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
python scripts/fetch_reference_data.py --config "${config_path}" --all
python scripts/fetch_lqt_data.py --config "${config_path}"
python scripts/prepare_inputs.py --config "${config_path}"
bash scripts/run_niistat_node_wm.sh --config "${config_path}"
Rscript scripts/run_lqt_edges.R --config "${config_path}"
python scripts/compute_dat_impact_scores.py --config "${config_path}"
python scripts/collect_results.py --config "${config_path}"
```

## Configuration

Change working path, input files, outcome, covariates, binary threshold, and
analysis table names in `notebooks/00_project_config.ipynb`, then rerun its
write-config cell. The scripts infer cohort size from
`derivatives/qc/subject_manifest.csv`.

`compute_dat_impact_scores.py` implements the SDC/FDC-style impact layer:
10-fold discovery-validation, key lesion/DAT node-edge selection, patient-level
impact scores, and six ordered mRS model comparisons:
clinical-only, clinical + lesion impact, clinical + DAT node impact,
clinical + DAT edge impact, clinical + DAT node + DAT edge impact, and
clinical + lesion impact + DAT node + DAT edge impact. The primary prediction
comparison uses 10-fold out-of-sample ordered mRS probabilities and bootstrap
pairwise metric differences. In-sample likelihood comparisons are kept as
auxiliary model-fit summaries.

Large imaging data, LQT resources, and derivatives are excluded from git.
