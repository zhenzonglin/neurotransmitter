# NT-CLSM DAT Pilot

This repository runs a DAT-focused NT-CLSM pilot for acute ischemic stroke.
The workflow is lesion-first and uses existing projects for the main analyses:
NiiStat for node and voxelwise CLSM, LQT-R plus DSI Studio for edge
disconnectivity, and bctpy/BCT-style correction for network-level edge
statistics. NeMo code is retained as an optional legacy branch, but it is not
used in the main analysis.

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

```bash
python scripts/fetch_reference_data.py --config config/dat_config.yaml --all
python scripts/fetch_lqt_data.py --config config/dat_config.yaml
python scripts/prepare_inputs.py --config config/dat_config.yaml
matlab -batch "run('scripts/run_niistat_node_wm.m')"
Rscript scripts/run_lqt_edges.R
python scripts/collect_results.py --config config/dat_config.yaml
```

Optional legacy NeMo commands remain in `scripts/run_nemo_edges.py`, but NeMo
requires authenticated requester-pays S3 access for the database and is no
longer part of the main DAT-edge pipeline.

Large imaging data, LQT resources, NeMo database files, and derivatives are
excluded from git.
