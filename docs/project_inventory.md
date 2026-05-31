# Project Inventory

This file records the cleaned project layout after removing DAT-only legacy
outputs, NiiStat/NeMo-era helper code, and pilot subject expansion logic.

## Main Scripts

- `scripts/fetch_reference_data.py`: download AtlasPack, Hansen maps, and Alves/Functionnectome maps.
- `scripts/fetch_lqt_data.py`: download LQT public resources.
- `scripts/prepare_inputs.py`: build QC tables and shared atlas files.
- `scripts/run_lqt_edges.R`: compute lesion-only LQT edge disconnection.
- `scripts/run_ml_profile_analysis.py`: screen neurotransmitter systems in training folds, integrate selected voxel maps, recompute node/edge damage, and evaluate prediction models.
- `scripts/generate_html_report.py`: generate the flow-style HTML report and figures.
- `scripts/compute_impact_scores.py`: reusable impact-score and prediction functions imported by `run_ml_profile_analysis.py`.

## Main Results

- `derivatives/qc/`: active subject manifest and lesion QC tables.
- `derivatives/shared/lesion_node_load.csv`: lesion-only node load features.
- `derivatives/shared/lqt_edge_disconnection.csv`: lesion-only structural disconnection edge features.
- `derivatives/shared/edge_tract_voxels_2mm.npz`: edge-by-voxel tract mask matrix for Alves-weighted edge damage.
- `derivatives/ml_profile/`: fold-specific NT selection, integrated profiles, NTDC scores, LSM maps, fixed D1/D2/DAT exploratory outputs, and prediction models.
- `derivatives/reports/`: HTML flow report and generated PNG figures.

## Removed Legacy Items

- DAT-only NiiStat node and WM voxelwise outputs.
- DAT-only edge CLSM matrices and integrated-model outputs.
- Integrated 13-NT profile branch.
- Single-neurotransmitter NTDC prediction branch.
- Old subject-level machine-learning branch.
- Old `collect_results.py`, NiiStat runner scripts, post-processing scripts, and temporary expansion helper.
- Pilot-only table duplication logic.
- Local NiiStat and bctpy external clones.
- Duplicate top-level DAT raw and processed maps.
