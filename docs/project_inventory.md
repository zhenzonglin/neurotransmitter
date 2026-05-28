# Project Inventory

This file records the cleaned project layout after removing DAT-only legacy
outputs and NiiStat/NeMo-era helper code.

## Main Scripts

- `scripts/fetch_reference_data.py`: download AtlasPack, Hansen maps, and Alves/Functionnectome maps.
- `scripts/fetch_lqt_data.py`: download LQT public resources.
- `scripts/prepare_inputs.py`: build QC tables and shared atlas files.
- `scripts/run_lqt_edges.R`: compute lesion-only LQT edge disconnection.
- `scripts/run_multi_nt_analysis.py`: run node, Alves-weighted edge, impact-score, and prediction analyses for every configured neurotransmitter system.
- `scripts/run_nt_profile_analysis.py`: build an integrated 13-NT voxel profile and run node/edge LSM plus prediction analyses.
- `scripts/compute_impact_scores.py`: reusable impact-score and prediction functions imported by `run_multi_nt_analysis.py`.

## Main Results

- `derivatives/qc/`: active subject manifest and lesion QC tables.
- `derivatives/shared/lesion_node_load.csv`: lesion-only node load features.
- `derivatives/shared/lqt_edge_disconnection.csv`: lesion-only structural disconnection edge features.
- `derivatives/shared/edge_tract_voxels_2mm.npz`: edge-by-voxel tract mask matrix for Alves-weighted edge damage.
- `derivatives/nt/<nt_id>/`: per-neurotransmitter maps, node features, Alves-weighted edge features, impact scores, and models.
- `derivatives/nt/summary/`: cross-neurotransmitter summary tables and run report.
- `derivatives/nt_profile/integrated_13nt/`: integrated 13-NT profile maps, node/edge LSM statistics, impact scores, and prediction models.

## Removed Legacy Items

- DAT-only NiiStat node and WM voxelwise outputs.
- DAT-only edge CLSM matrices and integrated-model outputs.
- Old `collect_results.py`, NiiStat runner scripts, post-processing scripts, and temporary expansion helper.
- Local NiiStat and bctpy external clones.
- Duplicate top-level DAT raw and processed maps.
