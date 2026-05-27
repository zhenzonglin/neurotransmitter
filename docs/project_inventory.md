# Project Inventory

This file records the cleaned project layout after removing DAT-only legacy
outputs and NiiStat/NeMo-era helper code.

## Main Scripts

- `scripts/fetch_reference_data.py`: download AtlasPack, Hansen maps, and Alves/Functionnectome maps.
- `scripts/fetch_lqt_data.py`: download LQT public resources.
- `scripts/prepare_inputs.py`: build QC tables and shared atlas files.
- `scripts/run_lqt_edges.R`: compute lesion-only LQT edge disconnection.
- `scripts/run_multi_nt_analysis.py`: run node, edge, WM, impact-score, and prediction analyses for every configured neurotransmitter system.
- `scripts/compute_impact_scores.py`: reusable impact-score and prediction functions imported by `run_multi_nt_analysis.py`.

## Main Results

- `derivatives/qc/`: active subject manifest and lesion QC tables.
- `derivatives/shared/lesion_node_load.csv`: lesion-only node load features.
- `derivatives/shared/lqt_edge_disconnection.csv`: lesion-only structural disconnection edge features.
- `derivatives/nt/<nt_id>/`: per-neurotransmitter maps, features, impact scores, and models.
- `derivatives/nt/summary/`: cross-neurotransmitter summary tables and run report.

## Removed Legacy Items

- DAT-only NiiStat node and WM voxelwise outputs.
- DAT-only edge CLSM matrices and integrated-model outputs.
- Old `collect_results.py`, NiiStat runner scripts, post-processing scripts, and temporary expansion helper.
- Local NiiStat and bctpy external clones.
- Duplicate top-level DAT raw and processed maps.
