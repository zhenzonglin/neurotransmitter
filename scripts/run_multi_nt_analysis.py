#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from scipy import sparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from build_edge_tract_matrix import build_edge_matrix  # noqa: E402
from compute_impact_scores import (  # noqa: E402
    compute_lesion_node_load,
    fit_ordinal_impact_model,
    load_lesion_feature_tables,
    run_cross_validated_impact,
    run_cross_validated_prediction,
    write_full_sample_keys,
)
from nt_analysis.config import analysis_table, ensure_dir, load_config, project_path  # noqa: E402
from nt_analysis.images import resample_like  # noqa: E402
from nt_analysis.tables import write_csv  # noqa: E402


def neurotransmitter_specs(config: dict) -> list[dict[str, object]]:
    """Return configured neurotransmitter maps."""
    specs = config.get("neurotransmitters", [])
    if not specs:
        raise RuntimeError("no neurotransmitters are configured")
    return specs


def nt_root(config: dict, nt_id: str) -> Path:
    """Return one neurotransmitter output folder."""
    return ensure_dir(project_path(config, "derivatives", "nt", nt_id))


def raw_paths(config: dict, spec: dict[str, object]) -> tuple[Path, Path]:
    """Return Hansen and Alves raw map paths."""
    raw_dir = project_path(config, config["atlases"]["raw_dir"])
    hansen = raw_dir / "hansen" / str(spec["hansen_file"])
    alves = raw_dir / "alves" / f"functionnectome_anat_{spec['alves_name']}.nii.gz"
    missing = [str(path) for path in [hansen, alves] if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing neurotransmitter maps: {missing}")
    return hansen, alves


def prepare_nt_maps(config: dict, spec: dict[str, object], reference_2mm: Path, out_dir: Path) -> dict[str, Path]:
    """Resample Hansen and Alves maps into the analysis space."""
    nt_id = str(spec["id"])
    hansen, alves = raw_paths(config, spec)
    atlas_dir = ensure_dir(out_dir / "atlases")
    gray_2mm = atlas_dir / f"{nt_id}_hansen_gray_2mm.nii.gz"
    wm_2mm = atlas_dir / f"{nt_id}_alves_wm_2mm.nii.gz"
    resample_like(hansen, reference_2mm, gray_2mm, "continuous")
    resample_like(alves, reference_2mm, wm_2mm, "continuous")
    return {"gray_2mm": gray_2mm, "wm_2mm": wm_2mm}


def compute_roi_table(config: dict, gray_2mm: Path, out_dir: Path) -> pd.DataFrame:
    """Compute ROI-level Hansen map means."""
    atlas_img = nib.load(str(project_path(config, config["atlases"]["outputs"]["atlas4s156_2mm"])))
    map_img = nib.load(str(gray_2mm))
    atlas = np.rint(atlas_img.get_fdata()).astype(int)
    values = map_img.get_fdata()
    rows = []
    for roi in [int(x) for x in sorted(np.unique(atlas)) if x > 0]:
        mask = atlas == roi
        roi_values = values[mask]
        rows.append(
            {
                "roi": roi,
                "nt_mean": float(np.nanmean(roi_values)),
                "nt_coverage": float(np.mean(np.isfinite(roi_values) & (roi_values != 0))),
                "voxel_count": int(mask.sum()),
            }
        )
    roi_table = pd.DataFrame(rows)
    write_csv(roi_table, out_dir / "node" / "nt_roi_156.csv")
    return roi_table


def build_node_damage(lesion_node: pd.DataFrame, roi_table: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    """Create node-level neurotransmitter damage features."""
    node_cols = [f"node_{int(roi):03d}" for roi in roi_table["roi"]]
    weights = roi_table["nt_mean"].to_numpy(dtype=float)
    node = lesion_node[["subject_id", *node_cols]].copy()
    node[node_cols] = node[node_cols].to_numpy(dtype=float) * weights[None, :]
    write_csv(node, out_dir / "node" / "nt_node_damage.csv")
    return node


def build_edge_damage(config: dict, wm_2mm: Path, manifest: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    """Create edge features from lesion voxels, tract masks, and Alves WM maps."""
    shared_dir = project_path(config, config["outputs"]["edge_dir"])
    matrix_path = shared_dir / "edge_tract_voxels_2mm.npz"
    edge_path = shared_dir / "edge_tract_voxels_2mm_edges.csv"
    if not matrix_path.exists() or not edge_path.exists():
        # 首次运行时生成边-体素掩膜矩阵
        build_edge_matrix(config)
    edge_matrix = sparse.load_npz(matrix_path).astype(np.float32)
    edge_names = pd.read_csv(edge_path)["edge"].tolist()
    wm = np.nan_to_num(nib.load(str(wm_2mm)).get_fdata(), nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32).ravel()
    rows = []
    for row in manifest[["subject_id", "lesion_path"]].itertuples(index=False):
        lesion = (nib.load(str(row.lesion_path)).get_fdata().ravel() != 0).astype(np.float32)
        weighted_voxels = lesion * wm
        values = edge_matrix @ weighted_voxels
        rows.append({"subject_id": row.subject_id, **dict(zip(edge_names, np.asarray(values).ravel().astype(float)))})
    edge = pd.DataFrame(rows)
    write_csv(edge, out_dir / "edge" / "nt_edge_lqt.csv")
    return edge


def nt_config(config: dict, out_dir: Path) -> dict:
    """Create a config copy that writes model outputs inside one NT folder."""
    config_copy = copy.deepcopy(config)
    config_copy["outputs"]["model_dir"] = str(out_dir / "models")
    return config_copy


def run_one_nt(config: dict, spec: dict[str, object], manifest: pd.DataFrame, lesion_node: pd.DataFrame, lesion_edge: pd.DataFrame) -> dict[str, object]:
    """Run one neurotransmitter analysis."""
    nt_id = str(spec["id"])
    out_dir = nt_root(config, nt_id)
    for name in ["atlases", "node", "edge", "impact", "models"]:
        ensure_dir(out_dir / name)

    maps = prepare_nt_maps(config, spec, Path(manifest.loc[0, "lesion_path"]), out_dir)
    roi_table = compute_roi_table(config, maps["gray_2mm"], out_dir)
    node = build_node_damage(lesion_node, roi_table, out_dir)
    edge = build_edge_damage(config, maps["wm_2mm"], manifest, out_dir)

    config_one = nt_config(config, out_dir)
    impact_dir = ensure_dir(out_dir / "impact")
    scores = run_cross_validated_impact(config_one, manifest, node, edge, lesion_node, lesion_edge, impact_dir, nt_id)
    write_full_sample_keys(config_one, manifest, node, edge, lesion_node, lesion_edge, impact_dir, nt_id)
    fit_ordinal_impact_model(config_one, scores, impact_dir)
    run_cross_validated_prediction(config_one, scores, ensure_dir(out_dir / "models"))
    return {"nt_id": nt_id, "label": spec.get("label", nt_id), "output_dir": str(out_dir)}


def summarize_outputs(config: dict, rows: list[dict[str, object]]) -> None:
    """Write cross-neurotransmitter summary tables."""
    summary_dir = ensure_dir(project_path(config, "derivatives", "nt", "summary"))
    write_csv(pd.DataFrame(rows), summary_dir / "nt_run_manifest.csv")
    performance = []
    primary = []
    for row in rows:
        nt_id = str(row["nt_id"])
        out_dir = nt_root(config, nt_id)
        perf_path = out_dir / "models" / "model_prediction_performance.csv"
        perf = pd.read_csv(perf_path)
        perf.insert(0, "nt_id", nt_id)
        performance.append(perf)
        pair_path = out_dir / "models" / "model_prediction_pairwise_bootstrap.csv"
        pair = pd.read_csv(pair_path)
        pair.insert(0, "nt_id", nt_id)
        primary.append(pair[pair["model_a"] == "Clinical"].copy())
    if performance:
        write_csv(pd.concat(performance, ignore_index=True), summary_dir / "nt_prediction_performance.csv")
    if primary:
        primary_df = pd.concat(primary, ignore_index=True)
        write_csv(primary_df, summary_dir / "nt_prediction_vs_clinical_bootstrap.csv")
    else:
        primary_df = pd.DataFrame()
    if performance:
        write_run_report(summary_dir, pd.DataFrame(rows), pd.concat(performance, ignore_index=True), primary_df)


def write_run_report(summary_dir: Path, manifest: pd.DataFrame, performance: pd.DataFrame, bootstrap: pd.DataFrame) -> None:
    """Write a compact Markdown run report."""
    files = [path for path in summary_dir.parent.rglob("*") if path.is_file()]
    lines = [
        "# Multi-Neurotransmitter Run Report",
        "",
        "## Files",
        "",
        f"- neurotransmitter systems: {manifest.shape[0]}",
        f"- performance rows: {performance.shape[0]}",
        f"- clinical-vs-model bootstrap rows: {bootstrap.shape[0]}",
        f"- total files under derivatives/nt: {len(files)}",
        "",
        "## Summary Tables",
        "",
    ]
    for name in ["nt_run_manifest.csv", "nt_prediction_performance.csv", "nt_prediction_vs_clinical_bootstrap.csv"]:
        lines.append(f"- `{summary_dir / name}`")
    lines.extend(["", "## Best Ordinal Log Loss Per System", ""])
    header = ["nt_id", "best_model", "ordinal_log_loss", "rps", "c_index", "auc"]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("| " + " | ".join(["---"] * len(header)) + " |")
    rows = []
    for nt_id, group in performance.groupby("nt_id"):
        best = group.sort_values("ordinal_log_loss").iloc[0]
        rows.append(
            [
                str(nt_id),
                str(best["model"]),
                f"{best['ordinal_log_loss']:.6f}",
                f"{best['ranked_probability_score']:.6f}",
                f"{best['ordinal_c_index']:.6f}",
                f"{best['binary_auc_mrs_le_threshold']:.6f}",
            ]
        )
    for row in sorted(rows):
        lines.append("| " + " | ".join(row) + " |")
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- Lower ordinal log loss and RPS are better.",
            "- Higher ordinal C-index and binary AUC are better.",
            "- Primary comparison uses 10-fold out-of-sample prediction tables.",
        ]
    )
    (summary_dir / "nt_run_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run multi-neurotransmitter lesion analysis.")
    parser.add_argument("--config", default="config/dat_config.yaml")
    parser.add_argument("--nt", nargs="*", default=None, help="Optional neurotransmitter IDs to run.")
    parser.add_argument("--summary-only", action="store_true", help="Only refresh cross-neurotransmitter summary tables.")
    args = parser.parse_args()

    config = load_config(args.config)
    specs = neurotransmitter_specs(config)
    if args.nt:
        selected = set(args.nt)
        specs = [spec for spec in specs if str(spec["id"]) in selected]
    if args.summary_only:
        rows = [{"nt_id": str(spec["id"]), "label": spec.get("label", spec["id"]), "output_dir": str(nt_root(config, str(spec["id"])))} for spec in specs]
        summarize_outputs(config, rows)
        return

    manifest = pd.read_csv(project_path(config, config["outputs"]["qc_dir"], "subject_manifest.csv"))
    lesion_node = compute_lesion_node_load(config, manifest)
    _, lesion_edge = load_lesion_feature_tables(config, manifest)
    rows = []
    for index, spec in enumerate(specs, start=1):
        print(f"running {spec['id']} ({index}/{len(specs)})")
        rows.append(run_one_nt(config, spec, manifest, lesion_node, lesion_edge))
    summarize_outputs(config, rows)


if __name__ == "__main__":
    main()
