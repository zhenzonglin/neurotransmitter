#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
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
    fit_fast_mass_univariate,
    fit_ordinal_impact_model,
    load_lesion_feature_tables,
    run_cross_validated_impact,
    run_cross_validated_prediction,
    write_full_sample_keys,
)
from nt_analysis.config import analysis_covariates, ensure_dir, load_config, outcome_column, project_path  # noqa: E402
from nt_analysis.images import save_img  # noqa: E402
from nt_analysis.tables import write_csv  # noqa: E402


def neurotransmitter_ids(config: dict) -> list[str]:
    """Return configured neurotransmitter IDs."""
    return [str(spec["id"]) for spec in config.get("neurotransmitters", [])]


def output_root(config: dict, profile_name: str) -> Path:
    """Return the integrated profile output folder."""
    return ensure_dir(project_path(config, "derivatives", "nt_profile", profile_name))


def map_paths(config: dict, nt_id: str) -> tuple[Path, Path]:
    """Return resampled gray and WM maps for one neurotransmitter."""
    nt_dir = project_path(config, "derivatives", "nt", nt_id, "atlases")
    gray = nt_dir / f"{nt_id}_hansen_gray_2mm.nii.gz"
    wm = nt_dir / f"{nt_id}_alves_wm_2mm.nii.gz"
    missing = [str(path) for path in [gray, wm] if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing resampled neurotransmitter maps: {missing}")
    return gray, wm


def robust_scale(data: np.ndarray, mask: np.ndarray, lower: float, upper: float) -> tuple[np.ndarray, dict[str, float]]:
    """Scale one map to 0-1 within a valid mask."""
    valid = np.isfinite(data) & mask
    values = data[valid]
    if values.size == 0:
        return np.zeros(data.shape, dtype=np.float32), {"n_valid": 0, "low": np.nan, "high": np.nan}
    low, high = np.nanpercentile(values, [lower, upper])
    if not np.isfinite(high - low) or high <= low:
        scaled = np.zeros(data.shape, dtype=np.float32)
    else:
        scaled = np.zeros(data.shape, dtype=np.float32)
        scaled[valid] = np.clip((data[valid] - low) / (high - low), 0.0, 1.0)
    return scaled, {"n_valid": int(values.size), "low": float(low), "high": float(high)}


def as_3d(data: np.ndarray, path: Path) -> np.ndarray:
    """Return a 3D image array."""
    data = np.squeeze(data)
    if data.ndim != 3:
        raise ValueError(f"expected a 3D map after squeezing: {path}")
    return data


def save_profile_img(data: np.ndarray, reference_img: nib.Nifti1Image, output: Path) -> Path:
    """Save one integrated profile image."""
    img = nib.Nifti1Image(data.astype(np.float32), reference_img.affine, reference_img.header)
    img.set_data_dtype(np.float32)
    return save_img(img, output)


def build_integrated_maps(config: dict, out_dir: Path, lower: float, upper: float) -> tuple[Path, Path, pd.DataFrame]:
    """Build integrated 13-NT gray and WM profile maps."""
    nt_ids = neurotransmitter_ids(config)
    atlas_img = nib.load(str(project_path(config, config["atlases"]["outputs"]["atlas4s156_2mm"])))
    atlas = np.rint(atlas_img.get_fdata()).astype(int)
    gray_mask = atlas > 0
    gray_stack = []
    wm_stack = []
    rows = []
    for nt_id in nt_ids:
        gray_path, wm_path = map_paths(config, nt_id)
        gray_img = nib.load(str(gray_path))
        wm_img = nib.load(str(wm_path))
        gray_data = as_3d(gray_img.get_fdata(), gray_path)
        wm_data = as_3d(wm_img.get_fdata(), wm_path)
        wm_mask = np.isfinite(wm_data) & (wm_data != 0)
        gray_scaled, gray_meta = robust_scale(gray_data, gray_mask, lower, upper)
        wm_scaled, wm_meta = robust_scale(wm_data, wm_mask, lower, upper)
        gray_stack.append(gray_scaled)
        wm_stack.append(wm_scaled)
        rows.append({"nt_id": nt_id, "map_type": "gray", **gray_meta})
        rows.append({"nt_id": nt_id, "map_type": "wm", **wm_meta})

    gray_density = np.mean(np.stack(gray_stack, axis=0), axis=0).astype(np.float32)
    wm_density = np.mean(np.stack(wm_stack, axis=0), axis=0).astype(np.float32)
    atlas_dir = ensure_dir(out_dir / "atlases")
    gray_out = save_profile_img(gray_density, atlas_img, atlas_dir / "nt_profile_gray_density_2mm.nii.gz")
    wm_out = save_profile_img(wm_density, atlas_img, atlas_dir / "nt_profile_wm_density_2mm.nii.gz")
    scaling = pd.DataFrame(rows)
    write_csv(scaling, atlas_dir / "nt_profile_scaling.csv")
    method = {
        "profile_name": out_dir.name,
        "nt_ids": nt_ids,
        "scaling": "robust percentile min-max per neurotransmitter map",
        "lower_percentile": lower,
        "upper_percentile": upper,
        "integration": "mean of 13 scaled voxelwise neurotransmitter maps",
        "gray_mask": "atlas4s156 nonzero voxels",
        "wm_mask": "finite nonzero Alves map voxels per neurotransmitter",
    }
    (atlas_dir / "nt_profile_method.json").write_text(json.dumps(method, indent=2), encoding="utf-8")
    return gray_out, wm_out, scaling


def build_roi_profile(config: dict, gray_profile: Path, out_dir: Path) -> pd.DataFrame:
    """Compute ROI means for the integrated gray profile."""
    atlas_img = nib.load(str(project_path(config, config["atlases"]["outputs"]["atlas4s156_2mm"])))
    atlas = np.rint(atlas_img.get_fdata()).astype(int)
    profile = nib.load(str(gray_profile)).get_fdata()
    rows = []
    for roi in [int(value) for value in sorted(np.unique(atlas)) if value > 0]:
        mask = atlas == roi
        values = profile[mask]
        rows.append(
            {
                "roi": roi,
                "profile_density_mean": float(np.nanmean(values)),
                "profile_density_sum": float(np.nansum(values)),
                "voxel_count": int(mask.sum()),
            }
        )
    table = pd.DataFrame(rows)
    write_csv(table, out_dir / "node" / "nt_profile_roi_156.csv")
    return table


def build_node_profile_damage(config: dict, gray_profile: Path, manifest: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    """Build voxel-weighted node damage from the integrated gray profile."""
    atlas_img = nib.load(str(project_path(config, config["atlases"]["outputs"]["atlas4s156_2mm"])))
    atlas = np.rint(atlas_img.get_fdata()).astype(int)
    profile = np.nan_to_num(nib.load(str(gray_profile)).get_fdata(), nan=0.0, posinf=0.0, neginf=0.0)
    labels = [int(value) for value in sorted(np.unique(atlas)) if value > 0]
    roi_masks = {roi: atlas == roi for roi in labels}
    path_cache: dict[str, dict[str, float]] = {}
    rows = []
    for row in manifest[["subject_id", "lesion_path"]].itertuples(index=False):
        lesion_path = str(row.lesion_path)
        if lesion_path not in path_cache:
            lesion = nib.load(lesion_path).get_fdata() != 0
            values = {}
            for roi, mask in roi_masks.items():
                # ROI内逐体素计算 lesion × profile，避免只用ROI平均值
                values[f"node_{roi:03d}"] = float(np.sum(lesion[mask] * profile[mask]) / max(mask.sum(), 1))
            path_cache[lesion_path] = values
        rows.append({"subject_id": row.subject_id, **path_cache[lesion_path]})
    node = pd.DataFrame(rows)
    write_csv(node, out_dir / "node" / "nt_profile_node_damage.csv")
    return node


def build_edge_profile_damage(config: dict, wm_profile: Path, manifest: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    """Build edge damage from lesion voxels, tract masks, and the integrated WM profile."""
    shared_dir = project_path(config, config["outputs"]["edge_dir"])
    matrix_path = shared_dir / "edge_tract_voxels_2mm.npz"
    edge_path = shared_dir / "edge_tract_voxels_2mm_edges.csv"
    if not matrix_path.exists() or not edge_path.exists():
        # 首次运行时生成边-体素掩膜矩阵
        build_edge_matrix(config)
    edge_matrix = sparse.load_npz(matrix_path).astype(np.float32)
    edge_names = pd.read_csv(edge_path)["edge"].tolist()
    profile = np.nan_to_num(nib.load(str(wm_profile)).get_fdata(), nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32).ravel()
    rows = []
    path_cache: dict[str, np.ndarray] = {}
    for row in manifest[["subject_id", "lesion_path"]].itertuples(index=False):
        lesion_path = str(row.lesion_path)
        if lesion_path not in path_cache:
            lesion = (nib.load(lesion_path).get_fdata().ravel() != 0).astype(np.float32)
            path_cache[lesion_path] = np.asarray(edge_matrix @ (lesion * profile)).ravel().astype(float)
        rows.append({"subject_id": row.subject_id, **dict(zip(edge_names, path_cache[lesion_path]))})
    edge = pd.DataFrame(rows)
    write_csv(edge, out_dir / "edge" / "nt_profile_edge_lqt.csv")
    return edge


def profile_config(config: dict, out_dir: Path) -> dict:
    """Create a config copy that writes model outputs inside the profile folder."""
    config_copy = copy.deepcopy(config)
    config_copy["outputs"]["model_dir"] = str(out_dir / "models")
    return config_copy


def write_lsm_tables(config: dict, manifest: pd.DataFrame, node: pd.DataFrame, edge: pd.DataFrame, out_dir: Path) -> None:
    """Write full-sample node and edge LSM statistics."""
    outcome = outcome_column(config)
    covariates = analysis_covariates(config, "model_covariates")
    node_stats = fit_fast_mass_univariate(node, manifest, outcome, covariates)
    edge_stats = fit_fast_mass_univariate(edge, manifest, outcome, covariates)
    write_csv(node_stats, out_dir / "node" / "nt_profile_node_lsm_stats.csv")
    write_csv(edge_stats, out_dir / "edge" / "nt_profile_edge_lsm_stats.csv")


def write_report(out_dir: Path, manifest: pd.DataFrame, node: pd.DataFrame, edge: pd.DataFrame) -> None:
    """Write a compact run report."""
    perf_path = out_dir / "models" / "model_prediction_performance.csv"
    perf = pd.read_csv(perf_path) if perf_path.exists() else pd.DataFrame()
    lines = [
        "# Integrated 13-NT Profile Run Report",
        "",
        "## Method",
        "",
        "- Each Hansen gray map and Alves WM map was robustly scaled to 0-1.",
        "- The integrated profile is the voxelwise mean of the 13 scaled maps.",
        "- Node damage uses ROI-wise sums of `lesion * gray_profile`.",
        "- Edge damage uses `lesion * edge_tract_mask * wm_profile`.",
        "",
        "## Output Shapes",
        "",
        f"- subjects: {manifest.shape[0]}",
        f"- node table: {node.shape[0]} x {node.shape[1]}",
        f"- edge table: {edge.shape[0]} x {edge.shape[1]}",
        f"- prediction rows: {perf.shape[0] if not perf.empty else 0}",
        "",
    ]
    if not perf.empty:
        lines.extend(["## Prediction Performance", "", "```text", perf.to_string(index=False), "```", ""])
    (out_dir / "nt_profile_run_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run integrated 13-neurotransmitter profile analysis.")
    parser.add_argument("--config", default="config/dat_config.yaml")
    parser.add_argument("--profile-name", default="integrated_13nt")
    parser.add_argument("--lower-percentile", type=float, default=1.0)
    parser.add_argument("--upper-percentile", type=float, default=99.0)
    args = parser.parse_args()

    config = load_config(args.config)
    out_dir = output_root(config, args.profile_name)
    for name in ["atlases", "node", "edge", "impact", "models"]:
        ensure_dir(out_dir / name)

    manifest = pd.read_csv(project_path(config, config["outputs"]["qc_dir"], "subject_manifest.csv"))
    if "base_subject_id" not in manifest.columns:
        manifest["base_subject_id"] = manifest["subject_id"]
    if "repeat_id" not in manifest.columns:
        manifest["repeat_id"] = 1

    gray_profile, wm_profile, _ = build_integrated_maps(config, out_dir, args.lower_percentile, args.upper_percentile)
    build_roi_profile(config, gray_profile, out_dir)
    node = build_node_profile_damage(config, gray_profile, manifest, out_dir)
    edge = build_edge_profile_damage(config, wm_profile, manifest, out_dir)
    lesion_node, lesion_edge = load_lesion_feature_tables(config, manifest)
    write_lsm_tables(config, manifest, node, edge, out_dir)

    config_one = profile_config(config, out_dir)
    impact_dir = ensure_dir(out_dir / "impact")
    scores = run_cross_validated_impact(config_one, manifest, node, edge, lesion_node, lesion_edge, impact_dir, args.profile_name)
    write_full_sample_keys(config_one, manifest, node, edge, lesion_node, lesion_edge, impact_dir, args.profile_name)
    fit_ordinal_impact_model(config_one, scores, impact_dir)
    run_cross_validated_prediction(config_one, scores, ensure_dir(out_dir / "models"))
    write_report(out_dir, manifest, node, edge)


if __name__ == "__main__":
    main()
