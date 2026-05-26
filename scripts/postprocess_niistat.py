#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from scipy.stats import norm
from statsmodels.stats.multitest import multipletests

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nt_analysis.config import ensure_dir, load_config, project_path
from nt_analysis.stats import fit_mass_univariate
from nt_analysis.tables import write_csv


def save_like(data: np.ndarray, ref_img: nib.Nifti1Image, output: Path) -> None:
    """Save a NIfTI image with a reference header."""
    img = nib.Nifti1Image(data.astype(np.float32), ref_img.affine, ref_img.header)
    img.set_data_dtype(np.float32)
    output.parent.mkdir(parents=True, exist_ok=True)
    nib.save(img, str(output))


def two_tailed_p(z: np.ndarray) -> np.ndarray:
    """Convert Z values to two-tailed p values."""
    return 2.0 * norm.sf(np.abs(z))


def fdr_map(p_values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Compute FDR q values inside a mask."""
    q_values = np.ones_like(p_values, dtype=np.float32)
    valid = mask & np.isfinite(p_values)
    if np.any(valid):
        q_values[valid] = multipletests(p_values[valid], method="fdr_bh")[1].astype(np.float32)
    return q_values


def find_one(pattern: Path) -> Path:
    """Find one file matching a glob pattern."""
    matches = sorted(glob.glob(str(pattern)))
    if len(matches) != 1:
        raise RuntimeError(f"expected one file for {pattern}, found {len(matches)}")
    return Path(matches[0])


def residualize(values: np.ndarray, covariates: np.ndarray) -> np.ndarray:
    """Residualize values by intercept and covariates."""
    design = np.column_stack([np.ones(covariates.shape[0]), covariates])
    q, _ = np.linalg.qr(design)
    return values - q @ (q.T @ values)


def beta_from_matrix(features: np.ndarray, outcome: np.ndarray, covariates: np.ndarray) -> np.ndarray:
    """Estimate OLS beta for many features."""
    y_res = residualize(outcome.reshape(-1, 1), covariates).ravel()
    x_res = residualize(features, covariates)
    numerator = np.sum(x_res * y_res[:, None], axis=0)
    denominator = np.sum(x_res * x_res, axis=0)
    beta = np.full(features.shape[1], np.nan, dtype=np.float64)
    valid = denominator > np.finfo(float).eps
    beta[valid] = numerator[valid] / denominator[valid]
    return beta


def node_beta_stats(config: dict) -> pd.DataFrame:
    """Compute true node-level OLS beta values."""
    node_dir = project_path(config, config["outputs"]["node_dir"])
    features = pd.read_csv(node_dir / "dat_node_damage_66x156.csv")
    phenotype = pd.read_csv(project_path(config, config["outputs"]["qc_dir"], "subject_manifest.csv")).rename(columns={"mrs_3m": "outcome"})
    stats = fit_mass_univariate(features, phenotype, "outcome", ["age", "sex", "nihss"])
    stats["roi"] = stats["feature"].str.replace("node_", "", regex=False).astype(int)
    return stats[["roi", "feature", "beta", "t", "p", "n", "q"]]


def postprocess_node(config: dict) -> None:
    """Standardize node-level NiiStat outputs."""
    node_dir = ensure_dir(project_path(config, config["outputs"]["node_dir"]))
    result_dir = node_dir / "niistat_node_results"
    z_path = find_one(result_dir / "Zdat_node_clsm*.nii")
    atlas_path = project_path(config, config["atlases"]["outputs"]["atlas4s156_2mm"])
    atlas_img = nib.load(str(atlas_path))
    z_img = nib.load(str(z_path))
    atlas = np.rint(atlas_img.get_fdata()).astype(int)
    z_data = z_img.get_fdata()
    labels = [int(x) for x in sorted(np.unique(atlas)) if x > 0]

    rows = []
    p_values = []
    for label in labels:
        roi_z = z_data[atlas == label]
        z_value = float(np.nanmean(roi_z)) if roi_z.size else np.nan
        p_value = float(two_tailed_p(np.array([z_value]))[0]) if np.isfinite(z_value) else np.nan
        p_values.append(p_value)
        rows.append({"roi": label, "feature": f"node_{label:03d}", "z": z_value, "p": p_value, "stat_type": "niistat_z"})

    stats = pd.DataFrame(rows)
    valid = stats["p"].notna()
    stats["q"] = np.nan
    if valid.any():
        stats.loc[valid, "q"] = multipletests(stats.loc[valid, "p"], method="fdr_bh")[1]
    beta_stats = node_beta_stats(config)
    write_csv(beta_stats, node_dir / "dat_node_clsm_beta_stats.csv")
    stats = stats.merge(
        beta_stats.rename(columns={"t": "beta_t", "p": "beta_p", "n": "beta_n", "q": "beta_q"})[
            ["roi", "feature", "beta", "beta_t", "beta_p", "beta_n", "beta_q"]
        ],
        on=["roi", "feature"],
        how="left",
    )
    write_csv(stats, node_dir / "dat_node_clsm_stats.csv")

    z_output = node_dir / "dat_node_clsm_z_map.nii.gz"
    save_like(z_data, z_img, z_output)
    beta_data = np.zeros_like(z_data, dtype=np.float32)
    beta_lookup = dict(zip(beta_stats["roi"], beta_stats["beta"]))
    for label in labels:
        beta_value = beta_lookup.get(label, np.nan)
        if np.isfinite(beta_value):
            beta_data[atlas == label] = beta_value
    # 保存真实OLS beta图
    save_like(beta_data, z_img, node_dir / "dat_node_clsm_beta_map.nii.gz")


def wm_beta_map(config: dict, mask: np.ndarray, ref_img: nib.Nifti1Image) -> np.ndarray:
    """Compute true DAT-WM voxelwise OLS beta map."""
    wm_dir = project_path(config, config["outputs"]["wm_voxel_dir"])
    subjects = pd.read_csv(wm_dir / "dat_wm_voxel_niistat_subjects.csv")
    matrix = np.load(wm_dir / "dat_wm_voxel_matrix.npy", mmap_mode="r")
    if matrix.shape[0] != subjects.shape[0]:
        raise RuntimeError("DAT-WM matrix row count does not match subject table")
    y = subjects["mrs_3m"].to_numpy(dtype=float)
    covariates = subjects[["age", "sex", "nihss"]].to_numpy(dtype=float)
    valid_rows = np.isfinite(y) & np.isfinite(covariates).all(axis=1)
    flat_mask = mask.ravel()
    # 只在DAT-WM有效区域计算beta
    features = np.asarray(matrix[valid_rows][:, flat_mask], dtype=np.float64)
    beta_values = beta_from_matrix(features, y[valid_rows], covariates[valid_rows])
    beta_data = np.zeros(int(np.prod(ref_img.shape)), dtype=np.float32)
    beta_data[flat_mask] = np.nan_to_num(beta_values, nan=0.0).astype(np.float32)
    return beta_data.reshape(ref_img.shape)


def postprocess_wm(config: dict) -> None:
    """Standardize DAT-WM voxelwise NiiStat outputs."""
    wm_dir = ensure_dir(project_path(config, config["outputs"]["wm_voxel_dir"]))
    result_dir = wm_dir / "niistat_wm_results"
    z_path = find_one(result_dir / "Zdat_wm_voxel_clsm*.nii")
    mask_path = project_path(config, config["atlases"]["outputs"]["dat_wm_mask_2mm"])
    z_img = nib.load(str(z_path))
    mask_img = nib.load(str(mask_path))
    z_data = z_img.get_fdata()
    mask = mask_img.get_fdata() != 0
    p_data = np.ones_like(z_data, dtype=np.float32)
    p_data[mask] = two_tailed_p(z_data[mask]).astype(np.float32)
    q_data = fdr_map(p_data, mask)
    beta_data = wm_beta_map(config, mask, z_img)

    save_like(z_data, z_img, wm_dir / "dat_wm_voxel_z.nii.gz")
    save_like(beta_data, z_img, wm_dir / "dat_wm_voxel_beta.nii.gz")
    save_like(p_data, z_img, wm_dir / "dat_wm_voxel_p.nii.gz")
    save_like(q_data, z_img, wm_dir / "dat_wm_voxel_q.nii.gz")


def main() -> None:
    parser = argparse.ArgumentParser(description="Postprocess NiiStat DAT CLSM outputs.")
    parser.add_argument("--config", default="config/dat_config.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    postprocess_node(config)
    postprocess_wm(config)


if __name__ == "__main__":
    main()
