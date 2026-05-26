#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from nilearn.image import resample_to_img
from scipy.io import savemat

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nt_analysis.config import analysis_covariates, analysis_table, ensure_dir, load_config, outcome_column, project_path, require_columns
from nt_analysis.ids import normalize_subject_id, parse_lesion_subject
from nt_analysis.images import lesion_volume_ml, resample_like, save_img
from nt_analysis.tables import write_csv


def load_phenotype(config: dict) -> pd.DataFrame:
    """Load and normalize phenotype data."""
    path = project_path(config, config["inputs"]["phenotype_file"])
    df = pd.read_excel(path, sheet_name=config["inputs"]["phenotype_sheet"])
    id_col = config["inputs"]["phenotype_id_column"]
    outcome = outcome_column(config)
    df["subject_id"] = df[id_col].map(normalize_subject_id)
    covariate_map = config["inputs"].get("covariates", {})
    rename = {source: target for target, source in covariate_map.items()}
    rename[config["inputs"]["outcome_column"]] = outcome
    df = df.rename(columns=rename)
    keep = ["subject_id", outcome, *covariate_map.keys()]
    out = df[[c for c in keep if c in df.columns]].copy()
    for col, mapping in config["inputs"].get("categorical_maps", {}).items():
        if col in out.columns:
            # 按配置转换分类变量
            out[col] = out[col].astype(str).str.strip().map(mapping).fillna(out[col])
    for col in [outcome, *covariate_map.keys()]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def build_manifest(config: dict) -> pd.DataFrame:
    """Build a subject-level lesion manifest."""
    pattern = project_path(config, config["inputs"]["lesion_glob"])
    paths = sorted(glob.glob(str(pattern)))
    rows = []
    for path in paths:
        subject_id = parse_lesion_subject(path, config["inputs"]["lesion_subject_regex"])
        img = nib.load(path)
        rows.append(
            {
                "subject_id": subject_id,
                "lesion_path": path,
                "shape": "x".join(str(x) for x in img.shape[:3]),
                "voxel_volume_mm3": abs(float(np.linalg.det(img.affine[:3, :3]))),
                "lesion_volume_ml": lesion_volume_ml(path),
            }
        )
    return pd.DataFrame(rows)


def prepare_reference_images(config: dict, reference_2mm: Path) -> None:
    """Resample atlas and DAT maps into analysis spaces."""
    atlas = project_path(config, config["atlases"]["atlas4s156"]["nii"])
    dat_gray = project_path(config, config["atlases"]["hansen_dat"]["nii"])
    dat_wm = project_path(config, config["atlases"]["dat_wm"]["nii"])
    lqt_ref = project_path(config, config["lqt"]["data_dir"], "MNI152_T1_1mm.nii.gz")
    outputs = config["atlases"]["outputs"]

    # 生成2mm分析图谱
    resample_like(atlas, reference_2mm, project_path(config, outputs["atlas4s156_2mm"]), "nearest")
    resample_like(dat_gray, reference_2mm, project_path(config, outputs["dat_gray_2mm"]), "continuous")
    resample_like(dat_wm, reference_2mm, project_path(config, outputs["dat_wm_2mm"]), "continuous")

    # 只为LQT连接矩阵生成1mm atlas，不生成患者1mm病灶
    resample_like(atlas, lqt_ref, project_path(config, outputs["atlas4s156_1mm_lqt"]), "nearest")

    # 用原始DAT-WM非零支持域生成mask，避免插值背景泄漏
    dat_wm_raw_img = nib.load(str(dat_wm))
    raw_mask = (np.isfinite(dat_wm_raw_img.get_fdata()) & (dat_wm_raw_img.get_fdata() != 0)).astype(np.uint8)
    raw_mask_img = nib.Nifti1Image(raw_mask, dat_wm_raw_img.affine, dat_wm_raw_img.header)
    raw_mask_img.set_data_dtype(np.uint8)
    reference_img = nib.load(str(reference_2mm))
    mask_img = resample_to_img(raw_mask_img, reference_img, interpolation="nearest", force_resample=True, copy_header=True)
    mask = (mask_img.get_fdata() != 0).astype(np.uint8)
    out = nib.Nifti1Image(mask, reference_img.affine, reference_img.header)
    out.set_data_dtype(np.uint8)
    save_img(out, project_path(config, outputs["dat_wm_mask_2mm"]))


def compute_lqt_node_table(config: dict) -> None:
    """Create 4S156 node labels and coordinates for LQT."""
    output = project_path(config, config["atlases"]["outputs"]["atlas4s156_lqt_nodes"])
    atlas_img = nib.load(str(project_path(config, config["atlases"]["outputs"]["atlas4s156_1mm_lqt"])))
    atlas = np.rint(atlas_img.get_fdata()).astype(int)
    tsv = pd.read_csv(project_path(config, config["atlases"]["atlas4s156"]["tsv"]), sep="\t")
    rows = []
    for label in [int(x) for x in sorted(np.unique(atlas)) if x > 0]:
        idx = np.argwhere(atlas == label)
        if idx.size == 0:
            continue
        xyz = nib.affines.apply_affine(atlas_img.affine, idx).mean(axis=0)
        meta = tsv.loc[tsv["index"] == label]
        name = f"roi_{label:03d}" if meta.empty else str(meta["label"].iloc[0])
        network = "unknown" if meta.empty else str(meta["network_label"].iloc[0])
        rows.append(
            {
                "roi": label,
                "label": name,
                "network": network,
                "x": float(xyz[0]),
                "y": float(xyz[1]),
                "z": float(xyz[2]),
            }
        )
    write_csv(pd.DataFrame(rows), output)


def compute_node_features(config: dict, manifest: pd.DataFrame, phenotype: pd.DataFrame) -> None:
    """Compute node-level DAT damage features."""
    node_dir = ensure_dir(project_path(config, config["outputs"]["node_dir"]))
    outcome = outcome_column(config)
    covariates = analysis_covariates(config, "niistat_covariates")
    feature_file = analysis_table(config, "node_damage", "dat_node_damage.csv")
    atlas_img = nib.load(str(project_path(config, config["atlases"]["outputs"]["atlas4s156_2mm"])))
    dat_img = nib.load(str(project_path(config, config["atlases"]["outputs"]["dat_gray_2mm"])))
    atlas = np.rint(atlas_img.get_fdata()).astype(int)
    dat = dat_img.get_fdata()
    labels = [int(x) for x in sorted(np.unique(atlas)) if x > 0]
    roi_rows = []
    for label in labels:
        mask = atlas == label
        values = dat[mask]
        roi_rows.append(
            {
                "roi": label,
                "dat_mean": float(np.nanmean(values)),
                "dat_coverage": float(np.mean(np.isfinite(values) & (values != 0))),
                "voxel_count": int(mask.sum()),
            }
        )
    dat_roi = pd.DataFrame(roi_rows)
    write_csv(dat_roi, node_dir / "dat_roi_156.csv")

    feature_rows = []
    for row in manifest.itertuples(index=False):
        img = nib.load(row.lesion_path)
        lesion = img.get_fdata() != 0
        values = {"subject_id": row.subject_id}
        for roi in labels:
            mask = atlas == roi
            lesion_load = float(np.sum(lesion & mask) / max(mask.sum(), 1))
            dat_mean = float(dat_roi.loc[dat_roi["roi"] == roi, "dat_mean"].iloc[0])
            values[f"node_{roi:03d}"] = lesion_load * dat_mean
        feature_rows.append(values)
    features = pd.DataFrame(feature_rows)
    write_csv(features, node_dir / feature_file)

    required = [outcome, *covariates]
    require_columns(required, list(phenotype.columns), "phenotype")
    merged = phenotype.merge(features, on="subject_id").dropna(subset=required)
    write_csv(merged[["subject_id", *required]], node_dir / "dat_node_niistat_subjects.csv")
    nuisance = merged[covariates].to_numpy(dtype=float) if covariates else np.empty((merged.shape[0], 0))
    mat = {
        "les": merged[[f"node_{roi:03d}" for roi in labels]].to_numpy(dtype=float),
        "beh": merged[[outcome]].to_numpy(dtype=float),
        "beh_names": np.array([outcome], dtype=object),
        "nuisance": nuisance,
        "logical_mask": np.ones(len(labels), dtype=bool),
        "roi_names": np.array([f"node_{roi:03d}" for roi in labels], dtype=object),
    }
    savemat(node_dir / "dat_node_niistat_input.mat", mat)


def prepare_wm_voxel_input(config: dict, manifest: pd.DataFrame, phenotype: pd.DataFrame) -> None:
    """Prepare full-volume DAT-WM voxelwise input for NiiStat core."""
    wm_dir = ensure_dir(project_path(config, config["outputs"]["wm_voxel_dir"]))
    outcome = outcome_column(config)
    covariates = analysis_covariates(config, "niistat_covariates")
    mask_img = nib.load(str(project_path(config, config["atlases"]["outputs"]["dat_wm_mask_2mm"])))
    mask = mask_img.get_fdata().ravel() != 0
    required = [outcome, *covariates]
    require_columns(required, list(phenotype.columns), "phenotype")
    merged = phenotype.merge(manifest, on="subject_id").dropna(subset=required)
    write_csv(merged[["subject_id", *required]], wm_dir / "dat_wm_voxel_niistat_subjects.csv")
    matrix = np.zeros((merged.shape[0], int(mask.size)), dtype=np.float32)
    for row_index, row in enumerate(merged.itertuples(index=False)):
        img = nib.load(row.lesion_path)
        matrix[row_index, :] = (img.get_fdata().ravel() != 0)
    np.save(wm_dir / "dat_wm_voxel_matrix.npy", matrix)
    np.save(wm_dir / "dat_wm_mask_index.npy", np.flatnonzero(mask))
    savemat(
        wm_dir / "dat_wm_voxel_niistat_input.mat",
        {
            "les": matrix,
            "beh": merged[[outcome]].to_numpy(dtype=float),
            "beh_names": np.array([outcome], dtype=object),
            "nuisance": merged[covariates].to_numpy(dtype=float) if covariates else np.empty((merged.shape[0], 0)),
            "logical_mask": mask.astype(bool),
            "mask_shape": np.array(mask_img.shape, dtype=np.int32),
            "mask_index": np.flatnonzero(mask).astype(np.int64) + 1,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare inputs for DAT NT-CLSM.")
    parser.add_argument("--config", default="config/dat_config.yaml")
    args = parser.parse_args()
    config = load_config(args.config)

    for key in ["qc_dir", "node_dir", "wm_voxel_dir", "edge_dir", "model_dir"]:
        ensure_dir(project_path(config, config["outputs"][key]))

    manifest = build_manifest(config)
    phenotype = load_phenotype(config)
    merged = manifest.merge(phenotype, on="subject_id", how="left")
    qc_dir = project_path(config, config["outputs"]["qc_dir"])
    write_csv(manifest, qc_dir / "lesion_qc.csv")
    write_csv(merged, qc_dir / "subject_manifest.csv")
    outcome = outcome_column(config)
    covariates = list(config["inputs"].get("covariates", {}).keys())
    keep = ["subject_id", "lesion_path", outcome, *covariates]
    write_csv(merged[[column for column in keep if column in merged.columns]], qc_dir / "phenotype_merge_qc.csv")

    reference_2mm = Path(manifest.loc[0, "lesion_path"])
    prepare_reference_images(config, reference_2mm)
    compute_lqt_node_table(config)
    compute_node_features(config, manifest, phenotype)
    prepare_wm_voxel_input(config, manifest, phenotype)


if __name__ == "__main__":
    main()
