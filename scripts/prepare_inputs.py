#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nt_analysis.config import analysis_covariates, ensure_dir, load_config, outcome_column, project_path
from nt_analysis.ids import normalize_subject_id, parse_lesion_subject
from nt_analysis.images import lesion_volume_ml, resample_like
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
    out = df[[column for column in keep if column in df.columns]].copy()
    for column, mapping in config["inputs"].get("categorical_maps", {}).items():
        if column in out.columns:
            # 按配置转换分类变量
            out[column] = out[column].astype(str).str.strip().map(mapping).fillna(out[column])
    for column in [outcome, *covariate_map.keys()]:
        if column in out.columns:
            out[column] = pd.to_numeric(out[column], errors="coerce")
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
                "shape": "x".join(str(value) for value in img.shape[:3]),
                "voxel_volume_mm3": abs(float(np.linalg.det(img.affine[:3, :3]))),
                "lesion_volume_ml": lesion_volume_ml(path),
            }
        )
    return pd.DataFrame(rows)


def apply_lesion_qc(config: dict, manifest: pd.DataFrame) -> pd.DataFrame:
    """Apply lesion-level inclusion rules."""
    qc_cfg = config.get("qc", {})
    exclude_empty = bool(qc_cfg.get("exclude_empty_lesion", True))
    min_volume = float(qc_cfg.get("min_lesion_volume_ml", 0.0))
    out = manifest.copy()
    out["is_empty_lesion"] = out["lesion_volume_ml"] <= min_volume
    out["included_by_lesion_qc"] = ~(exclude_empty & out["is_empty_lesion"])
    # 空病灶保留在QC表中，但不进入主分析
    out["lesion_qc_exclusion_reason"] = np.where(out["included_by_lesion_qc"], "", "empty_lesion")
    return out


def merge_manifest_phenotype(manifest: pd.DataFrame, phenotype: pd.DataFrame) -> pd.DataFrame:
    """Merge lesion manifest and phenotype without subject duplication."""
    merged = manifest.merge(phenotype, on="subject_id", how="left")
    if merged["subject_id"].duplicated().any():
        duplicated = sorted(merged.loc[merged["subject_id"].duplicated(), "subject_id"].astype(str).unique())
        raise ValueError(f"duplicated subject_id after merge: {duplicated[:10]}")
    return merged


def prepare_reference_images(config: dict, reference_2mm: Path) -> None:
    """Resample atlas images into analysis spaces."""
    atlas = project_path(config, config["atlases"]["atlas4s156"]["nii"])
    lqt_ref = project_path(config, config["lqt"]["data_dir"], "MNI152_T1_1mm.nii.gz")
    outputs = config["atlases"]["outputs"]

    # 生成2mm分析图谱
    resample_like(atlas, reference_2mm, project_path(config, outputs["atlas4s156_2mm"]), "nearest")

    # LQT只需要1mm atlas，不生成患者1mm病灶
    resample_like(atlas, lqt_ref, project_path(config, outputs["atlas4s156_1mm_lqt"]), "nearest")


def compute_lqt_node_table(config: dict) -> None:
    """Create 4S156 node labels and coordinates for LQT."""
    output = project_path(config, config["atlases"]["outputs"]["atlas4s156_lqt_nodes"])
    atlas_img = nib.load(str(project_path(config, config["atlases"]["outputs"]["atlas4s156_1mm_lqt"])))
    atlas = np.rint(atlas_img.get_fdata()).astype(int)
    tsv = pd.read_csv(project_path(config, config["atlases"]["atlas4s156"]["tsv"]), sep="\t")
    rows = []
    for label in [int(value) for value in sorted(np.unique(atlas)) if value > 0]:
        index = np.argwhere(atlas == label)
        if index.size == 0:
            continue
        xyz = nib.affines.apply_affine(atlas_img.affine, index).mean(axis=0)
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare shared inputs for multi-NT analysis.")
    parser.add_argument("--config", default="config/dat_config.yaml")
    args = parser.parse_args()
    config = load_config(args.config)

    for key in ["qc_dir", "node_dir", "edge_dir"]:
        ensure_dir(project_path(config, config["outputs"][key]))

    manifest = build_manifest(config)
    lesion_qc = apply_lesion_qc(config, manifest)
    active_manifest = lesion_qc.loc[lesion_qc["included_by_lesion_qc"]].copy()
    if active_manifest.empty:
        raise RuntimeError("no lesion masks passed lesion QC")
    phenotype = load_phenotype(config)
    analysis_manifest = merge_manifest_phenotype(active_manifest, phenotype)
    qc_dir = project_path(config, config["outputs"]["qc_dir"])
    write_csv(lesion_qc, qc_dir / "lesion_qc.csv")
    write_csv(analysis_manifest, qc_dir / "subject_manifest.csv")

    outcome = outcome_column(config)
    covariates = analysis_covariates(config, "model_covariates")
    keep = ["subject_id", "lesion_path", outcome, *covariates]
    write_csv(analysis_manifest[[column for column in keep if column in analysis_manifest.columns]], qc_dir / "phenotype_merge_qc.csv")

    reference_2mm = Path(active_manifest.iloc[0]["lesion_path"])
    prepare_reference_images(config, reference_2mm)
    compute_lqt_node_table(config)


if __name__ == "__main__":
    main()
