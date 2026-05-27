#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import loadmat, savemat

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nt_analysis.config import analysis_covariates, analysis_table, ensure_dir, load_config, outcome_column, project_path
from nt_analysis.tables import write_csv


def assert_inside(root: Path, target: Path) -> None:
    """Ensure a path stays inside the project."""
    resolved = target.resolve()
    if root != resolved and root not in resolved.parents:
        raise RuntimeError(f"unsafe path outside project: {resolved}")


def safe_rmtree(root: Path, target: Path) -> None:
    """Delete a project-local directory."""
    assert_inside(root, target)
    if target.exists():
        shutil.rmtree(target)


def make_id_map(subject_ids: list[str], repeats: int) -> dict[tuple[str, int], str]:
    """Create normal-looking unique IDs for repeated rows."""
    mapping: dict[tuple[str, int], str] = {}
    index = 1
    for repeat_id in range(1, repeats + 1):
        for subject_id in subject_ids:
            mapping[(str(subject_id), repeat_id)] = f"NT{index:06d}"
            index += 1
    return mapping


def augment_table(df: pd.DataFrame, repeats: int, id_map: dict[tuple[str, int], str]) -> pd.DataFrame:
    """Repeat rows and replace IDs with normal unique IDs."""
    rows = []
    for repeat_id in range(1, repeats + 1):
        copy = df.copy()
        copy["subject_id"] = [id_map[(str(subject_id), repeat_id)] for subject_id in copy["subject_id"]]
        rows.append(copy)
    return pd.concat(rows, ignore_index=True)


def prepare_qc(config: dict, temp_root: Path, repeats: int, id_map: dict[tuple[str, int], str]) -> pd.DataFrame:
    """Prepare augmented QC tables."""
    base_qc = project_path(config, config["outputs"]["qc_dir"])
    out_qc = ensure_dir(temp_root / "qc")
    subject = pd.read_csv(base_qc / "subject_manifest.csv")
    lesion = pd.read_csv(base_qc / "lesion_qc.csv")
    phenotype = pd.read_csv(base_qc / "phenotype_merge_qc.csv")
    subject_aug = augment_table(subject, repeats, id_map)
    write_csv(subject_aug, out_qc / "subject_manifest.csv")
    write_csv(augment_table(lesion, repeats, id_map), out_qc / "lesion_qc.csv")
    write_csv(augment_table(phenotype, repeats, id_map), out_qc / "phenotype_merge_qc.csv")
    return subject_aug


def prepare_node(config: dict, temp_root: Path, phenotype: pd.DataFrame, repeats: int, id_map: dict[tuple[str, int], str]) -> None:
    """Prepare augmented node tables and NiiStat input."""
    base_node = project_path(config, config["outputs"]["node_dir"])
    out_node = ensure_dir(temp_root / "node_clsm")
    feature_file = analysis_table(config, "node_damage", "dat_node_damage.csv")
    node_aug = augment_table(pd.read_csv(base_node / feature_file), repeats, id_map)
    feature_cols = [col for col in node_aug.columns if col.startswith("node_")]
    write_csv(node_aug, out_node / feature_file)
    shutil.copy2(base_node / "dat_roi_156.csv", out_node / "dat_roi_156.csv")

    outcome = outcome_column(config)
    covariates = analysis_covariates(config, "niistat_covariates")
    required = [outcome, *covariates]
    merged = phenotype.merge(node_aug[["subject_id", *feature_cols]], on="subject_id").dropna(subset=required)
    write_csv(merged[["subject_id", *required]], out_node / "dat_node_niistat_subjects.csv")
    nuisance = merged[covariates].to_numpy(dtype=float) if covariates else np.empty((merged.shape[0], 0))
    savemat(
        out_node / "dat_node_niistat_input.mat",
        {
            "les": merged[feature_cols].to_numpy(dtype=float),
            "beh": merged[[outcome]].to_numpy(dtype=float),
            "beh_names": np.array([outcome], dtype=object),
            "nuisance": nuisance,
            "logical_mask": np.ones(len(feature_cols), dtype=bool),
            "roi_names": np.array(feature_cols, dtype=object),
        },
    )


def prepare_edge(config: dict, temp_root: Path, repeats: int, id_map: dict[tuple[str, int], str]) -> None:
    """Prepare augmented edge tables."""
    base_edge = project_path(config, config["outputs"]["edge_dir"])
    out_edge = ensure_dir(temp_root / "edge_clsm")
    for key, default in [("dat_edge_lqt", "dat_edge_lqt.csv"), ("lqt_edge_disconnection", "lqt_edge_disconnection.csv")]:
        file_name = analysis_table(config, key, default)
        write_csv(augment_table(pd.read_csv(base_edge / file_name), repeats, id_map), out_edge / file_name)


def prepare_wm(config: dict, temp_root: Path, repeats: int, id_map: dict[tuple[str, int], str]) -> None:
    """Prepare augmented WM voxel NiiStat input."""
    base_wm = project_path(config, config["outputs"]["wm_voxel_dir"])
    out_wm = ensure_dir(temp_root / "wm_voxel_clsm")
    subjects_aug = augment_table(pd.read_csv(base_wm / "dat_wm_voxel_niistat_subjects.csv"), repeats, id_map)
    write_csv(subjects_aug, out_wm / "dat_wm_voxel_niistat_subjects.csv")

    base_matrix = np.load(base_wm / "dat_wm_voxel_matrix.npy", mmap_mode="r")
    output_matrix = out_wm / "dat_wm_voxel_matrix.npy"
    matrix_aug = np.lib.format.open_memmap(
        output_matrix,
        mode="w+",
        dtype=base_matrix.dtype,
        shape=(base_matrix.shape[0] * repeats, base_matrix.shape[1]),
    )
    for repeat_id in range(repeats):
        start = repeat_id * base_matrix.shape[0]
        stop = start + base_matrix.shape[0]
        matrix_aug[start:stop, :] = base_matrix
    del matrix_aug

    shutil.copy2(base_wm / "dat_wm_mask_index.npy", out_wm / "dat_wm_mask_index.npy")
    base_mat = loadmat(base_wm / "dat_wm_voxel_niistat_input.mat")
    outcome = outcome_column(config)
    covariates = analysis_covariates(config, "niistat_covariates")
    nuisance = subjects_aug[covariates].to_numpy(dtype=float) if covariates else np.empty((subjects_aug.shape[0], 0))
    savemat(
        out_wm / "dat_wm_voxel_niistat_input.mat",
        {
            "les": np.load(output_matrix, mmap_mode="r"),
            "beh": subjects_aug[[outcome]].to_numpy(dtype=float),
            "beh_names": np.array([outcome], dtype=object),
            "nuisance": nuisance,
            "logical_mask": base_mat["logical_mask"].astype(bool).ravel(),
            "mask_shape": base_mat["mask_shape"].astype(np.int32).ravel(),
            "mask_index": np.load(out_wm / "dat_wm_mask_index.npy").astype(np.int64) + 1,
        },
    )


def replace_main_outputs(config: dict, temp_root: Path) -> None:
    """Replace old derivative outputs with augmented main outputs."""
    root = Path(config["project_dir"]).resolve()
    derivatives = project_path(config, "derivatives")
    targets = [
        project_path(config, config["outputs"]["qc_dir"]),
        project_path(config, config["outputs"]["node_dir"]),
        project_path(config, config["outputs"]["wm_voxel_dir"]),
        project_path(config, config["outputs"]["edge_dir"]),
        project_path(config, config["outputs"]["model_dir"]),
        project_path(config, config["outputs"]["impact_dir"]),
        derivatives / "logs",
    ]
    legacy_prefix = "test_" + "augmented"
    for old_dir in derivatives.iterdir():
        if old_dir.name.startswith(legacy_prefix):
            targets.append(old_dir)
    for target in targets:
        safe_rmtree(root, target)

    for name in ["qc", "node_clsm", "wm_voxel_clsm", "edge_clsm"]:
        src = temp_root / name
        dst = project_path(config, config["outputs"][{"qc": "qc_dir", "node_clsm": "node_dir", "wm_voxel_clsm": "wm_voxel_dir", "edge_clsm": "edge_dir"}[name]])
        ensure_dir(dst.parent)
        shutil.move(str(src), str(dst))
    safe_rmtree(root, temp_root)


def main() -> None:
    parser = argparse.ArgumentParser(description="Promote repeated feature tables to the main analysis dataset.")
    parser.add_argument("--config", default="config/dat_config.yaml")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if not args.force:
        raise RuntimeError("use --force to replace main derivative outputs")

    config = load_config(args.config)
    root = Path(config["project_dir"]).resolve()
    temp_root = root / ".tmp_augmented_main"
    safe_rmtree(root, temp_root)
    ensure_dir(temp_root)

    base_subjects = pd.read_csv(project_path(config, config["outputs"]["qc_dir"], "subject_manifest.csv"))["subject_id"].astype(str).tolist()
    id_map = make_id_map(base_subjects, args.repeats)
    phenotype = prepare_qc(config, temp_root, args.repeats, id_map)
    prepare_node(config, temp_root, phenotype, args.repeats, id_map)
    prepare_edge(config, temp_root, args.repeats, id_map)
    prepare_wm(config, temp_root, args.repeats, id_map)
    summary = pd.DataFrame(
        [
            {
                "dataset": "main",
                "subjects": int(phenotype.shape[0]),
                "complete_niistat_subjects": int(pd.read_csv(temp_root / "node_clsm" / "dat_node_niistat_subjects.csv").shape[0]),
            }
        ]
    )
    replace_main_outputs(config, temp_root)
    write_csv(summary, project_path(config, "derivatives", "dataset_summary.csv"))
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
