#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from nt_analysis.config import ensure_dir, load_config, project_path  # noqa: E402
from nt_analysis.tables import write_csv  # noqa: E402
from run_ml_profile_analysis import load_edge_matrix  # noqa: E402


def entropy(values: np.ndarray) -> float:
    """Return normalized entropy."""
    total = float(np.sum(values))
    if total <= np.finfo(float).eps:
        return 0.0
    p = values / total
    p = p[p > 0]
    if p.size <= 1:
        return 0.0
    return float(-np.sum(p * np.log(p)) / np.log(values.size))


def summarize_weights(weight_paths: list[Path], level: str) -> pd.DataFrame:
    """Summarize fold-wise weights."""
    if not weight_paths:
        raise FileNotFoundError(f"missing {level} fold weight files")
    data = pd.concat([pd.read_csv(path) for path in weight_paths], ignore_index=True)
    feature_col = "roi_id" if level == "node" else "edge"
    data["sign"] = np.sign(data["beta_ridge"].astype(float))
    summary = (
        data.groupby([feature_col, "nt_id"], as_index=False)
        .agg(
            mean_beta=("beta_ridge", "mean"),
            mean_abs_beta=("beta_ridge", lambda x: float(np.mean(np.abs(x)))),
            weight=("weight_ridge", "mean"),
            selection_frequency=("weight_ridge", lambda x: float(np.mean(np.asarray(x) > 0))),
            positive_frequency=("sign", lambda x: float(np.mean(np.asarray(x) > 0))),
            negative_frequency=("sign", lambda x: float(np.mean(np.asarray(x) < 0))),
        )
    )
    return summary


def wide_weight_table(summary: pd.DataFrame, feature_col: str, nt_ids: list[str]) -> pd.DataFrame:
    """Convert long weights to one row per feature."""
    table = summary.pivot_table(index=feature_col, columns="nt_id", values="weight", fill_value=0.0).reset_index()
    for nt_id in nt_ids:
        if nt_id not in table.columns:
            table[nt_id] = 0.0
    return table[[feature_col, *nt_ids]]


def export_roi_maps(config: dict, out_dir: Path, nt_ids: list[str], roi_weights: pd.DataFrame) -> None:
    """Export ROI 4D and summary maps."""
    atlas_path = project_path(config, config["atlases"]["outputs"]["atlas4s156_2mm"])
    atlas_img = nib.load(str(atlas_path))
    atlas = np.rint(atlas_img.get_fdata()).astype(np.int16)
    shape = atlas.shape
    four_d = np.zeros((*shape, len(nt_ids)), dtype=np.float32)
    top_map = np.zeros(shape, dtype=np.float32)
    entropy_map = np.zeros(shape, dtype=np.float32)
    max_map = np.zeros(shape, dtype=np.float32)
    rows = []
    for _, row in roi_weights.iterrows():
        roi_id = int(row["roi_id"])
        # 递质名可数字开头，不能按属性读取
        weights = np.asarray([float(row.get(nt_id, 0.0)) for nt_id in nt_ids], dtype=np.float32)
        mask = atlas == roi_id
        for nt_index, value in enumerate(weights):
            four_d[..., nt_index][mask] = value
        top_index = int(np.argmax(weights)) if weights.size else 0
        top_map[mask] = float(top_index + 1) if weights[top_index] > 0 else 0.0
        entropy_value = entropy(weights)
        entropy_map[mask] = entropy_value
        max_map[mask] = float(np.max(weights)) if weights.size else 0.0
        rows.append({"roi_id": roi_id, "top_nt": nt_ids[top_index], "top_nt_index": top_index + 1, "max_weight": float(np.max(weights)), "entropy": entropy_value})

    img = nib.Nifti1Image(four_d, atlas_img.affine, atlas_img.header)
    img.set_data_dtype(np.float32)
    nib.save(img, str(out_dir / f"roi_nt_weight_{len(nt_ids)}nt_4d.nii.gz"))
    for nt_index, nt_id in enumerate(nt_ids):
        nt_img = nib.Nifti1Image(four_d[..., nt_index], atlas_img.affine, atlas_img.header)
        nt_img.set_data_dtype(np.float32)
        nib.save(nt_img, str(out_dir / f"roi_nt_weight_{nt_id}.nii.gz"))
    for values, name in [
        (top_map, "roi_top_nt_map.nii.gz"),
        (entropy_map, "roi_nt_entropy_map.nii.gz"),
        (max_map, "roi_max_nt_weight_map.nii.gz"),
    ]:
        out_img = nib.Nifti1Image(values.astype(np.float32), atlas_img.affine, atlas_img.header)
        out_img.set_data_dtype(np.float32)
        nib.save(out_img, str(out_dir / name))
    write_csv(pd.DataFrame(rows), out_dir / "roi_nt_summary.csv")


def export_edge_maps(config: dict, out_dir: Path, nt_ids: list[str], edge_weights: pd.DataFrame) -> None:
    """Export edge matrices and projection maps."""
    edge_matrix, edge_names = load_edge_matrix(config)
    reference_path = pd.read_csv(project_path(config, config["outputs"]["qc_dir"], "subject_manifest.csv")).loc[0, "lesion_path"]
    ref_img = nib.load(str(reference_path))
    labels = sorted({int(part) for edge in edge_names for part in edge.split("_")[1:]})
    label_index = {label: idx for idx, label in enumerate(labels)}
    edge_lookup = edge_weights.set_index("edge")
    for nt_id in nt_ids:
        matrix = np.zeros((len(labels), len(labels)), dtype=np.float32)
        vector = np.zeros(len(edge_names), dtype=np.float32)
        for edge_index, edge_name in enumerate(edge_names):
            left, right = [int(value) for value in edge_name.split("_")[1:]]
            value = float(edge_lookup.loc[edge_name, nt_id]) if edge_name in edge_lookup.index else 0.0
            matrix[label_index[left], label_index[right]] = value
            matrix[label_index[right], label_index[left]] = value
            vector[edge_index] = value
        matrix_df = pd.DataFrame(matrix, index=labels, columns=labels)
        matrix_df.index.name = "roi_id"
        write_csv(matrix_df.reset_index(), out_dir / f"edge_nt_weight_matrix_{nt_id}.csv")
        selected = (vector > 0).astype(np.float32)
        numerator = np.asarray(edge_matrix.T.dot(vector)).ravel().astype(np.float32)
        denominator = np.asarray(edge_matrix.T.dot(selected)).ravel().astype(np.float32)
        projection = np.zeros_like(numerator, dtype=np.float32)
        mask = denominator > 0
        projection[mask] = numerator[mask] / denominator[mask]
        img = nib.Nifti1Image(projection.reshape(ref_img.shape), ref_img.affine, ref_img.header)
        img.set_data_dtype(np.float32)
        nib.save(img, str(out_dir / f"edge_nt_weight_projection_{nt_id}.nii.gz"))


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Export prognostic NTDC atlas maps.")
    parser.add_argument("--config", default="config/dat_config.yaml")
    parser.add_argument("--jobs", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    """Export maps."""
    args = parse_args()
    config = load_config(args.config)
    atlas_cfg = config.get("prognostic_ntdc_atlas", {})
    out_dir = ensure_dir(project_path(config, atlas_cfg.get("output_dir", "derivatives/prognostic_ntdc_atlas")))
    nt = pd.read_csv(out_dir / "nt_table.csv")
    nt_ids = nt["nt_id"].astype(str).tolist()
    node_summary = summarize_weights(sorted((out_dir / "fold_weights").glob("fold_*_node_nt_weights.csv")), "node")
    edge_summary = summarize_weights(sorted((out_dir / "fold_weights").glob("fold_*_edge_nt_weights.csv")), "edge")
    write_csv(node_summary, out_dir / "roi_nt_weight_long.csv")
    write_csv(edge_summary, out_dir / "edge_nt_weight_long.csv")
    roi_weights = wide_weight_table(node_summary, "roi_id", nt_ids)
    edge_weights = wide_weight_table(edge_summary, "edge", nt_ids)
    write_csv(roi_weights, out_dir / "roi_nt_weight_atlas.csv")
    write_csv(edge_weights, out_dir / "edge_nt_weight_atlas.csv")
    export_roi_maps(config, out_dir, nt_ids, roi_weights)
    export_edge_maps(config, out_dir, nt_ids, edge_weights)
    print(f"exported prognostic NTDC maps to {out_dir}", flush=True)


if __name__ == "__main__":
    main()
