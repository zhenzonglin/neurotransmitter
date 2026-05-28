#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from nibabel.streamlines import load as load_trk
from nilearn.image import resample_to_img
from scipy import sparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nt_analysis.config import ensure_dir, load_config, project_path  # noqa: E402
from nt_analysis.tables import write_csv  # noqa: E402


def sample_label_near(atlas: np.ndarray, voxel: np.ndarray, radius: int) -> int:
    """Sample a label near one endpoint."""
    center = np.rint(voxel).astype(int)
    shape = np.asarray(atlas.shape)
    if np.any(center < 0) or np.any(center >= shape):
        return 0
    label = int(atlas[tuple(center)])
    if label > 0 or radius <= 0:
        return label

    # 端点不在ROI内时，查找附近最近的非零标签
    low = np.maximum(center - radius, 0)
    high = np.minimum(center + radius + 1, shape)
    block = atlas[low[0] : high[0], low[1] : high[1], low[2] : high[2]]
    nonzero = np.argwhere(block > 0)
    if nonzero.size == 0:
        return 0
    coords = nonzero + low
    distances = np.sum((coords - center) ** 2, axis=1)
    nearest = coords[int(np.argmin(distances))]
    return int(atlas[tuple(nearest)])


def streamline_voxels(points: np.ndarray, reference_img: nib.Nifti1Image, hcp_origin: np.ndarray) -> np.ndarray:
    """Convert one streamline into unique 2mm voxel indices."""
    world = points + hcp_origin[None, :]
    vox = np.rint(nib.affines.apply_affine(np.linalg.inv(reference_img.affine), world)).astype(np.int32)
    shape = np.asarray(reference_img.shape[:3], dtype=np.int32)
    ok = np.all((vox >= 0) & (vox < shape[None, :]), axis=1)
    if not np.any(ok):
        return np.empty(0, dtype=np.int32)
    vox = vox[ok]
    flat = np.ravel_multi_index((vox[:, 0], vox[:, 1], vox[:, 2]), tuple(shape))
    return np.unique(flat.astype(np.int32))


def make_edge_names(labels: list[int]) -> tuple[list[str], dict[tuple[int, int], int]]:
    """Create upper-triangle edge names."""
    edge_names = []
    edge_index = {}
    for i, left in enumerate(labels[:-1]):
        for right in labels[i + 1 :]:
            edge_index[(left, right)] = len(edge_names)
            edge_names.append(f"edge_{left:03d}_{right:03d}")
    return edge_names, edge_index


def resample_atlas_to_hcp(config: dict, hcp_ref: Path, output: Path) -> Path:
    """Resample the 4S156 atlas to the HCP tractography grid."""
    source = nib.load(str(project_path(config, config["atlases"]["outputs"]["atlas4s156_1mm_lqt"])))
    target = nib.load(str(hcp_ref))
    resampled = resample_to_img(source, target, interpolation="nearest", force_resample=True, copy_header=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    nib.save(resampled, str(output))
    return output


def build_edge_matrix(config: dict, force: bool = False, max_streamlines: int | None = None) -> None:
    """Build an edge-by-voxel tract mask matrix in 2mm lesion space."""
    shared_dir = ensure_dir(project_path(config, config["outputs"]["edge_dir"]))
    output_npz = shared_dir / "edge_tract_voxels_2mm.npz"
    output_edges = shared_dir / "edge_tract_voxels_2mm_edges.csv"
    output_qc = shared_dir / "edge_tract_voxels_2mm_qc.csv"
    if output_npz.exists() and output_edges.exists() and not force:
        print(f"exists {output_npz}")
        return

    hcp_ref = project_path(config, config["lqt"]["data_dir"], "HCP842_QA.nii.gz")
    trk_path = project_path(config, config["lqt"]["data_dir"], "Tractography_Atlas", "all_tracts_1mm.trk.gz")
    atlas_hcp = resample_atlas_to_hcp(config, hcp_ref, shared_dir / "atlas4s156_hcp842_1mm.nii.gz")
    reference_2mm = Path(pd.read_csv(project_path(config, config["outputs"]["qc_dir"], "subject_manifest.csv")).loc[0, "lesion_path"])

    hcp_img = nib.load(str(hcp_ref))
    atlas_img = nib.load(str(atlas_hcp))
    ref_img = nib.load(str(reference_2mm))
    atlas = np.rint(atlas_img.get_fdata()).astype(np.int16)
    labels = [int(value) for value in sorted(np.unique(atlas)) if value > 0]
    edge_names, edge_index = make_edge_names(labels)
    hcp_origin = hcp_img.affine[:3, 3].astype(float)
    hcp_inv = np.linalg.inv(hcp_img.affine)

    rows = []
    cols = []
    skipped_same = 0
    skipped_label = 0
    kept = 0
    trk = load_trk(str(trk_path), lazy_load=True)
    for stream_index, points in enumerate(trk.streamlines, start=1):
        if max_streamlines is not None and stream_index > max_streamlines:
            break
        world_end = points[[0, -1], :] + hcp_origin[None, :]
        endpoint_vox = nib.affines.apply_affine(hcp_inv, world_end)
        left = sample_label_near(atlas, endpoint_vox[0], radius=2)
        right = sample_label_near(atlas, endpoint_vox[1], radius=2)
        if left == 0 or right == 0:
            skipped_label += 1
            continue
        if left == right:
            skipped_same += 1
            continue
        edge = tuple(sorted((left, right)))
        idx = edge_index.get(edge)
        if idx is None:
            skipped_label += 1
            continue
        voxels = streamline_voxels(points, ref_img, hcp_origin)
        if voxels.size == 0:
            skipped_label += 1
            continue
        rows.append(np.full(voxels.size, idx, dtype=np.int32))
        cols.append(voxels)
        kept += 1
        if kept % 10000 == 0:
            print(f"kept {kept} streamlines")

    if rows:
        row = np.concatenate(rows)
        col = np.concatenate(cols)
        data = np.ones(row.size, dtype=np.uint8)
    else:
        row = np.empty(0, dtype=np.int32)
        col = np.empty(0, dtype=np.int32)
        data = np.empty(0, dtype=np.uint8)

    matrix = sparse.coo_matrix((data, (row, col)), shape=(len(edge_names), int(np.prod(ref_img.shape[:3]))), dtype=np.uint8).tocsr()
    matrix.sum_duplicates()
    matrix.data[:] = 1
    sparse.save_npz(output_npz, matrix, compressed=True)
    write_csv(pd.DataFrame({"edge": edge_names}), output_edges)
    write_csv(
        pd.DataFrame(
            [
                {
                    "n_edges": len(edge_names),
                    "n_voxels": int(np.prod(ref_img.shape[:3])),
                    "n_streamlines_header": int(trk.header.get("nb_streamlines", 0)),
                    "n_streamlines_kept": kept,
                    "n_streamlines_skipped_label": skipped_label,
                    "n_streamlines_skipped_same_label": skipped_same,
                    "nnz": int(matrix.nnz),
                    "max_streamlines": max_streamlines,
                }
            ]
        ),
        output_qc,
    )
    print(f"wrote {output_npz}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build edge tract voxel matrix.")
    parser.add_argument("--config", default="config/dat_config.yaml")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--max-streamlines", type=int, default=None)
    args = parser.parse_args()
    config = load_config(args.config)
    build_edge_matrix(config, force=args.force, max_streamlines=args.max_streamlines)


if __name__ == "__main__":
    main()
