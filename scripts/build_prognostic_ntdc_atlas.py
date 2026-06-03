#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from compute_impact_scores import add_cv_group, compute_lesion_node_load, load_lesion_feature_tables  # noqa: E402
from nt_analysis.config import analysis_covariates, ensure_dir, load_config, outcome_column, project_path  # noqa: E402
from nt_analysis.tables import write_csv  # noqa: E402
from run_ml_profile_analysis import (  # noqa: E402
    atlas_info,
    load_edge_matrix,
    load_lesion_indices,
    load_map_arrays,
    prepare_maps,
    neurotransmitter_specs,
)


worker_context: dict[str, object] = {}


def selected_specs(config: dict) -> list[dict[str, object]]:
    """Return selected neurotransmitter specs."""
    specs = neurotransmitter_specs(config)
    selected = config.get("prognostic_ntdc_atlas", {}).get("nt_ids")
    if not selected:
        return specs
    wanted = {str(value) for value in selected}
    out = [spec for spec in specs if str(spec["id"]) in wanted]
    if not out:
        raise RuntimeError("no configured neurotransmitters matched prognostic_ntdc_atlas.nt_ids")
    return out


def load_manifest(config: dict, max_subjects: int | None) -> pd.DataFrame:
    """Load active subjects."""
    manifest_path = project_path(config, config["outputs"]["qc_dir"], "subject_manifest.csv")
    manifest = pd.read_csv(manifest_path, dtype={"subject_id": str})
    manifest = add_cv_group(config, manifest)
    outcome = outcome_column(config)
    covariates = analysis_covariates(config, "model_covariates")
    required = ["subject_id", "cv_group", "lesion_path", outcome, *covariates]
    missing = [column for column in required if column not in manifest.columns]
    if missing:
        raise KeyError(f"missing columns in subject_manifest.csv: {missing}")
    data = manifest.dropna(subset=required).copy()
    data["subject_id"] = data["subject_id"].astype(str)
    if max_subjects is not None:
        data = data.head(int(max_subjects)).copy()
    if data.empty:
        raise RuntimeError("no eligible subjects were found")
    return data


def write_lesion_tables(config: dict, data: pd.DataFrame, out_dir: Path) -> None:
    """Write aligned structure-only damage tables."""
    lesion_node = compute_lesion_node_load(config, data)
    _, lesion_edge = load_lesion_feature_tables(config, data)
    ids = data["subject_id"].astype(str).tolist()
    lesion_node["subject_id"] = lesion_node["subject_id"].astype(str)
    lesion_edge["subject_id"] = lesion_edge["subject_id"].astype(str)
    lesion_node = lesion_node.set_index("subject_id").loc[ids].reset_index()
    lesion_edge = lesion_edge.set_index("subject_id").loc[ids].reset_index()
    write_csv(lesion_node, out_dir / "lesion_node_damage.csv")
    write_csv(lesion_edge, out_dir / "lesion_edge_damage.csv")


def compute_node_damage(
    data: pd.DataFrame,
    lesion_indices: dict[str, np.ndarray],
    atlas_flat: np.ndarray,
    labels: list[int],
    gray_arrays: np.ndarray,
    out_dir: Path,
) -> tuple[Path, pd.DataFrame]:
    """Build subject by ROI by NT damage."""
    n_subjects = data.shape[0]
    n_roi = len(labels)
    n_nt = gray_arrays.shape[0]
    output = out_dir / f"nt_node_damage_{n_nt}nt.npy"
    values = np.lib.format.open_memmap(output, mode="w+", dtype=np.float32, shape=(n_subjects, n_roi, n_nt))
    denom = np.zeros((n_roi, n_nt), dtype=np.float32)
    max_label = int(max(labels))
    label_to_index = {int(label): idx for idx, label in enumerate(labels)}
    atlas_positive = atlas_flat > 0
    for nt_index in range(n_nt):
        gray = gray_arrays[nt_index]
        for roi_index, label in enumerate(labels):
            mask = atlas_flat == label
            denom[roi_index, nt_index] = float(np.sum(np.abs(gray[mask])))
    safe_denom = denom.copy()
    safe_denom[safe_denom <= np.finfo(np.float32).eps] = 1.0

    for subject_index, subject_id in enumerate(data["subject_id"].astype(str).tolist()):
        idx = lesion_indices[subject_id]
        label_values = atlas_flat[idx]
        valid = label_values > 0
        valid_idx = idx[valid]
        valid_labels = label_values[valid].astype(int)
        for nt_index in range(n_nt):
            sums = np.bincount(valid_labels, weights=gray_arrays[nt_index, valid_idx], minlength=max_label + 1)
            for label, roi_index in label_to_index.items():
                values[subject_index, roi_index, nt_index] = float(sums[label] / safe_denom[roi_index, nt_index])
        if (subject_index + 1) % 100 == 0 or subject_index + 1 == n_subjects:
            print(f"node damage subjects {subject_index + 1}/{n_subjects}", flush=True)
    values.flush()

    rows = []
    for roi_index, label in enumerate(labels):
        for nt_index in range(n_nt):
            rows.append({"roi_id": int(label), "nt_index": int(nt_index), "denominator": float(denom[roi_index, nt_index])})
    qc = pd.DataFrame(rows)
    write_csv(qc, out_dir / "nt_node_denominator_qc.csv")
    return output, qc


def init_edge_worker(context: dict[str, object]) -> None:
    """Initialize forked edge workers."""
    worker_context.clear()
    worker_context.update(context)


def compute_edge_chunk(job: tuple[int, int]) -> dict[str, object]:
    """Build one subject chunk for edge damage."""
    start, end = job
    data = worker_context["data"]
    subject_ids = data["subject_id"].astype(str).tolist()
    lesion_indices = worker_context["lesion_indices"]
    edge_matrix = worker_context["edge_matrix"]
    wm_arrays = worker_context["wm_arrays"]
    denom = worker_context["edge_denom"]
    fraction_path = Path(worker_context["fraction_path"])
    sum_path = Path(worker_context["sum_path"])
    shape = tuple(worker_context["edge_shape"])
    n_edges, n_nt = int(shape[1]), int(shape[2])
    fraction_values = np.lib.format.open_memmap(fraction_path, mode="r+", dtype=np.float32, shape=shape)
    sum_values = np.lib.format.open_memmap(sum_path, mode="r+", dtype=np.float32, shape=shape)
    safe = denom.copy()
    safe[safe <= np.finfo(np.float32).eps] = 1.0
    for local_index, subject_index in enumerate(range(start, end)):
        subject_id = subject_ids[subject_index]
        idx = lesion_indices[subject_id]
        subject_sum = np.zeros((n_edges, n_nt), dtype=np.float32)
        if idx.size:
            sub_matrix = edge_matrix[:, idx]
            for nt_index in range(n_nt):
                weighted = sub_matrix.dot(wm_arrays[nt_index, idx])
                subject_sum[:, nt_index] = np.asarray(weighted).ravel().astype(np.float32)
        sum_values[subject_index, :, :] = subject_sum
        fraction_values[subject_index, :, :] = subject_sum / safe
        if local_index % 50 == 0:
            sum_values.flush()
            fraction_values.flush()
    sum_values.flush()
    fraction_values.flush()
    return {"start": start, "end": end, "n_subjects": end - start}


def compute_edge_damage(
    data: pd.DataFrame,
    lesion_indices: dict[str, np.ndarray],
    edge_matrix,
    edge_names: list[str],
    wm_arrays: np.ndarray,
    nt_ids: list[str],
    out_dir: Path,
    jobs: int,
    chunk_size: int,
) -> tuple[Path, Path, pd.DataFrame]:
    """Build subject by edge by NT damage."""
    n_subjects = data.shape[0]
    n_edges = len(edge_names)
    n_nt = len(nt_ids)
    fraction_path = out_dir / f"nt_edge_damage_fraction_{n_nt}nt.npy"
    sum_path = out_dir / f"nt_edge_damage_sum_{n_nt}nt.npy"
    fraction_values = np.lib.format.open_memmap(fraction_path, mode="w+", dtype=np.float32, shape=(n_subjects, n_edges, n_nt))
    sum_values = np.lib.format.open_memmap(sum_path, mode="w+", dtype=np.float32, shape=(n_subjects, n_edges, n_nt))
    fraction_values.flush()
    sum_values.flush()

    edge_denom = np.zeros((n_edges, n_nt), dtype=np.float32)
    abs_edge = edge_matrix.astype(np.float32)
    for nt_index in range(n_nt):
        weighted = abs_edge.dot(np.abs(wm_arrays[nt_index]).astype(np.float32))
        edge_denom[:, nt_index] = np.asarray(weighted).ravel().astype(np.float32)
    positive = edge_denom[edge_denom > np.finfo(np.float32).eps]
    min_percentile = float(worker_context.get("denominator_min_percentile", 5.0)) if worker_context else 5.0
    threshold = float(np.percentile(positive, min_percentile)) if positive.size else 0.0
    rows = []
    for edge_index, edge_name in enumerate(edge_names):
        for nt_index, nt_id in enumerate(nt_ids):
            denominator = float(edge_denom[edge_index, nt_index])
            rows.append(
                {
                    "edge": edge_name,
                    "nt_id": nt_id,
                    "denominator": denominator,
                    "low_denominator": bool(denominator < threshold),
                    "denominator_min_percentile": min_percentile,
                    "denominator_threshold": threshold,
                }
            )
    qc = pd.DataFrame(rows)
    write_csv(qc, out_dir / "nt_edge_denominator_qc.csv")

    chunks = [(start, min(start + int(chunk_size), n_subjects)) for start in range(0, n_subjects, int(chunk_size))]
    context = {
        "data": data[["subject_id"]].copy(),
        "lesion_indices": lesion_indices,
        "edge_matrix": edge_matrix,
        "wm_arrays": wm_arrays,
        "edge_denom": edge_denom,
        "fraction_path": str(fraction_path),
        "sum_path": str(sum_path),
        "edge_shape": (n_subjects, n_edges, n_nt),
    }
    jobs = max(1, min(int(jobs), len(chunks)))
    if jobs == 1:
        init_edge_worker(context)
        for chunk in chunks:
            result = compute_edge_chunk(chunk)
            print(f"edge damage subjects {result['end']}/{n_subjects}", flush=True)
    else:
        mp_context = mp.get_context("fork")
        with ProcessPoolExecutor(max_workers=jobs, mp_context=mp_context, initializer=init_edge_worker, initargs=(context,)) as executor:
            futures = [executor.submit(compute_edge_chunk, chunk) for chunk in chunks]
            for future in as_completed(futures):
                result = future.result()
                print(f"edge damage subjects {result['end']}/{n_subjects}", flush=True)
    return fraction_path, sum_path, qc


def write_metadata(
    out_dir: Path,
    data: pd.DataFrame,
    labels: list[int],
    edge_names: list[str],
    nt_ids: list[str],
    ref_img: nib.Nifti1Image,
) -> None:
    """Write array metadata."""
    write_csv(data, out_dir / "subject_table.csv")
    write_csv(pd.DataFrame({"roi_id": labels}), out_dir / "roi_table.csv")
    edge_rows = []
    for edge in edge_names:
        parts = edge.split("_")
        edge_rows.append({"edge": edge, "roi_a": int(parts[1]), "roi_b": int(parts[2])})
    write_csv(pd.DataFrame(edge_rows), out_dir / "edge_table.csv")
    write_csv(pd.DataFrame({"nt_index": list(range(len(nt_ids))), "nt_id": nt_ids}), out_dir / "nt_table.csv")
    metadata = {
        "n_subjects": int(data.shape[0]),
        "n_roi": int(len(labels)),
        "n_edges": int(len(edge_names)),
        "n_nt": int(len(nt_ids)),
        "image_shape": list(ref_img.shape),
        "nt_ids": nt_ids,
    }
    (out_dir / "feature_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Build prognostic NTDC atlas feature arrays.")
    parser.add_argument("--config", default="config/dat_config.yaml")
    parser.add_argument("--jobs", type=int, default=None)
    parser.add_argument("--chunk-size", type=int, default=None)
    parser.add_argument("--max-subjects", type=int, default=None)
    parser.add_argument("--force-maps", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Run feature building."""
    args = parse_args()
    config = load_config(args.config)
    atlas_cfg = config.get("prognostic_ntdc_atlas", {})
    out_dir = ensure_dir(project_path(config, atlas_cfg.get("output_dir", "derivatives/prognostic_ntdc_atlas")))
    jobs = int(args.jobs if args.jobs is not None else atlas_cfg.get("jobs", config.get("resources", {}).get("atlas_build_jobs", 1)))
    chunk_size = int(args.chunk_size if args.chunk_size is not None else atlas_cfg.get("chunk_size", 256))
    data = load_manifest(config, args.max_subjects)
    specs = selected_specs(config)
    reference_2mm = Path(data["lesion_path"].iloc[0])
    maps = prepare_maps(config, specs, reference_2mm, out_dir, args.force_maps)
    nt_ids, gray_arrays, wm_arrays, ref_img = load_map_arrays(maps)
    atlas_flat, labels, _ = atlas_info(config)
    lesion_indices = load_lesion_indices(data)
    edge_matrix, edge_names = load_edge_matrix(config)
    write_metadata(out_dir, data, labels, edge_names, nt_ids, ref_img)
    write_lesion_tables(config, data, out_dir)
    print(f"building node NT damage for {data.shape[0]} subjects, {len(labels)} ROI, {len(nt_ids)} NT", flush=True)
    compute_node_damage(data, lesion_indices, atlas_flat, labels, gray_arrays, out_dir)
    print(f"building edge NT damage with jobs={jobs}, chunk_size={chunk_size}", flush=True)
    global worker_context
    worker_context["denominator_min_percentile"] = float(atlas_cfg.get("denominator_min_percentile", 5))
    compute_edge_damage(data, lesion_indices, edge_matrix, edge_names, wm_arrays, nt_ids, out_dir, jobs, chunk_size)
    print(f"wrote prognostic NTDC feature arrays to {out_dir}", flush=True)


if __name__ == "__main__":
    main()
