#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import ElasticNetCV
from sklearn.model_selection import GroupKFold
from statsmodels.stats.multitest import multipletests

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from build_edge_tract_matrix import build_edge_matrix  # noqa: E402
from compute_impact_scores import (  # noqa: E402
    add_cv_group,
    align_probabilities,
    bootstrap_metric_delta,
    compute_lesion_node_load,
    fit_fast_mass_univariate,
    fit_ordered_model,
    impact_from_weights,
    load_lesion_feature_tables,
    metric_directions,
    prediction_metrics,
    select_weights,
)
from nt_analysis.config import analysis_covariates, ensure_dir, load_config, outcome_column, project_path  # noqa: E402
from nt_analysis.images import resample_like  # noqa: E402
from nt_analysis.tables import write_csv  # noqa: E402


def neurotransmitter_specs(config: dict) -> list[dict[str, object]]:
    """Return configured neurotransmitter maps."""
    specs = config.get("neurotransmitters", [])
    if not specs:
        raise RuntimeError("no neurotransmitters are configured")
    return specs


def model_specs(covariates: list[str], ntdc_col: str = "ntdc", ntdc_label: str = "NTDC") -> list[tuple[str, list[str]]]:
    """Return final prediction models."""
    return [
        ("Clinical", covariates),
        ("Clinical + SDC", covariates + ["sdc"]),
        (f"Clinical + {ntdc_label}", covariates + [ntdc_col]),
        (f"Clinical + SDC + {ntdc_label}", covariates + ["sdc", ntdc_col]),
    ]


def raw_map_paths(config: dict, spec: dict[str, object]) -> tuple[Path, Path]:
    """Return raw Hansen and Alves map paths."""
    raw_dir = project_path(config, config["atlases"]["raw_dir"])
    hansen = raw_dir / "hansen" / str(spec["hansen_file"])
    alves = raw_dir / "alves" / f"functionnectome_anat_{spec['alves_name']}.nii.gz"
    missing = [str(path) for path in [hansen, alves] if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing neurotransmitter maps: {missing}")
    return hansen, alves


def prepare_maps(config: dict, specs: list[dict[str, object]], reference_2mm: Path, out_dir: Path, force: bool) -> pd.DataFrame:
    """Resample all maps to the lesion grid."""
    map_dir = ensure_dir(out_dir / "maps")
    rows = []
    for spec in specs:
        nt_id = str(spec["id"])
        hansen, alves = raw_map_paths(config, spec)
        gray_2mm = map_dir / f"{nt_id}_hansen_gray_2mm.nii.gz"
        wm_2mm = map_dir / f"{nt_id}_alves_wm_2mm.nii.gz"
        if force or not gray_2mm.exists():
            resample_like(hansen, reference_2mm, gray_2mm, "continuous")
        if force or not wm_2mm.exists():
            resample_like(alves, reference_2mm, wm_2mm, "continuous")
        rows.append(
            {
                "nt_id": nt_id,
                "label": spec.get("label", nt_id),
                "hansen_gray_2mm": str(gray_2mm),
                "alves_wm_2mm": str(wm_2mm),
            }
        )
    maps = pd.DataFrame(rows)
    write_csv(maps, out_dir / "map_manifest.csv")
    return maps


def load_map_arrays(maps: pd.DataFrame) -> tuple[list[str], np.ndarray, np.ndarray, nib.Nifti1Image]:
    """Load map arrays into memory."""
    nt_ids = maps["nt_id"].astype(str).tolist()
    gray_arrays = []
    wm_arrays = []
    ref_img = nib.load(str(maps["hansen_gray_2mm"].iloc[0]))
    for row in maps.itertuples(index=False):
        gray = np.nan_to_num(nib.load(str(row.hansen_gray_2mm)).get_fdata(), nan=0.0, posinf=0.0, neginf=0.0)
        wm = np.nan_to_num(nib.load(str(row.alves_wm_2mm)).get_fdata(), nan=0.0, posinf=0.0, neginf=0.0)
        gray_arrays.append(gray.astype(np.float32).ravel())
        wm_arrays.append(wm.astype(np.float32).ravel())
    return nt_ids, np.vstack(gray_arrays), np.vstack(wm_arrays), ref_img


def atlas_info(config: dict) -> tuple[np.ndarray, list[int], np.ndarray]:
    """Load atlas labels and ROI voxel counts."""
    atlas_img = nib.load(str(project_path(config, config["atlases"]["outputs"]["atlas4s156_2mm"])))
    atlas = np.rint(atlas_img.get_fdata()).astype(np.int16)
    atlas_flat = atlas.ravel()
    labels = [int(value) for value in sorted(np.unique(atlas_flat)) if value > 0]
    counts = np.asarray([(atlas_flat == label).sum() for label in labels], dtype=float)
    counts[counts <= 0] = 1.0
    return atlas_flat, labels, counts


def load_lesion_indices(manifest: pd.DataFrame) -> dict[str, np.ndarray]:
    """Load lesion voxel indices for active subjects."""
    indices = {}
    for row in manifest[["subject_id", "lesion_path"]].itertuples(index=False):
        lesion = nib.load(str(row.lesion_path)).get_fdata().ravel() != 0
        indices[str(row.subject_id)] = np.flatnonzero(lesion).astype(np.int32)
    return indices


def load_edge_matrix(config: dict) -> tuple[sparse.csc_matrix, list[str]]:
    """Load the edge-by-voxel tract matrix."""
    shared_dir = project_path(config, config["outputs"]["edge_dir"])
    matrix_path = shared_dir / "edge_tract_voxels_2mm.npz"
    edge_path = shared_dir / "edge_tract_voxels_2mm_edges.csv"
    if not matrix_path.exists() or not edge_path.exists():
        build_edge_matrix(config)
    matrix = sparse.load_npz(matrix_path).astype(np.float32).tocsc()
    edge_names = pd.read_csv(edge_path)["edge"].astype(str).tolist()
    return matrix, edge_names


def compute_screening_features(
    manifest: pd.DataFrame,
    nt_ids: list[str],
    gray_arrays: np.ndarray,
    wm_arrays: np.ndarray,
    atlas_flat: np.ndarray,
    lesion_indices: dict[str, np.ndarray],
    out_dir: Path,
    force: bool,
) -> pd.DataFrame:
    """Compute low-dimensional NT screening features."""
    output = out_dir / "screening_features.csv"
    if output.exists() and not force:
        cached = pd.read_csv(output)
        if cached.shape[0] == manifest.shape[0] and set(cached["subject_id"]) == set(manifest["subject_id"]):
            return cached

    gray_mask = atlas_flat > 0
    rows = []
    for subject_id in manifest["subject_id"].astype(str).tolist():
        idx = lesion_indices[subject_id]
        row = {"subject_id": subject_id}
        atlas_idx = idx[atlas_flat[idx] > 0]
        for nt_index, nt_id in enumerate(nt_ids):
            gray = gray_arrays[nt_index]
            wm = wm_arrays[nt_index]
            gray_denom = float(np.sum(np.abs(gray[gray_mask])))
            wm_denom = float(np.sum(np.abs(wm)))
            if gray_denom <= np.finfo(float).eps:
                gray_denom = 1.0
            if wm_denom <= np.finfo(float).eps:
                wm_denom = 1.0
            row[f"screen_node_{nt_id}"] = float(np.sum(gray[atlas_idx]) / gray_denom) if atlas_idx.size else 0.0
            row[f"screen_edge_{nt_id}"] = float(np.sum(wm[idx]) / wm_denom) if idx.size else 0.0
        rows.append(row)
    screening = pd.DataFrame(rows)
    write_csv(screening, output)
    return screening


def zscore_train(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Z-score by training-set statistics."""
    mean = np.nanmean(values, axis=0)
    sd = np.nanstd(values, axis=0, ddof=1)
    sd[~np.isfinite(sd) | (sd <= np.finfo(float).eps)] = 1.0
    return (values - mean[None, :]) / sd[None, :], mean, sd


def residualize(values: np.ndarray, design: np.ndarray) -> np.ndarray:
    """Residualize values by a design matrix."""
    coef, *_ = np.linalg.lstsq(design, values, rcond=None)
    return values - design @ coef


def elastic_net_nt_weights(
    train: pd.DataFrame,
    feature_cols: list[str],
    nt_ids: list[str],
    covariates: list[str],
    outcome: str,
    groups: np.ndarray,
    config: dict,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    """Select neurotransmitter systems inside one training fold."""
    ml_cfg = config.get("ml_profile", {})
    inner_splits = int(ml_cfg.get("inner_splits", 5))
    l1_ratios = [float(value) for value in ml_cfg.get("l1_ratios", [0.1, 0.5, 0.9, 1.0])]
    alphas = np.asarray([float(value) for value in ml_cfg.get("alphas", np.logspace(-4, 0, 20))], dtype=float)

    x_raw = train[feature_cols].to_numpy(dtype=float)
    x_z, _, _ = zscore_train(x_raw)
    design = np.column_stack([np.ones(train.shape[0]), train[covariates].to_numpy(dtype=float)])
    y = train[outcome].to_numpy(dtype=float).reshape(-1, 1)
    x_res = residualize(x_z, design)
    y_res = residualize(y, design).ravel()

    unique_groups = np.unique(groups)
    n_splits = max(2, min(inner_splits, len(unique_groups)))
    inner_cv = list(GroupKFold(n_splits=n_splits).split(x_res, groups=groups))
    model = ElasticNetCV(l1_ratio=l1_ratios, alphas=alphas, cv=inner_cv, max_iter=100000, tol=1e-3, random_state=int(config.get("impact", {}).get("random_state", 42)))
    model.fit(x_res, y_res)
    coef = np.asarray(model.coef_, dtype=float)
    feature_table = pd.DataFrame({"feature": feature_cols, "coef": coef})
    feature_table["nt_id"] = feature_table["feature"].str.replace("screen_node_", "", regex=False).str.replace("screen_edge_", "", regex=False)

    beta_rows = []
    for nt_id in nt_ids:
        node_coef = float(feature_table.loc[feature_table["feature"] == f"screen_node_{nt_id}", "coef"].sum())
        edge_coef = float(feature_table.loc[feature_table["feature"] == f"screen_edge_{nt_id}", "coef"].sum())
        beta_rows.append({"nt_id": nt_id, "node_coef": node_coef, "edge_coef": edge_coef, "beta": node_coef + edge_coef})
    beta = pd.DataFrame(beta_rows)

    fallback = False
    if beta["beta"].abs().sum() <= np.finfo(float).eps:
        # 全部收缩为0时，使用训练折内残差相关最高的递质
        corr_rows = []
        for feature_index, feature in enumerate(feature_cols):
            x = x_res[:, feature_index]
            value = 0.0
            if np.nanstd(x) > np.finfo(float).eps and np.nanstd(y_res) > np.finfo(float).eps:
                value = float(np.corrcoef(x, y_res)[0, 1])
            if not np.isfinite(value):
                value = 0.0
            nt_id = feature.replace("screen_node_", "").replace("screen_edge_", "")
            corr_rows.append({"nt_id": nt_id, "corr": value})
        corr = pd.DataFrame(corr_rows).groupby("nt_id")["corr"].sum().reindex(nt_ids).fillna(0.0)
        best_nt = str(corr.abs().idxmax())
        beta["beta"] = 0.0
        beta.loc[beta["nt_id"] == best_nt, "beta"] = float(corr.loc[best_nt])
        fallback = True

    denom = float(beta["beta"].abs().sum())
    if denom <= np.finfo(float).eps:
        beta["profile_weight"] = 0.0
    else:
        beta["profile_weight"] = beta["beta"] / denom
    beta["selected"] = beta["profile_weight"].abs() > np.finfo(float).eps
    meta = {
        "alpha": float(model.alpha_),
        "l1_ratio": float(model.l1_ratio_),
        "selected_count": int(beta["selected"].sum()),
        "fallback": fallback,
    }
    return feature_table, beta, meta


def save_profile_images(weights: np.ndarray, gray_arrays: np.ndarray, wm_arrays: np.ndarray, ref_img: nib.Nifti1Image, out_dir: Path, fold: int) -> tuple[np.ndarray, np.ndarray]:
    """Save fold-specific integrated profile images."""
    profile_dir = ensure_dir(out_dir / "profiles")
    gray_flat = weights @ gray_arrays
    wm_flat = weights @ wm_arrays
    gray_img = nib.Nifti1Image(gray_flat.reshape(ref_img.shape).astype(np.float32), ref_img.affine, ref_img.header)
    wm_img = nib.Nifti1Image(wm_flat.reshape(ref_img.shape).astype(np.float32), ref_img.affine, ref_img.header)
    gray_img.set_data_dtype(np.float32)
    wm_img.set_data_dtype(np.float32)
    nib.save(gray_img, str(profile_dir / f"fold_{fold:02d}_ntdc_hansen_profile.nii.gz"))
    nib.save(wm_img, str(profile_dir / f"fold_{fold:02d}_ntdc_alves_profile.nii.gz"))
    return gray_flat.astype(np.float32), wm_flat.astype(np.float32)


def robust_minmax(values: np.ndarray) -> np.ndarray:
    """Robustly scale a map to 0-1."""
    values = np.nan_to_num(values.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    finite = values[np.isfinite(values)]
    nonzero = finite[np.abs(finite) > np.finfo(float).eps]
    if nonzero.size == 0:
        return np.zeros_like(values, dtype=np.float32)
    low, high = np.percentile(nonzero, [1, 99])
    if not np.isfinite(high - low) or high <= low:
        low, high = float(np.min(nonzero)), float(np.max(nonzero))
    if high <= low:
        return np.zeros_like(values, dtype=np.float32)
    scaled = np.clip((values - low) / (high - low), 0.0, 1.0)
    return scaled.astype(np.float32)


def save_fixed_dopamine_profile(nt_ids: list[str], gray_arrays: np.ndarray, wm_arrays: np.ndarray, ref_img: nib.Nifti1Image, out_dir: Path) -> tuple[np.ndarray, np.ndarray] | None:
    """Save an exploratory D1-D2-DAT integrated profile."""
    target = ["d1", "d2", "dat"]
    index = [nt_ids.index(nt_id) for nt_id in target if nt_id in nt_ids]
    profile_dir = ensure_dir(out_dir / "exploratory_profiles")
    if len(index) != len(target):
        write_csv(pd.DataFrame({"requested_nt": target, "status": ["missing" if nt_id not in nt_ids else "ok" for nt_id in target]}), profile_dir / "dopamine_d1_d2_dat_manifest.csv")
        return None
    gray_profile = np.mean([robust_minmax(gray_arrays[idx]) for idx in index], axis=0).astype(np.float32)
    wm_profile = np.mean([robust_minmax(wm_arrays[idx]) for idx in index], axis=0).astype(np.float32)
    gray_img = nib.Nifti1Image(gray_profile.reshape(ref_img.shape), ref_img.affine, ref_img.header)
    wm_img = nib.Nifti1Image(wm_profile.reshape(ref_img.shape), ref_img.affine, ref_img.header)
    gray_img.set_data_dtype(np.float32)
    wm_img.set_data_dtype(np.float32)
    nib.save(gray_img, str(profile_dir / "dopamine_d1_d2_dat_hansen_profile.nii.gz"))
    nib.save(wm_img, str(profile_dir / "dopamine_d1_d2_dat_alves_profile.nii.gz"))
    write_csv(pd.DataFrame({"nt_id": target, "weight": [1.0 / len(target)] * len(target)}), profile_dir / "dopamine_d1_d2_dat_manifest.csv")
    return gray_profile, wm_profile


def compute_profile_node_damage(
    subject_ids: list[str],
    lesion_indices: dict[str, np.ndarray],
    atlas_flat: np.ndarray,
    labels: list[int],
    roi_counts: np.ndarray,
    gray_flat: np.ndarray,
) -> pd.DataFrame:
    """Compute voxel-weighted node damage for one integrated profile."""
    max_label = int(max(labels))
    rows = []
    for subject_id in subject_ids:
        idx = lesion_indices[str(subject_id)]
        label_values = atlas_flat[idx]
        valid = label_values > 0
        sums = np.bincount(label_values[valid].astype(int), weights=gray_flat[idx][valid], minlength=max_label + 1)
        row = {"subject_id": subject_id}
        for label, count in zip(labels, roi_counts):
            row[f"node_{label:03d}"] = float(sums[label] / count)
        rows.append(row)
    return pd.DataFrame(rows)


def compute_profile_edge_damage(
    subject_ids: list[str],
    lesion_indices: dict[str, np.ndarray],
    edge_matrix: sparse.csc_matrix,
    edge_names: list[str],
    wm_flat: np.ndarray,
) -> pd.DataFrame:
    """Compute edge damage for one integrated WM profile."""
    rows = []
    for subject_id in subject_ids:
        idx = lesion_indices[str(subject_id)]
        if idx.size:
            values = edge_matrix[:, idx].dot(wm_flat[idx])
            values = np.asarray(values).ravel().astype(float)
        else:
            values = np.zeros(len(edge_names), dtype=float)
        rows.append({"subject_id": subject_id, **dict(zip(edge_names, values))})
    return pd.DataFrame(rows)


def align_by_subject(df: pd.DataFrame, subject_ids: list[str]) -> pd.DataFrame:
    """Order a table by subject ID."""
    return df.set_index("subject_id").loc[subject_ids].reset_index()


def train_z_apply(train_values: pd.Series, test_values: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Apply train-set z score to train and test series."""
    mean = float(train_values.mean(skipna=True))
    sd = float(train_values.std(skipna=True))
    if not np.isfinite(sd) or sd <= np.finfo(float).eps:
        sd = 1.0
    return (train_values - mean) / sd, (test_values - mean) / sd


def save_node_weight_map(weights: pd.DataFrame, atlas_flat: np.ndarray, labels: list[int], ref_img: nib.Nifti1Image, out_dir: Path, fold: int, prefix: str) -> None:
    """Save ROI-weight maps in lesion space."""
    map_dir = ensure_dir(out_dir / "lsm_maps")
    values = np.zeros_like(atlas_flat, dtype=np.float32)
    lookup = dict(zip(weights["feature"].astype(str), weights["weight"].astype(float)))
    for label in labels:
        values[atlas_flat == label] = float(lookup.get(f"node_{label:03d}", 0.0))
    img = nib.Nifti1Image(values.reshape(ref_img.shape), ref_img.affine, ref_img.header)
    img.set_data_dtype(np.float32)
    nib.save(img, str(map_dir / f"fold_{fold:02d}_{prefix}_node_lsm_weight_map.nii.gz"))


def save_edge_weight_projection(weights: pd.DataFrame, edge_matrix: sparse.csc_matrix, edge_names: list[str], ref_img: nib.Nifti1Image, out_dir: Path, fold: int, prefix: str) -> None:
    """Save edge-weight projection maps in lesion space."""
    map_dir = ensure_dir(out_dir / "lsm_maps")
    edge_weight = np.zeros(len(edge_names), dtype=np.float32)
    lookup = dict(zip(weights["feature"].astype(str), weights["weight"].astype(float)))
    for index, edge_name in enumerate(edge_names):
        edge_weight[index] = float(lookup.get(edge_name, 0.0))
    selected = (edge_weight != 0).astype(np.float32)
    numerator = np.asarray(edge_matrix.T.dot(edge_weight)).ravel().astype(np.float32)
    denominator = np.asarray(edge_matrix.T.dot(selected)).ravel().astype(np.float32)
    values = np.zeros_like(numerator, dtype=np.float32)
    mask = denominator > 0
    values[mask] = numerator[mask] / denominator[mask]
    img = nib.Nifti1Image(values.reshape(ref_img.shape), ref_img.affine, ref_img.header)
    img.set_data_dtype(np.float32)
    nib.save(img, str(map_dir / f"fold_{fold:02d}_{prefix}_edge_lsm_projection_map.nii.gz"))


def summarize_lsm_maps(out_dir: Path, ref_img: nib.Nifti1Image) -> None:
    """Write fold-averaged LSM maps."""
    map_dir = ensure_dir(out_dir / "lsm_maps")
    for prefix in ["ntdc", "dopamine"]:
        for pattern, output_name in [
            (f"fold_*_{prefix}_node_lsm_weight_map.nii.gz", f"{prefix}_node_lsm_weight_mean_map.nii.gz"),
            (f"fold_*_{prefix}_edge_lsm_projection_map.nii.gz", f"{prefix}_edge_lsm_projection_mean_map.nii.gz"),
        ]:
            paths = sorted(map_dir.glob(pattern))
            if not paths:
                continue
            stack = np.stack([nib.load(str(path)).get_fdata().astype(np.float32) for path in paths], axis=0)
            mean_img = nib.Nifti1Image(np.mean(stack, axis=0).astype(np.float32), ref_img.affine, ref_img.header)
            freq_img = nib.Nifti1Image(np.mean(stack != 0, axis=0).astype(np.float32), ref_img.affine, ref_img.header)
            mean_img.set_data_dtype(np.float32)
            freq_img.set_data_dtype(np.float32)
            nib.save(mean_img, str(map_dir / output_name))
            nib.save(freq_img, str(map_dir / output_name.replace("_mean_map", "_selection_frequency_map")))


def compute_fold_scores(
    config: dict,
    fold: int,
    data: pd.DataFrame,
    train_ids: list[str],
    test_ids: list[str],
    lesion_node: pd.DataFrame,
    lesion_edge: pd.DataFrame,
    profile_node_train: pd.DataFrame,
    profile_node_test: pd.DataFrame,
    profile_edge_train: pd.DataFrame,
    profile_edge_test: pd.DataFrame,
    out_dir: Path,
    atlas_flat: np.ndarray,
    labels: list[int],
    ref_img: nib.Nifti1Image,
    edge_matrix: sparse.csc_matrix,
    edge_names: list[str],
    score_col: str = "ntdc",
    map_prefix: str = "ntdc",
    weight_prefix: str = "profile",
    impact_prefix: str = "profile",
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int]]:
    """Learn LSM weights in train data and score train/test rows."""
    impact_cfg = config.get("impact", {})
    outcome = outcome_column(config)
    covariates = analysis_covariates(config, "model_covariates")
    stat_name = str(impact_cfg.get("weight_stat", "t"))
    q_threshold = float(impact_cfg.get("q_threshold", 0.05))
    node_top_k = int(impact_cfg.get("node_top_k", 20))
    edge_top_k = int(impact_cfg.get("edge_top_k", 200))
    use_q = bool(impact_cfg.get("use_q_if_available", True))
    weight_dir = ensure_dir(out_dir / "fold_weights")

    train_pheno = align_by_subject(data, train_ids)
    test_pheno = align_by_subject(data, test_ids)
    train_lesion_node = align_by_subject(lesion_node, train_ids)
    test_lesion_node = align_by_subject(lesion_node, test_ids)
    train_lesion_edge = align_by_subject(lesion_edge, train_ids)
    test_lesion_edge = align_by_subject(lesion_edge, test_ids)

    lesion_node_stats = fit_fast_mass_univariate(train_lesion_node, train_pheno, outcome, covariates)
    lesion_edge_stats = fit_fast_mass_univariate(train_lesion_edge, train_pheno, outcome, covariates)
    profile_node_stats = fit_fast_mass_univariate(profile_node_train, train_pheno, outcome, covariates)
    profile_edge_stats = fit_fast_mass_univariate(profile_edge_train, train_pheno, outcome, covariates)

    lesion_node_weights = select_weights(lesion_node_stats, stat_name, node_top_k, q_threshold, use_q)
    lesion_edge_weights = select_weights(lesion_edge_stats, stat_name, edge_top_k, q_threshold, use_q)
    profile_node_weights = select_weights(profile_node_stats, stat_name, node_top_k, q_threshold, use_q)
    profile_edge_weights = select_weights(profile_edge_stats, stat_name, edge_top_k, q_threshold, use_q)
    write_csv(lesion_node_weights, weight_dir / f"fold_{fold:02d}_lesion_node_weights.csv")
    write_csv(lesion_edge_weights, weight_dir / f"fold_{fold:02d}_lesion_edge_weights.csv")
    write_csv(profile_node_weights, weight_dir / f"fold_{fold:02d}_{weight_prefix}_node_weights.csv")
    write_csv(profile_edge_weights, weight_dir / f"fold_{fold:02d}_{weight_prefix}_edge_weights.csv")
    save_node_weight_map(profile_node_weights, atlas_flat, labels, ref_img, out_dir, fold, map_prefix)
    save_edge_weight_projection(profile_edge_weights, edge_matrix, edge_names, ref_img, out_dir, fold, map_prefix)

    train_lesion_node_impact = impact_from_weights(train_lesion_node, lesion_node_weights)
    test_lesion_node_impact = impact_from_weights(test_lesion_node, lesion_node_weights)
    train_lesion_edge_impact = impact_from_weights(train_lesion_edge, lesion_edge_weights)
    test_lesion_edge_impact = impact_from_weights(test_lesion_edge, lesion_edge_weights)
    train_profile_node_impact = impact_from_weights(profile_node_train, profile_node_weights)
    test_profile_node_impact = impact_from_weights(profile_node_test, profile_node_weights)
    train_profile_edge_impact = impact_from_weights(profile_edge_train, profile_edge_weights)
    test_profile_edge_impact = impact_from_weights(profile_edge_test, profile_edge_weights)

    train_lesion_node_z, test_lesion_node_z = train_z_apply(train_lesion_node_impact, test_lesion_node_impact)
    train_lesion_edge_z, test_lesion_edge_z = train_z_apply(train_lesion_edge_impact, test_lesion_edge_impact)
    train_profile_node_z, test_profile_node_z = train_z_apply(train_profile_node_impact, test_profile_node_impact)
    train_profile_edge_z, test_profile_edge_z = train_z_apply(train_profile_edge_impact, test_profile_edge_impact)

    train_scores = train_pheno[["subject_id", "cv_group", outcome, *covariates]].copy()
    test_scores = test_pheno[["subject_id", "cv_group", outcome, *covariates]].copy()
    train_scores["fold"] = fold
    test_scores["fold"] = fold
    train_scores["lesion_node_impact"] = train_lesion_node_impact.to_numpy()
    train_scores["lesion_edge_impact"] = train_lesion_edge_impact.to_numpy()
    test_scores["lesion_node_impact"] = test_lesion_node_impact.to_numpy()
    test_scores["lesion_edge_impact"] = test_lesion_edge_impact.to_numpy()
    train_scores[f"{impact_prefix}_node_impact"] = train_profile_node_impact.to_numpy()
    train_scores[f"{impact_prefix}_edge_impact"] = train_profile_edge_impact.to_numpy()
    test_scores[f"{impact_prefix}_node_impact"] = test_profile_node_impact.to_numpy()
    test_scores[f"{impact_prefix}_edge_impact"] = test_profile_edge_impact.to_numpy()
    train_scores["sdc"] = train_lesion_node_z.to_numpy() + train_lesion_edge_z.to_numpy()
    test_scores["sdc"] = test_lesion_node_z.to_numpy() + test_lesion_edge_z.to_numpy()
    train_scores[score_col] = train_profile_node_z.to_numpy() + train_profile_edge_z.to_numpy()
    test_scores[score_col] = test_profile_node_z.to_numpy() + test_profile_edge_z.to_numpy()

    counts = {
        "selected_lesion_nodes": int((lesion_node_weights["weight"] != 0).sum()),
        "selected_lesion_edges": int((lesion_edge_weights["weight"] != 0).sum()),
        f"selected_{impact_prefix}_nodes": int((profile_node_weights["weight"] != 0).sum()),
        f"selected_{impact_prefix}_edges": int((profile_edge_weights["weight"] != 0).sum()),
    }
    return train_scores, test_scores, counts


def predict_fold(train: pd.DataFrame, test: pd.DataFrame, labels: list[int], threshold: float, covariates: list[str], outcome: str, ntdc_col: str = "ntdc", ntdc_label: str = "NTDC") -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Fit final prediction models in one outer fold."""
    rows = []
    status_rows = []
    fold = int(test["fold"].iloc[0])
    for model_name, predictors in model_specs(covariates, ntdc_col, ntdc_label):
        fit, status, error = fit_ordered_model(train, outcome, predictors)
        status_rows.append({"model": model_name, "fold": fold, "n_train": int(train.shape[0]), "n_test": int(test.shape[0]), "status": status, "error": error})
        if fit is None:
            continue
        probabilities = fit.model.predict(fit.params, exog=test[predictors].astype(float))
        aligned = align_probabilities(fit, np.asarray(probabilities, dtype=float), labels)
        expected = aligned @ np.asarray(labels, dtype=float)
        binary_probability = aligned[:, [index for index, label in enumerate(labels) if label <= threshold]].sum(axis=1)
        for row_index, subject_id in enumerate(test["subject_id"].tolist()):
            item = {
                "subject_id": subject_id,
                "model": model_name,
                "fold": fold,
                "observed_mrs": int(test.iloc[row_index][outcome]),
                "expected_mrs": float(expected[row_index]),
                "prob_mrs_le_threshold": float(binary_probability[row_index]),
            }
            for label_index, label in enumerate(labels):
                item[f"prob_{label}"] = float(aligned[row_index, label_index])
            rows.append(item)
    return rows, status_rows


def summarize_predictions(config: dict, predictions: pd.DataFrame, out_dir: Path, ntdc_col: str = "ntdc", ntdc_label: str = "NTDC", file_prefix: str = "") -> None:
    """Write prediction metrics and paired bootstrap comparisons."""
    binary_cfg = config.get("analysis", {}).get("binary_outcome", {})
    threshold = float(binary_cfg.get("threshold", 2))
    labels = [int(label) for label in sorted(predictions["observed_mrs"].unique())]
    covariates = analysis_covariates(config, "model_covariates")
    specs = model_specs(covariates, ntdc_col, ntdc_label)
    performance_rows = []
    for model_name, _ in specs:
        model_pred = predictions[predictions["model"] == model_name].copy()
        performance_rows.append({"model": model_name, "n": int(model_pred.shape[0]), **prediction_metrics(model_pred, labels, threshold)})
    performance = pd.DataFrame(performance_rows)
    write_csv(performance, out_dir / f"{file_prefix}model_prediction_performance.csv")

    directions = metric_directions()
    n_bootstrap = int(config.get("impact", {}).get("prediction_bootstrap", 1000))
    random_state = int(config.get("impact", {}).get("random_state", 42))
    pair_rows = []
    for index, (model_a, _) in enumerate(specs):
        pred_a = predictions[predictions["model"] == model_a].copy()
        for model_b, _ in specs[index + 1 :]:
            pred_b = predictions[predictions["model"] == model_b].copy()
            for metric, direction in directions.items():
                delta, ci_low, ci_high, p_value = bootstrap_metric_delta(pred_a, pred_b, labels, threshold, metric, n_bootstrap, random_state + index + len(pair_rows))
                pair_rows.append(
                    {
                        "model_a": model_a,
                        "model_b": model_b,
                        "metric": metric,
                        "higher_is_better": direction == "higher",
                        "delta_b_minus_a": delta,
                        "ci_low": ci_low,
                        "ci_high": ci_high,
                        "p_bootstrap": p_value,
                        "n_bootstrap": n_bootstrap,
                    }
                )
    pairwise = pd.DataFrame(pair_rows)
    pairwise["p_fdr_bh"] = np.nan
    pairwise["p_bonferroni"] = np.nan
    for metric in pairwise["metric"].unique():
        mask = (pairwise["metric"] == metric) & pairwise["p_bootstrap"].notna()
        if mask.any():
            pairwise.loc[mask, "p_fdr_bh"] = multipletests(pairwise.loc[mask, "p_bootstrap"], method="fdr_bh")[1]
            pairwise.loc[mask, "p_bonferroni"] = multipletests(pairwise.loc[mask, "p_bootstrap"], method="bonferroni")[1]
    write_csv(pairwise, out_dir / f"{file_prefix}model_prediction_pairwise_bootstrap.csv")


def write_run_report(out_dir: Path, selection_summary: pd.DataFrame, performance: pd.DataFrame) -> None:
    """Write a compact run report."""
    lines = [
        "# NTDC Run Report",
        "",
        "## Models",
        "",
        "- Clinical",
        "- Clinical + SDC",
        "- Clinical + NTDC",
        "- Clinical + SDC + NTDC",
        "",
        "## Selected Neurotransmitters",
        "",
        selection_summary.to_string(index=False),
        "",
        "## Prediction Performance",
        "",
        performance.to_string(index=False),
        "",
    ]
    (out_dir / "ml_profile_run_report.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run fold-specific NTDC prediction.")
    parser.add_argument("--config", default="config/dat_config.yaml")
    parser.add_argument("--force-maps", action="store_true")
    parser.add_argument("--force-screening", action="store_true")
    parser.add_argument("--max-subjects", type=int, default=None)
    parser.add_argument("--output-subdir", default="ml_profile")
    return parser.parse_args()


def main() -> None:
    """Run the NTDC pipeline."""
    args = parse_args()
    config = load_config(args.config)
    out_dir = ensure_dir(project_path(config, "derivatives", args.output_subdir))
    model_dir = ensure_dir(out_dir / "models")
    specs = neurotransmitter_specs(config)
    manifest = pd.read_csv(project_path(config, config["outputs"]["qc_dir"], "subject_manifest.csv"))
    manifest = add_cv_group(config, manifest)
    outcome = outcome_column(config)
    covariates = analysis_covariates(config, "model_covariates")
    required = ["subject_id", "cv_group", "lesion_path", outcome, *covariates]
    data = manifest.dropna(subset=[column for column in required if column in manifest.columns]).copy()
    if args.max_subjects is not None:
        data = data.head(int(args.max_subjects)).copy()
    data[outcome] = data[outcome].astype(int)

    reference_2mm = Path(data["lesion_path"].iloc[0])
    maps = prepare_maps(config, specs, reference_2mm, out_dir, args.force_maps)
    nt_ids, gray_arrays, wm_arrays, ref_img = load_map_arrays(maps)
    dopamine_profile = save_fixed_dopamine_profile(nt_ids, gray_arrays, wm_arrays, ref_img, out_dir)
    atlas_flat, labels, roi_counts = atlas_info(config)
    lesion_indices = load_lesion_indices(data)
    edge_matrix, edge_names = load_edge_matrix(config)
    lesion_node = compute_lesion_node_load(config, data)
    _, lesion_edge = load_lesion_feature_tables(config, data)
    lesion_node = align_by_subject(lesion_node, data["subject_id"].astype(str).tolist())
    lesion_edge = align_by_subject(lesion_edge, data["subject_id"].astype(str).tolist())
    screening = compute_screening_features(data, nt_ids, gray_arrays, wm_arrays, atlas_flat, lesion_indices, out_dir, args.force_screening)
    data = data.merge(screening, on="subject_id", how="inner")

    feature_cols = [f"screen_node_{nt_id}" for nt_id in nt_ids] + [f"screen_edge_{nt_id}" for nt_id in nt_ids]
    binary_cfg = config.get("analysis", {}).get("binary_outcome", {})
    threshold = float(binary_cfg.get("threshold", 2))
    labels_outcome = [int(label) for label in sorted(data[outcome].unique())]
    n_splits = min(int(config.get("impact", {}).get("n_splits", 10)), data["cv_group"].nunique())
    if n_splits < 2:
        raise RuntimeError("at least two cross-validation groups are required")

    splitter = GroupKFold(n_splits=n_splits)
    fold_rows = []
    selection_rows = []
    feature_coef_rows = []
    score_rows = []
    prediction_rows = []
    status_rows = []
    dopamine_score_rows = []
    dopamine_prediction_rows = []
    dopamine_status_rows = []
    dopamine_fold_rows = []

    for fold, (train_idx, test_idx) in enumerate(splitter.split(data, groups=data["cv_group"].astype(str)), start=1):
        train = data.iloc[train_idx].copy()
        test = data.iloc[test_idx].copy()
        train_ids = train["subject_id"].astype(str).tolist()
        test_ids = test["subject_id"].astype(str).tolist()
        feature_table, beta_table, meta = elastic_net_nt_weights(train, feature_cols, nt_ids, covariates, outcome, train["cv_group"].astype(str).to_numpy(), config)
        weights = beta_table.set_index("nt_id").reindex(nt_ids)["profile_weight"].fillna(0.0).to_numpy(dtype=np.float32)
        gray_profile, wm_profile = save_profile_images(weights, gray_arrays, wm_arrays, ref_img, out_dir, fold)

        profile_node_train = compute_profile_node_damage(train_ids, lesion_indices, atlas_flat, labels, roi_counts, gray_profile)
        profile_node_test = compute_profile_node_damage(test_ids, lesion_indices, atlas_flat, labels, roi_counts, gray_profile)
        profile_edge_train = compute_profile_edge_damage(train_ids, lesion_indices, edge_matrix, edge_names, wm_profile)
        profile_edge_test = compute_profile_edge_damage(test_ids, lesion_indices, edge_matrix, edge_names, wm_profile)
        train_scores, test_scores, selected_counts = compute_fold_scores(
            config,
            fold,
            data,
            train_ids,
            test_ids,
            lesion_node,
            lesion_edge,
            profile_node_train,
            profile_node_test,
            profile_edge_train,
            profile_edge_test,
            out_dir,
            atlas_flat,
            labels,
            ref_img,
            edge_matrix,
            edge_names,
        )
        fold_pred, fold_status = predict_fold(train_scores, test_scores, labels_outcome, threshold, covariates, outcome)
        prediction_rows.extend(fold_pred)
        status_rows.extend(fold_status)
        score_rows.append(test_scores)

        if dopamine_profile is not None:
            dopamine_gray, dopamine_wm = dopamine_profile
            dopamine_node_train = compute_profile_node_damage(train_ids, lesion_indices, atlas_flat, labels, roi_counts, dopamine_gray)
            dopamine_node_test = compute_profile_node_damage(test_ids, lesion_indices, atlas_flat, labels, roi_counts, dopamine_gray)
            dopamine_edge_train = compute_profile_edge_damage(train_ids, lesion_indices, edge_matrix, edge_names, dopamine_wm)
            dopamine_edge_test = compute_profile_edge_damage(test_ids, lesion_indices, edge_matrix, edge_names, dopamine_wm)
            dopamine_train_scores, dopamine_test_scores, dopamine_counts = compute_fold_scores(
                config,
                fold,
                data,
                train_ids,
                test_ids,
                lesion_node,
                lesion_edge,
                dopamine_node_train,
                dopamine_node_test,
                dopamine_edge_train,
                dopamine_edge_test,
                out_dir,
                atlas_flat,
                labels,
                ref_img,
                edge_matrix,
                edge_names,
                score_col="dopamine_ntdc",
                map_prefix="dopamine",
                weight_prefix="dopamine",
                impact_prefix="dopamine",
            )
            dopamine_pred, dopamine_status = predict_fold(dopamine_train_scores, dopamine_test_scores, labels_outcome, threshold, covariates, outcome, "dopamine_ntdc", "D1/D2/DAT")
            dopamine_prediction_rows.extend(dopamine_pred)
            dopamine_status_rows.extend(dopamine_status)
            dopamine_score_rows.append(dopamine_test_scores)
            dopamine_fold_rows.append({"fold": fold, **dopamine_counts})

        beta_table["fold"] = fold
        feature_table["fold"] = fold
        for key, value in meta.items():
            beta_table[key] = value
            feature_table[key] = value
        selection_rows.append(beta_table)
        feature_coef_rows.append(feature_table)
        fold_rows.append(
            {
                "fold": fold,
                "n_train": int(train.shape[0]),
                "n_test": int(test.shape[0]),
                "n_train_groups": int(train["cv_group"].nunique()),
                "n_test_groups": int(test["cv_group"].nunique()),
                "selected_nt": "|".join(beta_table.loc[beta_table["selected"], "nt_id"].astype(str).tolist()),
                **meta,
                **selected_counts,
            }
        )
        print(f"finished fold {fold}/{n_splits}")

    scores = pd.concat(score_rows, ignore_index=True).drop_duplicates(subset=["subject_id"], keep="last")
    selections = pd.concat(selection_rows, ignore_index=True)
    feature_coefs = pd.concat(feature_coef_rows, ignore_index=True)
    selection_summary = (
        selections.groupby("nt_id")
        .agg(
            selection_frequency=("selected", "mean"),
            mean_beta=("beta", "mean"),
            mean_abs_beta=("beta", lambda x: float(np.mean(np.abs(x)))),
            mean_profile_weight=("profile_weight", "mean"),
            mean_abs_profile_weight=("profile_weight", lambda x: float(np.mean(np.abs(x)))),
            selected_folds=("selected", "sum"),
        )
        .reset_index()
        .sort_values(["selection_frequency", "mean_abs_profile_weight"], ascending=False)
    )
    predictions = pd.DataFrame(prediction_rows)
    write_csv(pd.DataFrame(fold_rows), out_dir / "fold_manifest.csv")
    write_csv(selections, out_dir / "selection_folds.csv")
    write_csv(feature_coefs, out_dir / "selection_feature_coefficients.csv")
    write_csv(selection_summary, out_dir / "selection_summary.csv")
    write_csv(scores, out_dir / "profile_scores.csv")
    write_csv(predictions, model_dir / "model_prediction_cv.csv")
    write_csv(pd.DataFrame(status_rows), model_dir / "model_prediction_fold_status.csv")
    summarize_predictions(config, predictions, model_dir)
    if dopamine_prediction_rows:
        dopamine_dir = ensure_dir(out_dir / "exploratory_profiles")
        dopamine_scores = pd.concat(dopamine_score_rows, ignore_index=True).drop_duplicates(subset=["subject_id"], keep="last")
        dopamine_predictions = pd.DataFrame(dopamine_prediction_rows)
        write_csv(dopamine_scores, dopamine_dir / "dopamine_d1_d2_dat_scores.csv")
        write_csv(pd.DataFrame(dopamine_fold_rows), dopamine_dir / "dopamine_d1_d2_dat_fold_summary.csv")
        write_csv(dopamine_predictions, dopamine_dir / "dopamine_d1_d2_dat_model_prediction_cv.csv")
        write_csv(pd.DataFrame(dopamine_status_rows), dopamine_dir / "dopamine_d1_d2_dat_model_prediction_fold_status.csv")
        summarize_predictions(config, dopamine_predictions, dopamine_dir, "dopamine_ntdc", "D1/D2/DAT", "dopamine_d1_d2_dat_")
    summarize_lsm_maps(out_dir, ref_img)
    performance = pd.read_csv(model_dir / "model_prediction_performance.csv")
    write_run_report(out_dir, selection_summary, performance)


if __name__ == "__main__":
    main()
