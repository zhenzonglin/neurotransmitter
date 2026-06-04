#!/usr/bin/env python3
from __future__ import annotations

import argparse
import multiprocessing as mp
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.svm import LinearSVC
from statsmodels.stats.multitest import multipletests

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from compute_impact_scores import (  # noqa: E402
    align_probabilities,
    bootstrap_metric_delta,
    fit_fast_mass_univariate,
    fit_ordered_model,
    impact_from_weights,
    metric_directions,
    prediction_metrics,
    select_weights,
)
from nt_analysis.config import analysis_covariates, ensure_dir, load_config, outcome_column, project_path  # noqa: E402
from nt_analysis.tables import write_csv  # noqa: E402


model_context: dict[str, object] = {}
edge_context: dict[str, object] = {}


def load_inputs(config: dict, max_subjects: int | None) -> dict[str, object]:
    """Load feature arrays and metadata."""
    atlas_cfg = config.get("prognostic_ntdc_atlas", {})
    out_dir = ensure_dir(project_path(config, atlas_cfg.get("output_dir", "derivatives/prognostic_ntdc_atlas")))
    subject = pd.read_csv(out_dir / "subject_table.csv", dtype={"subject_id": str})
    if max_subjects is not None:
        subject = subject.head(int(max_subjects)).copy()
    roi = pd.read_csv(out_dir / "roi_table.csv")
    edge = pd.read_csv(out_dir / "edge_table.csv")
    nt = pd.read_csv(out_dir / "nt_table.csv")
    n_nt = nt.shape[0]
    support_qc, support_summary = edge_support_tables(out_dir, edge)
    supported_edge_names = support_qc.loc[support_qc["tract_supported"], "edge"].astype(str).tolist()
    supported_index = support_qc.index[support_qc["tract_supported"]].to_numpy(dtype=int)
    if len(supported_edge_names) == 0:
        raise RuntimeError("no tract-supported edges were found")
    lesion_node = pd.read_csv(out_dir / "lesion_node_damage.csv", dtype={"subject_id": str})
    lesion_edge = pd.read_csv(out_dir / "lesion_edge_damage.csv", dtype={"subject_id": str})
    ids = subject["subject_id"].astype(str).tolist()
    lesion_node["subject_id"] = lesion_node["subject_id"].astype(str)
    lesion_edge["subject_id"] = lesion_edge["subject_id"].astype(str)
    lesion_node = lesion_node.set_index("subject_id").loc[ids].reset_index()
    lesion_edge = lesion_edge.set_index("subject_id").loc[ids].reset_index()
    missing_edges = [name for name in supported_edge_names if name not in lesion_edge.columns]
    if missing_edges:
        raise KeyError(f"missing supported edge columns in lesion_edge_damage.csv: {missing_edges[:10]}")
    lesion_edge = lesion_edge[["subject_id", *supported_edge_names]].copy()
    node_damage = np.load(out_dir / f"nt_node_damage_{n_nt}nt.npy", mmap_mode="r")[: subject.shape[0]]
    # 只让有纤维体素和递质分母的边进入模型
    edge_damage_full = np.load(out_dir / f"nt_edge_damage_fraction_{n_nt}nt.npy", mmap_mode="r")
    edge_sum_full = np.load(out_dir / f"nt_edge_damage_sum_{n_nt}nt.npy", mmap_mode="r")
    edge_damage = np.asarray(edge_damage_full[: subject.shape[0], supported_index, :], dtype=np.float32)
    edge_sum = np.asarray(edge_sum_full[: subject.shape[0], supported_index, :], dtype=np.float32)
    edge = edge.iloc[supported_index].reset_index(drop=True)
    write_csv(support_qc, out_dir / "edge_support_qc.csv")
    write_csv(support_summary, out_dir / "edge_support_summary.csv")
    return {
        "out_dir": out_dir,
        "subject": subject,
        "roi": roi,
        "edge": edge,
        "nt": nt,
        "lesion_node": lesion_node,
        "lesion_edge": lesion_edge,
        "node_damage": node_damage,
        "edge_damage": edge_damage,
        "edge_sum": edge_sum,
        "edge_support_qc": support_qc,
        "edge_support_summary": support_summary,
    }


def edge_support_tables(out_dir: Path, edge: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build tract-supported edge QC tables."""
    qc_path = out_dir / "nt_edge_denominator_qc.csv"
    if not qc_path.exists():
        raise FileNotFoundError(f"missing edge denominator QC: {qc_path}")
    qc = pd.read_csv(qc_path)
    required = {"edge", "nt_id", "denominator", "low_denominator"}
    missing = required - set(qc.columns)
    if missing:
        raise KeyError(f"missing columns in nt_edge_denominator_qc.csv: {sorted(missing)}")
    qc["edge"] = qc["edge"].astype(str)
    qc["low_denominator_bool"] = qc["low_denominator"].astype(str).str.lower().isin(["true", "1", "yes"])
    edge_order = edge["edge"].astype(str).tolist()
    grouped = (
        qc.assign(positive_denominator=qc["denominator"].astype(float) > np.finfo(np.float32).eps)
        .groupby("edge", as_index=False)
        .agg(
            n_nt=("nt_id", "count"),
            positive_nt=("positive_denominator", "sum"),
            low_nt=("low_denominator_bool", "sum"),
            median_denominator=("denominator", "median"),
            max_denominator=("denominator", "max"),
        )
    )
    grouped["tract_supported"] = grouped["positive_nt"].astype(int) > 0
    grouped["all_zero_denominator"] = grouped["positive_nt"].astype(int) == 0
    support = pd.DataFrame({"edge": edge_order}).merge(grouped, on="edge", how="left")
    support["n_nt"] = support["n_nt"].fillna(0).astype(int)
    support["positive_nt"] = support["positive_nt"].fillna(0).astype(int)
    support["low_nt"] = support["low_nt"].fillna(0).astype(int)
    support["median_denominator"] = support["median_denominator"].fillna(0.0).astype(float)
    support["max_denominator"] = support["max_denominator"].fillna(0.0).astype(float)
    support["tract_supported"] = support["tract_supported"].fillna(False).astype(bool)
    support["all_zero_denominator"] = support["all_zero_denominator"].fillna(True).astype(bool)
    summary = pd.DataFrame(
        [
            {
                "theoretical_edges": int(support.shape[0]),
                "tract_supported_edges": int(support["tract_supported"].sum()),
                "excluded_unsupported_edges": int((~support["tract_supported"]).sum()),
                "edges_with_any_low_denominator": int((support["low_nt"] > 0).sum()),
                "nt_systems": int(qc["nt_id"].nunique()),
            }
        ]
    )
    return support, summary


def align_by_subject(df: pd.DataFrame, subject_ids: list[str]) -> pd.DataFrame:
    """Align rows by subject ID."""
    out = df.copy()
    out["subject_id"] = out["subject_id"].astype(str)
    return out.set_index("subject_id").loc[[str(value) for value in subject_ids]].reset_index()


def train_z_apply(train_values: np.ndarray, test_values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Apply train-set z score."""
    train_values = np.asarray(train_values, dtype=float)
    test_values = np.asarray(test_values, dtype=float)
    if train_values.size == 0:
        # 空特征集直接返回，避免小样本日志警告
        n_features = train_values.shape[1] if train_values.ndim > 1 else 0
        return train_values, test_values, np.zeros(n_features), np.ones(n_features)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mean = np.nanmean(train_values, axis=0)
        sd = np.nanstd(train_values, axis=0, ddof=1)
    mean = np.where(np.isfinite(mean), mean, 0.0)
    sd = np.where(np.isfinite(sd) & (sd > np.finfo(float).eps), sd, 1.0)
    return (train_values - mean) / sd, (test_values - mean) / sd, mean, sd


def residualize_train_test(train_x: np.ndarray, test_x: np.ndarray, train_base: np.ndarray, test_base: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Residualize test data with train-set coefficients."""
    train_x = np.nan_to_num(np.asarray(train_x, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    test_x = np.nan_to_num(np.asarray(test_x, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    train_base = np.nan_to_num(np.asarray(train_base, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    test_base = np.nan_to_num(np.asarray(test_base, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    train_design = np.column_stack([np.ones(train_base.shape[0]), train_base])
    test_design = np.column_stack([np.ones(test_base.shape[0]), test_base])
    try:
        coef, *_ = np.linalg.lstsq(train_design, train_x, rcond=None)
        return train_x - train_design @ coef, test_x - test_design @ coef
    except np.linalg.LinAlgError:
        mean = np.nanmean(train_x, axis=0)
        mean = np.where(np.isfinite(mean), mean, 0.0)
        return train_x - mean, test_x - mean


def fit_ridge_location(
    train_x: np.ndarray,
    test_x: np.ndarray,
    train_base: np.ndarray,
    test_base: np.ndarray,
    cov_train: np.ndarray,
    cov_test: np.ndarray,
    y_train: np.ndarray,
    ridge_c: float,
    max_iter: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Fit one ridge model and return coefficients and scores."""
    beta = np.zeros(train_x.shape[1], dtype=np.float32)
    if len(np.unique(y_train)) < 2 or train_x.shape[0] <= 2:
        return beta, np.zeros(train_x.shape[0], dtype=np.float32), np.zeros(test_x.shape[0], dtype=np.float32), 0.0
    train_res, test_res = residualize_train_test(train_x, test_x, train_base, test_base)
    train_res_z, test_res_z, _, _ = train_z_apply(train_res, test_res)
    cov_train_z, cov_test_z, _, _ = train_z_apply(cov_train, cov_test)
    if np.nanstd(train_res_z) <= np.finfo(float).eps:
        return beta, np.zeros(train_x.shape[0], dtype=np.float32), np.zeros(test_x.shape[0], dtype=np.float32), 0.0
    x_model = np.column_stack([cov_train_z, train_res_z])
    try:
        model = LogisticRegression(penalty="l2", C=float(ridge_c), solver="lbfgs", max_iter=int(max_iter))
        model.fit(x_model, y_train)
        beta = model.coef_[0, cov_train.shape[1] :].astype(np.float32)
    except Exception:
        return beta, np.zeros(train_x.shape[0], dtype=np.float32), np.zeros(test_x.shape[0], dtype=np.float32), 0.0
    denom = float(np.sum(np.abs(beta)))
    train_score = train_res_z @ beta if denom > np.finfo(float).eps else np.zeros(train_x.shape[0], dtype=np.float32)
    test_score = test_res_z @ beta if denom > np.finfo(float).eps else np.zeros(test_x.shape[0], dtype=np.float32)
    return beta, train_score.astype(np.float32), test_score.astype(np.float32), denom


def fit_sensitivity_models(x: np.ndarray, cov: np.ndarray, y: np.ndarray, random_state: int, max_iter: int) -> dict[str, np.ndarray]:
    """Fit sensitivity models for one location."""
    x_z, _, _, _ = train_z_apply(x, x)
    cov_z, _, _, _ = train_z_apply(cov, cov)
    design = np.column_stack([cov_z, x_z])
    n_cov = cov.shape[1]
    out: dict[str, np.ndarray] = {}
    if len(np.unique(y)) < 2:
        return out
    try:
        model = LogisticRegression(penalty="elasticnet", solver="saga", l1_ratio=0.5, C=1.0, max_iter=max_iter, random_state=random_state)
        model.fit(design, y)
        out["elasticnet_logistic"] = model.coef_[0, n_cov:].astype(float)
    except Exception:
        pass
    try:
        model = LinearSVC(C=1.0, max_iter=max_iter, random_state=random_state)
        model.fit(design, y)
        out["linear_svm"] = model.coef_[0, n_cov:].astype(float)
    except Exception:
        pass
    try:
        model = GradientBoostingClassifier(random_state=random_state)
        model.fit(design, y)
        out["gradient_boosting"] = model.feature_importances_[n_cov:].astype(float)
    except Exception:
        pass
    return out


def binary_outcome(config: dict, values: pd.Series) -> np.ndarray:
    """Build binary good-outcome labels."""
    binary_cfg = config.get("analysis", {}).get("binary_outcome", {})
    threshold = float(binary_cfg.get("threshold", 2))
    positive_if_less_equal = bool(binary_cfg.get("positive_if_less_equal", True))
    numeric = values.astype(float).to_numpy()
    return (numeric <= threshold).astype(int) if positive_if_less_equal else (numeric > threshold).astype(int)


def fold_sdc(config: dict, data: pd.DataFrame, train_ids: list[str], test_ids: list[str], lesion_node: pd.DataFrame, lesion_edge: pd.DataFrame) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series, pd.DataFrame, pd.DataFrame]:
    """Learn structure-only SDC in one fold."""
    impact_cfg = config.get("impact", {})
    outcome = outcome_column(config)
    covariates = analysis_covariates(config, "model_covariates")
    train_pheno = align_by_subject(data, train_ids)
    train_node = align_by_subject(lesion_node, train_ids)
    test_node = align_by_subject(lesion_node, test_ids)
    train_edge = align_by_subject(lesion_edge, train_ids)
    test_edge = align_by_subject(lesion_edge, test_ids)
    node_stats = fit_fast_mass_univariate(train_node, train_pheno, outcome, covariates)
    edge_stats = fit_fast_mass_univariate(train_edge, train_pheno, outcome, covariates)
    node_weights = select_weights(node_stats, str(impact_cfg.get("weight_stat", "t")), int(impact_cfg.get("node_top_k", 20)), float(impact_cfg.get("q_threshold", 0.05)), bool(impact_cfg.get("use_q_if_available", True)))
    edge_weights = select_weights(edge_stats, str(impact_cfg.get("weight_stat", "t")), int(impact_cfg.get("edge_top_k", 200)), float(impact_cfg.get("q_threshold", 0.05)), bool(impact_cfg.get("use_q_if_available", True)))
    train_node_impact = impact_from_weights(train_node, node_weights)
    test_node_impact = impact_from_weights(test_node, node_weights)
    train_edge_impact = impact_from_weights(train_edge, edge_weights)
    test_edge_impact = impact_from_weights(test_edge, edge_weights)
    train_node_z, test_node_z, _, _ = train_z_apply(train_node_impact.to_numpy(dtype=float), test_node_impact.to_numpy(dtype=float))
    train_edge_z, test_edge_z, _, _ = train_z_apply(train_edge_impact.to_numpy(dtype=float), test_edge_impact.to_numpy(dtype=float))
    return (
        pd.Series(train_node_z + train_edge_z, index=train_node.index),
        pd.Series(test_node_z + test_edge_z, index=test_node.index),
        train_node_impact,
        train_edge_impact,
        node_weights,
        edge_weights,
    )


def init_edge_context(context: dict[str, object]) -> None:
    """Initialize edge worker context."""
    edge_context.clear()
    edge_context.update(context)


def fit_edge_chunk(job: tuple[int, int]) -> dict[str, object]:
    """Fit ridge models for one edge chunk."""
    start, end = job
    edge_damage = edge_context["edge_damage"]
    lesion_edge_values = edge_context["lesion_edge_values"]
    train_idx = edge_context["train_idx"]
    test_idx = edge_context["test_idx"]
    cov_train = edge_context["cov_train"]
    cov_test = edge_context["cov_test"]
    y_train = edge_context["y_train"]
    lesion_volume_train = edge_context["lesion_volume_train"]
    lesion_volume_test = edge_context["lesion_volume_test"]
    sdc_train = edge_context["sdc_train"]
    sdc_test = edge_context["sdc_test"]
    ridge_c = edge_context["ridge_c"]
    max_iter = edge_context["max_iter"]
    n_nt = edge_damage.shape[2]
    beta = np.zeros((end - start, n_nt), dtype=np.float32)
    train_sum = np.zeros(len(train_idx), dtype=np.float32)
    test_sum = np.zeros(len(test_idx), dtype=np.float32)
    denom_sum = 0.0
    for offset, edge_index in enumerate(range(start, end)):
        train_x = np.asarray(edge_damage[train_idx, edge_index, :], dtype=np.float32)
        test_x = np.asarray(edge_damage[test_idx, edge_index, :], dtype=np.float32)
        local_train = np.asarray(lesion_edge_values[train_idx, edge_index], dtype=float).reshape(-1, 1)
        local_test = np.asarray(lesion_edge_values[test_idx, edge_index], dtype=float).reshape(-1, 1)
        train_base = np.column_stack([lesion_volume_train, local_train, sdc_train])
        test_base = np.column_stack([lesion_volume_test, local_test, sdc_test])
        local_beta, local_train_score, local_test_score, local_denom = fit_ridge_location(train_x, test_x, train_base, test_base, cov_train, cov_test, y_train, ridge_c, max_iter)
        beta[offset, :] = local_beta
        train_sum += local_train_score
        test_sum += local_test_score
        denom_sum += local_denom
    return {"start": start, "end": end, "beta": beta, "train_sum": train_sum, "test_sum": test_sum, "denom_sum": denom_sum}


def run_fold(job: tuple[int, list[int], list[int]]) -> dict[str, object]:
    """Run one outer fold."""
    fold, train_idx, test_idx = job
    config = model_context["config"]
    atlas_cfg = config.get("prognostic_ntdc_atlas", {})
    data = model_context["data"]
    roi = model_context["roi"]
    edge = model_context["edge"]
    nt = model_context["nt"]
    lesion_node = model_context["lesion_node"]
    lesion_edge = model_context["lesion_edge"]
    node_damage = model_context["node_damage"]
    edge_damage = model_context["edge_damage"]
    out_dir = model_context["out_dir"]
    covariates = model_context["covariates"]
    outcome = model_context["outcome"]
    fold_edge_jobs = int(model_context["fold_edge_jobs"])
    edge_chunk_size = int(model_context["edge_chunk_size"])
    random_state = int(config.get("impact", {}).get("random_state", 42)) + int(fold)
    ridge_c = float(atlas_cfg.get("ridge_c", 1.0))
    max_iter = int(atlas_cfg.get("max_iter", 5000))
    train = data.iloc[train_idx].copy().reset_index(drop=True)
    test = data.iloc[test_idx].copy().reset_index(drop=True)
    train_ids = train["subject_id"].astype(str).tolist()
    test_ids = test["subject_id"].astype(str).tolist()
    sdc_train, sdc_test, _, _, lesion_node_weights, lesion_edge_weights = fold_sdc(config, data, train_ids, test_ids, lesion_node, lesion_edge)
    y_train = binary_outcome(config, train[outcome])
    cov_train = train[covariates].astype(float).to_numpy()
    cov_test = test[covariates].astype(float).to_numpy()
    lesion_volume_train = train["lesion_volume_ml"].astype(float).to_numpy().reshape(-1, 1) if "lesion_volume_ml" in train.columns else np.zeros((train.shape[0], 1))
    lesion_volume_test = test["lesion_volume_ml"].astype(float).to_numpy().reshape(-1, 1) if "lesion_volume_ml" in test.columns else np.zeros((test.shape[0], 1))
    train_sdc = sdc_train.to_numpy(dtype=float).reshape(-1, 1)
    test_sdc = sdc_test.to_numpy(dtype=float).reshape(-1, 1)
    lesion_node_values = lesion_node.drop(columns=["subject_id"]).to_numpy(dtype=float)
    lesion_edge_values = lesion_edge.drop(columns=["subject_id"]).to_numpy(dtype=float)

    n_roi = len(roi)
    n_nt = len(nt)
    node_beta = np.zeros((n_roi, n_nt), dtype=np.float32)
    node_train_sum = np.zeros(train.shape[0], dtype=np.float32)
    node_test_sum = np.zeros(test.shape[0], dtype=np.float32)
    node_denom = 0.0
    for roi_index in range(n_roi):
        train_x = np.asarray(node_damage[train_idx, roi_index, :], dtype=np.float32)
        test_x = np.asarray(node_damage[test_idx, roi_index, :], dtype=np.float32)
        local_train = lesion_node_values[train_idx, roi_index].reshape(-1, 1)
        local_test = lesion_node_values[test_idx, roi_index].reshape(-1, 1)
        train_base = np.column_stack([lesion_volume_train, local_train, train_sdc])
        test_base = np.column_stack([lesion_volume_test, local_test, test_sdc])
        beta, train_score, test_score, denom = fit_ridge_location(train_x, test_x, train_base, test_base, cov_train, cov_test, y_train, ridge_c, max_iter)
        node_beta[roi_index, :] = beta
        node_train_sum += train_score
        node_test_sum += test_score
        node_denom += denom

    n_edges = len(edge)
    edge_beta = np.zeros((n_edges, n_nt), dtype=np.float32)
    edge_train_sum = np.zeros(train.shape[0], dtype=np.float32)
    edge_test_sum = np.zeros(test.shape[0], dtype=np.float32)
    edge_denom = 0.0
    edge_chunks = [(start, min(start + edge_chunk_size, n_edges)) for start in range(0, n_edges, edge_chunk_size)]
    context = {
        "edge_damage": edge_damage,
        "lesion_edge_values": lesion_edge_values,
        "train_idx": np.asarray(train_idx, dtype=int),
        "test_idx": np.asarray(test_idx, dtype=int),
        "cov_train": cov_train,
        "cov_test": cov_test,
        "y_train": y_train,
        "lesion_volume_train": lesion_volume_train,
        "lesion_volume_test": lesion_volume_test,
        "sdc_train": train_sdc,
        "sdc_test": test_sdc,
        "ridge_c": ridge_c,
        "max_iter": max_iter,
    }
    if fold_edge_jobs <= 1:
        init_edge_context(context)
        iterator = [fit_edge_chunk(chunk) for chunk in edge_chunks]
    else:
        mp_context = mp.get_context("fork")
        with ProcessPoolExecutor(max_workers=fold_edge_jobs, mp_context=mp_context, initializer=init_edge_context, initargs=(context,)) as executor:
            iterator = [future.result() for future in as_completed([executor.submit(fit_edge_chunk, chunk) for chunk in edge_chunks])]
    for result in iterator:
        start, end = int(result["start"]), int(result["end"])
        edge_beta[start:end, :] = result["beta"]
        edge_train_sum += result["train_sum"]
        edge_test_sum += result["test_sum"]
        edge_denom += float(result["denom_sum"])
        print(f"fold {fold}: edge models finished {end}/{n_edges}", flush=True)

    node_train_raw = node_train_sum / node_denom if node_denom > np.finfo(float).eps else np.zeros(train.shape[0], dtype=np.float32)
    node_test_raw = node_test_sum / node_denom if node_denom > np.finfo(float).eps else np.zeros(test.shape[0], dtype=np.float32)
    edge_train_raw = edge_train_sum / edge_denom if edge_denom > np.finfo(float).eps else np.zeros(train.shape[0], dtype=np.float32)
    edge_test_raw = edge_test_sum / edge_denom if edge_denom > np.finfo(float).eps else np.zeros(test.shape[0], dtype=np.float32)
    node_train_z, node_test_z, _, _ = train_z_apply(node_train_raw, node_test_raw)
    edge_train_z, edge_test_z, _, _ = train_z_apply(edge_train_raw, edge_test_raw)

    train_scores = train[["subject_id", "cv_group", outcome, *covariates]].copy()
    test_scores = test[["subject_id", "cv_group", outcome, *covariates]].copy()
    train_scores["fold"] = fold
    test_scores["fold"] = fold
    train_scores["sdc"] = train_sdc.ravel()
    test_scores["sdc"] = test_sdc.ravel()
    train_scores["residual_node_ntdc"] = node_train_raw
    test_scores["residual_node_ntdc"] = node_test_raw
    train_scores["residual_edge_ntdc"] = edge_train_raw
    test_scores["residual_edge_ntdc"] = edge_test_raw
    train_scores["residual_ntdc"] = node_train_z + edge_train_z
    test_scores["residual_ntdc"] = node_test_z + edge_test_z

    fold_dir = ensure_dir(out_dir / "fold_weights")
    node_rows = []
    for roi_index, roi_id in enumerate(roi["roi_id"].astype(int).tolist()):
        denom = float(np.sum(np.abs(node_beta[roi_index, :])))
        for nt_index, nt_id in enumerate(nt["nt_id"].astype(str).tolist()):
            beta = float(node_beta[roi_index, nt_index])
            node_rows.append({"fold": fold, "roi_id": int(roi_id), "nt_id": nt_id, "beta_ridge": beta, "weight_ridge": abs(beta) / denom if denom > np.finfo(float).eps else 0.0})
    edge_rows = []
    edge_names = edge["edge"].astype(str).tolist()
    for edge_index, edge_name in enumerate(edge_names):
        denom = float(np.sum(np.abs(edge_beta[edge_index, :])))
        for nt_index, nt_id in enumerate(nt["nt_id"].astype(str).tolist()):
            beta = float(edge_beta[edge_index, nt_index])
            edge_rows.append({"fold": fold, "edge": edge_name, "nt_id": nt_id, "beta_ridge": beta, "weight_ridge": abs(beta) / denom if denom > np.finfo(float).eps else 0.0})
    write_csv(pd.DataFrame(node_rows), fold_dir / f"fold_{fold:02d}_node_nt_weights.csv")
    write_csv(pd.DataFrame(edge_rows), fold_dir / f"fold_{fold:02d}_edge_nt_weights.csv")
    write_csv(lesion_node_weights, fold_dir / f"fold_{fold:02d}_lesion_node_weights.csv")
    write_csv(lesion_edge_weights, fold_dir / f"fold_{fold:02d}_lesion_edge_weights.csv")

    sensitivity = sensitivity_for_fold(fold, train_idx, train, node_damage, edge_damage, node_beta, edge_beta, lesion_node_values, lesion_edge_values, train_sdc, cov_train, y_train, lesion_volume_train, roi, edge, nt, atlas_cfg, random_state)
    if not sensitivity.empty:
        write_csv(sensitivity, fold_dir / f"fold_{fold:02d}_sensitivity_importance.csv")

    fold_pred, fold_status = predict_models(config, train_scores, test_scores)
    print(f"finished fold {fold}", flush=True)
    return {
        "fold": fold,
        "train_scores": train_scores,
        "test_scores": test_scores,
        "predictions": fold_pred,
        "status": fold_status,
        "summary": {
            "fold": fold,
            "n_train": int(train.shape[0]),
            "n_test": int(test.shape[0]),
            "node_beta_nonzero": int(np.sum(node_beta != 0)),
            "edge_beta_nonzero": int(np.sum(edge_beta != 0)),
            "edge_jobs": int(fold_edge_jobs),
        },
    }


def sensitivity_for_fold(
    fold: int,
    train_idx: list[int],
    train: pd.DataFrame,
    node_damage,
    edge_damage,
    node_beta: np.ndarray,
    edge_beta: np.ndarray,
    lesion_node_values: np.ndarray,
    lesion_edge_values: np.ndarray,
    train_sdc: np.ndarray,
    cov_train: np.ndarray,
    y_train: np.ndarray,
    lesion_volume_train: np.ndarray,
    roi: pd.DataFrame,
    edge: pd.DataFrame,
    nt: pd.DataFrame,
    atlas_cfg: dict,
    random_state: int,
) -> pd.DataFrame:
    """Run sensitivity models on top ridge locations."""
    top_nodes = int(atlas_cfg.get("sensitivity_top_nodes", 156))
    top_edges = int(atlas_cfg.get("sensitivity_top_edges", 200))
    max_iter = int(atlas_cfg.get("max_iter", 5000))
    node_rank = np.argsort(-np.sum(np.abs(node_beta), axis=1))[:top_nodes]
    edge_rank = np.argsort(-np.sum(np.abs(edge_beta), axis=1))[:top_edges]
    rows = []
    nt_ids = nt["nt_id"].astype(str).tolist()
    for roi_index in node_rank:
        train_x = np.asarray(node_damage[train_idx, roi_index, :], dtype=np.float32)
        local_train = lesion_node_values[train_idx, roi_index].reshape(-1, 1)
        train_base = np.column_stack([lesion_volume_train, local_train, train_sdc])
        residual, _ = residualize_train_test(train_x, train_x, train_base, train_base)
        models = fit_sensitivity_models(residual, cov_train, y_train, random_state, max_iter)
        for model, values in models.items():
            for nt_index, nt_id in enumerate(nt_ids):
                rows.append({"fold": fold, "level": "node", "feature": int(roi.iloc[roi_index]["roi_id"]), "nt_id": nt_id, "model": model, "importance": float(values[nt_index])})
    for edge_index in edge_rank:
        train_x = np.asarray(edge_damage[train_idx, edge_index, :], dtype=np.float32)
        local_train = lesion_edge_values[train_idx, edge_index].reshape(-1, 1)
        train_base = np.column_stack([lesion_volume_train, local_train, train_sdc])
        residual, _ = residualize_train_test(train_x, train_x, train_base, train_base)
        models = fit_sensitivity_models(residual, cov_train, y_train, random_state, max_iter)
        edge_name = str(edge.iloc[edge_index]["edge"])
        for model, values in models.items():
            for nt_index, nt_id in enumerate(nt_ids):
                rows.append({"fold": fold, "level": "edge", "feature": edge_name, "nt_id": nt_id, "model": model, "importance": float(values[nt_index])})
    return pd.DataFrame(rows)


def model_specs(covariates: list[str]) -> list[tuple[str, list[str]]]:
    """Return prediction models."""
    return [
        ("Clinical", covariates),
        ("Clinical + residual NTDC", covariates + ["residual_ntdc"]),
        ("Clinical + SDC", covariates + ["sdc"]),
        ("Clinical + SDC + residual NTDC", covariates + ["sdc", "residual_ntdc"]),
    ]


def predict_models(config: dict, train: pd.DataFrame, test: pd.DataFrame) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Fit ordered prediction models."""
    outcome = outcome_column(config)
    covariates = analysis_covariates(config, "model_covariates")
    labels = [int(label) for label in sorted(pd.concat([train[outcome], test[outcome]]).dropna().unique())]
    threshold = float(config.get("analysis", {}).get("binary_outcome", {}).get("threshold", 2))
    rows = []
    status_rows = []
    fold = int(test["fold"].iloc[0])
    for model_name, predictors in model_specs(covariates):
        fit, status, error = fit_ordered_model(train, outcome, predictors)
        status_rows.append({"fold": fold, "model": model_name, "n_train": int(train.shape[0]), "n_test": int(test.shape[0]), "status": status, "error": error})
        if fit is None:
            continue
        probabilities = fit.model.predict(fit.params, exog=test[predictors].astype(float))
        aligned = align_probabilities(fit, np.asarray(probabilities, dtype=float), labels)
        expected = aligned @ np.asarray(labels, dtype=float)
        binary_prob = aligned[:, [index for index, label in enumerate(labels) if label <= threshold]].sum(axis=1)
        for row_index, subject_id in enumerate(test["subject_id"].astype(str).tolist()):
            item = {
                "subject_id": subject_id,
                "model": model_name,
                "fold": fold,
                "observed_mrs": int(test.iloc[row_index][outcome]),
                "expected_mrs": float(expected[row_index]),
                "prob_mrs_le_threshold": float(binary_prob[row_index]),
            }
            for label_index, label in enumerate(labels):
                item[f"prob_{label}"] = float(aligned[row_index, label_index])
            rows.append(item)
    return rows, status_rows


def summarize_predictions(config: dict, predictions: pd.DataFrame, out_dir: Path) -> None:
    """Write prediction summaries."""
    if predictions.empty or "observed_mrs" not in predictions.columns:
        write_csv(pd.DataFrame(), out_dir / "prediction_performance.csv")
        write_csv(pd.DataFrame(), out_dir / "pairwise_bootstrap.csv")
        return
    labels = [int(label) for label in sorted(predictions["observed_mrs"].dropna().unique())]
    threshold = float(config.get("analysis", {}).get("binary_outcome", {}).get("threshold", 2))
    performance_rows = []
    for model in predictions["model"].drop_duplicates().tolist():
        model_pred = predictions[predictions["model"] == model].copy()
        performance_rows.append({"model": model, "n": int(model_pred.shape[0]), **prediction_metrics(model_pred, labels, threshold)})
    performance = pd.DataFrame(performance_rows)
    write_csv(performance, out_dir / "prediction_performance.csv")
    n_bootstrap = int(config.get("prognostic_ntdc_atlas", {}).get("prediction_bootstrap", config.get("impact", {}).get("prediction_bootstrap", 1000)))
    random_state = int(config.get("impact", {}).get("random_state", 42))
    pair_rows = []
    models = predictions["model"].drop_duplicates().tolist()
    directions = metric_directions()
    for index, model_a in enumerate(models):
        pred_a = predictions[predictions["model"] == model_a].copy()
        for model_b in models[index + 1 :]:
            pred_b = predictions[predictions["model"] == model_b].copy()
            for metric, direction in directions.items():
                delta, ci_low, ci_high, p_value = bootstrap_metric_delta(pred_a, pred_b, labels, threshold, metric, n_bootstrap, random_state + index + len(pair_rows))
                pair_rows.append({"model_a": model_a, "model_b": model_b, "metric": metric, "higher_is_better": direction == "higher", "delta_b_minus_a": delta, "ci_low": ci_low, "ci_high": ci_high, "p_bootstrap": p_value, "n_bootstrap": n_bootstrap})
    pairwise = pd.DataFrame(pair_rows)
    pairwise["p_fdr_bh"] = np.nan
    if not pairwise.empty:
        for metric in pairwise["metric"].dropna().unique():
            mask = (pairwise["metric"] == metric) & pairwise["p_bootstrap"].notna()
            if mask.any():
                pairwise.loc[mask, "p_fdr_bh"] = multipletests(pairwise.loc[mask, "p_bootstrap"], method="fdr_bh")[1]
    write_csv(pairwise, out_dir / "pairwise_bootstrap.csv")


def write_agreement(out_dir: Path) -> None:
    """Summarize sensitivity model agreement."""
    paths = sorted((out_dir / "fold_weights").glob("fold_*_sensitivity_importance.csv"))
    if not paths:
        write_csv(pd.DataFrame(), out_dir / "model_agreement_summary.csv")
        return
    sensitivity = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
    write_csv(sensitivity, out_dir / "sensitivity_model_importance.csv")
    summary = (
        sensitivity.assign(abs_importance=lambda df: df["importance"].abs())
        .groupby(["level", "feature", "nt_id"], as_index=False)
        .agg(
            sensitivity_mean_abs_importance=("abs_importance", "mean"),
            sensitivity_model_count=("model", "nunique"),
            sensitivity_fold_count=("fold", "nunique"),
        )
    )
    write_csv(summary, out_dir / "model_agreement_summary.csv")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run ridge prognostic NTDC atlas models.")
    parser.add_argument("--config", default="config/dat_config.yaml")
    parser.add_argument("--jobs", type=int, default=None)
    parser.add_argument("--edge-jobs", type=int, default=None)
    parser.add_argument("--edge-chunk-size", type=int, default=128)
    parser.add_argument("--max-subjects", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    """Run model learning."""
    args = parse_args()
    config = load_config(args.config)
    atlas_cfg = config.get("prognostic_ntdc_atlas", {})
    loaded = load_inputs(config, args.max_subjects)
    out_dir = loaded["out_dir"]
    data = loaded["subject"].copy()
    outcome = outcome_column(config)
    covariates = analysis_covariates(config, "model_covariates")
    data[outcome] = data[outcome].astype(int)
    n_splits = min(int(atlas_cfg.get("n_splits", config.get("impact", {}).get("n_splits", 10))), data["cv_group"].nunique())
    if n_splits < 2:
        raise RuntimeError("at least two CV groups are required")
    jobs = int(args.jobs if args.jobs is not None else atlas_cfg.get("fold_jobs", config.get("resources", {}).get("fold_jobs", 1)))
    edge_jobs = int(args.edge_jobs if args.edge_jobs is not None else atlas_cfg.get("edge_model_jobs", config.get("resources", {}).get("edge_model_jobs", 1)))
    jobs = max(1, min(jobs, n_splits))
    fold_edge_jobs = max(1, edge_jobs // jobs)
    fold_jobs = [
        (fold, train_idx.tolist(), test_idx.tolist())
        for fold, (train_idx, test_idx) in enumerate(GroupKFold(n_splits=n_splits).split(data, groups=data["cv_group"].astype(str)), start=1)
    ]
    model_context.clear()
    model_context.update(
        {
            "config": config,
            "data": data,
            "roi": loaded["roi"],
            "edge": loaded["edge"],
            "nt": loaded["nt"],
            "lesion_node": loaded["lesion_node"],
            "lesion_edge": loaded["lesion_edge"],
            "node_damage": loaded["node_damage"],
            "edge_damage": loaded["edge_damage"],
            "out_dir": out_dir,
            "covariates": covariates,
            "outcome": outcome,
            "fold_edge_jobs": fold_edge_jobs,
            "edge_chunk_size": int(args.edge_chunk_size),
        }
    )
    print(f"running {n_splits} folds with fold_jobs={jobs}, edge_jobs_per_fold={fold_edge_jobs}", flush=True)
    if jobs == 1:
        results = [run_fold(job) for job in fold_jobs]
    else:
        mp_context = mp.get_context("fork")
        with ProcessPoolExecutor(max_workers=jobs, mp_context=mp_context) as executor:
            results = [future.result() for future in as_completed([executor.submit(run_fold, job) for job in fold_jobs])]
    scores = pd.concat([result["test_scores"] for result in sorted(results, key=lambda item: item["fold"])], ignore_index=True)
    predictions = pd.DataFrame([row for result in results for row in result["predictions"]])
    status = pd.DataFrame([row for result in results for row in result["status"]])
    summary = pd.DataFrame([result["summary"] for result in results])
    write_csv(scores, out_dir / "residual_ntdc_scores.csv")
    write_csv(predictions, out_dir / "prediction_cv.csv")
    write_csv(status, out_dir / "prediction_status.csv")
    write_csv(summary, out_dir / "fold_runtime.csv")
    summarize_predictions(config, predictions, out_dir)
    write_agreement(out_dir)
    print(f"wrote prognostic NTDC models to {out_dir}", flush=True)


if __name__ == "__main__":
    main()
