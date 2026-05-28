#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import nibabel as nib
import pandas as pd
from scipy import stats as scipy_stats
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import GroupKFold
from statsmodels.miscmodels.ordinal_model import OrderedModel
from statsmodels.stats.multitest import multipletests

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nt_analysis.config import analysis_covariates, analysis_table, ensure_dir, outcome_column, project_path, require_columns
from nt_analysis.tables import write_csv


def compute_lesion_node_load(config: dict, phenotype: pd.DataFrame) -> pd.DataFrame:
    """Compute lesion-only node load features."""
    node_dir = ensure_dir(project_path(config, config["outputs"]["node_dir"]))
    output = node_dir / "lesion_node_load.csv"
    atlas_img = nib.load(str(project_path(config, config["atlases"]["outputs"]["atlas4s156_2mm"])))
    atlas = np.rint(atlas_img.get_fdata()).astype(int)
    labels = [int(x) for x in sorted(np.unique(atlas)) if x > 0]
    require_columns(["subject_id", "lesion_path"], list(phenotype.columns), "phenotype")
    path_cache: dict[str, dict[str, float]] = {}
    rows = []
    for row in phenotype[["subject_id", "lesion_path"]].itertuples(index=False):
        lesion_path = str(row.lesion_path)
        if lesion_path not in path_cache:
            lesion = nib.load(lesion_path).get_fdata() != 0
            values = {}
            for roi in labels:
                mask = atlas == roi
                values[f"node_{roi:03d}"] = float(np.sum(lesion & mask) / max(mask.sum(), 1))
            path_cache[lesion_path] = values
        rows.append({"subject_id": row.subject_id, **path_cache[lesion_path]})
    lesion_node = pd.DataFrame(rows)
    write_csv(lesion_node, output)
    return lesion_node


def load_lesion_feature_tables(config: dict, phenotype: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load lesion-only node and edge features."""
    lesion_node = compute_lesion_node_load(config, phenotype)
    edge_path = project_path(config, config["outputs"]["edge_dir"], analysis_table(config, "lqt_edge_disconnection", "lqt_edge_disconnection.csv"))
    lesion_edge = pd.read_csv(edge_path)
    return lesion_node, lesion_edge


def residualize(values: np.ndarray, design: np.ndarray) -> np.ndarray:
    """Residualize columns by a design matrix."""
    q, _ = np.linalg.qr(design)
    return values - q @ (q.T @ values)


def fit_fast_mass_univariate(
    features: pd.DataFrame,
    phenotype: pd.DataFrame,
    outcome: str,
    covariates: list[str],
) -> pd.DataFrame:
    """Fit adjusted OLS for many features using vectorized residualization."""
    feature_cols = [col for col in features.columns if col.startswith("node_") or col.startswith("edge_")]
    require_columns(["subject_id", outcome, *covariates], list(phenotype.columns), "phenotype")
    merged = phenotype[["subject_id", outcome, *covariates]].merge(features, on="subject_id")
    data = merged.dropna(subset=[outcome, *covariates]).copy()
    y = data[outcome].to_numpy(dtype=float)
    x_cov = data[covariates].to_numpy(dtype=float) if covariates else np.empty((data.shape[0], 0))
    feature_matrix = data[feature_cols].fillna(0).to_numpy(dtype=float)

    # 协变量先残差化，再逐特征估计效应
    design = np.column_stack([np.ones(data.shape[0]), x_cov])
    y_res = residualize(y.reshape(-1, 1), design).ravel()
    x_res = residualize(feature_matrix, design)
    denominator = np.sum(x_res * x_res, axis=0)
    numerator = np.sum(x_res * y_res[:, None], axis=0)
    beta = np.full(len(feature_cols), np.nan, dtype=float)
    valid = denominator > np.finfo(float).eps
    beta[valid] = numerator[valid] / denominator[valid]

    resid = y_res[:, None] - x_res * np.nan_to_num(beta, nan=0.0)[None, :]
    df = max(data.shape[0] - len(covariates) - 2, 1)
    sigma2 = np.sum(resid * resid, axis=0) / df
    se = np.sqrt(sigma2 / np.where(valid, denominator, np.nan))
    t_values = beta / se
    p_values = 2.0 * scipy_stats.t.sf(np.abs(t_values), df=df)
    p_values[~np.isfinite(p_values)] = np.nan

    out = pd.DataFrame(
        {
            "feature": feature_cols,
            "beta": beta,
            "t": t_values,
            "p": p_values,
            "n": int(data.shape[0]),
        }
    )
    ok = out["p"].notna()
    out["q"] = np.nan
    if ok.any():
        out.loc[ok, "q"] = multipletests(out.loc[ok, "p"], method="fdr_bh")[1]
    return out


def select_weights(stats: pd.DataFrame, stat_name: str, top_k: int, q_threshold: float, use_q: bool) -> pd.DataFrame:
    """Select key features and return signed weights."""
    values = stats.copy()
    values["weight"] = pd.to_numeric(values[stat_name], errors="coerce")
    values = values[np.isfinite(values["weight"])].copy()
    values["selected"] = False
    if values.empty:
        return values

    selected = pd.Series(False, index=values.index)
    if use_q and "q" in values.columns:
        selected = values["q"].fillna(1.0) <= q_threshold
    if not selected.any():
        top_k = min(int(top_k), values.shape[0])
        selected.loc[values["weight"].abs().nlargest(top_k).index] = True
    values.loc[selected, "selected"] = True
    values.loc[~values["selected"], "weight"] = 0.0
    return values


def impact_from_weights(features: pd.DataFrame, weights: pd.DataFrame) -> pd.Series:
    """Compute weighted average feature burden."""
    selected = weights.loc[weights["weight"] != 0, ["feature", "weight"]].copy()
    if selected.empty:
        return pd.Series(np.nan, index=features.index)
    feature_cols = selected["feature"].tolist()
    matrix = features[feature_cols].fillna(0).to_numpy(dtype=float)
    weight = selected["weight"].to_numpy(dtype=float)
    denom = np.sum(np.abs(weight))
    if denom <= np.finfo(float).eps:
        return pd.Series(np.nan, index=features.index)
    return pd.Series(matrix @ weight / denom, index=features.index)


def zscore(values: pd.Series) -> pd.Series:
    """Return a stable z score."""
    sd = values.std(skipna=True)
    if not np.isfinite(sd) or sd <= np.finfo(float).eps:
        return values * np.nan
    return (values - values.mean(skipna=True)) / sd


def add_cv_group(config: dict, data: pd.DataFrame) -> pd.DataFrame:
    """Add the cross-validation group column."""
    out = data.copy()
    group_col = str(config.get("analysis", {}).get("cv_group_column", "subject_id"))
    if group_col not in out.columns:
        group_col = "subject_id"
    # 默认每个真实被试独立分组
    out["cv_group"] = out[group_col].astype(str)
    return out


def standard_score_name(config: dict, key: str, default: str) -> str:
    """Return configured model score column names."""
    return str(config.get("analysis", {}).get(key, default))


def standard_score_label(config: dict, key: str, default: str) -> str:
    """Return configured model display labels."""
    return str(config.get("analysis", {}).get(key, default))


def run_cross_validated_impact(
    config: dict,
    phenotype: pd.DataFrame,
    node: pd.DataFrame,
    edge: pd.DataFrame,
    lesion_node: pd.DataFrame,
    lesion_edge: pd.DataFrame,
    out_dir: Path,
    dataset_name: str,
) -> pd.DataFrame:
    """Run 10-fold discovery-validation impact scoring."""
    impact_cfg = config.get("impact", {})
    outcome = outcome_column(config)
    covariates = analysis_covariates(config, "model_covariates")
    stat_name = str(impact_cfg.get("weight_stat", "t"))
    n_splits = int(impact_cfg.get("n_splits", 10))
    q_threshold = float(impact_cfg.get("q_threshold", 0.05))
    node_top_k = int(impact_cfg.get("node_top_k", 20))
    edge_top_k = int(impact_cfg.get("edge_top_k", 200))
    use_q = bool(impact_cfg.get("use_q_if_available", True))

    require_columns(["subject_id", outcome, *covariates], list(phenotype.columns), "phenotype")
    group_source = add_cv_group(config, phenotype)
    merged_ids = group_source[["subject_id", "cv_group", outcome, *covariates]].dropna(subset=[outcome, *covariates])
    groups = merged_ids["cv_group"].astype(str).to_numpy()
    unique_groups = np.unique(groups)
    n_splits = min(n_splits, len(unique_groups))
    if n_splits < 2:
        raise RuntimeError("at least two subject groups are required for cross-validation")

    scores = merged_ids[["subject_id", "cv_group", outcome, *covariates]].copy()
    scores["fold"] = np.nan
    scores["lesion_node_impact"] = np.nan
    scores["lesion_edge_impact"] = np.nan
    scores["nt_node_impact"] = np.nan
    scores["nt_edge_impact"] = np.nan
    fold_rows = []
    gkf = GroupKFold(n_splits=n_splits)

    for fold_id, (train_idx, test_idx) in enumerate(gkf.split(merged_ids, groups=groups), start=1):
        train_ids = merged_ids.iloc[train_idx]["subject_id"].tolist()
        test_ids = merged_ids.iloc[test_idx]["subject_id"].tolist()
        train_pheno = phenotype[phenotype["subject_id"].isin(train_ids)]
        train_node = node[node["subject_id"].isin(train_ids)]
        train_edge = edge[edge["subject_id"].isin(train_ids)]
        train_lesion_node = lesion_node[lesion_node["subject_id"].isin(train_ids)]
        train_lesion_edge = lesion_edge[lesion_edge["subject_id"].isin(train_ids)]
        test_node = node[node["subject_id"].isin(test_ids)].reset_index(drop=True)
        test_edge = edge[edge["subject_id"].isin(test_ids)].reset_index(drop=True)
        test_lesion_node = lesion_node[lesion_node["subject_id"].isin(test_ids)].reset_index(drop=True)
        test_lesion_edge = lesion_edge[lesion_edge["subject_id"].isin(test_ids)].reset_index(drop=True)

        lesion_node_stats = fit_fast_mass_univariate(train_lesion_node, train_pheno, outcome, covariates)
        lesion_edge_stats = fit_fast_mass_univariate(train_lesion_edge, train_pheno, outcome, covariates)
        node_stats = fit_fast_mass_univariate(train_node, train_pheno, outcome, covariates)
        edge_stats = fit_fast_mass_univariate(train_edge, train_pheno, outcome, covariates)
        lesion_node_weights = select_weights(lesion_node_stats, stat_name, node_top_k, q_threshold, use_q)
        lesion_edge_weights = select_weights(lesion_edge_stats, stat_name, edge_top_k, q_threshold, use_q)
        node_weights = select_weights(node_stats, stat_name, node_top_k, q_threshold, use_q)
        edge_weights = select_weights(edge_stats, stat_name, edge_top_k, q_threshold, use_q)

        lesion_node_impact = impact_from_weights(test_lesion_node, lesion_node_weights)
        lesion_edge_impact = impact_from_weights(test_lesion_edge, lesion_edge_weights)
        node_impact = impact_from_weights(test_node, node_weights)
        edge_impact = impact_from_weights(test_edge, edge_weights)
        test_frame = pd.DataFrame(
            {
                "subject_id": test_node["subject_id"],
                "fold": fold_id,
                "lesion_node_impact": lesion_node_impact.to_numpy(),
                "lesion_edge_impact": lesion_edge_impact.to_numpy(),
                "nt_node_impact": node_impact.to_numpy(),
                "nt_edge_impact": edge_impact.to_numpy(),
            }
        )
        scores = scores.merge(test_frame, on="subject_id", how="left", suffixes=("", "_new"))
        for col in ["fold", "lesion_node_impact", "lesion_edge_impact", "nt_node_impact", "nt_edge_impact"]:
            scores[col] = scores[f"{col}_new"].combine_first(scores[col])
            scores = scores.drop(columns=[f"{col}_new"])

        fold_rows.append(
            {
                "dataset": dataset_name,
                "fold": fold_id,
                "n_train": len(train_ids),
                "n_test": len(test_ids),
                "n_train_groups": len(set(merged_ids.iloc[train_idx]["cv_group"])),
                "n_test_groups": len(set(merged_ids.iloc[test_idx]["cv_group"])),
                "selected_lesion_nodes": int((lesion_node_weights["weight"] != 0).sum()),
                "selected_lesion_edges": int((lesion_edge_weights["weight"] != 0).sum()),
                "selected_nt_nodes": int((node_weights["weight"] != 0).sum()),
                "selected_nt_edges": int((edge_weights["weight"] != 0).sum()),
                "selected_nodes": int((node_weights["weight"] != 0).sum()),
                "selected_edges": int((edge_weights["weight"] != 0).sum()),
            }
        )

    scores["sdc"] = zscore(scores["lesion_node_impact"]) + zscore(scores["lesion_edge_impact"])
    scores["ntdc"] = zscore(scores["nt_node_impact"]) + zscore(scores["nt_edge_impact"])
    scores["dataset"] = dataset_name
    write_csv(scores, out_dir / "nt_impact_scores.csv")
    write_csv(scores[["subject_id", "fold", "lesion_node_impact", "lesion_edge_impact", "sdc", "dataset"]], out_dir / "lesion_impact_scores.csv")
    write_csv(pd.DataFrame(fold_rows), out_dir / "nt_impact_fold_summary.csv")
    return scores


def write_full_sample_keys(
    config: dict,
    phenotype: pd.DataFrame,
    node: pd.DataFrame,
    edge: pd.DataFrame,
    lesion_node: pd.DataFrame,
    lesion_edge: pd.DataFrame,
    out_dir: Path,
    dataset_name: str,
) -> None:
    """Write descriptive full-sample key node and edge tables."""
    impact_cfg = config.get("impact", {})
    outcome = outcome_column(config)
    covariates = analysis_covariates(config, "model_covariates")
    stat_name = str(impact_cfg.get("weight_stat", "t"))
    q_threshold = float(impact_cfg.get("q_threshold", 0.05))
    use_q = bool(impact_cfg.get("use_q_if_available", True))
    lesion_node_stats = fit_fast_mass_univariate(lesion_node, phenotype, outcome, covariates)
    lesion_edge_stats = fit_fast_mass_univariate(lesion_edge, phenotype, outcome, covariates)
    node_stats = fit_fast_mass_univariate(node, phenotype, outcome, covariates)
    edge_stats = fit_fast_mass_univariate(edge, phenotype, outcome, covariates)
    lesion_node_weights = select_weights(lesion_node_stats, stat_name, int(impact_cfg.get("node_top_k", 20)), q_threshold, use_q)
    lesion_edge_weights = select_weights(lesion_edge_stats, stat_name, int(impact_cfg.get("edge_top_k", 200)), q_threshold, use_q)
    node_weights = select_weights(node_stats, stat_name, int(impact_cfg.get("node_top_k", 20)), q_threshold, use_q)
    edge_weights = select_weights(edge_stats, stat_name, int(impact_cfg.get("edge_top_k", 200)), q_threshold, use_q)
    lesion_node_weights["dataset"] = dataset_name
    lesion_edge_weights["dataset"] = dataset_name
    node_weights["dataset"] = dataset_name
    edge_weights["dataset"] = dataset_name
    write_csv(lesion_node_weights.sort_values("weight", key=lambda x: x.abs(), ascending=False), out_dir / "key_lesion_nodes.csv")
    write_csv(lesion_edge_weights.sort_values("weight", key=lambda x: x.abs(), ascending=False), out_dir / "key_lesion_edges.csv")
    write_csv(node_weights.sort_values("weight", key=lambda x: x.abs(), ascending=False), out_dir / "key_nt_nodes.csv")
    write_csv(edge_weights.sort_values("weight", key=lambda x: x.abs(), ascending=False), out_dir / "key_nt_edges.csv")


def fit_ordered_model(data: pd.DataFrame, outcome: str, predictors: list[str]) -> tuple[object | None, str, str]:
    """Fit one ordered logistic model."""
    try:
        model = OrderedModel(data[outcome].astype(int), data[predictors].astype(float), distr="logit")
        fit = model.fit(method="bfgs", disp=False)
        return fit, "ok", ""
    except Exception as error:  # noqa: BLE001
        return None, "failed", str(error)


def model_specs(covariates: list[str], config: dict | None = None) -> list[tuple[str, list[str]]]:
    """Return the ordered model set."""
    config = config or {}
    sdc_col = standard_score_name(config, "sdc_predictor", "sdc")
    ntdc_col = standard_score_name(config, "ntdc_predictor", "ntdc")
    ntdc_label = standard_score_label(config, "ntdc_label", "NTDC")
    return [
        ("Clinical", covariates),
        ("Clinical + SDC", covariates + [sdc_col]),
        (f"Clinical + {ntdc_label}", covariates + [ntdc_col]),
        (f"Clinical + SDC + {ntdc_label}", covariates + [sdc_col, ntdc_col]),
    ]


def align_probabilities(fit: object, probabilities: np.ndarray, labels: list[int]) -> np.ndarray:
    """Align model probabilities to global outcome labels."""
    aligned = np.zeros((probabilities.shape[0], len(labels)), dtype=float)
    fit_labels = [int(label) for label in getattr(fit.model, "labels", range(probabilities.shape[1]))]
    label_index = {label: index for index, label in enumerate(labels)}
    for source_index, label in enumerate(fit_labels):
        if label in label_index:
            aligned[:, label_index[label]] = probabilities[:, source_index]
    row_sum = aligned.sum(axis=1)
    missing = row_sum <= np.finfo(float).eps
    if np.any(missing):
        aligned[missing, :] = 1.0 / len(labels)
        row_sum = aligned.sum(axis=1)
    return aligned / row_sum[:, None]


def ordinal_c_index(observed: np.ndarray, expected: np.ndarray) -> float:
    """Compute a simple concordance index for ordered outcomes."""
    observed = np.asarray(observed, dtype=int)
    expected = np.asarray(expected, dtype=float)
    valid = np.isfinite(observed) & np.isfinite(expected)
    observed = observed[valid]
    expected = expected[valid]
    if observed.size < 2:
        return np.nan
    levels = sorted(np.unique(observed))
    level_index = {level: index for index, level in enumerate(levels)}
    order = np.argsort(expected, kind="mergesort")
    y = observed[order]
    score = expected[order]
    previous = np.zeros(len(levels), dtype=float)
    concordant = 0.0
    comparable = 0.0
    start = 0
    while start < len(y):
        stop = start + 1
        while stop < len(y) and score[stop] == score[start]:
            stop += 1
        group = y[start:stop]
        group_counts = np.zeros(len(levels), dtype=float)
        for value in group:
            idx = level_index[int(value)]
            # 更高预测分对应更差预后
            concordant += previous[:idx].sum()
            comparable += previous.sum() - previous[idx]
            group_counts[idx] += 1
        group_comparable = 0.0
        for left in range(len(levels)):
            for right in range(left + 1, len(levels)):
                group_comparable += group_counts[left] * group_counts[right]
        concordant += 0.5 * group_comparable
        comparable += group_comparable
        previous += group_counts
        start = stop
    if comparable <= 0:
        return np.nan
    return float(concordant / comparable)


def prediction_metrics(df: pd.DataFrame, labels: list[int], threshold: float) -> dict[str, float]:
    """Compute ordinal and binary prediction metrics."""
    prob_cols = [f"prob_{label}" for label in labels]
    probabilities = df[prob_cols].to_numpy(dtype=float)
    probabilities = np.clip(probabilities, 1e-15, 1.0)
    probabilities = probabilities / probabilities.sum(axis=1, keepdims=True)
    observed = df["observed_mrs"].to_numpy(dtype=int)
    label_index = {label: index for index, label in enumerate(labels)}
    true_index = np.array([label_index[value] for value in observed], dtype=int)
    expected = probabilities @ np.asarray(labels, dtype=float)

    # 有序结局用完整概率分布评分
    log_loss = float(-np.mean(np.log(probabilities[np.arange(probabilities.shape[0]), true_index])))
    pred_cdf = np.cumsum(probabilities, axis=1)[:, :-1]
    obs_cdf = (observed[:, None] <= np.asarray(labels[:-1])[None, :]).astype(float)
    rps = float(np.mean(np.sum((pred_cdf - obs_cdf) ** 2, axis=1) / max(len(labels) - 1, 1)))
    mae = float(np.mean(np.abs(expected - observed)))
    c_index = ordinal_c_index(observed.astype(float), expected)
    slope = np.nan
    if np.nanstd(expected) > np.finfo(float).eps:
        slope = float(np.polyfit(expected, observed.astype(float), 1)[0])

    binary_observed = (observed <= threshold).astype(int)
    binary_probability = df["prob_mrs_le_threshold"].to_numpy(dtype=float)
    auc = np.nan
    if len(np.unique(binary_observed)) == 2:
        auc = float(roc_auc_score(binary_observed, binary_probability))
    brier = float(brier_score_loss(binary_observed, binary_probability))
    return {
        "ordinal_log_loss": log_loss,
        "ranked_probability_score": rps,
        "expected_mrs_mae": mae,
        "ordinal_c_index": c_index,
        "expected_mrs_calibration_slope": slope,
        "expected_mrs_calibration_error": float(abs(slope - 1.0)) if np.isfinite(slope) else np.nan,
        "binary_auc_mrs_le_threshold": auc,
        "binary_brier_mrs_le_threshold": brier,
    }


def prediction_metric_value(df: pd.DataFrame, labels: list[int], threshold: float, metric: str) -> float:
    """Compute one prediction metric."""
    prob_cols = [f"prob_{label}" for label in labels]
    probabilities = df[prob_cols].to_numpy(dtype=float)
    probabilities = np.clip(probabilities, 1e-15, 1.0)
    probabilities = probabilities / probabilities.sum(axis=1, keepdims=True)
    observed = df["observed_mrs"].to_numpy(dtype=int)
    label_index = {label: index for index, label in enumerate(labels)}
    true_index = np.array([label_index[value] for value in observed], dtype=int)
    expected = probabilities @ np.asarray(labels, dtype=float)
    if metric == "ordinal_log_loss":
        return float(-np.mean(np.log(probabilities[np.arange(probabilities.shape[0]), true_index])))
    if metric == "ranked_probability_score":
        pred_cdf = np.cumsum(probabilities, axis=1)[:, :-1]
        obs_cdf = (observed[:, None] <= np.asarray(labels[:-1])[None, :]).astype(float)
        return float(np.mean(np.sum((pred_cdf - obs_cdf) ** 2, axis=1) / max(len(labels) - 1, 1)))
    if metric == "expected_mrs_mae":
        return float(np.mean(np.abs(expected - observed)))
    if metric == "ordinal_c_index":
        return ordinal_c_index(observed.astype(float), expected)
    if metric == "expected_mrs_calibration_error":
        if np.nanstd(expected) <= np.finfo(float).eps:
            return np.nan
        slope = float(np.polyfit(expected, observed.astype(float), 1)[0])
        return float(abs(slope - 1.0))
    binary_observed = (observed <= threshold).astype(int)
    binary_probability = df["prob_mrs_le_threshold"].to_numpy(dtype=float)
    if metric == "binary_auc_mrs_le_threshold":
        if len(np.unique(binary_observed)) < 2:
            return np.nan
        return float(roc_auc_score(binary_observed, binary_probability))
    if metric == "binary_brier_mrs_le_threshold":
        return float(brier_score_loss(binary_observed, binary_probability))
    raise KeyError(f"unknown prediction metric: {metric}")


def metric_directions() -> dict[str, str]:
    """Return metric optimization directions."""
    return {
        "ordinal_log_loss": "lower",
        "ranked_probability_score": "lower",
        "expected_mrs_mae": "lower",
        "ordinal_c_index": "higher",
        "expected_mrs_calibration_error": "lower",
        "binary_auc_mrs_le_threshold": "higher",
        "binary_brier_mrs_le_threshold": "lower",
    }


def bootstrap_metric_delta(
    data_a: pd.DataFrame,
    data_b: pd.DataFrame,
    labels: list[int],
    threshold: float,
    metric: str,
    n_bootstrap: int,
    random_state: int,
) -> tuple[float, float, float, float]:
    """Bootstrap a paired metric difference."""
    merged = data_a.merge(data_b, on=["subject_id", "observed_mrs"], suffixes=("_a", "_b"))
    boot_values = []
    rng = np.random.default_rng(random_state)
    indices = np.arange(merged.shape[0])
    prob_cols_a = {f"prob_{label}_a": f"prob_{label}" for label in labels}
    prob_cols_b = {f"prob_{label}_b": f"prob_{label}" for label in labels}

    def metric_for_suffix(sample: pd.DataFrame, suffix: str) -> float:
        if suffix == "a":
            renamed = sample.rename(columns={**prob_cols_a, "prob_mrs_le_threshold_a": "prob_mrs_le_threshold"})
        else:
            renamed = sample.rename(columns={**prob_cols_b, "prob_mrs_le_threshold_b": "prob_mrs_le_threshold"})
        return prediction_metric_value(renamed[["observed_mrs", "prob_mrs_le_threshold", *[f"prob_{label}" for label in labels]]], labels, threshold, metric)

    observed_delta = metric_for_suffix(merged, "b") - metric_for_suffix(merged, "a")
    for _ in range(n_bootstrap):
        sample = merged.iloc[rng.choice(indices, size=len(indices), replace=True)]
        delta = metric_for_suffix(sample, "b") - metric_for_suffix(sample, "a")
        if np.isfinite(delta):
            boot_values.append(delta)
    if not boot_values:
        return observed_delta, np.nan, np.nan, np.nan
    boot = np.asarray(boot_values, dtype=float)
    ci_low, ci_high = np.percentile(boot, [2.5, 97.5])
    p_value = 2.0 * min(np.mean(boot <= 0), np.mean(boot >= 0))
    return float(observed_delta), float(ci_low), float(ci_high), float(min(p_value, 1.0))


def run_cross_validated_prediction(config: dict, scores: pd.DataFrame, out_dir: Path) -> None:
    """Run 10-fold out-of-sample prediction and bootstrap model comparisons."""
    outcome = outcome_column(config)
    covariates = analysis_covariates(config, "model_covariates")
    specs = model_specs(covariates, config)
    all_predictors = sorted({term for _, terms in specs for term in terms})
    binary_cfg = config.get("analysis", {}).get("binary_outcome", {})
    threshold = float(binary_cfg.get("threshold", 2))
    impact_cfg = config.get("impact", {})
    n_bootstrap = int(impact_cfg.get("prediction_bootstrap", 1000))
    random_state = int(impact_cfg.get("random_state", 42))

    data = scores.dropna(subset=[outcome, "fold", *all_predictors]).copy()
    data[outcome] = data[outcome].astype(int)
    data["fold"] = data["fold"].astype(int)
    labels = [int(label) for label in sorted(data[outcome].unique())]
    rows = []
    status_rows = []

    for model_name, predictors in specs:
        for fold in sorted(data["fold"].unique()):
            train = data[data["fold"] != fold].copy()
            test = data[data["fold"] == fold].copy()
            fit, status, error = fit_ordered_model(train, outcome, predictors)
            status_rows.append({"model": model_name, "fold": int(fold), "n_train": int(train.shape[0]), "n_test": int(test.shape[0]), "status": status, "error": error})
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
                    "fold": int(fold),
                    "observed_mrs": int(test.iloc[row_index][outcome]),
                    "expected_mrs": float(expected[row_index]),
                    "prob_mrs_le_threshold": float(binary_probability[row_index]),
                }
                for label_index, label in enumerate(labels):
                    item[f"prob_{label}"] = float(aligned[row_index, label_index])
                rows.append(item)

    predictions = pd.DataFrame(rows)
    write_csv(predictions, out_dir / "model_prediction_cv.csv")
    write_csv(pd.DataFrame(status_rows), out_dir / "model_prediction_fold_status.csv")

    performance_rows = []
    for model_name, _ in specs:
        model_pred = predictions[predictions["model"] == model_name].copy()
        metrics = prediction_metrics(model_pred, labels, threshold)
        performance_rows.append({"model": model_name, "n": int(model_pred.shape[0]), **metrics})
    performance = pd.DataFrame(performance_rows)
    write_csv(performance, out_dir / "model_prediction_performance.csv")

    pair_rows = []
    directions = metric_directions()
    for index, (model_a, _) in enumerate(specs):
        pred_a = predictions[predictions["model"] == model_a].copy()
        for model_b, _ in specs[index + 1 :]:
            pred_b = predictions[predictions["model"] == model_b].copy()
            for metric, direction in directions.items():
                delta, ci_low, ci_high, p_value = bootstrap_metric_delta(
                    pred_a,
                    pred_b,
                    labels,
                    threshold,
                    metric,
                    n_bootstrap,
                    random_state + index + len(pair_rows),
                )
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
    write_csv(pairwise, out_dir / "model_prediction_pairwise_bootstrap.csv")


def write_pairwise_model_comparison(
    specs: list[tuple[str, list[str]]],
    fits: dict[str, object | None],
    n_rows: int,
    model_dir: Path,
) -> None:
    """Write pairwise model comparisons."""
    rows = []
    predictor_map = {name: set(predictors) for name, predictors in specs}
    loglike_obs: dict[str, np.ndarray] = {}
    for model_name, fit in fits.items():
        if fit is None:
            continue
        try:
            loglike_obs[model_name] = np.asarray(fit.model.loglikeobs(fit.params), dtype=float)
        except Exception:  # noqa: BLE001
            continue

    for index, (model_a, predictors_a) in enumerate(specs):
        for model_b, predictors_b in specs[index + 1 :]:
            fit_a = fits.get(model_a)
            fit_b = fits.get(model_b)
            pred_a = predictor_map[model_a]
            pred_b = predictor_map[model_b]
            row = {
                "model_a": model_a,
                "model_b": model_b,
                "n": n_rows,
                "predictors_a": "|".join(predictors_a),
                "predictors_b": "|".join(predictors_b),
                "status": "ok" if fit_a is not None and fit_b is not None else "failed",
            }
            if fit_a is not None and fit_b is not None:
                row.update(
                    {
                        "aic_a": float(fit_a.aic),
                        "aic_b": float(fit_b.aic),
                        "delta_aic_b_minus_a": float(fit_b.aic - fit_a.aic),
                        "bic_a": float(fit_a.bic),
                        "bic_b": float(fit_b.bic),
                        "delta_bic_b_minus_a": float(fit_b.bic - fit_a.bic),
                        "llf_a": float(fit_a.llf),
                        "llf_b": float(fit_b.llf),
                        "delta_llf_b_minus_a": float(fit_b.llf - fit_a.llf),
                    }
                )
                if pred_a < pred_b:
                    reduced_name, full_name = model_a, model_b
                    reduced_fit, full_fit = fit_a, fit_b
                    relation = "model_a_nested_in_model_b"
                    lr_df = len(pred_b - pred_a)
                elif pred_b < pred_a:
                    reduced_name, full_name = model_b, model_a
                    reduced_fit, full_fit = fit_b, fit_a
                    relation = "model_b_nested_in_model_a"
                    lr_df = len(pred_a - pred_b)
                else:
                    reduced_name, full_name = "", ""
                    reduced_fit, full_fit = None, None
                    relation = "not_nested"
                    lr_df = 0

                row["relation"] = relation
                row["lr_reduced_model"] = reduced_name
                row["lr_full_model"] = full_name
                if reduced_fit is not None and full_fit is not None and lr_df > 0:
                    lr_stat = max(2.0 * (full_fit.llf - reduced_fit.llf), 0.0)
                    row["lr_stat"] = float(lr_stat)
                    row["lr_df"] = int(lr_df)
                    row["p_uncorrected"] = float(scipy_stats.chi2.sf(lr_stat, lr_df))
                    row["test_type"] = "nested_likelihood_ratio"
                else:
                    row["lr_stat"] = np.nan
                    row["lr_df"] = np.nan
                    row["p_uncorrected"] = np.nan
                    row["test_type"] = "descriptive_non_nested"

                if model_a in loglike_obs and model_b in loglike_obs:
                    diff = loglike_obs[model_b] - loglike_obs[model_a]
                    row["paired_mean_loglik_diff_b_minus_a"] = float(np.mean(diff))
                    row["paired_median_loglik_diff_b_minus_a"] = float(np.median(diff))
                    if np.allclose(diff, 0):
                        row["paired_p_uncorrected"] = 1.0
                    else:
                        row["paired_p_uncorrected"] = float(scipy_stats.wilcoxon(diff, alternative="two-sided").pvalue)
                    row["paired_test_type"] = "paired_loglik_wilcoxon"
            rows.append(row)

    pairwise = pd.DataFrame(rows)
    valid = pairwise["p_uncorrected"].notna()
    pairwise["p_fdr_bh"] = np.nan
    pairwise["p_bonferroni"] = np.nan
    if valid.any():
        pairwise.loc[valid, "p_fdr_bh"] = multipletests(pairwise.loc[valid, "p_uncorrected"], method="fdr_bh")[1]
        pairwise.loc[valid, "p_bonferroni"] = multipletests(pairwise.loc[valid, "p_uncorrected"], method="bonferroni")[1]
    paired_valid = pairwise["paired_p_uncorrected"].notna() if "paired_p_uncorrected" in pairwise.columns else pd.Series(False, index=pairwise.index)
    pairwise["paired_p_fdr_bh"] = np.nan
    pairwise["paired_p_bonferroni"] = np.nan
    if paired_valid.any():
        pairwise.loc[paired_valid, "paired_p_fdr_bh"] = multipletests(pairwise.loc[paired_valid, "paired_p_uncorrected"], method="fdr_bh")[1]
        pairwise.loc[paired_valid, "paired_p_bonferroni"] = multipletests(pairwise.loc[paired_valid, "paired_p_uncorrected"], method="bonferroni")[1]
    write_csv(pairwise, model_dir / "model_pairwise_comparison.csv")


def fit_ordinal_impact_model(config: dict, scores: pd.DataFrame, impact_dir: Path) -> None:
    """Fit ordered mRS model comparisons with out-of-fold impact scores."""
    outcome = outcome_column(config)
    covariates = analysis_covariates(config, "model_covariates")
    model_dir = ensure_dir(project_path(config, config["outputs"]["model_dir"]))
    specs = model_specs(covariates, config)
    all_predictors = sorted({term for _, terms in specs for term in terms})
    data = scores.dropna(subset=[outcome, *all_predictors]).copy()
    data[outcome] = data[outcome].astype(int)

    fits = {}
    rows = []
    term_rows = []
    for model_name, predictors in specs:
        fit, status, error = fit_ordered_model(data, outcome, predictors)
        fits[model_name] = fit
        row = {
            "model": model_name,
            "n": int(data.shape[0]),
            "n_predictors": len(predictors),
            "predictors": "|".join(predictors),
            "status": status,
            "error": error,
        }
        if fit is not None:
            row.update(
                {
                    "aic": float(fit.aic),
                    "bic": float(fit.bic),
                    "llf": float(fit.llf),
                    "df_model": float(fit.df_model),
                }
            )
            for term, value in fit.params.items():
                term_rows.append(
                    {
                        "model": model_name,
                        "term": term,
                        "coef": float(value),
                        "se": float(fit.bse.get(term, np.nan)),
                        "z": float(fit.tvalues.get(term, np.nan)),
                        "p": float(fit.pvalues.get(term, np.nan)),
                        "is_predictor": term in predictors,
                    }
                )
        rows.append(row)

    comparison = pd.DataFrame(rows)
    base = fits.get("Clinical")
    if base is not None:
        for index, row in comparison.iterrows():
            fit = fits.get(row["model"])
            if fit is None:
                continue
            comparison.loc[index, "delta_aic_vs_clinical"] = float(fit.aic - base.aic)
            comparison.loc[index, "delta_bic_vs_clinical"] = float(fit.bic - base.bic)
            comparison.loc[index, "delta_llf_vs_clinical"] = float(fit.llf - base.llf)
            if row["model"] == "Clinical":
                comparison.loc[index, "lr_stat_vs_clinical"] = np.nan
                comparison.loc[index, "lr_df_vs_clinical"] = np.nan
                comparison.loc[index, "lr_p_vs_clinical"] = np.nan
            else:
                lr_stat = max(2.0 * (fit.llf - base.llf), 0.0)
                lr_df = max(int(round(fit.df_model - base.df_model)), 1)
                comparison.loc[index, "lr_stat_vs_clinical"] = float(lr_stat)
                comparison.loc[index, "lr_df_vs_clinical"] = int(lr_df)
                comparison.loc[index, "lr_p_vs_clinical"] = float(scipy_stats.chi2.sf(lr_stat, lr_df))

    write_csv(comparison, model_dir / "model_comparison_ordinal.csv")
    write_csv(pd.DataFrame(term_rows), model_dir / "model_comparison_terms.csv")
    write_pairwise_model_comparison(specs, fits, int(data.shape[0]), model_dir)

    ntdc_label = standard_score_label(config, "ntdc_label", "NTDC")
    nt_model = comparison[comparison["model"] == f"Clinical + SDC + {ntdc_label}"].copy()
    if base is not None and not nt_model.empty:
        nt_model = nt_model.rename(
            columns={
                "aic": "full_aic",
                "llf": "full_llf",
                "lr_stat_vs_clinical": "lr_stat",
                "lr_df_vs_clinical": "lr_df",
                "lr_p_vs_clinical": "lr_p",
            }
        )
        nt_model["base_aic"] = float(base.aic)
        nt_model["base_llf"] = float(base.llf)
        nt_model["model"] = f"ordinal_mrs_{ntdc_label.lower().replace('-', '_')}"
        write_csv(nt_model[["model", "n", "base_aic", "full_aic", "base_llf", "full_llf", "lr_stat", "lr_df", "lr_p", "status"]], impact_dir / "nt_impact_model_performance.csv")
    terms = pd.DataFrame(term_rows)
    write_csv(terms[terms["model"] == f"Clinical + SDC + {ntdc_label}"], impact_dir / "nt_impact_ordinal_model.csv")
