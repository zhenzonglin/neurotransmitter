#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from compute_impact_scores import (  # noqa: E402
    align_probabilities,
    bootstrap_metric_delta,
    compute_lesion_node_load,
    fit_ordered_model,
    impact_from_weights,
    load_lesion_feature_tables,
    metric_directions,
    prediction_metrics,
)
from nt_analysis.config import analysis_covariates, ensure_dir, load_config, outcome_column, project_path  # noqa: E402
from nt_analysis.tables import write_csv  # noqa: E402
from run_ml_profile_analysis import (  # noqa: E402
    align_by_subject,
    atlas_info,
    compute_profile_edge_damage,
    compute_profile_node_damage,
    load_edge_matrix,
    load_lesion_indices,
    train_z_apply,
)


def read_scores(config: dict, profile_subdir: str, max_subjects: int | None) -> pd.DataFrame:
    """Load the subject list used by the completed NTDC run."""
    score_path = project_path(config, "derivatives", profile_subdir, "profile_scores.csv")
    if not score_path.exists():
        raise FileNotFoundError(f"missing profile scores: {score_path}")
    scores = pd.read_csv(score_path, dtype={"subject_id": str})
    if max_subjects is not None:
        scores = scores.head(int(max_subjects)).copy()

    manifest = pd.read_csv(project_path(config, config["outputs"]["qc_dir"], "subject_manifest.csv"), dtype={"subject_id": str})
    keep_cols = ["subject_id", "lesion_path"]
    merged = scores.merge(manifest[keep_cols], on="subject_id", how="left")
    if merged["lesion_path"].isna().any():
        missing = merged.loc[merged["lesion_path"].isna(), "subject_id"].head(10).tolist()
        raise RuntimeError(f"subjects missing lesion_path: {missing}")
    return merged


def zscore_pair(train: pd.Series, test: pd.Series, name: str) -> tuple[pd.Series, pd.Series]:
    """Apply train-set z score to one predictor."""
    train_z, test_z = train_z_apply(train.astype(float), test.astype(float))
    train_z.name = name
    test_z.name = name
    return train_z, test_z


def add_interactions(train: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Add train-standardized interaction features."""
    # 交互项只使用训练折均值和标准差
    train = train.copy()
    test = test.copy()
    if "nihss" in train.columns:
        train_nihss_z, test_nihss_z = zscore_pair(train["nihss"], test["nihss"], "nihss_z")
        train["nihss_z"] = train_nihss_z.to_numpy()
        test["nihss_z"] = test_nihss_z.to_numpy()
        train["ntdc_x_nihss"] = train["ntdc"] * train["nihss_z"]
        test["ntdc_x_nihss"] = test["ntdc"] * test["nihss_z"]
        train["nt_node_x_nihss"] = train["nt_node_z"] * train["nihss_z"]
        test["nt_node_x_nihss"] = test["nt_node_z"] * test["nihss_z"]

    if "lesion_volume_ml" in train.columns:
        train_volume_z, test_volume_z = zscore_pair(train["lesion_volume_ml"], test["lesion_volume_ml"], "lesion_volume_z")
        train["lesion_volume_z"] = train_volume_z.to_numpy()
        test["lesion_volume_z"] = test_volume_z.to_numpy()
        train["ntdc_x_lesion_volume"] = train["ntdc"] * train["lesion_volume_z"]
        test["ntdc_x_lesion_volume"] = test["ntdc"] * test["lesion_volume_z"]
        train["nt_edge_x_lesion_volume"] = train["nt_edge_z"] * train["lesion_volume_z"]
        test["nt_edge_x_lesion_volume"] = test["nt_edge_z"] * test["lesion_volume_z"]

    return train, test


def read_fold_weights(profile_dir: Path, fold: int, prefix: str, level: str) -> pd.DataFrame:
    """Read saved LSM weights for one fold."""
    path = profile_dir / "fold_weights" / f"fold_{fold:02d}_{prefix}_{level}_weights.csv"
    if not path.exists():
        raise FileNotFoundError(f"missing fold weights: {path}")
    return pd.read_csv(path)


def reconstruct_fold_scores(
    config: dict,
    profile_dir: Path,
    fold: int,
    data: pd.DataFrame,
    train_ids: list[str],
    test_ids: list[str],
    lesion_indices: dict[str, np.ndarray],
    atlas_flat: np.ndarray,
    labels: list[int],
    roi_counts: np.ndarray,
    edge_matrix,
    edge_names: list[str],
    lesion_node: pd.DataFrame,
    lesion_edge: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Reconstruct fold-specific component scores from saved profiles and weights."""
    gray_path = profile_dir / "profiles" / f"fold_{fold:02d}_ntdc_hansen_profile.nii.gz"
    wm_path = profile_dir / "profiles" / f"fold_{fold:02d}_ntdc_alves_profile.nii.gz"
    if not gray_path.exists() or not wm_path.exists():
        raise FileNotFoundError(f"missing fold profile images for fold {fold}")

    gray_flat = np.nan_to_num(nib.load(str(gray_path)).get_fdata(), nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32).ravel()
    wm_flat = np.nan_to_num(nib.load(str(wm_path)).get_fdata(), nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32).ravel()

    train_pheno = align_by_subject(data, train_ids)
    test_pheno = align_by_subject(data, test_ids)
    train_lesion_node = align_by_subject(lesion_node, train_ids)
    test_lesion_node = align_by_subject(lesion_node, test_ids)
    train_lesion_edge = align_by_subject(lesion_edge, train_ids)
    test_lesion_edge = align_by_subject(lesion_edge, test_ids)

    profile_node_train = compute_profile_node_damage(train_ids, lesion_indices, atlas_flat, labels, roi_counts, gray_flat)
    profile_node_test = compute_profile_node_damage(test_ids, lesion_indices, atlas_flat, labels, roi_counts, gray_flat)
    profile_edge_train = compute_profile_edge_damage(train_ids, lesion_indices, edge_matrix, edge_names, wm_flat)
    profile_edge_test = compute_profile_edge_damage(test_ids, lesion_indices, edge_matrix, edge_names, wm_flat)

    lesion_node_weights = read_fold_weights(profile_dir, fold, "lesion", "node")
    lesion_edge_weights = read_fold_weights(profile_dir, fold, "lesion", "edge")
    profile_node_weights = read_fold_weights(profile_dir, fold, "profile", "node")
    profile_edge_weights = read_fold_weights(profile_dir, fold, "profile", "edge")

    train_lesion_node_impact = impact_from_weights(train_lesion_node, lesion_node_weights)
    test_lesion_node_impact = impact_from_weights(test_lesion_node, lesion_node_weights)
    train_lesion_edge_impact = impact_from_weights(train_lesion_edge, lesion_edge_weights)
    test_lesion_edge_impact = impact_from_weights(test_lesion_edge, lesion_edge_weights)
    train_nt_node_impact = impact_from_weights(profile_node_train, profile_node_weights)
    test_nt_node_impact = impact_from_weights(profile_node_test, profile_node_weights)
    train_nt_edge_impact = impact_from_weights(profile_edge_train, profile_edge_weights)
    test_nt_edge_impact = impact_from_weights(profile_edge_test, profile_edge_weights)

    train_lesion_node_z, test_lesion_node_z = zscore_pair(train_lesion_node_impact, test_lesion_node_impact, "lesion_node_z")
    train_lesion_edge_z, test_lesion_edge_z = zscore_pair(train_lesion_edge_impact, test_lesion_edge_impact, "lesion_edge_z")
    train_nt_node_z, test_nt_node_z = zscore_pair(train_nt_node_impact, test_nt_node_impact, "nt_node_z")
    train_nt_edge_z, test_nt_edge_z = zscore_pair(train_nt_edge_impact, test_nt_edge_impact, "nt_edge_z")

    covariates = analysis_covariates(config, "model_covariates")
    outcome = outcome_column(config)
    cols = ["subject_id", "cv_group", outcome, *covariates]
    train = train_pheno[cols].copy()
    test = test_pheno[cols].copy()
    train["fold"] = fold
    test["fold"] = fold
    train["lesion_node_impact"] = train_lesion_node_impact.to_numpy()
    test["lesion_node_impact"] = test_lesion_node_impact.to_numpy()
    train["lesion_edge_impact"] = train_lesion_edge_impact.to_numpy()
    test["lesion_edge_impact"] = test_lesion_edge_impact.to_numpy()
    train["profile_node_impact"] = train_nt_node_impact.to_numpy()
    test["profile_node_impact"] = test_nt_node_impact.to_numpy()
    train["profile_edge_impact"] = train_nt_edge_impact.to_numpy()
    test["profile_edge_impact"] = test_nt_edge_impact.to_numpy()
    train["lesion_node_z"] = train_lesion_node_z.to_numpy()
    test["lesion_node_z"] = test_lesion_node_z.to_numpy()
    train["lesion_edge_z"] = train_lesion_edge_z.to_numpy()
    test["lesion_edge_z"] = test_lesion_edge_z.to_numpy()
    train["nt_node_z"] = train_nt_node_z.to_numpy()
    test["nt_node_z"] = test_nt_node_z.to_numpy()
    train["nt_edge_z"] = train_nt_edge_z.to_numpy()
    test["nt_edge_z"] = test_nt_edge_z.to_numpy()
    train["sdc"] = train["lesion_node_z"] + train["lesion_edge_z"]
    test["sdc"] = test["lesion_node_z"] + test["lesion_edge_z"]
    train["ntdc"] = train["nt_node_z"] + train["nt_edge_z"]
    test["ntdc"] = test["nt_node_z"] + test["nt_edge_z"]
    train, test = add_interactions(train, test)
    return train, test


def component_model_specs(covariates: list[str], available: set[str]) -> list[tuple[str, list[str]]]:
    """Return component-wise prediction models."""
    candidates = [
        ("Clinical", covariates),
        ("Clinical + SDC", covariates + ["sdc"]),
        ("Clinical + NTDC", covariates + ["ntdc"]),
        ("Clinical + NTDC components", covariates + ["nt_node_z", "nt_edge_z"]),
        ("Clinical + SDC + NTDC", covariates + ["sdc", "ntdc"]),
        ("Clinical + SDC + NTDC components", covariates + ["lesion_node_z", "lesion_edge_z", "nt_node_z", "nt_edge_z"]),
        ("Clinical + NTDC interactions", covariates + ["ntdc", "ntdc_x_nihss", "ntdc_x_lesion_volume"]),
        (
            "Clinical + components + interactions",
            covariates
            + [
                "lesion_node_z",
                "lesion_edge_z",
                "nt_node_z",
                "nt_edge_z",
                "ntdc_x_nihss",
                "ntdc_x_lesion_volume",
                "nt_node_x_nihss",
                "nt_edge_x_lesion_volume",
            ],
        ),
    ]
    return [(name, predictors) for name, predictors in candidates if all(col in available for col in predictors)]


def predict_models(train: pd.DataFrame, test: pd.DataFrame, labels: list[int], threshold: float, models: list[tuple[str, list[str]]], outcome: str) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Fit ordered models and predict one test fold."""
    rows = []
    status_rows = []
    fold = int(test["fold"].iloc[0])
    for model_name, predictors in models:
        fit, status, error = fit_ordered_model(train, outcome, predictors)
        status_rows.append({"model": model_name, "fold": fold, "n_train": int(train.shape[0]), "n_test": int(test.shape[0]), "status": status, "error": error})
        if fit is None:
            continue
        probabilities = fit.model.predict(fit.params, exog=test[predictors].astype(float))
        aligned = align_probabilities(fit, np.asarray(probabilities, dtype=float), labels)
        expected = aligned @ np.asarray(labels, dtype=float)
        binary_probability = aligned[:, [index for index, label in enumerate(labels) if label <= threshold]].sum(axis=1)
        for row_index, subject_id in enumerate(test["subject_id"].astype(str).tolist()):
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


def summarize_prediction_tables(config: dict, predictions: pd.DataFrame, out_dir: Path, models: list[tuple[str, list[str]]]) -> None:
    """Write performance and pairwise bootstrap tables."""
    binary_cfg = config.get("analysis", {}).get("binary_outcome", {})
    threshold = float(binary_cfg.get("threshold", 2))
    labels = [int(label) for label in sorted(predictions["observed_mrs"].unique())]
    performance_rows = []
    for model_name, _ in models:
        model_pred = predictions[predictions["model"] == model_name].copy()
        if model_pred.empty:
            continue
        performance_rows.append({"model": model_name, "n": int(model_pred.shape[0]), **prediction_metrics(model_pred, labels, threshold)})
    performance = pd.DataFrame(performance_rows)
    write_csv(performance, out_dir / "component_model_prediction_performance.csv")

    directions = metric_directions()
    n_bootstrap = int(config.get("impact", {}).get("prediction_bootstrap", 1000))
    random_state = int(config.get("impact", {}).get("random_state", 42))
    pair_rows = []
    available_models = performance["model"].tolist()
    for index, model_a in enumerate(available_models):
        pred_a = predictions[predictions["model"] == model_a].copy()
        for model_b in available_models[index + 1 :]:
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
    write_csv(pairwise, out_dir / "component_model_prediction_pairwise_bootstrap.csv")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run component-wise and interaction-aware prediction from completed NTDC outputs.")
    parser.add_argument("--config", default="config/dat_config.yaml")
    parser.add_argument("--profile-subdir", default="ml_profile")
    parser.add_argument("--output-subdir", default="component_prediction")
    parser.add_argument("--max-subjects", type=int, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    profile_dir = project_path(config, "derivatives", args.profile_subdir)
    out_dir = ensure_dir(project_path(config, "derivatives", args.output_subdir))
    data = read_scores(config, args.profile_subdir, args.max_subjects)
    outcome = outcome_column(config)
    covariates = analysis_covariates(config, "model_covariates")
    data[outcome] = data[outcome].astype(int)
    data["fold"] = data["fold"].astype(int)

    atlas_flat, labels, roi_counts = atlas_info(config)
    lesion_indices = load_lesion_indices(data)
    edge_matrix, edge_names = load_edge_matrix(config)
    lesion_node = compute_lesion_node_load(config, data)
    _, lesion_edge = load_lesion_feature_tables(config, data)
    lesion_node = align_by_subject(lesion_node, data["subject_id"].astype(str).tolist())
    lesion_edge = align_by_subject(lesion_edge, data["subject_id"].astype(str).tolist())

    binary_cfg = config.get("analysis", {}).get("binary_outcome", {})
    threshold = float(binary_cfg.get("threshold", 2))
    labels_outcome = [int(label) for label in sorted(data[outcome].unique())]
    score_rows = []
    prediction_rows = []
    status_rows = []
    model_list = None

    for fold in sorted(data["fold"].unique()):
        train_ids = data.loc[data["fold"] != fold, "subject_id"].astype(str).tolist()
        test_ids = data.loc[data["fold"] == fold, "subject_id"].astype(str).tolist()
        train, test = reconstruct_fold_scores(
            config,
            profile_dir,
            int(fold),
            data,
            train_ids,
            test_ids,
            lesion_indices,
            atlas_flat,
            labels,
            roi_counts,
            edge_matrix,
            edge_names,
            lesion_node,
            lesion_edge,
        )
        available = set(train.columns) & set(test.columns)
        models = component_model_specs(covariates, available)
        model_list = models if model_list is None else model_list
        fold_pred, fold_status = predict_models(train, test, labels_outcome, threshold, models, outcome)
        prediction_rows.extend(fold_pred)
        status_rows.extend(fold_status)
        score_rows.append(test)
        print(f"finished component fold {fold}", flush=True)

    scores = pd.concat(score_rows, ignore_index=True).drop_duplicates(subset=["subject_id"], keep="last")
    predictions = pd.DataFrame(prediction_rows)
    write_csv(scores, out_dir / "component_scores.csv")
    write_csv(predictions, out_dir / "component_model_prediction_cv.csv")
    write_csv(pd.DataFrame(status_rows), out_dir / "component_model_prediction_fold_status.csv")
    summarize_prediction_tables(config, predictions, out_dir, model_list or [])
    print(f"component_output={out_dir}")


if __name__ == "__main__":
    main()
