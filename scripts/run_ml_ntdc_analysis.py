#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNetCV
from sklearn.model_selection import GroupKFold
from statsmodels.stats.multitest import multipletests

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from compute_impact_scores import (  # noqa: E402
    add_cv_group,
    align_probabilities,
    bootstrap_metric_delta,
    fit_ordered_model,
    metric_directions,
    prediction_metrics,
)
from nt_analysis.config import analysis_covariates, ensure_dir, load_config, outcome_column, project_path  # noqa: E402
from nt_analysis.tables import write_csv  # noqa: E402


def neurotransmitter_ids(config: dict) -> list[str]:
    """Return configured neurotransmitter IDs."""
    return [str(spec["id"]) for spec in config.get("neurotransmitters", [])]


def model_specs(covariates: list[str]) -> list[tuple[str, list[str]]]:
    """Return the final ML-NTDC model set."""
    return [
        ("Clinical", covariates),
        ("Clinical + SDC", covariates + ["sdc"]),
        ("Clinical + ML-NTDC", covariates + ["ml_ntdc"]),
        ("Clinical + SDC + ML-NTDC", covariates + ["sdc", "ml_ntdc"]),
    ]


def load_ntdc_wide(config: dict) -> pd.DataFrame:
    """Load SDC and all neurotransmitter-specific NTDC scores."""
    nt_ids = neurotransmitter_ids(config)
    merged = None
    for nt_id in nt_ids:
        path = project_path(config, "derivatives", "nt", nt_id, "impact", "nt_impact_scores.csv")
        df = pd.read_csv(path)
        df = add_cv_group(config, df)
        keep = ["subject_id", "cv_group", outcome_column(config), *analysis_covariates(config, "model_covariates"), "sdc", "ntdc"]
        df = df[[col for col in keep if col in df.columns]].copy()
        df = df.rename(columns={"ntdc": f"ntdc_{nt_id}"})
        if merged is None:
            merged = df
        else:
            merged = merged.merge(df[["subject_id", f"ntdc_{nt_id}"]], on="subject_id", how="inner")
    if merged is None:
        raise RuntimeError("no neurotransmitter NTDC tables found")
    return merged


def zscore_train_test(train: np.ndarray, test: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Apply training-set z scores to train and test matrices."""
    mean = np.nanmean(train, axis=0)
    sd = np.nanstd(train, axis=0, ddof=1)
    sd[~np.isfinite(sd) | (sd <= np.finfo(float).eps)] = 1.0
    train_z = (train - mean[None, :]) / sd[None, :]
    test_z = (test - mean[None, :]) / sd[None, :]
    return train_z, test_z, mean, sd


def residualize_train_test(values_train: np.ndarray, values_test: np.ndarray, design_train: np.ndarray, design_test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Residualize train and test values by training-set baseline design."""
    coef, *_ = np.linalg.lstsq(design_train, values_train, rcond=None)
    return values_train - design_train @ coef, values_test - design_test @ coef


def elastic_net_score(
    train: pd.DataFrame,
    test: pd.DataFrame,
    feature_cols: list[str],
    base_cols: list[str],
    outcome: str,
    groups: np.ndarray,
    config: dict,
) -> tuple[pd.Series, pd.Series, pd.DataFrame, dict[str, object]]:
    """Select NTDC features in training data and score train/test rows."""
    ml_cfg = config.get("ml_ntdc", {})
    inner_splits = int(ml_cfg.get("inner_splits", 5))
    l1_ratios = [float(value) for value in ml_cfg.get("l1_ratios", [0.1, 0.5, 0.9, 1.0])]
    alphas = np.asarray([float(value) for value in ml_cfg.get("alphas", np.logspace(-4, 0, 20))], dtype=float)

    x_train_raw = train[feature_cols].to_numpy(dtype=float)
    x_test_raw = test[feature_cols].to_numpy(dtype=float)
    x_train_z, x_test_z, _, _ = zscore_train_test(x_train_raw, x_test_raw)
    design_train = np.column_stack([np.ones(train.shape[0]), train[base_cols].to_numpy(dtype=float)])
    design_test = np.column_stack([np.ones(test.shape[0]), test[base_cols].to_numpy(dtype=float)])
    y_train = train[outcome].to_numpy(dtype=float).reshape(-1, 1)

    # 只筛选Clinical+SDC之外的递质增量信息
    y_res, _ = residualize_train_test(y_train, y_train, design_train, design_train)
    x_train_res, x_test_res = residualize_train_test(x_train_z, x_test_z, design_train, design_test)
    unique_groups = np.unique(groups)
    n_splits = max(2, min(inner_splits, len(unique_groups)))
    inner_cv = list(GroupKFold(n_splits=n_splits).split(x_train_res, groups=groups))
    model = ElasticNetCV(l1_ratio=l1_ratios, alphas=alphas, cv=inner_cv, max_iter=100000, tol=1e-3, random_state=int(config.get("impact", {}).get("random_state", 42)))
    model.fit(x_train_res, y_res.ravel())
    coef = np.asarray(model.coef_, dtype=float)
    fallback = False
    if np.sum(np.abs(coef)) <= np.finfo(float).eps:
        # 全部收缩为0时，退回到训练集内最强残差相关递质
        corr = np.asarray([np.corrcoef(x_train_res[:, idx], y_res.ravel())[0, 1] if np.nanstd(x_train_res[:, idx]) > 0 else 0.0 for idx in range(len(feature_cols))])
        corr[~np.isfinite(corr)] = 0.0
        best = int(np.argmax(np.abs(corr)))
        coef[best] = corr[best]
        fallback = True
    denom = np.sum(np.abs(coef))
    train_score = x_train_res @ coef / denom
    test_score = x_test_res @ coef / denom
    mean = float(np.mean(train_score))
    sd = float(np.std(train_score, ddof=1))
    if not np.isfinite(sd) or sd <= np.finfo(float).eps:
        sd = 1.0
    train_score = (train_score - mean) / sd
    test_score = (test_score - mean) / sd

    rows = []
    for feature, value in zip(feature_cols, coef):
        rows.append({"feature": feature, "nt_id": feature.replace("ntdc_", ""), "coef": float(value), "selected": bool(abs(value) > np.finfo(float).eps)})
    meta = {
        "alpha": float(model.alpha_),
        "l1_ratio": float(model.l1_ratio_),
        "selected_count": int(np.sum(np.abs(coef) > np.finfo(float).eps)),
        "fallback": fallback,
    }
    return pd.Series(train_score, index=train.index), pd.Series(test_score, index=test.index), pd.DataFrame(rows), meta


def predict_outer_fold(train: pd.DataFrame, test: pd.DataFrame, labels: list[int], threshold: float, specs: list[tuple[str, list[str]]], outcome: str) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Fit final ordered models in one outer fold."""
    rows = []
    status_rows = []
    for model_name, predictors in specs:
        fit, status, error = fit_ordered_model(train, outcome, predictors)
        status_rows.append({"model": model_name, "fold": int(test["fold"].iloc[0]), "n_train": int(train.shape[0]), "n_test": int(test.shape[0]), "status": status, "error": error})
        if fit is None:
            continue
        probabilities = fit.model.predict(fit.params, exog=test[predictors].astype(float))
        aligned = align_probabilities(fit, np.asarray(probabilities, dtype=float), labels)
        expected = aligned @ np.asarray(labels, dtype=float)
        binary_probability = aligned[:, [idx for idx, label in enumerate(labels) if label <= threshold]].sum(axis=1)
        for row_index, subject_id in enumerate(test["subject_id"].tolist()):
            item = {
                "subject_id": subject_id,
                "model": model_name,
                "fold": int(test.iloc[row_index]["fold"]),
                "observed_mrs": int(test.iloc[row_index][outcome]),
                "expected_mrs": float(expected[row_index]),
                "prob_mrs_le_threshold": float(binary_probability[row_index]),
            }
            for label_index, label in enumerate(labels):
                item[f"prob_{label}"] = float(aligned[row_index, label_index])
            rows.append(item)
    return rows, status_rows


def summarize_predictions(config: dict, predictions: pd.DataFrame, out_dir: Path) -> None:
    """Write prediction performance and paired bootstrap comparisons."""
    binary_cfg = config.get("analysis", {}).get("binary_outcome", {})
    threshold = float(binary_cfg.get("threshold", 2))
    labels = [int(label) for label in sorted(predictions["observed_mrs"].unique())]
    specs = model_specs(analysis_covariates(config, "model_covariates"))
    performance_rows = []
    for model_name, _ in specs:
        model_pred = predictions[predictions["model"] == model_name].copy()
        performance_rows.append({"model": model_name, "n": int(model_pred.shape[0]), **prediction_metrics(model_pred, labels, threshold)})
    performance = pd.DataFrame(performance_rows)
    write_csv(performance, out_dir / "model_prediction_performance.csv")

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
    write_csv(pairwise, out_dir / "model_prediction_pairwise_bootstrap.csv")


def write_report(out_dir: Path, selection_summary: pd.DataFrame, performance: pd.DataFrame) -> None:
    """Write a compact ML-NTDC run report."""
    lines = [
        "# ML-NTDC Run Report",
        "",
        "## Models",
        "",
        "- Clinical",
        "- Clinical + SDC",
        "- Clinical + ML-NTDC",
        "- Clinical + SDC + ML-NTDC",
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
    (out_dir / "ml_ntdc_run_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run nested ML neurotransmitter selection and ML-NTDC prediction.")
    parser.add_argument("--config", default="config/dat_config.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    out_dir = ensure_dir(project_path(config, "derivatives", "nt_ml", "ml_ntdc"))
    model_dir = ensure_dir(out_dir / "models")

    data = load_ntdc_wide(config)
    outcome = outcome_column(config)
    covariates = analysis_covariates(config, "model_covariates")
    feature_cols = [f"ntdc_{nt_id}" for nt_id in neurotransmitter_ids(config)]
    base_cols = [*covariates, "sdc"]
    require_cols = ["subject_id", "cv_group", outcome, *base_cols, *feature_cols]
    data = data.dropna(subset=require_cols).copy()
    data[outcome] = data[outcome].astype(int)
    labels = [int(label) for label in sorted(data[outcome].unique())]
    binary_cfg = config.get("analysis", {}).get("binary_outcome", {})
    threshold = float(binary_cfg.get("threshold", 2))

    outer_splits = min(int(config.get("impact", {}).get("n_splits", 10)), data["cv_group"].nunique())
    splitter = GroupKFold(n_splits=outer_splits)
    predictions = []
    statuses = []
    score_rows = []
    selection_rows = []
    specs = model_specs(covariates)
    for fold, (train_idx, test_idx) in enumerate(splitter.split(data, groups=data["cv_group"].astype(str)), start=1):
        train = data.iloc[train_idx].copy()
        test = data.iloc[test_idx].copy()
        train_groups = train["cv_group"].astype(str).to_numpy()
        train_score, test_score, coef_table, meta = elastic_net_score(train, test, feature_cols, base_cols, outcome, train_groups, config)
        train["ml_ntdc"] = train_score
        test["ml_ntdc"] = test_score
        train["fold"] = fold
        test["fold"] = fold
        # 只保存外层测试折的折外ML-NTDC分数
        score_rows.append(test[["subject_id", "cv_group", outcome, *covariates, "sdc", "ml_ntdc", "fold"]])
        coef_table["fold"] = fold
        for key, value in meta.items():
            coef_table[key] = value
        selection_rows.append(coef_table)
        fold_pred, fold_status = predict_outer_fold(train, test, labels, threshold, specs, outcome)
        predictions.extend(fold_pred)
        statuses.extend(fold_status)
        print(f"finished outer fold {fold}/{outer_splits}")

    scores = pd.concat(score_rows, ignore_index=True).drop_duplicates(subset=["subject_id"], keep="last")
    selection = pd.concat(selection_rows, ignore_index=True)
    selection_summary = (
        selection.groupby("nt_id")
        .agg(selection_frequency=("selected", "mean"), mean_coef=("coef", "mean"), mean_abs_coef=("coef", lambda x: float(np.mean(np.abs(x)))), selected_folds=("selected", "sum"))
        .reset_index()
        .sort_values(["selection_frequency", "mean_abs_coef"], ascending=False)
    )
    predictions_df = pd.DataFrame(predictions)
    write_csv(scores, out_dir / "ml_ntdc_scores.csv")
    write_csv(selection, out_dir / "ml_ntdc_selection_folds.csv")
    write_csv(selection_summary, out_dir / "ml_ntdc_selection_summary.csv")
    write_csv(predictions_df, model_dir / "model_prediction_cv.csv")
    write_csv(pd.DataFrame(statuses), model_dir / "model_prediction_fold_status.csv")
    summarize_predictions(config, predictions_df, model_dir)
    performance = pd.read_csv(model_dir / "model_prediction_performance.csv")
    write_report(out_dir, selection_summary, performance)


if __name__ == "__main__":
    main()
