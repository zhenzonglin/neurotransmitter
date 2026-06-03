#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import sys
from datetime import datetime
from itertools import combinations
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from jinja2 import Template
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import GroupKFold, StratifiedKFold
from statsmodels.stats.multitest import multipletests

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nt_analysis.config import analysis_covariates, ensure_dir, load_config, project_path  # noqa: E402
from nt_analysis.tables import write_csv  # noqa: E402


palette = ["#0077BB", "#33BBEE", "#009988", "#EE7733", "#CC3311", "#EE3377", "#BBBBBB", "#000000"]


def normalize_subject_id_flexible(value: object) -> str:
    """Normalize TMS or numeric subject IDs."""
    if pd.isna(value):
        return ""
    if isinstance(value, float) and value.is_integer():
        text = str(int(value))
    else:
        text = str(value).strip()
    text = text.replace(" ", "")
    upper = text.upper().replace("-", "").replace("_", "")
    if upper.startswith("TMS"):
        digits = "".join(ch for ch in upper if ch.isdigit())
        return f"TMS{int(digits):03d}" if digits else upper
    if upper.endswith(".0") and upper[:-2].isdigit():
        upper = upper[:-2]
    return upper


def normalize_qc_value(value: object) -> str:
    """Normalize exclusion values."""
    if pd.isna(value):
        return ""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    text = str(value).strip()
    if text.endswith(".0") and text[:-2].isdigit():
        return text[:-2]
    return text


def parse_endpoint_specs(values: list[str]) -> list[dict[str, str]]:
    """Parse endpoint specs."""
    specs = []
    for value in values:
        parts = value.split(":")
        if len(parts) != 3:
            raise ValueError(f"endpoint spec must be label:outcome_column:stroke_column, got {value}")
        specs.append({"label": parts[0], "outcome_column": parts[1], "stroke_column": parts[2]})
    return specs


def load_feature_table(config: dict, feature_subdir: str, feature_table: str | None) -> pd.DataFrame:
    """Load completed SDC/NTDC impact features."""
    if feature_table:
        path = Path(feature_table)
        if not path.is_absolute():
            path = project_path(config, feature_table)
    else:
        path = project_path(config, "derivatives", feature_subdir, "profile_scores.csv")
    if not path.exists():
        raise FileNotFoundError(f"missing feature table: {path}")
    data = pd.read_csv(path, dtype={"subject_id": str})
    data["subject_id"] = data["subject_id"].map(normalize_subject_id_flexible)
    return data


def load_phenotype(config: dict, endpoint_specs: list[dict[str, str]]) -> pd.DataFrame:
    """Load endpoint and recurrence columns."""
    path = project_path(config, config["inputs"]["phenotype_file"])
    sheet = config["inputs"]["phenotype_sheet"]
    id_col = config["inputs"]["phenotype_id_column"]
    df = pd.read_excel(path, sheet_name=sheet)
    if id_col not in df.columns:
        raise KeyError(f"missing phenotype id column: {id_col}")
    df["subject_id"] = df[id_col].map(normalize_subject_id_flexible)
    keep = ["subject_id"]
    for spec in endpoint_specs:
        for column in [spec["outcome_column"], spec["stroke_column"]]:
            if column in df.columns:
                keep.append(column)
    keep = list(dict.fromkeys(keep))
    return df[keep].copy()


def train_z_apply_frame(train: pd.DataFrame, test: pd.DataFrame, predictors: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Z-score predictors with training-set statistics."""
    train_values = train[predictors].astype(float).to_numpy()
    test_values = test[predictors].astype(float).to_numpy()
    mean = np.nanmean(train_values, axis=0)
    sd = np.nanstd(train_values, axis=0, ddof=1)
    sd[~np.isfinite(sd) | (sd <= np.finfo(float).eps)] = 1.0
    return (train_values - mean[None, :]) / sd[None, :], (test_values - mean[None, :]) / sd[None, :]


def model_specs(clinical: list[str], selected: list[str]) -> list[tuple[str, list[str]]]:
    """Return fixed and automatic model sets."""
    return [
        ("Clinical", clinical),
        ("Clinical + lesion components", clinical + ["lesion_node_impact", "lesion_edge_impact"]),
        ("Clinical + NT components", clinical + ["profile_node_impact", "profile_edge_impact"]),
        ("Clinical + all components", clinical + ["lesion_node_impact", "lesion_edge_impact", "profile_node_impact", "profile_edge_impact"]),
        ("Clinical + auto-selected components", clinical + selected),
    ]


def component_subsets(candidates: list[str]) -> list[tuple[str, ...]]:
    """Enumerate all non-empty component subsets."""
    subsets = []
    for size in range(1, len(candidates) + 1):
        subsets.extend(combinations(candidates, size))
    return subsets


def select_components(train: pd.DataFrame, clinical: list[str], candidates: list[str], outcome: str, random_state: int) -> tuple[list[str], dict[str, object]]:
    """Select imaging components inside the training fold."""
    y = train[outcome].astype(int).to_numpy()
    if len(np.unique(y)) < 2:
        return [], {"selected": "", "selection_score": np.nan, "selection_method": "exhaustive_inner_cv", "status": "single_class"}
    min_class = int(pd.Series(y).value_counts().min())
    n_splits = max(2, min(5, min_class))
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    subset_rows = []
    for subset in component_subsets(candidates):
        predictors = clinical + list(subset)
        fold_losses = []
        for inner_train_idx, inner_test_idx in cv.split(train, y):
            inner_train = train.iloc[inner_train_idx].copy()
            inner_test = train.iloc[inner_test_idx].copy()
            prob, status = fit_predict_logistic(inner_train, inner_test, predictors, outcome, random_state)
            if prob is None:
                continue
            inner_y = inner_test[outcome].astype(int).to_numpy()
            fold_losses.append(float(log_loss(inner_y, np.clip(prob, 1e-15, 1 - 1e-15), labels=[0, 1])))
        score = float(np.mean(fold_losses)) if fold_losses else np.inf
        subset_rows.append({"subset": subset, "selection_score": score, "n_components": len(subset)})
    valid = [row for row in subset_rows if np.isfinite(row["selection_score"])]
    if not valid:
        return [], {"selected": "", "selection_score": np.nan, "selection_method": "exhaustive_inner_cv", "status": "failed"}
    best = min(valid, key=lambda row: (row["selection_score"], row["n_components"]))
    selected = list(best["subset"])
    return selected, {
        "selected": "|".join(selected),
        "selection_score": float(best["selection_score"]),
        "selection_method": "exhaustive_inner_cv",
        "status": "ok",
    }


def fit_predict_logistic(train: pd.DataFrame, test: pd.DataFrame, predictors: list[str], outcome: str, random_state: int) -> tuple[np.ndarray | None, str]:
    """Fit binary logistic regression and predict probabilities."""
    y_train = train[outcome].astype(int).to_numpy()
    if len(np.unique(y_train)) < 2:
        return None, "single_class"
    x_train, x_test = train_z_apply_frame(train, test, predictors)
    model = LogisticRegression(penalty="l2", C=1.0, solver="lbfgs", max_iter=5000, random_state=random_state)
    try:
        model.fit(x_train, y_train)
        return model.predict_proba(x_test)[:, 1], "ok"
    except Exception as error:  # noqa: BLE001
        return None, str(error)


def binary_metrics(data: pd.DataFrame) -> dict[str, float]:
    """Compute binary prediction metrics."""
    y = data["binary_good"].astype(int).to_numpy()
    p = np.clip(data["prob_good"].astype(float).to_numpy(), 1e-15, 1 - 1e-15)
    auc = np.nan
    if len(np.unique(y)) == 2:
        auc = float(roc_auc_score(y, p))
    pred = (p >= 0.5).astype(int)
    return {
        "binary_log_loss": float(log_loss(y, p, labels=[0, 1])),
        "binary_auc": auc,
        "binary_brier": float(brier_score_loss(y, p)),
        "accuracy": float(accuracy_score(y, pred)),
        "mean_prob_good": float(np.mean(p)),
    }


def metric_value(data: pd.DataFrame, metric: str) -> float:
    """Return one metric."""
    values = binary_metrics(data)
    return float(values[metric])


def bootstrap_pairwise(predictions: pd.DataFrame, n_bootstrap: int, random_state: int) -> pd.DataFrame:
    """Paired bootstrap model comparisons."""
    metrics = {"binary_log_loss": "lower", "binary_auc": "higher", "binary_brier": "lower", "accuracy": "higher"}
    models = predictions["model"].drop_duplicates().tolist()
    rows = []
    for index, model_a in enumerate(models):
        a = predictions[predictions["model"] == model_a]
        for model_b in models[index + 1 :]:
            b = predictions[predictions["model"] == model_b]
            merged = a.merge(b, on=["subject_id", "binary_good"], suffixes=("_a", "_b"))
            if merged.empty:
                continue
            rng = np.random.default_rng(random_state + index + len(rows))
            indices = np.arange(merged.shape[0])
            for metric, direction in metrics.items():
                def make_frame(sample: pd.DataFrame, suffix: str) -> pd.DataFrame:
                    return pd.DataFrame(
                        {
                            "subject_id": sample["subject_id"],
                            "binary_good": sample["binary_good"],
                            "prob_good": sample[f"prob_good_{suffix}"],
                        }
                    )

                observed = metric_value(make_frame(merged, "b"), metric) - metric_value(make_frame(merged, "a"), metric)
                boot = []
                for _ in range(n_bootstrap):
                    sample = merged.iloc[rng.choice(indices, size=len(indices), replace=True)]
                    value = metric_value(make_frame(sample, "b"), metric) - metric_value(make_frame(sample, "a"), metric)
                    if np.isfinite(value):
                        boot.append(value)
                if boot:
                    ci_low, ci_high = np.percentile(boot, [2.5, 97.5])
                    if direction == "lower":
                        p_value = 2 * min(np.mean(np.asarray(boot) <= 0), np.mean(np.asarray(boot) >= 0))
                    else:
                        p_value = 2 * min(np.mean(np.asarray(boot) >= 0), np.mean(np.asarray(boot) <= 0))
                    p_value = float(min(max(p_value, 0.0), 1.0))
                else:
                    ci_low, ci_high, p_value = np.nan, np.nan, np.nan
                rows.append(
                    {
                        "model_a": model_a,
                        "model_b": model_b,
                        "metric": metric,
                        "higher_is_better": direction == "higher",
                        "delta_b_minus_a": observed,
                        "ci_low": ci_low,
                        "ci_high": ci_high,
                        "p_bootstrap": p_value,
                        "n_bootstrap": n_bootstrap,
                    }
                )
    out = pd.DataFrame(rows)
    out["p_fdr_bh"] = np.nan
    out["p_bonferroni"] = np.nan
    for metric in out["metric"].unique():
        mask = (out["metric"] == metric) & out["p_bootstrap"].notna()
        if mask.any():
            out.loc[mask, "p_fdr_bh"] = multipletests(out.loc[mask, "p_bootstrap"], method="fdr_bh")[1]
            out.loc[mask, "p_bonferroni"] = multipletests(out.loc[mask, "p_bootstrap"], method="bonferroni")[1]
    return out


def build_endpoint_dataset(features: pd.DataFrame, phenotype: pd.DataFrame, spec: dict[str, str], clinical: list[str], candidates: list[str], exclude_values: set[str], threshold: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build one endpoint dataset and QC table."""
    outcome_col = spec["outcome_column"]
    stroke_col = spec["stroke_column"]
    if outcome_col not in phenotype.columns:
        raise KeyError(f"missing endpoint column: {outcome_col}")
    if stroke_col not in phenotype.columns:
        raise KeyError(f"missing recurrence column: {stroke_col}")

    endpoint = phenotype[["subject_id", outcome_col, stroke_col]].copy()
    endpoint = endpoint.rename(columns={outcome_col: "endpoint_mrs", stroke_col: "stroke_recurrence"})
    data = features.merge(endpoint, on="subject_id", how="left")
    data["endpoint_mrs"] = pd.to_numeric(data["endpoint_mrs"], errors="coerce")
    data["excluded_recurrence"] = data["stroke_recurrence"].map(normalize_qc_value).isin(exclude_values)
    data["has_endpoint"] = data["endpoint_mrs"].notna()
    data["included"] = data["has_endpoint"] & ~data["excluded_recurrence"]
    qc = pd.DataFrame(
        [
            {"item": "feature_rows", "value": int(features.shape[0])},
            {"item": "has_endpoint", "value": int(data["has_endpoint"].sum())},
            {"item": "excluded_recurrence", "value": int(data["excluded_recurrence"].sum())},
            {"item": "included", "value": int(data["included"].sum())},
            {"item": "binary_threshold_good_mrs_le", "value": int(threshold)},
        ]
    )
    data = data.loc[data["included"]].copy()
    required = ["subject_id", "endpoint_mrs", "cv_group", *clinical, *candidates]
    missing = [col for col in required if col not in data.columns]
    if missing:
        raise KeyError(f"missing required columns for endpoint {spec['label']}: {missing}")
    data = data.dropna(subset=required).copy()
    data["endpoint_mrs"] = data["endpoint_mrs"].astype(int)
    data["binary_good"] = (data["endpoint_mrs"] <= threshold).astype(int)
    return data, qc


def run_endpoint(data: pd.DataFrame, spec: dict[str, str], clinical: list[str], candidates: list[str], config: dict, out_dir: Path, args: argparse.Namespace, threshold: int) -> dict[str, object]:
    """Run one endpoint prediction analysis."""
    label = spec["label"]
    endpoint_dir = ensure_dir(out_dir / label)
    n_splits = min(int(config.get("impact", {}).get("n_splits", 10)), data["cv_group"].astype(str).nunique())
    if n_splits < 2:
        raise RuntimeError(f"endpoint {label} has fewer than two CV groups")
    groups = data["cv_group"].astype(str).to_numpy()
    splitter = GroupKFold(n_splits=n_splits)
    predictions = []
    selected_rows = []
    status_rows = []
    random_state = int(config.get("impact", {}).get("random_state", 42))

    for fold, (train_idx, test_idx) in enumerate(splitter.split(data, groups=groups), start=1):
        train = data.iloc[train_idx].copy()
        test = data.iloc[test_idx].copy()
        selected, meta = select_components(train, clinical, candidates, "binary_good", random_state + fold)
        selected_rows.append({"fold": fold, **meta})
        specs = model_specs(clinical, selected)
        available = set(train.columns) & set(test.columns)
        specs = [(name, predictors) for name, predictors in specs if all(col in available for col in predictors)]
        for model_name, predictors in specs:
            prob, status = fit_predict_logistic(train, test, predictors, "binary_good", random_state + fold)
            status_rows.append({"fold": fold, "model": model_name, "status": status, "predictors": "|".join(predictors)})
            if prob is None:
                continue
            for row_index, subject_id in enumerate(test["subject_id"].astype(str).tolist()):
                predictions.append(
                    {
                        "subject_id": subject_id,
                        "fold": fold,
                        "model": model_name,
                        "endpoint_mrs": int(test.iloc[row_index]["endpoint_mrs"]),
                        "binary_good": int(test.iloc[row_index]["binary_good"]),
                        "prob_good": float(prob[row_index]),
                    }
                )
        print(f"finished {label} fold {fold}/{n_splits}", flush=True)

    prediction_df = pd.DataFrame(predictions)
    selected_df = pd.DataFrame(selected_rows)
    status_df = pd.DataFrame(status_rows)
    performance = pd.DataFrame(
        [
            {"model": model, "n": int(group.shape[0]), **binary_metrics(group)}
            for model, group in prediction_df.groupby("model", sort=False)
        ]
    )
    pairwise = bootstrap_pairwise(prediction_df, int(config.get("impact", {}).get("prediction_bootstrap", 1000)), random_state)
    write_csv(prediction_df, endpoint_dir / "prediction_cv.csv")
    write_csv(performance, endpoint_dir / "prediction_performance.csv")
    write_csv(pairwise, endpoint_dir / "prediction_pairwise_bootstrap.csv")
    write_csv(selected_df, endpoint_dir / "selected_components_by_fold.csv")
    write_csv(status_df, endpoint_dir / "model_fold_status.csv")
    write_endpoint_report(spec, data, performance, pairwise, selected_df, prediction_df, endpoint_dir, threshold)
    best = performance.sort_values("binary_log_loss").iloc[0].to_dict() if not performance.empty else {}
    return {
        "endpoint": label,
        "base_endpoint": spec.get("base_label", label),
        "binary_good_threshold": threshold,
        "binary_target": f"mRS 0-{threshold} vs {threshold + 1}-6",
        "n": int(data.shape[0]),
        "best_model": best.get("model", ""),
        "best_binary_log_loss": best.get("binary_log_loss", np.nan),
        "best_auc": best.get("binary_auc", np.nan),
    }


def roc_points(y: np.ndarray, p: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Compute ROC points."""
    if len(np.unique(y)) < 2:
        return np.array([]), np.array([]), np.nan
    order = np.argsort(-p)
    y_sorted = y[order]
    p_sorted = p[order]
    distinct = np.r_[np.where(np.diff(p_sorted))[0], y_sorted.size - 1]
    tp = np.cumsum(y_sorted)[distinct]
    fp = (1 + distinct) - tp
    pos = y_sorted.sum()
    neg = y_sorted.size - pos
    fpr = np.r_[0.0, fp / neg, 1.0]
    tpr = np.r_[0.0, tp / pos, 1.0]
    return fpr, tpr, float(np.trapz(tpr, fpr))


def save_fig(fig: plt.Figure, path: Path) -> None:
    """Save and close a figure."""
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_performance(performance: pd.DataFrame, out_path: Path) -> None:
    """Plot endpoint model metrics."""
    panels = [("binary_log_loss", "Binary log loss"), ("binary_auc", "AUC"), ("binary_brier", "Brier"), ("accuracy", "Accuracy")]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    colors = [palette[i % len(palette)] for i in range(performance.shape[0])]
    for ax, (metric, title) in zip(axes.ravel(), panels):
        ax.bar(performance["model"], performance[metric], color=colors)
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=35)
    fig.tight_layout()
    save_fig(fig, out_path)


def plot_roc(predictions: pd.DataFrame, out_path: Path) -> pd.DataFrame:
    """Plot ROC curves."""
    fig, ax = plt.subplots(figsize=(6, 5))
    rows = []
    for index, (model, group) in enumerate(predictions.groupby("model", sort=False)):
        y = group["binary_good"].astype(int).to_numpy()
        p = group["prob_good"].astype(float).to_numpy()
        fpr, tpr, auc = roc_points(y, p)
        rows.append({"model": model, "n": int(group.shape[0]), "positive_n": int(y.sum()), "negative_n": int(y.size - y.sum()), "auc": auc})
        if fpr.size:
            ax.plot(fpr, tpr, color=palette[index % len(palette)], label=f"{model} (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], "--", color="#777777", linewidth=0.9)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("Out-of-sample ROC")
    if rows:
        ax.legend(loc="lower right", frameon=False)
    fig.tight_layout()
    save_fig(fig, out_path)
    return pd.DataFrame(rows)


def plot_pairwise(pairwise: pd.DataFrame, out_path: Path) -> None:
    """Plot pairwise binary log loss deltas."""
    data = pairwise[pairwise["metric"] == "binary_log_loss"].copy()
    if data.empty:
        return
    data["comparison"] = data["model_b"] + " vs " + data["model_a"]
    data = data.sort_values("delta_b_minus_a")
    fig, ax = plt.subplots(figsize=(8, max(3.5, 0.35 * len(data))))
    y = np.arange(data.shape[0])
    ax.errorbar(
        data["delta_b_minus_a"],
        y,
        xerr=[data["delta_b_minus_a"] - data["ci_low"], data["ci_high"] - data["delta_b_minus_a"]],
        fmt="o",
        color=palette[4],
        ecolor="#777777",
    )
    ax.axvline(0, color="#333333", linewidth=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(data["comparison"])
    ax.set_xlabel("Delta binary log loss")
    ax.set_title("Pairwise bootstrap")
    fig.tight_layout()
    save_fig(fig, out_path)


def table_html(df: pd.DataFrame, max_rows: int = 50) -> str:
    """Render a dataframe as HTML."""
    if df.empty:
        return "<p class=\"empty\">not available</p>"
    return df.head(max_rows).to_html(index=False, classes="data-table", border=0, float_format=lambda value: f"{value:.4f}" if isinstance(value, float) and math.isfinite(value) else str(value))


def write_endpoint_report(spec: dict[str, str], data: pd.DataFrame, performance: pd.DataFrame, pairwise: pd.DataFrame, selected: pd.DataFrame, predictions: pd.DataFrame, endpoint_dir: Path, threshold: int) -> None:
    """Write one endpoint HTML report."""
    figure_dir = ensure_dir(endpoint_dir / "figures")
    performance_png = figure_dir / "performance.png"
    roc_png = figure_dir / "roc.png"
    pairwise_png = figure_dir / "pairwise_binary_log_loss.png"
    plot_performance(performance, performance_png)
    roc_table = plot_roc(predictions, roc_png)
    plot_pairwise(pairwise, pairwise_png)
    write_csv(roc_table, endpoint_dir / "roc_auc.csv")
    selection_summary = selected["selected"].fillna("").str.split("|").explode()
    selection_summary = selection_summary[selection_summary != ""].value_counts().rename_axis("feature").reset_index(name="selected_folds")
    write_csv(selection_summary, endpoint_dir / "selected_component_summary.csv")
    counts = data["endpoint_mrs"].value_counts().sort_index().rename_axis("mrs").reset_index(name="n")

    template = Template(
        """<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <title>{{ label }} auto prediction</title>
  <style>
    body { margin: 0; font-family: Arial, "Microsoft YaHei", sans-serif; background: #f5f7fa; color: #17212b; }
    header { padding: 26px 34px; background: #fff; border-bottom: 1px solid #d9e1ea; }
    main { max-width: 1240px; margin: 0 auto; padding: 24px 28px 48px; }
    h1 { margin: 0 0 8px; font-size: 25px; }
    h2 { font-size: 19px; margin: 30px 0 12px; }
    h3 { font-size: 15px; margin: 20px 0 10px; }
    .meta { color: #526171; font-size: 13px; }
    .card { background: #fff; border: 1px solid #d9e1ea; border-radius: 8px; padding: 18px; margin: 16px 0; }
    .grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 18px; }
    figure { margin: 0; }
    img { width: 100%; border: 1px solid #d9e1ea; border-radius: 6px; background: #fff; }
    figcaption { margin-top: 6px; color: #526171; font-size: 12px; }
    .table-wrap { overflow-x: auto; border: 1px solid #d9e1ea; border-radius: 6px; }
    table.data-table { border-collapse: collapse; width: 100%; font-size: 12px; background: #fff; }
    table.data-table th, table.data-table td { padding: 7px 9px; border-bottom: 1px solid #edf1f5; text-align: left; white-space: nowrap; }
    table.data-table th { background: #eef3f8; color: #1c3346; }
    .empty { color: #7c8792; font-size: 12px; }
  </style>
</head>
<body>
  <header>
    <h1>{{ label }} Auto-selected Component Prediction</h1>
    <div class="meta">outcome: {{ outcome_column }} | recurrence exclusion: {{ stroke_column }} | binary target: mRS 0-{{ threshold }} vs {{ threshold + 1 }}-6 | generated: {{ generated_at }}</div>
  </header>
  <main>
    <section class="card">
      <h2>1. Model Performance</h2>
      <div class="grid">
        <figure><img src="figures/performance.png"><figcaption>Figure 1. Binary prediction metrics.</figcaption></figure>
        <figure><img src="figures/roc.png"><figcaption>Figure 2. Out-of-sample ROC.</figcaption></figure>
        <figure><img src="figures/pairwise_binary_log_loss.png"><figcaption>Figure 3. Pairwise bootstrap for binary log loss.</figcaption></figure>
      </div>
      <h3>prediction performance</h3>
      <div class="table-wrap">{{ performance_table }}</div>
      <h3>ROC AUC</h3>
      <div class="table-wrap">{{ roc_table }}</div>
      <h3>pairwise bootstrap</h3>
      <div class="table-wrap">{{ pairwise_table }}</div>
    </section>
    <section class="card">
      <h2>2. Automatic Selection</h2>
      <h3>selected component summary</h3>
      <div class="table-wrap">{{ selection_summary_table }}</div>
      <h3>selected components by fold</h3>
      <div class="table-wrap">{{ selected_table }}</div>
    </section>
    <section class="card">
      <h2>3. Endpoint Cohort</h2>
      <h3>mRS distribution</h3>
      <div class="table-wrap">{{ counts_table }}</div>
    </section>
  </main>
</body>
</html>"""
    )
    html = template.render(
        label=spec["label"],
        outcome_column=spec["outcome_column"],
        stroke_column=spec["stroke_column"],
        threshold=threshold,
        generated_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        performance_table=table_html(performance, 30),
        roc_table=table_html(roc_table, 30),
        pairwise_table=table_html(pairwise, 80),
        selected_table=table_html(selected, 30),
        selection_summary_table=table_html(selection_summary, 20),
        counts_table=table_html(counts, 20),
    )
    (endpoint_dir / "report.html").write_text(html, encoding="utf-8")


def write_index_report(out_dir: Path, summary: pd.DataFrame) -> None:
    """Write a combined report index."""
    template = Template(
        """<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <title>Multitimepoint Auto Prediction</title>
  <style>
    body { margin: 0; font-family: Arial, "Microsoft YaHei", sans-serif; background: #f5f7fa; color: #17212b; }
    header { padding: 26px 34px; background: #fff; border-bottom: 1px solid #d9e1ea; }
    main { max-width: 980px; margin: 0 auto; padding: 24px 28px 48px; }
    .card { background: #fff; border: 1px solid #d9e1ea; border-radius: 8px; padding: 18px; margin: 16px 0; }
    table { border-collapse: collapse; width: 100%; font-size: 13px; background: #fff; }
    th, td { padding: 8px 10px; border-bottom: 1px solid #edf1f5; text-align: left; }
    th { background: #eef3f8; }
  </style>
</head>
<body>
  <header><h1>Multitimepoint Auto Prediction</h1></header>
  <main>
    <section class="card">
      {{ summary_table }}
    </section>
  </main>
</body>
</html>"""
    )
    display = summary.copy()
    display["report"] = display["endpoint"].map(lambda value: f"<a href='{value}/report.html'>{value}</a>")
    html_table = display.to_html(index=False, escape=False)
    (out_dir / "index.html").write_text(template.render(summary_table=html_table), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run multitimepoint automatic component prediction and reports.")
    parser.add_argument("--config", default="config/dat_config.yaml")
    parser.add_argument("--feature-subdir", default="ml_profile")
    parser.add_argument("--feature-table", default=None)
    parser.add_argument("--output-subdir", default="multitimepoint_auto_prediction")
    parser.add_argument("--endpoints", nargs="+", default=["m3:m3_mRS:m3_stroke", "m6:m6_mRS:m6_stroke", "m12:m12_mRS:m12_stroke"])
    parser.add_argument("--stroke-exclude-values", nargs="+", default=["2"])
    parser.add_argument("--binary-good-threshold", type=int, default=None)
    parser.add_argument("--binary-good-thresholds", nargs="+", type=int, default=[1, 2])
    parser.add_argument("--max-subjects", type=int, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    out_dir = ensure_dir(project_path(config, "derivatives", args.output_subdir))
    endpoint_specs = parse_endpoint_specs(args.endpoints)
    features = load_feature_table(config, args.feature_subdir, args.feature_table)
    if args.max_subjects is not None:
        features = features.head(int(args.max_subjects)).copy()
    phenotype = load_phenotype(config, endpoint_specs)
    clinical = analysis_covariates(config, "model_covariates")
    candidates = ["lesion_node_impact", "lesion_edge_impact", "profile_node_impact", "profile_edge_impact"]
    exclude_values = {normalize_qc_value(value) for value in args.stroke_exclude_values}
    thresholds = [int(args.binary_good_threshold)] if args.binary_good_threshold is not None else [int(value) for value in args.binary_good_thresholds]
    thresholds = list(dict.fromkeys(thresholds))
    summary_rows = []

    for spec in endpoint_specs:
        for threshold in thresholds:
            run_spec = dict(spec)
            run_spec["base_label"] = spec["label"]
            run_spec["label"] = f"{spec['label']}_mrs0_{threshold}_vs_{threshold + 1}_6"
            try:
                data, qc = build_endpoint_dataset(features, phenotype, run_spec, clinical, candidates, exclude_values, threshold)
                endpoint_dir = ensure_dir(out_dir / run_spec["label"])
                write_csv(qc, endpoint_dir / "endpoint_qc.csv")
                write_csv(data[["subject_id", "endpoint_mrs", "binary_good", "stroke_recurrence", *clinical, *candidates]], endpoint_dir / "endpoint_analysis_table.csv")
                summary_rows.append(run_endpoint(data, run_spec, clinical, candidates, config, out_dir, args, threshold))
            except Exception as error:  # noqa: BLE001
                endpoint_dir = ensure_dir(out_dir / run_spec["label"])
                write_csv(pd.DataFrame([{"endpoint": run_spec["label"], "binary_good_threshold": threshold, "status": "failed", "error": str(error)}]), endpoint_dir / "endpoint_error.csv")
                summary_rows.append(
                    {
                        "endpoint": run_spec["label"],
                        "base_endpoint": spec["label"],
                        "binary_good_threshold": threshold,
                        "binary_target": f"mRS 0-{threshold} vs {threshold + 1}-6",
                        "n": 0,
                        "best_model": "",
                        "best_binary_log_loss": np.nan,
                        "best_auc": np.nan,
                        "status": f"failed: {error}",
                    }
                )
                print(f"failed {run_spec['label']}: {error}", flush=True)

    summary = pd.DataFrame(summary_rows)
    write_csv(summary, out_dir / "summary.csv")
    write_index_report(out_dir, summary)
    print(f"output_dir={out_dir}")
    print(f"index_html={out_dir / 'index.html'}")


if __name__ == "__main__":
    main()
