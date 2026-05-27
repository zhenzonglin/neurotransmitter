#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nt_analysis.config import analysis_covariates, analysis_table, ensure_dir, load_config, outcome_column, project_path, require_columns
from nt_analysis.stats import fit_mass_univariate_fast
from nt_analysis.tables import write_csv


def load_phenotype(config: dict) -> pd.DataFrame:
    """Load prepared phenotype table."""
    path = project_path(config, config["outputs"]["qc_dir"], "subject_manifest.csv")
    return pd.read_csv(path)


def run_lqt_edge_clsm(config: dict) -> None:
    """Run edge-wise CLSM on LQT DAT-weighted edge features."""
    edge_dir = ensure_dir(project_path(config, config["outputs"]["edge_dir"]))
    outcome = outcome_column(config)
    covariates = analysis_covariates(config, "model_covariates")
    phenotype = load_phenotype(config)
    features = pd.read_csv(edge_dir / analysis_table(config, "dat_edge_lqt", "dat_edge_lqt.csv"))
    stats = fit_mass_univariate_fast(features, phenotype, outcome, covariates)
    write_csv(stats, edge_dir / "dat_edge_clsm_stats_lqt.csv")

    edge_count = features.shape[1] - 1
    roi_count = int((1 + np.sqrt(1 + 8 * edge_count)) / 2)
    beta = np.zeros((roi_count, roi_count), dtype=float)
    pval = np.ones((roi_count, roi_count), dtype=float)
    qval = np.ones((roi_count, roi_count), dtype=float)
    for row in stats.itertuples(index=False):
        _, left, right = row.feature.split("_")
        i = int(left) - 1
        j = int(right) - 1
        beta[i, j] = row.beta
        pval[i, j] = row.p
        qval[i, j] = row.q
    pd.DataFrame(beta).to_csv(edge_dir / "dat_edge_beta_matrix_lqt.csv", index=False)
    pd.DataFrame(pval).to_csv(edge_dir / "dat_edge_p_matrix_lqt.csv", index=False)
    pd.DataFrame(qval).to_csv(edge_dir / "dat_edge_q_matrix_lqt.csv", index=False)


def select_model_features(stats: pd.DataFrame, prefix: str, top_k: int, q_threshold: float) -> list[str]:
    """Select compact features for the integrated model."""
    values = stats[stats["feature"].str.startswith(prefix)].copy()
    if values.empty:
        return []
    selected = values.loc[values["q"].fillna(1.0) <= q_threshold, "feature"].tolist()
    if not selected:
        selected = values.reindex(values["t"].abs().sort_values(ascending=False).index)["feature"].head(top_k).tolist()
    return selected


def run_integrated_model(config: dict) -> None:
    """Run a compact elastic-net integrated model."""
    model_dir = ensure_dir(project_path(config, config["outputs"]["model_dir"]))
    outcome = outcome_column(config)
    covariates = analysis_covariates(config, "model_covariates")
    binary = config.get("analysis", {}).get("binary_outcome", {})
    threshold = float(binary.get("threshold", 2))
    positive_if_less_equal = bool(binary.get("positive_if_less_equal", True))
    phenotype = load_phenotype(config)
    node = pd.read_csv(project_path(config, config["outputs"]["node_dir"], analysis_table(config, "node_damage", "dat_node_damage.csv")))
    edge_path = project_path(config, config["outputs"]["edge_dir"], analysis_table(config, "dat_edge_lqt", "dat_edge_lqt.csv"))
    if not edge_path.exists():
        return
    edge = pd.read_csv(edge_path)
    require_columns([outcome, *covariates], list(phenotype.columns), "phenotype")
    data = phenotype.merge(node, on="subject_id").merge(edge, on="subject_id").dropna(subset=[outcome])
    if positive_if_less_equal:
        data["binary_outcome"] = (data[outcome] <= threshold).astype(int)
    else:
        data["binary_outcome"] = (data[outcome] >= threshold).astype(int)
    node_stats_path = project_path(config, config["outputs"]["node_dir"], "dat_node_python_ols_stats.csv")
    edge_stats_path = project_path(config, config["outputs"]["edge_dir"], "dat_edge_clsm_stats_lqt.csv")
    node_stats = pd.read_csv(node_stats_path) if node_stats_path.exists() else fit_mass_univariate_fast(node, phenotype, outcome, covariates)
    edge_stats = pd.read_csv(edge_stats_path) if edge_stats_path.exists() else fit_mass_univariate_fast(edge, phenotype, outcome, covariates)
    impact_cfg = config.get("impact", {})
    q_threshold = float(impact_cfg.get("q_threshold", 0.05))
    node_top_k = int(impact_cfg.get("node_top_k", 20))
    edge_top_k = int(impact_cfg.get("edge_top_k", 200))
    # 高维模型先降到关键节点和边，避免测试流程被12090条边拖慢
    feature_cols = select_model_features(node_stats, "node_", node_top_k, q_threshold)
    feature_cols += select_model_features(edge_stats, "edge_", edge_top_k, q_threshold)
    feature_cols = [col for col in feature_cols if col in data.columns]
    if not feature_cols:
        return
    x = data[covariates + feature_cols].fillna(0).to_numpy(dtype=float)
    y = data["binary_outcome"].to_numpy(dtype=int)
    if len(np.unique(y)) < 2:
        return
    cv = StratifiedKFold(n_splits=min(5, np.bincount(y).min()), shuffle=True, random_state=42)
    model = LogisticRegressionCV(
        penalty="elasticnet",
        solver="saga",
        l1_ratios=[0.5],
        Cs=5,
        cv=cv,
        max_iter=5000,
        scoring="balanced_accuracy",
        n_jobs=1,
    )
    pred = cross_val_predict(model, x, y, cv=cv)
    model.fit(x, y)
    perf = pd.DataFrame(
        [
            {
                "model": "integrated_node_edge_lqt",
                "balanced_accuracy": balanced_accuracy_score(y, pred),
                "n": len(y),
                "n_features": len(feature_cols),
            }
        ]
    )
    write_csv(perf, model_dir / "integrated_model_performance.csv")
    coef = pd.DataFrame({"feature": covariates + feature_cols, "coef": model.coef_[0]})
    write_csv(coef[coef["feature"].str.startswith("node_") & (coef["coef"] != 0)], model_dir / "selected_nodes.csv")
    write_csv(coef[coef["feature"].str.startswith("edge_") & (coef["coef"] != 0)], model_dir / "selected_edges.csv")
    score = pd.DataFrame({"subject_id": data["subject_id"], "dat_integrated_score": model.decision_function(x)})
    write_csv(score, model_dir / "dat_integrated_score.csv")


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect DAT NT-CLSM results.")
    parser.add_argument("--config", default="config/dat_config.yaml")
    args = parser.parse_args()
    config = load_config(args.config)

    outcome = outcome_column(config)
    covariates = analysis_covariates(config, "model_covariates")
    node = pd.read_csv(project_path(config, config["outputs"]["node_dir"], analysis_table(config, "node_damage", "dat_node_damage.csv")))
    phenotype = load_phenotype(config)
    node_stats = fit_mass_univariate_fast(node, phenotype, outcome, covariates)
    # Python模型只做探索性对照，主结果来自NiiStat
    write_csv(node_stats, project_path(config, config["outputs"]["node_dir"], "dat_node_python_ols_stats.csv"))
    lqt_path = project_path(config, config["outputs"]["edge_dir"], analysis_table(config, "dat_edge_lqt", "dat_edge_lqt.csv"))
    if lqt_path.exists():
        run_lqt_edge_clsm(config)
    run_integrated_model(config)


if __name__ == "__main__":
    main()
