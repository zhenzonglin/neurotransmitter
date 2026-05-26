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

from nt_analysis.config import ensure_dir, load_config, project_path
from nt_analysis.stats import fit_mass_univariate
from nt_analysis.tables import write_csv


def load_phenotype(config: dict) -> pd.DataFrame:
    """Load prepared phenotype table."""
    path = project_path(config, config["outputs"]["qc_dir"], "subject_manifest.csv")
    return pd.read_csv(path)


def run_edge_clsm(config: dict, algorithm: str) -> None:
    """Run edge-wise mass-univariate CLSM on NeMo features."""
    edge_dir = ensure_dir(project_path(config, config["outputs"]["edge_dir"]))
    phenotype = load_phenotype(config).rename(columns={"mrs_3m": "outcome"})
    features = pd.read_csv(edge_dir / f"dat_edge_chaco_{algorithm}_66.csv")
    stats = fit_mass_univariate(features, phenotype, "outcome", ["age", "sex", "nihss", "lesion_volume_ml"])
    write_csv(stats, edge_dir / f"dat_edge_clsm_stats_{algorithm}.csv")

    # 保存矩阵形式结果
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
    pd.DataFrame(beta).to_csv(edge_dir / f"dat_edge_beta_matrix_{algorithm}.csv", index=False)
    pd.DataFrame(pval).to_csv(edge_dir / f"dat_edge_p_matrix_{algorithm}.csv", index=False)
    pd.DataFrame(qval).to_csv(edge_dir / f"dat_edge_q_matrix_{algorithm}.csv", index=False)


def run_lqt_edge_clsm(config: dict) -> None:
    """Run edge-wise CLSM on LQT DAT-weighted edge features."""
    edge_dir = ensure_dir(project_path(config, config["outputs"]["edge_dir"]))
    phenotype = load_phenotype(config).rename(columns={"mrs_3m": "outcome"})
    features = pd.read_csv(edge_dir / "dat_edge_lqt_66.csv")
    stats = fit_mass_univariate(features, phenotype, "outcome", ["age", "sex", "nihss", "lesion_volume_ml"])
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


def run_integrated_model(config: dict) -> None:
    """Run a compact elastic-net integrated model."""
    model_dir = ensure_dir(project_path(config, config["outputs"]["model_dir"]))
    phenotype = load_phenotype(config)
    node = pd.read_csv(project_path(config, config["outputs"]["node_dir"], "dat_node_damage_66x156.csv"))
    edge_path = project_path(config, config["outputs"]["edge_dir"], "dat_edge_lqt_66.csv")
    if not edge_path.exists():
        return
    edge = pd.read_csv(edge_path)
    data = phenotype.merge(node, on="subject_id").merge(edge, on="subject_id").dropna(subset=["mrs_3m"])
    data["mrs_binary"] = (data["mrs_3m"] <= 2).astype(int)
    feature_cols = [c for c in data.columns if c.startswith("node_") or c.startswith("edge_")]
    x = data[["age", "sex", "nihss", "lesion_volume_ml"] + feature_cols].fillna(0).to_numpy(dtype=float)
    y = data["mrs_binary"].to_numpy(dtype=int)
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
    perf = pd.DataFrame([{"model": "integrated_node_edge_lqt", "balanced_accuracy": balanced_accuracy_score(y, pred), "n": len(y)}])
    write_csv(perf, model_dir / "integrated_model_performance.csv")
    coef = pd.DataFrame({"feature": ["age", "sex", "nihss", "lesion_volume_ml"] + feature_cols, "coef": model.coef_[0]})
    write_csv(coef[coef["feature"].str.startswith("node_") & (coef["coef"] != 0)], model_dir / "selected_nodes.csv")
    write_csv(coef[coef["feature"].str.startswith("edge_") & (coef["coef"] != 0)], model_dir / "selected_edges.csv")
    score = pd.DataFrame({"subject_id": data["subject_id"], "dat_integrated_score": model.decision_function(x)})
    write_csv(score, model_dir / "dat_integrated_score.csv")


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect DAT NT-CLSM results.")
    parser.add_argument("--config", default="config/dat_config.yaml")
    args = parser.parse_args()
    config = load_config(args.config)

    node = pd.read_csv(project_path(config, config["outputs"]["node_dir"], "dat_node_damage_66x156.csv"))
    phenotype = load_phenotype(config).rename(columns={"mrs_3m": "outcome"})
    node_stats = fit_mass_univariate(node, phenotype, "outcome", ["age", "sex", "nihss", "lesion_volume_ml"])
    # Python模型只做探索性对照，主结果来自NiiStat
    write_csv(node_stats, project_path(config, config["outputs"]["node_dir"], "dat_node_python_ols_stats.csv"))
    lqt_path = project_path(config, config["outputs"]["edge_dir"], "dat_edge_lqt_66.csv")
    if lqt_path.exists():
        run_lqt_edge_clsm(config)
    for algorithm in ["ifod2act", "sdstream"]:
        path = project_path(config, config["outputs"]["edge_dir"], f"dat_edge_chaco_{algorithm}_66.csv")
        if path.exists():
            run_edge_clsm(config, algorithm)
    run_integrated_model(config)


if __name__ == "__main__":
    main()
