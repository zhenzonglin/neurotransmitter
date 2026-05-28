#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from jinja2 import Template

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nt_analysis.config import ensure_dir, load_config, project_path


plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 9,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)


palette = ["#0077BB", "#33BBEE", "#009988", "#EE7733", "#CC3311", "#EE3377", "#BBBBBB", "#000000"]
model_order = ["Clinical", "Clinical + SDC", "Clinical + NTDC", "Clinical + SDC + NTDC"]
ml_model_order = ["Clinical", "Clinical + SDC", "Clinical + ML-NTDC", "Clinical + SDC + ML-NTDC"]
metric_columns = [
    "ordinal_log_loss",
    "ranked_probability_score",
    "expected_mrs_mae",
    "ordinal_c_index",
    "binary_auc_mrs_le_threshold",
    "binary_brier_mrs_le_threshold",
]


def read_table(path: Path) -> pd.DataFrame:
    """读取结果表。"""
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def relpath(path: Path, base: Path) -> str:
    """生成HTML相对路径。"""
    return path.relative_to(base).as_posix()


def save_figure(fig: plt.Figure, path: Path) -> None:
    """保存图片并关闭对象。"""
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def fmt_float(value: float) -> str:
    """格式化小数。"""
    if pd.isna(value):
        return ""
    return f"{value:.4f}"


def table_html(df: pd.DataFrame, max_rows: int = 30, columns: Iterable[str] | None = None) -> str:
    """生成HTML表格。"""
    if df.empty:
        return "<p class=\"empty\">not available</p>"
    out = df.copy()
    if columns is not None:
        out = out[[column for column in columns if column in out.columns]]
    out = out.head(max_rows)
    return out.to_html(index=False, classes="data-table", border=0, float_format=fmt_float)


def ordered_categories(values: pd.Series, order: list[str]) -> pd.Series:
    """固定模型显示顺序。"""
    return pd.Categorical(values, categories=[item for item in order if item in set(values)], ordered=True)


def plot_cohort_qc(manifest: pd.DataFrame, lesion_qc: pd.DataFrame, output_path: Path) -> None:
    """绘制队列QC图。"""
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.2))

    if "mrs_3m" in manifest.columns:
        counts = manifest["mrs_3m"].dropna().astype(int).value_counts().sort_index()
        axes[0].bar(counts.index.astype(str), counts.values, color=palette[0])
        axes[0].set_xlabel("mRS at 3 months")
        axes[0].set_ylabel("Rows")
        axes[0].set_title("Outcome distribution")
    else:
        axes[0].axis("off")

    source = lesion_qc if not lesion_qc.empty else manifest
    if "lesion_volume_ml" in source.columns:
        axes[1].hist(source["lesion_volume_ml"].dropna(), bins=20, color=palette[2], edgecolor="white")
        axes[1].set_xlabel("Lesion volume (ml)")
        axes[1].set_ylabel("Rows")
        axes[1].set_title("Lesion volume")
    else:
        axes[1].axis("off")

    total_rows = len(manifest)
    group_col = "cv_group" if "cv_group" in manifest.columns else "subject_id"
    base_subjects = manifest[group_col].nunique()
    complete_cols = [column for column in ["mrs_3m", "age", "sex", "nihss", "lesion_volume_ml"] if column in manifest.columns]
    complete_rows = int(manifest[complete_cols].dropna().shape[0]) if complete_cols else total_rows
    axes[2].bar(["rows", "cv groups", "complete rows"], [total_rows, base_subjects, complete_rows], color=palette[:3])
    axes[2].set_ylabel("Count")
    axes[2].set_title("Analysis rows")
    axes[2].tick_params(axis="x", rotation=20)

    fig.tight_layout()
    save_figure(fig, output_path)


def plot_single_nt_heatmap(performance: pd.DataFrame, output_path: Path) -> None:
    """绘制单递质模型热图。"""
    if performance.empty:
        return
    data = performance.copy()
    data["model"] = ordered_categories(data["model"], model_order)
    pivot = data.pivot_table(index="nt_id", columns="model", values="ordinal_log_loss", observed=False)
    pivot = pivot[[column for column in model_order if column in pivot.columns]]

    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    image = ax.imshow(pivot.values, cmap="viridis", aspect="auto")
    ax.set_xticks(np.arange(pivot.shape[1]))
    ax.set_xticklabels(pivot.columns, rotation=35, ha="right")
    ax.set_yticks(np.arange(pivot.shape[0]))
    ax.set_yticklabels(pivot.index)
    ax.set_title("Single-NT model ordinal log loss")
    ax.set_xlabel("Model")
    ax.set_ylabel("Neurotransmitter")
    fig.colorbar(image, ax=ax, label="Ordinal log loss")
    fig.tight_layout()
    save_figure(fig, output_path)


def plot_single_nt_mean_metrics(performance: pd.DataFrame, output_path: Path) -> None:
    """绘制单递质平均指标。"""
    if performance.empty:
        return
    data = performance.copy()
    data["model"] = ordered_categories(data["model"], model_order)
    summary = data.groupby("model", observed=False)[metric_columns].mean().reset_index()
    summary = summary.dropna(subset=["model"])

    panels = [
        ("ordinal_log_loss", "Ordinal log loss"),
        ("ranked_probability_score", "RPS"),
        ("ordinal_c_index", "Ordinal C-index"),
        ("binary_auc_mrs_le_threshold", "Binary AUC"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(9, 6))
    for ax, (metric, title) in zip(axes.ravel(), panels):
        ax.bar(summary["model"].astype(str), summary[metric], color=palette[: len(summary)])
        ax.set_title(title)
        ax.set_ylabel(metric)
        ax.tick_params(axis="x", rotation=35)
    fig.tight_layout()
    save_figure(fig, output_path)


def plot_single_nt_delta_logloss(performance: pd.DataFrame, output_path: Path) -> None:
    """绘制单递质相对Clinical的差值。"""
    if performance.empty:
        return
    clinical = performance.loc[performance["model"] == "Clinical", ["nt_id", "ordinal_log_loss"]].rename(
        columns={"ordinal_log_loss": "clinical_log_loss"}
    )
    data = performance.merge(clinical, on="nt_id", how="left")
    data = data[data["model"] != "Clinical"].copy()
    data["delta_log_loss"] = data["ordinal_log_loss"] - data["clinical_log_loss"]
    data["model"] = ordered_categories(data["model"], model_order)
    data = data.sort_values(["model", "nt_id"])

    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    positions = np.arange(data["nt_id"].nunique())
    nt_ids = sorted(data["nt_id"].unique())
    width = 0.24
    models = [model for model in model_order if model != "Clinical" and model in set(data["model"].astype(str))]
    for idx, model in enumerate(models):
        subset = data[data["model"].astype(str) == model].set_index("nt_id").reindex(nt_ids)
        ax.bar(positions + (idx - 1) * width, subset["delta_log_loss"], width=width, label=model, color=palette[idx])
    ax.axhline(0, color="#444444", linewidth=0.8)
    ax.set_xticks(positions)
    ax.set_xticklabels(nt_ids, rotation=35, ha="right")
    ax.set_ylabel("Delta ordinal log loss")
    ax.set_xlabel("Neurotransmitter")
    ax.set_title("Single-NT delta vs Clinical")
    ax.legend(frameon=False)
    fig.tight_layout()
    save_figure(fig, output_path)


def plot_ml_selection(selection: pd.DataFrame, output_path: Path) -> None:
    """绘制机器学习筛选频率。"""
    if selection.empty:
        return
    data = selection.sort_values(["selection_frequency", "mean_abs_coef"], ascending=[True, True])
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.barh(data["nt_id"], data["selection_frequency"], color=palette[0])
    ax.set_xlabel("Selection frequency")
    ax.set_ylabel("Neurotransmitter")
    ax.set_title("ML-NTDC selection frequency")
    ax.set_xlim(0, max(1.0, float(data["selection_frequency"].max()) * 1.05))
    fig.tight_layout()
    save_figure(fig, output_path)


def plot_ml_model_performance(performance: pd.DataFrame, output_path: Path) -> None:
    """绘制ML模型指标。"""
    if performance.empty:
        return
    data = performance.copy()
    data["model"] = ordered_categories(data["model"], ml_model_order)
    data = data.sort_values("model")
    panels = [
        ("ordinal_log_loss", "Ordinal log loss"),
        ("ranked_probability_score", "RPS"),
        ("ordinal_c_index", "Ordinal C-index"),
        ("binary_auc_mrs_le_threshold", "Binary AUC"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(9, 6))
    for ax, (metric, title) in zip(axes.ravel(), panels):
        ax.bar(data["model"].astype(str), data[metric], color=palette[: len(data)])
        ax.set_title(title)
        ax.set_ylabel(metric)
        ax.tick_params(axis="x", rotation=35)
    fig.tight_layout()
    save_figure(fig, output_path)


def plot_ml_pairwise(pairwise: pd.DataFrame, output_path: Path) -> None:
    """绘制主要指标配对差值。"""
    if pairwise.empty:
        return
    metric_column = "metric" if "metric" in pairwise.columns else "metric_name"
    data = pairwise[pairwise[metric_column] == "ordinal_log_loss"].copy()
    if data.empty:
        return
    data["comparison"] = data["model_b"] + " vs " + data["model_a"]
    delta_column = "delta_mean" if "delta_mean" in data.columns else "delta_b_minus_a"
    data = data.sort_values(delta_column)

    fig, ax = plt.subplots(figsize=(7, max(3.2, 0.35 * len(data))))
    y = np.arange(len(data))
    xerr_low = data[delta_column] - data["ci_low"]
    xerr_high = data["ci_high"] - data[delta_column]
    ax.errorbar(data[delta_column], y, xerr=[xerr_low, xerr_high], fmt="o", color=palette[3], ecolor="#777777")
    ax.axvline(0, color="#444444", linewidth=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(data["comparison"])
    ax.set_xlabel("Delta ordinal log loss")
    ax.set_title("ML model pairwise bootstrap")
    fig.tight_layout()
    save_figure(fig, output_path)


def build_file_index(project_dir: Path) -> pd.DataFrame:
    """生成报告文件索引。"""
    patterns = [
        "derivatives/qc/*.csv",
        "derivatives/shared/*.csv",
        "derivatives/nt/summary/*.csv",
        "derivatives/nt_ml/ml_ntdc/*.csv",
        "derivatives/nt_ml/ml_ntdc/models/*.csv",
        "derivatives/reports/*.html",
        "derivatives/reports/*.csv",
        "derivatives/reports/figures/*.png",
    ]
    rows = []
    for pattern in patterns:
        for path in sorted(project_dir.glob(pattern)):
            if path.is_file():
                rows.append(
                    {
                        "path": path.relative_to(project_dir).as_posix(),
                        "size_kb": round(path.stat().st_size / 1024, 1),
                        "modified": datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d %H:%M"),
                    }
                )
    return pd.DataFrame(rows)


def make_shape_table(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """记录核心表维度。"""
    rows = []
    for name, table in tables.items():
        rows.append({"table": name, "rows": int(table.shape[0]), "columns": int(table.shape[1])})
    return pd.DataFrame(rows)


def render_html(context: dict[str, object], output_path: Path) -> None:
    """渲染HTML报告。"""
    template = Template(
        """<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>NT-CLSM Flow Report</title>
  <style>
    body { margin: 0; font-family: Arial, "Microsoft YaHei", sans-serif; color: #17212b; background: #f5f7fa; }
    header { padding: 28px 36px; background: #ffffff; border-bottom: 1px solid #d9e1ea; }
    h1 { margin: 0 0 8px; font-size: 26px; }
    h2 { margin: 32px 0 12px; font-size: 19px; }
    h3 { margin: 22px 0 10px; font-size: 15px; }
    main { max-width: 1180px; margin: 0 auto; padding: 24px 28px 48px; }
    .meta { color: #526171; font-size: 13px; }
    .card { background: #ffffff; border: 1px solid #d9e1ea; border-radius: 8px; padding: 18px; margin: 16px 0; }
    .flow { display: grid; grid-template-columns: repeat(5, 1fr); gap: 10px; align-items: stretch; }
    .step { border: 1px solid #cfd9e3; border-radius: 8px; padding: 12px; background: #fbfcfe; min-height: 78px; }
    .step b { display: block; margin-bottom: 6px; font-size: 13px; color: #0f4068; }
    .step span { display: block; font-size: 12px; color: #495867; line-height: 1.45; }
    .figure-grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 18px; }
    figure { margin: 0; }
    figure img { width: 100%; background: #ffffff; border: 1px solid #d9e1ea; border-radius: 6px; }
    figcaption { margin-top: 6px; color: #526171; font-size: 12px; }
    .table-wrap { overflow-x: auto; border: 1px solid #d9e1ea; border-radius: 6px; }
    table.data-table { border-collapse: collapse; width: 100%; font-size: 12px; background: #ffffff; }
    table.data-table th, table.data-table td { padding: 7px 9px; border-bottom: 1px solid #edf1f5; text-align: left; white-space: nowrap; }
    table.data-table th { background: #eef3f8; color: #1c3346; }
    .empty { color: #7c8792; font-size: 12px; }
    @media (max-width: 900px) {
      .flow { grid-template-columns: 1fr; }
      .figure-grid { grid-template-columns: 1fr; }
      main { padding: 18px; }
    }
  </style>
</head>
<body>
  <header>
    <h1>NT-CLSM Flow Report</h1>
    <div class="meta">project: {{ project_dir }} | generated: {{ generated_at }}</div>
  </header>
  <main>
    <section class="card">
      <h2>1. 运行流程</h2>
      <div class="flow">
        <div class="step"><b>Input</b><span>MNI 2mm lesion mask<br>phenotype table<br>atlas and NT maps</span></div>
        <div class="step"><b>Feature</b><span>lesion node load<br>LQT edge disconnection<br>NT node and edge damage</span></div>
        <div class="step"><b>LSM</b><span>node-level NT-LSM<br>edge-level NT-LSM<br>impact weights</span></div>
        <div class="step"><b>Prediction</b><span>10-fold out-of-sample prediction<br>Clinical, SDC, NTDC, ML-NTDC</span></div>
        <div class="step"><b>Output</b><span>metrics<br>bootstrap pairwise comparison<br>tables and figures</span></div>
      </div>
    </section>

    <section class="card">
      <h2>2. 输入与QC</h2>
      <div class="figure-grid">
        <figure>
          <img src="{{ figures.cohort_qc }}" alt="cohort qc">
          <figcaption>Figure 1. Cohort and lesion QC.</figcaption>
        </figure>
      </div>
      <h3>核心表维度</h3>
      <div class="table-wrap">{{ shape_table }}</div>
      <h3>subject manifest</h3>
      <div class="table-wrap">{{ manifest_table }}</div>
    </section>

    <section class="card">
      <h2>3. 单递质NTDC分析</h2>
      <div class="figure-grid">
        <figure>
          <img src="{{ figures.single_nt_heatmap }}" alt="single nt heatmap">
          <figcaption>Figure 2. Single-NT model ordinal log loss.</figcaption>
        </figure>
        <figure>
          <img src="{{ figures.single_nt_mean_metrics }}" alt="single nt metrics">
          <figcaption>Figure 3. Single-NT mean prediction metrics.</figcaption>
        </figure>
        <figure>
          <img src="{{ figures.single_nt_delta_logloss }}" alt="single nt delta">
          <figcaption>Figure 4. Single-NT delta ordinal log loss vs Clinical.</figcaption>
        </figure>
      </div>
      <h3>prediction performance</h3>
      <div class="table-wrap">{{ nt_performance_table }}</div>
      <h3>bootstrap comparison vs Clinical</h3>
      <div class="table-wrap">{{ nt_bootstrap_table }}</div>
    </section>

    <section class="card">
      <h2>4. ML-NTDC筛选与预测</h2>
      <div class="figure-grid">
        <figure>
          <img src="{{ figures.ml_selection_frequency }}" alt="ml selection">
          <figcaption>Figure 5. ML-NTDC selection frequency.</figcaption>
        </figure>
        <figure>
          <img src="{{ figures.ml_model_performance }}" alt="ml performance">
          <figcaption>Figure 6. ML-NTDC model prediction metrics.</figcaption>
        </figure>
        <figure>
          <img src="{{ figures.ml_pairwise_delta }}" alt="ml pairwise">
          <figcaption>Figure 7. ML model pairwise bootstrap for ordinal log loss.</figcaption>
        </figure>
      </div>
      <h3>selection summary</h3>
      <div class="table-wrap">{{ ml_selection_table }}</div>
      <h3>prediction performance</h3>
      <div class="table-wrap">{{ ml_performance_table }}</div>
      <h3>pairwise bootstrap</h3>
      <div class="table-wrap">{{ ml_pairwise_table }}</div>
    </section>

    <section class="card">
      <h2>5. 输出文件索引</h2>
      <div class="table-wrap">{{ file_index_table }}</div>
    </section>
  </main>
</body>
</html>"""
    )
    output_path.write_text(template.render(**context), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="Generate NT-CLSM flow HTML report.")
    parser.add_argument("--config", default="config/dat_config.yaml", help="Path to the YAML configuration file.")
    return parser.parse_args()


def main() -> None:
    """生成流程化报告。"""
    args = parse_args()
    config = load_config(args.config)
    project_dir = Path(config["project_dir"])
    report_dir = ensure_dir(project_path(config, "derivatives", "reports"))
    figure_dir = ensure_dir(report_dir / "figures")

    tables = {
        "subject_manifest": read_table(project_path(config, "derivatives", "qc", "subject_manifest.csv")),
        "lesion_qc": read_table(project_path(config, "derivatives", "qc", "lesion_qc.csv")),
        "edge_tract_voxels_qc": read_table(project_path(config, "derivatives", "shared", "edge_tract_voxels_2mm_qc.csv")),
        "nt_run_manifest": read_table(project_path(config, "derivatives", "nt", "summary", "nt_run_manifest.csv")),
        "nt_prediction_performance": read_table(project_path(config, "derivatives", "nt", "summary", "nt_prediction_performance.csv")),
        "nt_prediction_vs_clinical_bootstrap": read_table(
            project_path(config, "derivatives", "nt", "summary", "nt_prediction_vs_clinical_bootstrap.csv")
        ),
        "ml_ntdc_selection_summary": read_table(project_path(config, "derivatives", "nt_ml", "ml_ntdc", "ml_ntdc_selection_summary.csv")),
        "ml_prediction_performance": read_table(
            project_path(config, "derivatives", "nt_ml", "ml_ntdc", "models", "model_prediction_performance.csv")
        ),
        "ml_pairwise_bootstrap": read_table(
            project_path(config, "derivatives", "nt_ml", "ml_ntdc", "models", "model_prediction_pairwise_bootstrap.csv")
        ),
    }

    figure_paths = {
        "cohort_qc": figure_dir / "cohort_qc.png",
        "single_nt_heatmap": figure_dir / "single_nt_logloss_heatmap.png",
        "single_nt_mean_metrics": figure_dir / "single_nt_metric_bars.png",
        "single_nt_delta_logloss": figure_dir / "single_nt_delta_logloss.png",
        "ml_selection_frequency": figure_dir / "ml_selection_frequency.png",
        "ml_model_performance": figure_dir / "ml_model_performance.png",
        "ml_pairwise_delta": figure_dir / "ml_pairwise_delta.png",
    }

    plot_cohort_qc(tables["subject_manifest"], tables["lesion_qc"], figure_paths["cohort_qc"])
    plot_single_nt_heatmap(tables["nt_prediction_performance"], figure_paths["single_nt_heatmap"])
    plot_single_nt_mean_metrics(tables["nt_prediction_performance"], figure_paths["single_nt_mean_metrics"])
    plot_single_nt_delta_logloss(tables["nt_prediction_performance"], figure_paths["single_nt_delta_logloss"])
    plot_ml_selection(tables["ml_ntdc_selection_summary"], figure_paths["ml_selection_frequency"])
    plot_ml_model_performance(tables["ml_prediction_performance"], figure_paths["ml_model_performance"])
    plot_ml_pairwise(tables["ml_pairwise_bootstrap"], figure_paths["ml_pairwise_delta"])

    file_index = build_file_index(project_dir)
    file_index_path = report_dir / "report_file_index.csv"
    file_index.to_csv(file_index_path, index=False)
    tables["report_file_index"] = file_index

    figures = {key: relpath(path, report_dir) for key, path in figure_paths.items()}
    html_path = report_dir / "nt_clsm_flow_report.html"
    context = {
        "project_dir": project_dir.as_posix(),
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "figures": figures,
        "shape_table": table_html(make_shape_table(tables), max_rows=20),
        "manifest_table": table_html(tables["subject_manifest"], max_rows=12),
        "nt_performance_table": table_html(tables["nt_prediction_performance"], max_rows=24),
        "nt_bootstrap_table": table_html(tables["nt_prediction_vs_clinical_bootstrap"], max_rows=24),
        "ml_selection_table": table_html(tables["ml_ntdc_selection_summary"], max_rows=20),
        "ml_performance_table": table_html(tables["ml_prediction_performance"], max_rows=20),
        "ml_pairwise_table": table_html(tables["ml_pairwise_bootstrap"], max_rows=30),
        "file_index_table": table_html(file_index, max_rows=200),
    }
    render_html(context, html_path)
    print(f"report_html={html_path}")
    print(f"report_file_index={file_index_path}")


if __name__ == "__main__":
    main()
