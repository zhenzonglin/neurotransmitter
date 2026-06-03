#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from jinja2 import Template

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nt_analysis.config import ensure_dir, load_config, project_path  # noqa: E402


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


def read_table(path: Path) -> pd.DataFrame:
    """读取结果表。"""
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def fmt_float(value: float) -> str:
    """格式化小数。"""
    if pd.isna(value):
        return ""
    return f"{value:.4f}"


def table_html(df: pd.DataFrame, max_rows: int = 40) -> str:
    """生成HTML表格。"""
    if df.empty:
        return "<p class=\"empty\">not available</p>"
    return df.head(max_rows).to_html(index=False, classes="data-table", border=0, float_format=fmt_float)


def save_figure(fig: plt.Figure, path: Path) -> None:
    """保存图片。"""
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_performance(performance: pd.DataFrame, output: Path) -> None:
    """绘制模型性能。"""
    if performance.empty:
        return
    panels = [
        ("ordinal_log_loss", "Ordinal log loss"),
        ("ranked_probability_score", "RPS"),
        ("ordinal_c_index", "Ordinal C-index"),
        ("binary_auc_mrs_le_threshold", "Binary AUC"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    colors = [palette[index % len(palette)] for index in range(performance.shape[0])]
    for ax, (metric, title) in zip(axes.ravel(), panels):
        ax.bar(performance["model"], performance[metric], color=colors)
        ax.set_title(title)
        ax.set_ylabel(metric)
        ax.tick_params(axis="x", rotation=35)
    fig.tight_layout()
    save_figure(fig, output)


def roc_points(y_true: np.ndarray, scores: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """计算ROC坐标。"""
    if y_true.size == 0:
        return np.array([]), np.array([]), np.nan
    order = np.argsort(-scores)
    y_sorted = y_true[order].astype(int)
    score_sorted = scores[order]
    distinct = np.r_[np.where(np.diff(score_sorted))[0], y_sorted.size - 1]
    tp = np.cumsum(y_sorted)[distinct]
    fp = (1 + distinct) - tp
    positives = int(y_sorted.sum())
    negatives = int(y_sorted.size - positives)
    if positives == 0 or negatives == 0:
        return np.array([]), np.array([]), np.nan
    tpr = np.r_[0.0, tp / positives, 1.0]
    fpr = np.r_[0.0, fp / negatives, 1.0]
    auc = float(np.trapz(tpr, fpr))
    return fpr, tpr, auc


def plot_roc(predictions: pd.DataFrame, config: dict, output: Path) -> pd.DataFrame:
    """绘制ROC曲线。"""
    binary_cfg = config.get("analysis", {}).get("binary_outcome", {})
    threshold = float(binary_cfg.get("threshold", 2))
    positive_if_less_equal = bool(binary_cfg.get("positive_if_less_equal", True))
    required = {"model", "observed_mrs", "prob_mrs_le_threshold"}
    if predictions.empty or not required.issubset(predictions.columns):
        return pd.DataFrame()

    rows = []
    fig, ax = plt.subplots(figsize=(6.2, 5.3))
    for index, model in enumerate(predictions["model"].drop_duplicates().tolist()):
        data = predictions[predictions["model"] == model].dropna(subset=["observed_mrs", "prob_mrs_le_threshold"]).copy()
        observed = pd.to_numeric(data["observed_mrs"], errors="coerce").to_numpy(dtype=float)
        prob_good = pd.to_numeric(data["prob_mrs_le_threshold"], errors="coerce").to_numpy(dtype=float)
        ok = np.isfinite(observed) & np.isfinite(prob_good)
        observed = observed[ok]
        prob_good = prob_good[ok]
        if observed.size == 0:
            continue
        if positive_if_less_equal:
            y_true = (observed <= threshold).astype(int)
            scores = prob_good
            target = f"mRS <= {threshold:g}"
        else:
            y_true = (observed > threshold).astype(int)
            scores = 1.0 - prob_good
            target = f"mRS > {threshold:g}"
        fpr, tpr, auc = roc_points(y_true, scores)
        rows.append({"model": model, "n": int(y_true.size), "positive_target": target, "positive_n": int(y_true.sum()), "negative_n": int(y_true.size - y_true.sum()), "auc": auc})
        if fpr.size:
            ax.plot(fpr, tpr, color=palette[index % len(palette)], linewidth=1.8, label=f"{model} (AUC={auc:.3f})")

    ax.plot([0, 1], [0, 1], color="#777777", linewidth=0.9, linestyle="--")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("Component Model ROC")
    if rows:
        ax.legend(loc="lower right", frameon=False)
    fig.tight_layout()
    save_figure(fig, output)
    return pd.DataFrame(rows)


def plot_pairwise(pairwise: pd.DataFrame, output: Path) -> None:
    """绘制ordinal log loss差异。"""
    if pairwise.empty:
        return
    data = pairwise[pairwise["metric"] == "ordinal_log_loss"].copy()
    if data.empty:
        return
    data["comparison"] = data["model_b"] + " vs " + data["model_a"]
    data = data.sort_values("delta_b_minus_a")
    fig, ax = plt.subplots(figsize=(8, max(3.5, 0.38 * len(data))))
    y = np.arange(len(data))
    xerr_low = data["delta_b_minus_a"] - data["ci_low"]
    xerr_high = data["ci_high"] - data["delta_b_minus_a"]
    ax.errorbar(data["delta_b_minus_a"], y, xerr=[xerr_low, xerr_high], fmt="o", color=palette[4], ecolor="#777777")
    ax.axvline(0, color="#444444", linewidth=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(data["comparison"])
    ax.set_xlabel("Delta ordinal log loss")
    ax.set_title("Pairwise bootstrap")
    fig.tight_layout()
    save_figure(fig, output)


def build_file_index(project_dir: Path, component_subdir: str, report_subdir: str) -> pd.DataFrame:
    """生成文件索引。"""
    patterns = [
        f"derivatives/{component_subdir}/*.csv",
        f"derivatives/{report_subdir}/*.html",
        f"derivatives/{report_subdir}/*.csv",
        f"derivatives/{report_subdir}/figures/*.png",
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


def render_html(context: dict[str, object], output: Path) -> None:
    """渲染HTML。"""
    template = Template(
        """<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Component NTDC Report</title>
  <style>
    body { margin: 0; font-family: Arial, "Microsoft YaHei", sans-serif; color: #17212b; background: #f5f7fa; }
    header { padding: 28px 36px; background: #ffffff; border-bottom: 1px solid #d9e1ea; }
    h1 { margin: 0 0 8px; font-size: 26px; }
    h2 { margin: 30px 0 12px; font-size: 19px; }
    h3 { margin: 22px 0 10px; font-size: 15px; }
    main { max-width: 1240px; margin: 0 auto; padding: 24px 28px 48px; }
    .meta { color: #526171; font-size: 13px; }
    .card { background: #ffffff; border: 1px solid #d9e1ea; border-radius: 8px; padding: 18px; margin: 16px 0; }
    .flow { display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px; align-items: stretch; }
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
    @media (max-width: 900px) { .flow { grid-template-columns: 1fr; } .figure-grid { grid-template-columns: 1fr; } main { padding: 18px; } }
  </style>
</head>
<body>
  <header>
    <h1>Component NTDC Report</h1>
    <div class="meta">project: {{ project_dir }} | component: {{ component_subdir }} | generated: {{ generated_at }}</div>
  </header>
  <main>
    <section class="card">
      <h2>1. Analysis Flow</h2>
      <div class="flow">
        <div class="step"><b>Input</b><span>completed NTDC profiles<br>fold LSM weights<br>profile_scores.csv</span></div>
        <div class="step"><b>Components</b><span>lesion node / edge<br>NT node / edge<br>train-fold z scores</span></div>
        <div class="step"><b>Interactions</b><span>NTDC x NIHSS<br>NTDC x lesion volume<br>component interactions</span></div>
        <div class="step"><b>Prediction</b><span>10-fold out-of-sample<br>bootstrap comparison</span></div>
      </div>
    </section>
    <section class="card">
      <h2>2. Prediction Models</h2>
      <div class="figure-grid">
        <figure><img src="{{ figures.performance }}" alt="performance"><figcaption>Figure 1. Component model prediction metrics.</figcaption></figure>
        <figure><img src="{{ figures.roc }}" alt="roc"><figcaption>Figure 2. Out-of-sample ROC curves.</figcaption></figure>
        <figure><img src="{{ figures.pairwise }}" alt="pairwise"><figcaption>Figure 3. Pairwise bootstrap for ordinal log loss.</figcaption></figure>
      </div>
      <h3>prediction performance</h3>
      <div class="table-wrap">{{ performance_table }}</div>
      <h3>ROC AUC</h3>
      <div class="table-wrap">{{ roc_table }}</div>
      <h3>pairwise bootstrap</h3>
      <div class="table-wrap">{{ pairwise_table }}</div>
      <h3>model status</h3>
      <div class="table-wrap">{{ status_table }}</div>
    </section>
    <section class="card">
      <h2>3. Component Scores</h2>
      <div class="table-wrap">{{ score_table }}</div>
    </section>
    <section class="card">
      <h2>4. Output File Index</h2>
      <div class="table-wrap">{{ file_index_table }}</div>
    </section>
  </main>
</body>
</html>"""
    )
    output.write_text(template.render(**context), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="Generate component-wise NTDC report.")
    parser.add_argument("--config", default="config/dat_config.yaml")
    parser.add_argument("--component-subdir", default="component_prediction")
    parser.add_argument("--report-subdir", default="reports_component_prediction")
    return parser.parse_args()


def main() -> None:
    """生成component报告。"""
    args = parse_args()
    config = load_config(args.config)
    project_dir = Path(config["project_dir"])
    component_dir = project_path(config, "derivatives", args.component_subdir)
    report_dir = ensure_dir(project_path(config, "derivatives", args.report_subdir))
    figure_dir = ensure_dir(report_dir / "figures")

    performance = read_table(component_dir / "component_model_prediction_performance.csv")
    predictions = read_table(component_dir / "component_model_prediction_cv.csv")
    pairwise = read_table(component_dir / "component_model_prediction_pairwise_bootstrap.csv")
    status = read_table(component_dir / "component_model_prediction_fold_status.csv")
    scores = read_table(component_dir / "component_scores.csv")

    figures = {
        "performance": figure_dir / "component_model_performance.png",
        "roc": figure_dir / "component_model_roc.png",
        "pairwise": figure_dir / "component_model_pairwise_delta.png",
    }
    plot_performance(performance, figures["performance"])
    roc_metrics = plot_roc(predictions, config, figures["roc"])
    plot_pairwise(pairwise, figures["pairwise"])
    roc_metrics_path = report_dir / "component_model_roc_auc.csv"
    roc_metrics.to_csv(roc_metrics_path, index=False)
    file_index = build_file_index(project_dir, args.component_subdir, args.report_subdir)
    file_index_path = report_dir / "component_report_file_index.csv"
    file_index.to_csv(file_index_path, index=False)

    html_path = report_dir / "component_ntdc_report.html"
    render_html(
        {
            "project_dir": project_dir.as_posix(),
            "component_subdir": args.component_subdir,
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "figures": {key: path.relative_to(report_dir).as_posix() for key, path in figures.items()},
            "performance_table": table_html(performance, 30),
            "roc_table": table_html(roc_metrics, 30),
            "pairwise_table": table_html(pairwise, 80),
            "status_table": table_html(status, 60),
            "score_table": table_html(scores, 25),
            "file_index_table": table_html(file_index, 200),
        },
        html_path,
    )
    print(f"report_html={html_path}")
    print(f"report_file_index={file_index_path}")


if __name__ == "__main__":
    main()
