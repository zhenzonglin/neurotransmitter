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


def table_html(df: pd.DataFrame, max_rows: int = 30) -> str:
    """生成HTML表格。"""
    if df.empty:
        return "<p class=\"empty\">not available</p>"
    return df.head(max_rows).to_html(index=False, classes="data-table", border=0, float_format=fmt_float)


def save_figure(fig: plt.Figure, path: Path) -> None:
    """保存图片。"""
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_cohort_qc(manifest: pd.DataFrame, lesion_qc: pd.DataFrame, output: Path) -> None:
    """绘制队列QC。"""
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.2))
    if "mrs_3m" in manifest.columns:
        counts = manifest["mrs_3m"].dropna().astype(int).value_counts().sort_index()
        axes[0].bar(counts.index.astype(str), counts.values, color=palette[0])
        axes[0].set_xlabel("mRS at 3 months")
        axes[0].set_ylabel("Subjects")
        axes[0].set_title("Outcome distribution")
    else:
        axes[0].axis("off")

    source = lesion_qc if not lesion_qc.empty else manifest
    if "lesion_volume_ml" in source.columns:
        axes[1].hist(source["lesion_volume_ml"].dropna(), bins=24, color=palette[2], edgecolor="white")
        axes[1].set_xlabel("Lesion volume (ml)")
        axes[1].set_ylabel("Subjects")
        axes[1].set_title("Lesion volume")
    else:
        axes[1].axis("off")

    raw_rows = len(lesion_qc) if not lesion_qc.empty else len(manifest)
    if "included_by_lesion_qc" in lesion_qc.columns:
        included_rows = int(lesion_qc["included_by_lesion_qc"].astype(bool).sum())
    else:
        included_rows = len(manifest)
    if "is_empty_lesion" in lesion_qc.columns:
        empty_rows = int(lesion_qc["is_empty_lesion"].astype(bool).sum())
    else:
        empty_rows = 0
    complete_cols = [column for column in ["mrs_3m", "age", "sex", "nihss", "lesion_volume_ml"] if column in manifest.columns]
    complete_rows = int(manifest[complete_cols].dropna().shape[0]) if complete_cols else len(manifest)
    axes[2].bar(["raw", "included", "empty", "complete"], [raw_rows, included_rows, empty_rows, complete_rows], color=palette[:4])
    axes[2].set_ylabel("Count")
    axes[2].set_title("QC inclusion")
    fig.tight_layout()
    save_figure(fig, output)


def build_qc_summary(config: dict, manifest: pd.DataFrame, lesion_qc: pd.DataFrame) -> pd.DataFrame:
    """汇总病灶QC规则。"""
    qc_cfg = config.get("qc", {})
    min_volume = float(qc_cfg.get("min_lesion_volume_ml", 0.0))
    exclude_empty = bool(qc_cfg.get("exclude_empty_lesion", True))
    raw_rows = len(lesion_qc) if not lesion_qc.empty else len(manifest)
    included_rows = len(manifest)
    if "included_by_lesion_qc" in lesion_qc.columns:
        included_rows = int(lesion_qc["included_by_lesion_qc"].astype(bool).sum())
    empty_rows = 0
    if "is_empty_lesion" in lesion_qc.columns:
        empty_rows = int(lesion_qc["is_empty_lesion"].astype(bool).sum())
    elif "lesion_volume_ml" in lesion_qc.columns:
        empty_rows = int((lesion_qc["lesion_volume_ml"] <= min_volume).sum())
    excluded_rows = max(raw_rows - included_rows, 0)
    complete_cols = [column for column in ["mrs_3m", "age", "sex", "nihss", "lesion_volume_ml"] if column in manifest.columns]
    complete_rows = int(manifest[complete_cols].dropna().shape[0]) if complete_cols else len(manifest)
    return pd.DataFrame(
        [
            {"item": "raw_lesion_masks", "value": raw_rows, "rule": "all readable lesion masks"},
            {"item": "exclude_empty_lesion", "value": str(exclude_empty).lower(), "rule": f"lesion_volume_ml <= {min_volume:g}"},
            {"item": "empty_lesions", "value": empty_rows, "rule": "flagged in lesion_qc.csv"},
            {"item": "excluded_by_lesion_qc", "value": excluded_rows, "rule": "not included in subject_manifest.csv"},
            {"item": "included_after_lesion_qc", "value": included_rows, "rule": "used by downstream analysis scripts"},
            {"item": "complete_analysis_rows", "value": complete_rows, "rule": "non-missing outcome and configured covariates"},
        ]
    )


def plot_selection(selection: pd.DataFrame, output: Path) -> None:
    """绘制递质筛选频率。"""
    if selection.empty:
        return
    data = selection.sort_values(["selection_frequency", "mean_abs_profile_weight"], ascending=[True, True])
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.barh(data["nt_id"], data["selection_frequency"], color=palette[0])
    ax.set_xlabel("Selection frequency")
    ax.set_ylabel("Neurotransmitter")
    ax.set_title("NTDC neurotransmitter selection")
    ax.set_xlim(0, max(1.0, float(data["selection_frequency"].max()) * 1.05))
    fig.tight_layout()
    save_figure(fig, output)


def plot_fold_selection(folds: pd.DataFrame, output: Path) -> None:
    """绘制每折筛选数量。"""
    if folds.empty:
        return
    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    ax.bar(folds["fold"].astype(str), folds["selected_count"], color=palette[3])
    ax.set_xlabel("Outer fold")
    ax.set_ylabel("Selected NT systems")
    ax.set_title("Selected systems per fold")
    fig.tight_layout()
    save_figure(fig, output)


def plot_model_performance(performance: pd.DataFrame, output: Path) -> None:
    """绘制预测指标。"""
    if performance.empty:
        return
    panels = [
        ("ordinal_log_loss", "Ordinal log loss"),
        ("ranked_probability_score", "RPS"),
        ("ordinal_c_index", "Ordinal C-index"),
        ("binary_auc_mrs_le_threshold", "Binary AUC"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(9, 6))
    for ax, (metric, title) in zip(axes.ravel(), panels):
        ax.bar(performance["model"], performance[metric], color=palette[: len(performance)])
        ax.set_title(title)
        ax.set_ylabel(metric)
        ax.tick_params(axis="x", rotation=35)
    fig.tight_layout()
    save_figure(fig, output)


def plot_pairwise(pairwise: pd.DataFrame, output: Path) -> None:
    """绘制主要指标配对差值。"""
    if pairwise.empty:
        return
    data = pairwise[pairwise["metric"] == "ordinal_log_loss"].copy()
    if data.empty:
        return
    data["comparison"] = data["model_b"] + " vs " + data["model_a"]
    data = data.sort_values("delta_b_minus_a")
    fig, ax = plt.subplots(figsize=(7, max(3.2, 0.35 * len(data))))
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


def build_file_index(project_dir: Path) -> pd.DataFrame:
    """生成文件索引。"""
    patterns = [
        "derivatives/qc/*.csv",
        "derivatives/shared/*.csv",
        "derivatives/ml_profile/*.csv",
        "derivatives/ml_profile/models/*.csv",
        "derivatives/ml_profile/profiles/*.nii.gz",
        "derivatives/ml_profile/lsm_maps/*.nii.gz",
        "derivatives/ml_profile/exploratory_profiles/*.csv",
        "derivatives/ml_profile/exploratory_profiles/*.nii.gz",
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


def render_html(context: dict[str, object], output: Path) -> None:
    """渲染HTML。"""
    template = Template(
        """<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>NTDC Flow Report</title>
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
    .step { border: 1px solid #cfd9e3; border-radius: 8px; padding: 12px; background: #fbfcfe; min-height: 80px; }
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
    <h1>NTDC Flow Report</h1>
    <div class="meta">project: {{ project_dir }} | generated: {{ generated_at }}</div>
  </header>
  <main>
    <section class="card">
      <h2>1. 运行流程</h2>
      <div class="flow">
        <div class="step"><b>Input</b><span>MNI 2mm lesion mask<br>phenotype table<br>Hansen and Alves maps</span></div>
        <div class="step"><b>Screen</b><span>training-fold elastic-net<br>NT system weights</span></div>
        <div class="step"><b>Profile</b><span>fold-specific voxel-integrated gray and WM NT profiles</span></div>
        <div class="step"><b>Damage</b><span>profile node damage<br>profile edge damage<br>LSM impact weights</span></div>
        <div class="step"><b>Prediction</b><span>10-fold out-of-sample prediction<br>bootstrap model comparison</span></div>
      </div>
    </section>
    <section class="card">
      <h2>2. 输入与QC</h2>
      <div class="figure-grid">
        <figure><img src="{{ figures.cohort_qc }}" alt="cohort qc"><figcaption>Figure 1. Cohort and lesion QC.</figcaption></figure>
      </div>
      <h3>lesion QC summary</h3>
      <div class="table-wrap">{{ qc_summary_table }}</div>
      <h3>subject manifest</h3>
      <div class="table-wrap">{{ manifest_table }}</div>
    </section>
    <section class="card">
      <h2>3. 递质筛选与综合图谱</h2>
      <div class="figure-grid">
        <figure><img src="{{ figures.selection_frequency }}" alt="selection frequency"><figcaption>Figure 2. Neurotransmitter selection frequency.</figcaption></figure>
        <figure><img src="{{ figures.fold_selection }}" alt="fold selection"><figcaption>Figure 3. Selected systems per fold.</figcaption></figure>
      </div>
      <h3>selection summary</h3>
      <div class="table-wrap">{{ selection_table }}</div>
      <h3>fold manifest</h3>
      <div class="table-wrap">{{ fold_table }}</div>
    </section>
	    <section class="card">
	      <h2>4. 预测模型</h2>
	      <div class="figure-grid">
	        <figure><img src="{{ figures.model_performance }}" alt="model performance"><figcaption>Figure 4. Prediction metrics.</figcaption></figure>
	        <figure><img src="{{ figures.pairwise_delta }}" alt="pairwise delta"><figcaption>Figure 5. Pairwise bootstrap for ordinal log loss.</figcaption></figure>
      </div>
      <h3>prediction performance</h3>
      <div class="table-wrap">{{ performance_table }}</div>
	      <h3>pairwise bootstrap</h3>
	      <div class="table-wrap">{{ pairwise_table }}</div>
	    </section>
	    <section class="card">
	      <h2>5. D1/D2/DAT 探索预测</h2>
	      <div class="figure-grid">
	        <figure><img src="{{ figures.dopamine_performance }}" alt="dopamine performance"><figcaption>Figure 6. Exploratory D1/D2/DAT prediction metrics.</figcaption></figure>
	        <figure><img src="{{ figures.dopamine_pairwise_delta }}" alt="dopamine pairwise delta"><figcaption>Figure 7. Exploratory D1/D2/DAT pairwise bootstrap for ordinal log loss.</figcaption></figure>
	      </div>
	      <h3>D1/D2/DAT prediction performance</h3>
	      <div class="table-wrap">{{ dopamine_performance_table }}</div>
	      <h3>D1/D2/DAT pairwise bootstrap</h3>
	      <div class="table-wrap">{{ dopamine_pairwise_table }}</div>
	    </section>
	    <section class="card">
	      <h2>6. 输出文件索引</h2>
	      <div class="table-wrap">{{ file_index_table }}</div>
	    </section>
  </main>
</body>
</html>"""
    )
    output.write_text(template.render(**context), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="Generate NTDC flow HTML report.")
    parser.add_argument("--config", default="config/dat_config.yaml")
    return parser.parse_args()


def main() -> None:
    """生成流程化报告。"""
    args = parse_args()
    config = load_config(args.config)
    project_dir = Path(config["project_dir"])
    report_dir = ensure_dir(project_path(config, "derivatives", "reports"))
    figure_dir = ensure_dir(report_dir / "figures")
    ml_dir = project_path(config, "derivatives", "ml_profile")

    manifest = read_table(project_path(config, "derivatives", "qc", "subject_manifest.csv"))
    lesion_qc = read_table(project_path(config, "derivatives", "qc", "lesion_qc.csv"))
    selection = read_table(ml_dir / "selection_summary.csv")
    folds = read_table(ml_dir / "fold_manifest.csv")
    performance = read_table(ml_dir / "models" / "model_prediction_performance.csv")
    pairwise = read_table(ml_dir / "models" / "model_prediction_pairwise_bootstrap.csv")
    dopamine_dir = ml_dir / "exploratory_profiles"
    dopamine_performance = read_table(dopamine_dir / "dopamine_d1_d2_dat_model_prediction_performance.csv")
    dopamine_pairwise = read_table(dopamine_dir / "dopamine_d1_d2_dat_model_prediction_pairwise_bootstrap.csv")

    figure_paths = {
        "cohort_qc": figure_dir / "cohort_qc.png",
        "selection_frequency": figure_dir / "ml_profile_selection_frequency.png",
        "fold_selection": figure_dir / "ml_profile_fold_selection.png",
        "model_performance": figure_dir / "ml_profile_model_performance.png",
        "pairwise_delta": figure_dir / "ml_profile_pairwise_delta.png",
        "dopamine_performance": figure_dir / "dopamine_d1_d2_dat_model_performance.png",
        "dopamine_pairwise_delta": figure_dir / "dopamine_d1_d2_dat_pairwise_delta.png",
    }
    plot_cohort_qc(manifest, lesion_qc, figure_paths["cohort_qc"])
    plot_selection(selection, figure_paths["selection_frequency"])
    plot_fold_selection(folds, figure_paths["fold_selection"])
    plot_model_performance(performance, figure_paths["model_performance"])
    plot_pairwise(pairwise, figure_paths["pairwise_delta"])
    plot_model_performance(dopamine_performance, figure_paths["dopamine_performance"])
    plot_pairwise(dopamine_pairwise, figure_paths["dopamine_pairwise_delta"])

    qc_summary = build_qc_summary(config, manifest, lesion_qc)
    qc_summary_path = report_dir / "lesion_qc_summary.csv"
    qc_summary.to_csv(qc_summary_path, index=False)
    file_index = build_file_index(project_dir)
    file_index_path = report_dir / "report_file_index.csv"
    file_index.to_csv(file_index_path, index=False)
    figures = {key: path.relative_to(report_dir).as_posix() for key, path in figure_paths.items()}
    html_path = report_dir / "nt_clsm_flow_report.html"
    render_html(
        {
            "project_dir": project_dir.as_posix(),
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "figures": figures,
            "qc_summary_table": table_html(qc_summary, 20),
            "manifest_table": table_html(manifest, 12),
            "selection_table": table_html(selection, 20),
            "fold_table": table_html(folds, 20),
            "performance_table": table_html(performance, 20),
            "pairwise_table": table_html(pairwise, 40),
            "dopamine_performance_table": table_html(dopamine_performance, 20),
            "dopamine_pairwise_table": table_html(dopamine_pairwise, 40),
            "file_index_table": table_html(file_index, 200),
        },
        html_path,
    )
    print(f"report_html={html_path}")
    print(f"report_file_index={file_index_path}")


if __name__ == "__main__":
    main()
