#!/usr/bin/env python3
from __future__ import annotations

import argparse
import html
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nt_analysis.config import ensure_dir, load_config, project_path  # noqa: E402


palette = ["#0077BB", "#33BBEE", "#009988", "#EE7733", "#CC3311", "#EE3377", "#BBBBBB", "#000000"]


def read_table(path: Path) -> pd.DataFrame:
    """Read a table if it exists."""
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        # 小样本 smoke test 可能生成空结果表
        return pd.DataFrame()


def table_html(df: pd.DataFrame, max_rows: int = 20) -> str:
    """Render a compact HTML table."""
    if df.empty:
        return "<p>not available</p>"
    return df.head(max_rows).to_html(index=False, border=0, classes="data-table", escape=True)


def plot_roi_heatmap(roi_weights: pd.DataFrame, nt_ids: list[str], output: Path) -> None:
    """Plot ROI by NT weights."""
    if roi_weights.empty:
        return
    values = roi_weights[nt_ids].astype(float).to_numpy()
    fig, ax = plt.subplots(figsize=(10, 10))
    im = ax.imshow(values, aspect="auto", cmap="viridis")
    ax.set_xlabel("neurotransmitter")
    ax.set_ylabel("ROI")
    ax.set_xticks(np.arange(len(nt_ids)))
    ax.set_xticklabels(nt_ids, rotation=60, ha="right")
    ax.set_title("ROI x NT ridge weights")
    fig.colorbar(im, ax=ax, shrink=0.75)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_prediction(performance: pd.DataFrame, output: Path) -> None:
    """Plot prediction metrics."""
    if performance.empty:
        return
    metrics = [col for col in ["ordinal_log_loss", "ranked_probability_score", "ordinal_c_index", "binary_auc"] if col in performance.columns]
    if not metrics:
        return
    fig, axes = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), 4))
    axes = np.atleast_1d(axes)
    for ax, metric in zip(axes, metrics):
        ax.bar(performance["model"], performance[metric].astype(float), color=palette[: performance.shape[0]])
        ax.set_title(metric)
        ax.tick_params(axis="x", rotation=40)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_pairwise(pairwise: pd.DataFrame, output: Path) -> None:
    """Plot pairwise ordinal log loss deltas."""
    if pairwise.empty or "ordinal_log_loss" not in pairwise.get("metric", pd.Series(dtype=str)).astype(str).tolist():
        return
    data = pairwise[pairwise["metric"] == "ordinal_log_loss"].copy()
    if data.empty:
        return
    labels = data["model_b"] + " vs " + data["model_a"]
    y = np.arange(data.shape[0])
    fig, ax = plt.subplots(figsize=(10, max(4, data.shape[0] * 0.45)))
    ax.errorbar(data["delta_b_minus_a"], y, xerr=[data["delta_b_minus_a"] - data["ci_low"], data["ci_high"] - data["delta_b_minus_a"]], fmt="o", color="#CC3311", ecolor="#777777")
    ax.axvline(0, color="#333333", linewidth=1)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("delta ordinal log loss")
    ax.set_title("Pairwise bootstrap")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def build_file_index(paths: list[Path], project_dir: Path) -> pd.DataFrame:
    """Build output file index."""
    rows = []
    for path in paths:
        if path.exists():
            rows.append({"file": str(path.relative_to(project_dir)), "size_mb": round(path.stat().st_size / 1024 / 1024, 3)})
    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Generate prognostic NTDC atlas report.")
    parser.add_argument("--config", default="config/dat_config.yaml")
    return parser.parse_args()


def main() -> None:
    """Generate report."""
    args = parse_args()
    config = load_config(args.config)
    project_dir = Path(config["project_dir"])
    atlas_cfg = config.get("prognostic_ntdc_atlas", {})
    out_dir = project_path(config, atlas_cfg.get("output_dir", "derivatives/prognostic_ntdc_atlas"))
    report_dir = ensure_dir(project_path(config, atlas_cfg.get("report_dir", "derivatives/reports_prognostic_ntdc_atlas")))
    figure_dir = ensure_dir(report_dir / "figures")
    nt = read_table(out_dir / "nt_table.csv")
    nt_ids = nt["nt_id"].astype(str).tolist() if not nt.empty else []
    subject = read_table(out_dir / "subject_table.csv")
    roi_weights = read_table(out_dir / "roi_nt_weight_atlas.csv")
    edge_weights = read_table(out_dir / "edge_nt_weight_atlas.csv")
    performance = read_table(out_dir / "prediction_performance.csv")
    pairwise = read_table(out_dir / "pairwise_bootstrap.csv")
    edge_qc = read_table(out_dir / "nt_edge_denominator_qc.csv")
    edge_support = read_table(out_dir / "edge_support_qc.csv")
    edge_support_summary = read_table(out_dir / "edge_support_summary.csv")
    agreement = read_table(out_dir / "model_agreement_summary.csv")
    fold_runtime = read_table(out_dir / "fold_runtime.csv")
    metadata = {}
    metadata_path = out_dir / "feature_metadata.json"
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    endpoint = metadata.get("endpoint", {})

    figures = {
        "roi_heatmap": figure_dir / "roi_nt_weight_heatmap.png",
        "prediction": figure_dir / "prediction_performance.png",
        "pairwise": figure_dir / "pairwise_ordinal_log_loss.png",
    }
    plot_roi_heatmap(roi_weights, nt_ids, figures["roi_heatmap"])
    plot_prediction(performance, figures["prediction"])
    plot_pairwise(pairwise, figures["pairwise"])

    four_d = out_dir / f"roi_nt_weight_{len(nt_ids)}nt_4d.nii.gz"
    four_d_shape = ""
    if four_d.exists():
        four_d_shape = " x ".join(str(value) for value in nib.load(str(four_d)).shape)
    outputs = build_file_index(
        [
            four_d,
            out_dir / "roi_nt_weight_atlas.csv",
            out_dir / "edge_nt_weight_atlas.csv",
            out_dir / "residual_ntdc_scores.csv",
            out_dir / "prediction_cv.csv",
            out_dir / "prediction_performance.csv",
            out_dir / "pairwise_bootstrap.csv",
            out_dir / "edge_support_qc.csv",
            out_dir / "edge_support_summary.csv",
            out_dir / "model_agreement_summary.csv",
        ],
        project_dir,
    )
    theoretical_edges = metadata.get("n_edges", 12090)
    supported_edges = "NA"
    excluded_edges = "NA"
    if not edge_support_summary.empty:
        supported_edges = int(edge_support_summary.loc[0, "tract_supported_edges"])
        excluded_edges = int(edge_support_summary.loc[0, "excluded_unsupported_edges"])

    image_tags = []
    for name, path in figures.items():
        if path.exists():
            image_tags.append(f'<figure><img src="figures/{html.escape(path.name)}" alt="{html.escape(name)}"><figcaption>{html.escape(name)}</figcaption></figure>')

    html_text = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Prognostic NTDC Atlas Report</title>
<style>
body {{ font-family: Arial, sans-serif; margin: 32px; color: #102030; background: #f7f9fb; }}
section {{ background: white; border: 1px solid #d9e1e8; border-radius: 8px; padding: 22px; margin-bottom: 22px; }}
h1, h2 {{ color: #0b253f; }}
.grid {{ display: grid; grid-template-columns: repeat(6, 1fr); gap: 12px; }}
.metric {{ background: #eef4f8; border-radius: 6px; padding: 12px; }}
.metric b {{ display: block; font-size: 20px; }}
.data-table {{ border-collapse: collapse; width: 100%; font-size: 13px; }}
.data-table th, .data-table td {{ border-bottom: 1px solid #e5edf2; padding: 6px 8px; text-align: left; }}
img {{ max-width: 100%; border: 1px solid #d9e1e8; border-radius: 6px; background: white; }}
figure {{ margin: 0 0 22px 0; }}
code {{ background: #eef4f8; padding: 2px 5px; border-radius: 4px; }}
</style>
</head>
<body>
<h1>Prognostic NTDC Atlas Report</h1>
<section>
<h2>Run Summary</h2>
<p>endpoint: <code>{html.escape(str(endpoint.get("label", endpoint.get("id", "main"))))}</code> | outcome: <code>{html.escape(str(endpoint.get("outcome", "NA")))}</code> | recurrence exclusion: <code>{html.escape(str(endpoint.get("stroke_column", "NA")))} = {html.escape(str(endpoint.get("stroke_exclude_values", [2])))}</code></p>
<div class="grid">
<div class="metric"><span>subjects</span><b>{int(subject.shape[0]) if not subject.empty else metadata.get("n_subjects", "NA")}</b></div>
<div class="metric"><span>ROI</span><b>{metadata.get("n_roi", 156)}</b></div>
<div class="metric"><span>theoretical edges</span><b>{theoretical_edges}</b></div>
<div class="metric"><span>modeled edges</span><b>{supported_edges}</b></div>
<div class="metric"><span>excluded edges</span><b>{excluded_edges}</b></div>
<div class="metric"><span>NT systems</span><b>{len(nt_ids)}</b></div>
</div>
<p>4D atlas shape: <code>{html.escape(four_d_shape or "not available")}</code></p>
</section>
<section>
<h2>Workflow</h2>
<p>lesion -> SDC -> NT node/edge damage -> structure residualization -> ridge logistic atlas -> residual NTDC prediction</p>
</section>
<section>
<h2>Figures</h2>
{''.join(image_tags) if image_tags else '<p>not available</p>'}
</section>
<section>
<h2>Prediction Performance</h2>
{table_html(performance)}
</section>
<section>
<h2>Pairwise Bootstrap</h2>
{table_html(pairwise)}
</section>
<section>
<h2>Edge Support QC</h2>
{table_html(edge_support_summary)}
{table_html(edge_support)}
</section>
<section>
<h2>Edge Denominator QC</h2>
{table_html(edge_qc)}
</section>
<section>
<h2>Model Agreement</h2>
{table_html(agreement)}
</section>
<section>
<h2>Fold Runtime</h2>
{table_html(fold_runtime)}
</section>
<section>
<h2>Output Files</h2>
{table_html(outputs, 50)}
</section>
</body>
</html>
"""
    report_path = report_dir / "prognostic_ntdc_atlas_report.html"
    report_path.write_text(html_text, encoding="utf-8")
    print(f"wrote {report_path}", flush=True)


if __name__ == "__main__":
    main()
