#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import subprocess
import sys
from pathlib import Path

import yaml
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nt_analysis.config import ensure_dir, load_config, prognostic_endpoint_specs, project_path  # noqa: E402


def endpoint_config(base_config: dict, endpoint: dict, config_dir: Path) -> Path:
    """Write one endpoint-specific config."""
    cfg = copy.deepcopy(base_config)
    endpoint_id = str(endpoint["id"])
    atlas_cfg = cfg.setdefault("prognostic_ntdc_atlas", {})
    base_output = atlas_cfg.get("output_dir", "derivatives/prognostic_ntdc_atlas")
    base_report = atlas_cfg.get("report_dir", "derivatives/reports_prognostic_ntdc_atlas")
    cfg["project_dir"] = str(Path(base_config["project_dir"]))
    cfg.setdefault("inputs", {})["outcome_column"] = str(endpoint["outcome_column"])
    cfg.setdefault("analysis", {})["outcome"] = str(endpoint["outcome"])
    atlas_cfg["active_endpoint"] = dict(endpoint)
    atlas_cfg["output_dir"] = str(Path(base_output) / endpoint_id)
    atlas_cfg["report_dir"] = str(Path(base_report) / endpoint_id)
    path = config_dir / f"dat_config_{endpoint_id}.yaml"
    path.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return path


def run_command(args: list[str]) -> None:
    """Run one subprocess command."""
    print("running: " + " ".join(args), flush=True)
    subprocess.run(args, check=True)


def validate_endpoint_columns(config: dict, endpoints: list[dict]) -> None:
    """Fail early when endpoint columns are absent."""
    phenotype_path = project_path(config, config["inputs"]["phenotype_file"])
    phenotype = pd.read_excel(phenotype_path, sheet_name=config["inputs"]["phenotype_sheet"], nrows=1)
    available = set(phenotype.columns.astype(str))
    required = []
    for endpoint in endpoints:
        required.append(str(endpoint["outcome_column"]))
        required.append(str(endpoint["stroke_column"]))
    missing = [column for column in dict.fromkeys(required) if column not in available]
    if missing:
        raise KeyError(f"missing endpoint columns in phenotype file: {missing}")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run prognostic NTDC atlas for all configured endpoints.")
    parser.add_argument("--config", default="config/dat_config.yaml")
    parser.add_argument("--max-subjects", type=int, default=None)
    parser.add_argument("--build-jobs", type=int, default=None)
    parser.add_argument("--model-jobs", type=int, default=None)
    parser.add_argument("--edge-jobs", type=int, default=None)
    parser.add_argument("--chunk-size", type=int, default=None)
    parser.add_argument("--edge-chunk-size", type=int, default=128)
    parser.add_argument("--export-jobs", type=int, default=None)
    parser.add_argument("--force-maps", action="store_true")
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument("--skip-model", action="store_true")
    parser.add_argument("--skip-export", action="store_true")
    parser.add_argument("--skip-report", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Run all endpoints."""
    args = parse_args()
    base_config = load_config(args.config)
    atlas_cfg = base_config.get("prognostic_ntdc_atlas", {})
    config_dir = ensure_dir(project_path(base_config, atlas_cfg.get("output_dir", "derivatives/prognostic_ntdc_atlas"), "endpoint_configs"))
    endpoints = prognostic_endpoint_specs(base_config)
    if not endpoints:
        raise RuntimeError("no endpoints were configured")
    validate_endpoint_columns(base_config, endpoints)
    for endpoint in endpoints:
        endpoint_id = str(endpoint["id"])
        print(f"endpoint {endpoint_id}: {endpoint.get('label', endpoint_id)}", flush=True)
        endpoint_cfg = endpoint_config(base_config, endpoint, config_dir)
        if not args.skip_build:
            command = [sys.executable, "-u", "scripts/build_prognostic_ntdc_atlas.py", "--config", str(endpoint_cfg)]
            if args.build_jobs is not None:
                command += ["--jobs", str(args.build_jobs)]
            if args.chunk_size is not None:
                command += ["--chunk-size", str(args.chunk_size)]
            if args.max_subjects is not None:
                command += ["--max-subjects", str(args.max_subjects)]
            if args.force_maps:
                command += ["--force-maps"]
            run_command(command)
        if not args.skip_model:
            command = [sys.executable, "-u", "scripts/run_prognostic_ntdc_atlas.py", "--config", str(endpoint_cfg), "--edge-chunk-size", str(args.edge_chunk_size)]
            if args.model_jobs is not None:
                command += ["--jobs", str(args.model_jobs)]
            if args.edge_jobs is not None:
                command += ["--edge-jobs", str(args.edge_jobs)]
            if args.max_subjects is not None:
                command += ["--max-subjects", str(args.max_subjects)]
            run_command(command)
        if not args.skip_export:
            command = [sys.executable, "-u", "scripts/export_prognostic_ntdc_maps.py", "--config", str(endpoint_cfg)]
            if args.export_jobs is not None:
                command += ["--jobs", str(args.export_jobs)]
            run_command(command)
        if not args.skip_report:
            run_command([sys.executable, "-u", "scripts/generate_prognostic_ntdc_report.py", "--config", str(endpoint_cfg)])
    print("finished all prognostic NTDC endpoints", flush=True)


if __name__ == "__main__":
    main()
