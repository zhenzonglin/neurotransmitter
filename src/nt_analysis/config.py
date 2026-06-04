from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML configuration file."""
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    project_dir = Path(config["project_dir"]).expanduser()
    if not project_dir.is_absolute():
        project_dir = (path.resolve().parents[1] / project_dir).resolve()
    config["project_dir"] = str(project_dir)
    return config


def project_path(config: dict[str, Any], *parts: str | Path) -> Path:
    """Return an absolute path inside the project."""
    path = Path(config["project_dir"])
    for part in parts:
        path = path / part
    return path


def ensure_dir(path: str | Path) -> Path:
    """Create a directory if needed."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def outcome_column(config: dict[str, Any]) -> str:
    """Return the internal outcome column name."""
    return str(config.get("analysis", {}).get("outcome", "mrs_3m"))


def prognostic_endpoint_specs(config: dict[str, Any]) -> list[dict[str, Any]]:
    """Return configured prognostic endpoint specs."""
    atlas_cfg = config.get("prognostic_ntdc_atlas", {})
    endpoints = atlas_cfg.get("endpoints", [])
    if endpoints:
        return [dict(endpoint) for endpoint in endpoints]
    qc_cfg = config.get("qc", {})
    return [
        {
            "id": "main",
            "label": "main",
            "outcome_column": config.get("inputs", {}).get("outcome_column", "mRS_3m"),
            "outcome": outcome_column(config),
            "stroke_column": qc_cfg.get("m3_stroke_column", "m3_stroke"),
            "stroke_exclude_values": qc_cfg.get("m3_stroke_exclude_values", [2]),
        }
    ]


def active_prognostic_endpoint(config: dict[str, Any]) -> dict[str, Any]:
    """Return the active prognostic endpoint spec."""
    atlas_cfg = config.get("prognostic_ntdc_atlas", {})
    active = atlas_cfg.get("active_endpoint")
    if active:
        return dict(active)
    outcome = outcome_column(config)
    for endpoint in prognostic_endpoint_specs(config):
        if str(endpoint.get("outcome")) == outcome:
            return dict(endpoint)
    return prognostic_endpoint_specs(config)[0]


def analysis_covariates(config: dict[str, Any], key: str) -> list[str]:
    """Return configured covariates for one analysis layer."""
    values = config.get("analysis", {}).get(key, [])
    return [str(value) for value in values]


def analysis_table(config: dict[str, Any], key: str, default: str) -> str:
    """Return a configured output table name."""
    return str(config.get("analysis", {}).get("tables", {}).get(key, default))


def require_columns(columns: list[str], available: list[str], context: str) -> None:
    """Fail early when configured columns are missing."""
    missing = [column for column in columns if column not in available]
    if missing:
        raise KeyError(f"missing columns in {context}: {missing}")
