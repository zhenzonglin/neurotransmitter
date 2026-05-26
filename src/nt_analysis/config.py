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
