from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


def write_csv(df: pd.DataFrame, path: str | Path) -> Path:
    """Write a CSV file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def upper_triangle_edges(matrix: np.ndarray, labels: Iterable[int]) -> pd.DataFrame:
    """Flatten upper-triangle edges."""
    labels = list(labels)
    rows = []
    for i, left in enumerate(labels):
        for j, right in enumerate(labels):
            if j <= i:
                continue
            rows.append(
                {
                    "edge": f"edge_{int(left):03d}_{int(right):03d}",
                    "roi_i": int(left),
                    "roi_j": int(right),
                    "value": float(matrix[i, j]),
                }
            )
    return pd.DataFrame(rows)
