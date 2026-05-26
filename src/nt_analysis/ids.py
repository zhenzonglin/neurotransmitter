from __future__ import annotations

import re
from pathlib import Path


def normalize_subject_id(value: object) -> str:
    """Normalize subject IDs to TMS001 style."""
    text = str(value).strip().upper()
    text = text.replace("-", "").replace("_", "")
    match = re.search(r"TMS(\d{1,3})", text)
    if not match:
        raise ValueError(f"cannot parse subject id: {value}")
    return f"TMS{int(match.group(1)):03d}"


def parse_lesion_subject(path: str | Path, pattern: str) -> str:
    """Parse subject ID from a lesion filename."""
    name = Path(path).name
    match = re.search(pattern, name)
    if not match:
        raise ValueError(f"cannot parse lesion subject from {name}")
    return normalize_subject_id(match.group(1))
