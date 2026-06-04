from __future__ import annotations

import re
import math
from pathlib import Path


def normalize_subject_id(value: object) -> str:
    """Normalize TMS or numeric subject IDs."""
    if value is None:
        raise ValueError("cannot parse subject id: None")
    if isinstance(value, float):
        if math.isnan(value):
            raise ValueError("cannot parse subject id: nan")
        if value.is_integer():
            value = int(value)
    text = str(value).strip().upper().replace(" ", "")
    text = text.replace("-", "").replace("_", "")
    if text.endswith(".0") and text[:-2].isdigit():
        text = text[:-2]
    match = re.search(r"TMS(\d+)", text)
    if match:
        return f"TMS{int(match.group(1)):03d}"
    if text.isdigit():
        return text
    raise ValueError(f"cannot parse subject id: {value}")


def parse_lesion_subject(path: str | Path, pattern: str) -> str:
    """Parse subject ID from a lesion filename."""
    name = Path(path).name
    match = re.search(pattern, name)
    if not match:
        raise ValueError(f"cannot parse lesion subject from {name}")
    return normalize_subject_id(match.group(1))
