#!/usr/bin/env python3
from __future__ import annotations

import argparse
import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nt_analysis.config import ensure_dir, load_config, project_path
from nt_analysis.tables import write_csv


def algorithm_paths(config: dict, algorithm: str) -> dict[str, Path | str]:
    """Return NeMo paths for one algorithm."""
    data_dir = project_path(config, config["nemo"]["data_dir"])
    if algorithm == "ifod2act":
        return {
            "chunklist": data_dir / "nemo_chunklist.npz",
            "chunkdir": data_dir / "chunkfiles",
            "endpoints": data_dir / "nemo_endpoints.npy",
            "asum": data_dir / "nemo_Asum_endpoints.npz",
            "tracking_algorithm": "ifod2act",
        }
    if algorithm == "sdstream":
        return {
            "chunklist": data_dir / "nemo_sdstream_chunklist.npz",
            "chunkdir": data_dir / "chunkfiles_sdstream",
            "endpoints": data_dir / "nemo_sdstream_endpoints.npy",
            "asum": data_dir / "nemo_sdstream_Asum_endpoints.npz",
            "tracking_algorithm": "sdstream",
        }
    raise ValueError(f"unknown algorithm: {algorithm}")


def run_nemo_subject(config: dict, subject_id: str, lesion_path: Path, algorithm: str) -> Path:
    """Run NeMo for one subject."""
    paths = algorithm_paths(config, algorithm)
    required = [
        paths["chunklist"],
        paths["chunkdir"],
        paths["endpoints"],
        paths["asum"],
        project_path(config, config["nemo"]["data_dir"], "MNI152_T1_1mm_brain.nii.gz"),
    ]
    missing = [str(path) for path in required if not Path(path).exists()]
    if missing:
        # NeMo边分析依赖完整数据库，不能用空文件替代
        raise FileNotFoundError(
            "NeMo database is incomplete. Configure AWS requester-pays credentials and run "
            "`python scripts/fetch_reference_data.py --config config/dat_config.yaml --nemo --workers 4` first. "
            f"Missing: {missing[:5]}"
        )
    output_dir = ensure_dir(project_path(config, config["outputs"]["nemo_dir"], subject_id))
    outputbase = output_dir / f"{subject_id}_datwm_{algorithm}"
    result = Path(f"{outputbase}_chacoconn_mean.pkl")
    if result.exists() and result.stat().st_size > 0:
        return result
    command = [
        sys.executable,
        str(project_path(config, "external/nemo/nemo_lesion_to_chaco.py")),
        "--lesion",
        str(lesion_path),
        "--outputbase",
        str(outputbase),
        "--parcelvol",
        str(project_path(config, config["atlases"]["outputs"]["atlas4s156_1mm_nemo"])),
        "--pairwise",
        "--continuous_value",
        "--chunklist",
        str(paths["chunklist"]),
        "--chunkdir",
        str(paths["chunkdir"]),
        "--refvol",
        str(project_path(config, config["nemo"]["data_dir"], "MNI152_T1_1mm_brain.nii.gz")),
        "--endpoints",
        str(paths["endpoints"]),
        "--asum",
        str(paths["asum"]),
        "--tracking_algorithm",
        str(paths["tracking_algorithm"]),
        "--numthreads",
        str(config["nemo"].get("num_threads", 4)),
    ]
    # 调用NeMo成品脚本
    subprocess.run(command, check=True)
    return result


def load_chacoconn(path: Path) -> np.ndarray:
    """Load one NeMo chacoconn matrix."""
    with path.open("rb") as f:
        data = pickle.load(f)
    if sparse.issparse(data):
        data = data.toarray()
    return np.asarray(data, dtype=np.float32)


def collect_edge_table(config: dict, algorithm: str, results: dict[str, Path]) -> Path:
    """Flatten NeMo matrices into a subject by edge table."""
    edge_dir = ensure_dir(project_path(config, config["outputs"]["edge_dir"]))
    rows = []
    edge_names = None
    for subject_id, path in sorted(results.items()):
        matrix = load_chacoconn(path)
        labels = list(range(1, matrix.shape[0] + 1))
        values = {"subject_id": subject_id}
        names = []
        for i, left in enumerate(labels):
            for j, right in enumerate(labels):
                if j <= i:
                    continue
                name = f"edge_{left:03d}_{right:03d}"
                values[name] = float(matrix[i, j])
                names.append(name)
        if edge_names is None:
            edge_names = names
        rows.append(values)
    output = edge_dir / f"dat_edge_chaco_{algorithm}_66.csv"
    write_csv(pd.DataFrame(rows), output)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Run NeMo DAT-edge ChaCo.")
    parser.add_argument("--config", default="config/dat_config.yaml")
    parser.add_argument("--algorithm", choices=["ifod2act", "sdstream"], required=True)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    config = load_config(args.config)

    manifest = pd.read_csv(project_path(config, config["outputs"]["nemo_dir"], "nemo_input_manifest.csv"))
    if args.limit:
        manifest = manifest.head(args.limit)
    results = {}
    for row in manifest.itertuples(index=False):
        # 每例患者独立运行
        result = run_nemo_subject(config, row.subject_id, Path(row.dat_weighted_lesion_1mm), args.algorithm)
        results[row.subject_id] = result
    output = collect_edge_table(config, args.algorithm, results)
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
