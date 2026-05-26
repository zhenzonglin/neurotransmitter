#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import zipfile
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nt_analysis.config import ensure_dir, load_config, project_path


LQT_FILES = {
    "HCP842_QA.nii.gz": "https://ndownloader.figshare.com/files/27368315?private_link=2d830ec228a1c4bdf8aa",
    "MNI152_T1_1mm.nii.gz": "https://ndownloader.figshare.com/files/27368318?private_link=2d830ec228a1c4bdf8aa",
    "Tractography_Atlas.zip": "https://ndownloader.figshare.com/articles/14342426?private_link=4be1178860a7d8dad555",
    "All_Tracts.zip": "https://ndownloader.figshare.com/articles/14342450?private_link=83a8d620899ed9b198d3",
}


def sha256_file(path: Path) -> str:
    """Compute a file hash."""
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_url(url: str, output: Path, force: bool = False) -> dict[str, object]:
    """Download one public Figshare URL."""
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and output.stat().st_size > 0 and not force:
        return {"file": output.name, "path": str(output), "status": "exists", "size_bytes": output.stat().st_size}
    last_error = ""
    tmp = output.with_suffix(output.suffix + ".part")
    for _ in range(6):
        try:
            with requests.get(url, stream=True, timeout=180, headers={"User-Agent": "Mozilla/5.0"}) as response:
                response.raise_for_status()
                with tmp.open("wb") as f:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            f.write(chunk)
            if tmp.stat().st_size == 0:
                raise RuntimeError("downloaded zero-byte file")
            tmp.replace(output)
            return {
                "file": output.name,
                "path": str(output),
                "status": "downloaded",
                "size_bytes": output.stat().st_size,
                "sha256": sha256_file(output),
            }
        except Exception as error:  # noqa: BLE001
            last_error = str(error)
            if tmp.exists():
                tmp.unlink()
            time.sleep(10)
    raise RuntimeError(f"failed to download {url}: {last_error}")


def unzip_if_needed(zip_path: Path, output_dir: Path) -> None:
    """Extract a zip file once."""
    marker = output_dir / ".complete"
    if marker.exists():
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    # 公开模板是普通zip，解压后保留原始结构
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(output_dir)
    marker.write_text(time.strftime("%Y-%m-%dT%H:%M:%S"), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download open LQT/HCP842 resources.")
    parser.add_argument("--config", default="config/dat_config.yaml")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    data_dir = ensure_dir(project_path(config, config["lqt"]["data_dir"]))

    rows = []
    for name, url in LQT_FILES.items():
        print(f"checking {name}", flush=True)
        rows.append(download_url(url, data_dir / name, force=args.force))

    unzip_if_needed(data_dir / "Tractography_Atlas.zip", data_dir / "Tractography_Atlas")
    unzip_if_needed(data_dir / "All_Tracts.zip", data_dir / "Tractography_Atlas" / "All_Tracts")

    required = [
        data_dir / "HCP842_QA.nii.gz",
        data_dir / "MNI152_T1_1mm.nii.gz",
        data_dir / "Tractography_Atlas" / "HCP842_1mm.fib.gz",
        data_dir / "Tractography_Atlas" / "all_tracts_1mm.trk.gz",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"LQT resources are incomplete: {missing}")

    summary = {"files": rows, "required_ok": True, "data_dir": str(data_dir)}
    (data_dir / "download_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"wrote {data_dir / 'download_summary.json'}")


if __name__ == "__main__":
    main()
