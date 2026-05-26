#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nt_analysis.config import ensure_dir, load_config, project_path


ATLASPACK_URLS = {
    "repo": "https://github.com/PennLINC/AtlasPack.git",
    "nii": "tpl-MNI152NLin6Asym_atlas-4S156Parcels_res-01_dseg.nii.gz",
    "tsv": "atlas-4S156Parcels_dseg.tsv",
    "json": "tpl-MNI152NLin6Asym_atlas-4S156Parcels_dseg.json",
}

HANSEN_DAT_URL = (
    "https://raw.githubusercontent.com/netneurolab/hansen_receptors/main/"
    "data/PET_nifti_images/DAT_fpcit_hc174_dukart_spect.nii"
)

NEUROVAULT_DAT_API = "https://neurovault.org/api/images/802963/"


def sha256_file(path: Path) -> str:
    """Compute SHA256 for a file."""
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_url(url: str, output: Path) -> dict[str, object]:
    """Download one URL with a browser user agent."""
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and output.stat().st_size > 0:
        return {"path": str(output), "status": "exists", "size_bytes": output.stat().st_size, "sha256": sha256_file(output)}
    last_error = ""
    for _ in range(5):
        try:
            with requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=120, stream=True) as response:
                response.raise_for_status()
                tmp = output.with_suffix(output.suffix + ".part")
                with tmp.open("wb") as f:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            f.write(chunk)
                tmp.replace(output)
            break
        except Exception as error:  # noqa: BLE001
            last_error = str(error)
            time.sleep(5)
    if not output.exists() or output.stat().st_size == 0:
        raise RuntimeError(f"failed to download {url}: {last_error}")
    return {"path": str(output), "status": "downloaded", "size_bytes": output.stat().st_size, "sha256": sha256_file(output)}


def is_gzip_file(path: Path) -> bool:
    """Return whether a file starts with a gzip header."""
    if not path.exists() or path.stat().st_size == 0:
        return False
    with path.open("rb") as f:
        return f.read(2) == b"\x1f\x8b"


def run_command(command: list[str], cwd: Path | None = None) -> None:
    """Run an external command and fail loudly."""
    subprocess.run(command, cwd=cwd, check=True)


def download_atlaspack(config: dict, force: bool = False) -> list[dict[str, object]]:
    """Download AtlasPack files through DataLad/OSF."""
    raw_dir = ensure_dir(project_path(config, config["atlases"]["raw_dir"]))
    repo_dir = raw_dir / "atlaspack_datalad"
    datalad_bin = shutil.which("datalad")
    if datalad_bin is None:
        raise RuntimeError("datalad is required to download AtlasPack annex content")

    atlas_target = project_path(config, config["atlases"]["atlas4s156"]["nii"])
    if force and atlas_target.exists():
        atlas_target.unlink()

    if not repo_dir.exists():
        # AtlasPack的NIfTI是git-annex内容，必须用DataLad获取实体文件
        run_command([datalad_bin, "clone", ATLASPACK_URLS["repo"], str(repo_dir)])

    keys = ["nii", "tsv", "json"]
    paths = [ATLASPACK_URLS[key] for key in keys]
    run_command([datalad_bin, "get", *paths], cwd=repo_dir)

    outputs = {
        "nii": project_path(config, config["atlases"]["atlas4s156"]["nii"]),
        "tsv": project_path(config, config["atlases"]["atlas4s156"]["tsv"]),
        "json": project_path(config, config["atlases"]["atlas4s156"]["json"]),
    }
    sources = []
    for key in keys:
        src = repo_dir / ATLASPACK_URLS[key]
        dst = outputs[key]
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        if key == "nii" and not is_gzip_file(dst):
            raise RuntimeError(f"AtlasPack NIfTI is not a real gzip file: {dst}")
        sources.append(
            {
                "name": f"atlas4s156_{key}",
                "url": f"{ATLASPACK_URLS['repo']}::{ATLASPACK_URLS[key]}",
                "path": str(dst),
                "status": "downloaded" if force else "exists_or_downloaded",
                "size_bytes": dst.stat().st_size,
                "sha256": sha256_file(dst),
            }
        )
    return sources


def get_neurovault_file_url(api_url: str) -> str:
    """Read the DAT-WM download URL from NeuroVault metadata."""
    request = urllib.request.Request(api_url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request, timeout=120) as response:
        data = json.loads(response.read().decode("utf-8"))
    return data["file"].replace("http://", "https://")


def download_reference_maps(config: dict, force: bool = False) -> None:
    """Download AtlasPack, Hansen DAT and Functionnectome DAT-WM."""
    raw_dir = ensure_dir(project_path(config, config["atlases"]["raw_dir"]))
    sources = download_atlaspack(config, force=force)
    targets = [
        ("hansen_dat", HANSEN_DAT_URL, project_path(config, config["atlases"]["hansen_dat"]["nii"])),
    ]
    dat_wm_url = get_neurovault_file_url(NEUROVAULT_DAT_API)
    targets.append(("functionnectome_dat_wm", dat_wm_url, project_path(config, config["atlases"]["dat_wm"]["nii"])))
    for name, url, output in targets:
        if force and output.exists():
            output.unlink()
        result = download_url(url, output)
        result.update({"name": name, "url": url})
        sources.append(result)
    sources_path = raw_dir / "sources.json"
    sources_path.write_text(json.dumps(sources, indent=2), encoding="utf-8")
    print(f"wrote {sources_path}")


def make_s3_client():
    """Create an S3 client."""
    import boto3

    return boto3.client("s3")


def nemo_keys_for_algorithm(algorithm: str) -> dict[str, object]:
    """Return NeMo S3 keys for one tracking algorithm."""
    prefix = "" if algorithm == "ifod2act" else "nemo_sdstream_"
    chunk_dir = "chunkfiles" if algorithm == "ifod2act" else "chunkfiles_sdstream"
    return {
        "algorithm": algorithm,
        "chunk_dir": chunk_dir,
        "metadata": [
            f"{prefix}chunklist.npz" if algorithm != "ifod2act" else "nemo_chunklist.npz",
            f"{prefix}endpoints.npy" if algorithm != "ifod2act" else "nemo_endpoints.npy",
            f"{prefix}Asum_endpoints.npz" if algorithm != "ifod2act" else "nemo_Asum_endpoints.npz",
            f"{prefix}Asum_weighted_endpoints.npz" if algorithm != "ifod2act" else "nemo_Asum_weighted_endpoints.npz",
            f"{prefix}Asum_cumulative.npz" if algorithm != "ifod2act" else "nemo_Asum_cumulative.npz",
            f"{prefix}Asum_weighted_cumulative.npz" if algorithm != "ifod2act" else "nemo_Asum_weighted_cumulative.npz",
            f"{prefix}siftweights.npy" if algorithm != "ifod2act" else "nemo_siftweights.npy",
            f"{prefix}tracklengths.npy" if algorithm != "ifod2act" else "nemo_tracklengths.npy",
        ],
    }


def download_s3_key(
    client,
    bucket: str,
    key: str,
    output: Path,
    request_payer: str | None = None,
    retries: int = 3,
) -> dict[str, object]:
    """Download one S3 key with retries."""
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and output.stat().st_size > 0:
        return {"key": key, "local_path": str(output), "status": "exists", "size_bytes": output.stat().st_size}
    last_error = ""
    for _ in range(retries):
        try:
            tmp = output.with_suffix(output.suffix + ".part")
            extra_args = {"RequestPayer": request_payer} if request_payer else None
            if extra_args:
                client.download_file(bucket, key, str(tmp), ExtraArgs=extra_args)
            else:
                client.download_file(bucket, key, str(tmp))
            tmp.replace(output)
            return {"key": key, "local_path": str(output), "status": "downloaded", "size_bytes": output.stat().st_size}
        except Exception as error:  # noqa: BLE001
            last_error = str(error)
            time.sleep(5)
    return {"key": key, "local_path": str(output), "status": "failed", "size_bytes": 0, "error": last_error}


def append_manifest(path: Path, row: dict[str, object]) -> None:
    """Append one download row."""
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    fieldnames = ["algorithm", "key", "local_path", "status", "size_bytes", "error"]
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow({field: row.get(field, "") for field in fieldnames})


def get_chunk_indices(chunklist_path: Path) -> list[int]:
    """Read all chunk indices from a NeMo chunklist."""
    data = np.load(chunklist_path, allow_pickle=True)
    if "unique_chunks" in data:
        return [int(x) for x in data["unique_chunks"]]
    return list(range(len(data["chunkfilesize"])))


def download_nemo_database(config: dict, algorithms: list[str], workers: int = 4) -> None:
    """Download complete NeMo database for selected algorithms."""
    data_dir = ensure_dir(project_path(config, config["nemo"]["data_dir"]))
    manifest_path = data_dir / "download_manifest.csv"
    summary_path = data_dir / "download_summary.json"
    client = make_s3_client()
    bucket = config["nemo"]["s3_root"].replace("s3://", "").strip("/").split("/")[0]
    request_payer = config["nemo"].get("request_payer")
    started = time.strftime("%Y-%m-%dT%H:%M:%S")
    summary: dict[str, object] = {"started": started, "bucket": bucket, "algorithms": {}}

    shared = download_s3_key(
        client,
        bucket,
        "MNI152_T1_1mm_brain.nii.gz",
        data_dir / "MNI152_T1_1mm_brain.nii.gz",
        request_payer=request_payer,
    )
    shared["algorithm"] = "shared"
    append_manifest(manifest_path, shared)
    if shared["status"] == "failed":
        summary["failed"] = True
        summary["error"] = str(shared.get("error", "unknown error"))
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        raise RuntimeError(
            "failed to access NeMo requester-pays S3 bucket. "
            "Configure AWS credentials with billing permission, then rerun the download."
        )

    for algorithm in algorithms:
        info = nemo_keys_for_algorithm(algorithm)
        algorithm_rows = []
        print(f"downloading NeMo metadata: {algorithm}")
        for key in info["metadata"]:
            output = data_dir / key
            row = download_s3_key(client, bucket, key, output, request_payer=request_payer)
            row["algorithm"] = algorithm
            append_manifest(manifest_path, row)
            algorithm_rows.append(row)
            if row["status"] == "failed":
                raise RuntimeError(f"failed to download {key}: {row.get('error')}")

        chunklist_key = "nemo_chunklist.npz" if algorithm == "ifod2act" else "nemo_sdstream_chunklist.npz"
        chunk_indices = get_chunk_indices(data_dir / chunklist_key)
        chunk_dir = str(info["chunk_dir"])
        chunk_keys = [f"{chunk_dir}/chunk{idx:05d}.npz" for idx in chunk_indices]
        print(f"downloading {len(chunk_keys)} NeMo chunks: {algorithm}")

        ok_count = 0
        failed_count = 0
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(download_s3_key, client, bucket, key, data_dir / key, request_payer): key
                for key in chunk_keys
            }
            for index, future in enumerate(as_completed(futures), start=1):
                row = future.result()
                row["algorithm"] = algorithm
                append_manifest(manifest_path, row)
                ok_count += int(row["status"] in {"exists", "downloaded"})
                failed_count += int(row["status"] == "failed")
                if index % 100 == 0:
                    print(f"{algorithm}: {index}/{len(chunk_keys)} chunks checked")
        summary["algorithms"][algorithm] = {
            "metadata_files": len(info["metadata"]),
            "chunk_files": len(chunk_keys),
            "chunk_ok": ok_count,
            "chunk_failed": failed_count,
        }
        if failed_count:
            raise RuntimeError(f"{algorithm} failed chunks: {failed_count}")

    summary["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"wrote {summary_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download DAT pilot reference data.")
    parser.add_argument("--config", default="config/dat_config.yaml")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--maps", action="store_true")
    parser.add_argument("--nemo", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    config = load_config(args.config)
    if args.all or args.maps:
        download_reference_maps(config, force=args.force)
    if args.all or args.nemo:
        download_nemo_database(config, config["nemo"]["algorithms"], workers=args.workers)


if __name__ == "__main__":
    main()
