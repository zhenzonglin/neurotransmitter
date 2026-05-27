#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nt_analysis.config import ensure_dir, load_config, project_path


ATLASPACK_URLS = {
    "repo": "https://github.com/PennLINC/AtlasPack.git",
    "nii": "tpl-MNI152NLin6Asym_atlas-4S156Parcels_res-01_dseg.nii.gz",
    "tsv": "atlas-4S156Parcels_dseg.tsv",
    "json": "tpl-MNI152NLin6Asym_atlas-4S156Parcels_dseg.json",
}

HANSEN_BASE_URL = "https://raw.githubusercontent.com/netneurolab/hansen_receptors/main/data/PET_nifti_images"
NEUROVAULT_IMAGE_API = "https://neurovault.org/api/images/{image_id}/"


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
    keys = ["nii", "tsv", "json"]
    outputs = {
        "nii": project_path(config, config["atlases"]["atlas4s156"]["nii"]),
        "tsv": project_path(config, config["atlases"]["atlas4s156"]["tsv"]),
        "json": project_path(config, config["atlases"]["atlas4s156"]["json"]),
    }
    if not force and all(path.exists() and path.stat().st_size > 0 for path in outputs.values()):
        return [
            {
                "name": f"atlas4s156_{key}",
                "url": f"{ATLASPACK_URLS['repo']}::{ATLASPACK_URLS[key]}",
                "path": str(path),
                "status": "exists",
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for key, path in outputs.items()
        ]

    datalad_bin = shutil.which("datalad")
    if datalad_bin is None:
        raise RuntimeError("datalad is required to download AtlasPack annex content")

    atlas_target = outputs["nii"]
    if force and atlas_target.exists():
        atlas_target.unlink()

    if not repo_dir.exists():
        # AtlasPack的NIfTI是git-annex内容，必须用DataLad获取实体文件
        run_command([datalad_bin, "clone", ATLASPACK_URLS["repo"], str(repo_dir)])

    paths = [ATLASPACK_URLS[key] for key in keys]
    run_command([datalad_bin, "get", *paths], cwd=repo_dir)

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
    last_error = ""
    data = None
    for _ in range(5):
        try:
            response = requests.get(api_url, headers={"User-Agent": "Mozilla/5.0"}, timeout=120)
            response.raise_for_status()
            data = response.json()
            break
        except Exception as error:  # noqa: BLE001
            last_error = str(error)
            time.sleep(5)
    if data is None:
        try:
            request = urllib.request.Request(api_url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(request, timeout=120) as response:
                data = json.loads(response.read().decode("utf-8"))
        except Exception as error:  # noqa: BLE001
            raise RuntimeError(f"failed to read NeuroVault metadata {api_url}: {last_error}; {error}") from error
    return data["file"].replace("http://", "https://")


def neurotransmitter_specs(config: dict) -> list[dict[str, object]]:
    """Return configured neurotransmitter maps."""
    specs = config.get("neurotransmitters", [])
    if not specs:
        specs = [
            {
                "id": "dat",
                "label": "dopamine transporter",
                "hansen_file": "DAT_fpcit_hc174_dukart_spect.nii",
                "alves_name": "DAT",
                "neurovault_id": 802963,
            }
        ]
    return specs


def download_neurotransmitter_maps(config: dict, force: bool = False) -> list[dict[str, object]]:
    """Download Hansen and Alves maps for all configured neurotransmitters."""
    raw_dir = ensure_dir(project_path(config, config["atlases"]["raw_dir"]))
    hansen_dir = ensure_dir(raw_dir / "hansen")
    alves_dir = ensure_dir(raw_dir / "alves")
    sources = []
    for spec in neurotransmitter_specs(config):
        nt_id = str(spec["id"])
        hansen_file = str(spec["hansen_file"])
        alves_name = str(spec["alves_name"])
        neurovault_id = int(spec["neurovault_id"])
        hansen_url = f"{HANSEN_BASE_URL}/{hansen_file}"
        hansen_output = hansen_dir / hansen_file
        if force and hansen_output.exists():
            hansen_output.unlink()
        result = download_url(hansen_url, hansen_output)
        result.update({"name": f"{nt_id}_hansen", "url": hansen_url, "nt_id": nt_id, "source": "hansen"})
        sources.append(result)

        api_url = NEUROVAULT_IMAGE_API.format(image_id=neurovault_id)
        alves_url = get_neurovault_file_url(api_url)
        alves_output = alves_dir / f"functionnectome_anat_{alves_name}.nii.gz"
        if force and alves_output.exists():
            alves_output.unlink()
        result = download_url(alves_url, alves_output)
        result.update({"name": f"{nt_id}_alves", "url": alves_url, "nt_id": nt_id, "source": "alves"})
        sources.append(result)
    return sources


def download_reference_maps(config: dict, force: bool = False) -> None:
    """Download AtlasPack and configured neurotransmitter maps."""
    raw_dir = ensure_dir(project_path(config, config["atlases"]["raw_dir"]))
    sources = download_atlaspack(config, force=force)
    sources.extend(download_neurotransmitter_maps(config, force=force))
    sources_path = raw_dir / "sources.json"
    sources_path.write_text(json.dumps(sources, indent=2), encoding="utf-8")
    print(f"wrote {sources_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download DAT pilot reference data.")
    parser.add_argument("--config", default="config/dat_config.yaml")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--maps", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.all or args.maps:
        download_reference_maps(config, force=args.force)


if __name__ == "__main__":
    main()
