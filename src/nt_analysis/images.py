from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np
from nilearn.image import resample_to_img


def load_img(path: str | Path) -> nib.Nifti1Image:
    """Load a NIfTI image."""
    return nib.load(str(path))


def save_img(img: nib.Nifti1Image, path: str | Path) -> Path:
    """Save a NIfTI image."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(img, str(path))
    return path


def resample_like(
    source: str | Path,
    target: str | Path,
    output: str | Path,
    interpolation: str,
) -> Path:
    """Resample source image to target grid."""
    source_img = load_img(source)
    target_img = load_img(target)
    resampled = resample_to_img(
        source_img,
        target_img,
        interpolation=interpolation,
        force_resample=True,
        copy_header=True,
    )
    return save_img(resampled, output)


def binarize_img(path: str | Path, output: str | Path) -> Path:
    """Binarize nonzero voxels."""
    img = load_img(path)
    data = (img.get_fdata() != 0).astype(np.uint8)
    out = nib.Nifti1Image(data, img.affine, img.header)
    out.set_data_dtype(np.uint8)
    return save_img(out, output)


def multiply_images(left: str | Path, right: str | Path, output: str | Path) -> Path:
    """Multiply two images on the same grid."""
    left_img = load_img(left)
    right_img = load_img(right)
    left_data = left_img.get_fdata()
    right_data = right_img.get_fdata()
    if left_data.shape != right_data.shape:
        raise ValueError("image shapes do not match")
    data = left_data * right_data
    out = nib.Nifti1Image(data.astype(np.float32), left_img.affine, left_img.header)
    out.set_data_dtype(np.float32)
    return save_img(out, output)


def lesion_volume_ml(path: str | Path) -> float:
    """Compute lesion volume in ml."""
    img = load_img(path)
    data = img.get_fdata() != 0
    voxel_volume = abs(np.linalg.det(img.affine[:3, :3]))
    return float(data.sum() * voxel_volume / 1000.0)
