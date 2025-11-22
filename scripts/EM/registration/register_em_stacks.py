#!/usr/bin/env python3
"""
Jitter correction for cropped FIB-SEM ROIs.

Input:
  - A directory containing stacks produced by your CLAHE+ROI script, e.g.:
      ROI_soma1.tif
      ROI_soma2.tif
      ...
    Each file is a 3D stack with shape (Z, Y, X).

For each ROI stack, this script:
  1. Loads the volume.
  2. Estimates per-slice global XY shifts using phase correlation
     to a local z-window "template".
  3. Smooths the shift vector as a function of z.
  4. Applies the smoothed shifts to warp the stack and reduce jitter.
  5. Writes ROI_<Name>_aligned.tif in the output directory.

Dependencies:
  - numpy
  - scipy
  - scikit-image
  - tifffile

This is intentionally simple and robust:
  - Only global translation per slice (no rotation, no non-linear warping).
  - Local z-window templates to avoid cumulative drift.
"""

import argparse
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates, gaussian_filter1d
from skimage.registration import phase_cross_correlation
import tifffile


def load_stack(path: Path) -> np.ndarray:
    """Load a 3D (Z, Y, X) stack from a TIFF."""
    vol = tifffile.imread(str(path))
    if vol.ndim != 3:
        raise ValueError(f"Expected 3D stack for {path}, got shape {vol.shape}")
    return vol


def estimate_per_slice_shifts(
    vol: np.ndarray,
    window_radius: int = 3,
    blur_sigma: float = 1.0,
    upsample_factor: int = 10,
    max_abs_shift: float | None = None,
) -> np.ndarray:
    """
    Estimate global XY translation per slice.

    Strategy:
      - For each z, build a local template as the median of slices
        z-window_radius ... z+window_radius (clipped to valid range).
      - Compute shift that aligns slice z to this template via phase correlation.

    Parameters
    ----------
    vol : ndarray, shape (Z, Y, X)
    window_radius : int
        Radius for the local z-window used to construct templates.
    blur_sigma : float
        Gaussian blur (in pixels) applied in XY before registration.
        Helps robustness for noisy FIB-SEM.
    upsample_factor : int
        Subpixel resolution for phase_cross_correlation.
    max_abs_shift : float or None
        If not None, clamp |dx| and |dy| to this value. Larger estimates
        are treated as unreliable and set to 0.

    Returns
    -------
    shifts : ndarray, shape (Z, 2)
        Per-slice shifts [dy, dx], meaning slice z should be sampled
        from coordinates (y - dy, x - dx) in the original volume.
    """
    Z, Y, X = vol.shape
    vol_f = vol.astype(np.float32, copy=False)

    if blur_sigma is not None and blur_sigma > 0:
        vol_blur = np.empty_like(vol_f)
        for z in range(Z):
            vol_blur[z] = gaussian_filter(vol_f[z], sigma=blur_sigma)
    else:
        vol_blur = vol_f

    shifts = np.zeros((Z, 2), dtype=np.float32)

    for z in range(Z):
        z_min = max(0, z - window_radius)
        z_max = min(Z, z + window_radius + 1)
        template = np.median(vol_blur[z_min:z_max], axis=0)

        mov = vol_blur[z]

        # skip if almost constant
        if mov.std() < 1e-6 or template.std() < 1e-6:
            shifts[z] = 0.0
            continue

        shift, error, _ = phase_cross_correlation(
            template, mov, upsample_factor=upsample_factor
        )
        dy, dx = shift

        if max_abs_shift is not None:
            if abs(dy) > max_abs_shift or abs(dx) > max_abs_shift:
                # Treat as unreliable
                dy, dx = 0.0, 0.0

        shifts[z, 0] = dy
        shifts[z, 1] = dx

    return shifts


def smooth_shifts(shifts: np.ndarray, sigma_z: float = 2.0) -> np.ndarray:
    """
    Smooth per-slice shift vectors along z.

    Parameters
    ----------
    shifts : ndarray, shape (Z, 2)
        Per-slice [dy, dx] shifts.
    sigma_z : float
        Standard deviation (in slices) of Gaussian smoothing along z.

    Returns
    -------
    smoothed : ndarray, shape (Z, 2)
        Smoothed shifts.
    """
    smoothed = np.empty_like(shifts)
    for dim in range(2):
        smoothed[:, dim] = gaussian_filter1d(
            shifts[:, dim], sigma=sigma_z, mode="nearest"
        )
    return smoothed


def apply_shifts(
    vol: np.ndarray,
    shifts: np.ndarray,
    order: int = 1,
    mode: str = "nearest",
) -> np.ndarray:
    """
    Apply per-slice global translation to a stack using map_coordinates.

    Parameters
    ----------
    vol : ndarray, shape (Z, Y, X)
    shifts : ndarray, shape (Z, 2)
        Per-slice [dy, dx] shifts.
    order : int
        Interpolation order for map_coordinates (0=nearest, 1=linear, ...).
    mode : str
        Boundary mode for map_coordinates.

    Returns
    -------
    warped : ndarray, shape (Z, Y, X)
    """
    Z, Y, X = vol.shape
    yy, xx = np.meshgrid(np.arange(Y), np.arange(X), indexing="ij")

    warped = np.empty_like(vol)

    for z in range(Z):
        dy, dx = shifts[z]
        coords_y = yy - dy
        coords_x = xx - dx
        coords = np.array([coords_y, coords_x])
        warped[z] = map_coordinates(
            vol[z], coords, order=order, mode=mode
        )

    return warped


def process_roi_stack(
    in_path: Path,
    out_path: Path,
    window_radius: int,
    blur_sigma: float,
    upsample_factor: int,
    max_abs_shift: float | None,
    sigma_z: float,
):
    """
    Run jitter correction on a single ROI stack.
    """
    print(f"\n=== Processing {in_path.name} ===")
    vol = load_stack(in_path)
    print(f"  Input shape: {vol.shape} (Z, Y, X)")

    shifts = estimate_per_slice_shifts(
        vol,
        window_radius=window_radius,
        blur_sigma=blur_sigma,
        upsample_factor=upsample_factor,
        max_abs_shift=max_abs_shift,
    )
    print(f"  Raw shift stats (pixels): "
          f"dy [{shifts[:,0].min():.2f}, {shifts[:,0].max():.2f}], "
          f"dx [{shifts[:,1].min():.2f}, {shifts[:,1].max():.2f}]")

    shifts_smooth = smooth_shifts(shifts, sigma_z=sigma_z)
    print(f"  Smoothed shifts (first 5 slices):")
    for z in range(min(5, shifts_smooth.shape[0])):
        dy, dx = shifts_smooth[z]
        print(f"    z={z}: dy={dy:.3f}, dx={dx:.3f}")

    aligned = apply_shifts(vol, shifts_smooth, order=1, mode="nearest")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(str(out_path), aligned.astype(vol.dtype))
    print(f"  Wrote aligned stack: {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Jitter-correct ROI stacks (output from CLAHE+ROI crop script)."
    )
    parser.add_argument(
        "--roi-dir",
        required=True,
        help="Directory containing ROI_*.tif stacks.",
    )
    parser.add_argument(
        "--pattern",
        default="ROI_*.tif",
        help="Glob pattern for ROI stacks (default: ROI_*.tif).",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory to write aligned ROI stacks.",
    )
    parser.add_argument(
        "--window-radius",
        type=int,
        default=3,
        help="Radius (in slices) for local z-window used as template (default: 3).",
    )
    parser.add_argument(
        "--blur-sigma",
        type=float,
        default=1.0,
        help="Gaussian blur sigma (pixels) before registration (default: 1.0).",
    )
    parser.add_argument(
        "--upsample-factor",
        type=int,
        default=10,
        help="Subpixel upsample factor for phase_cross_correlation (default: 10).",
    )
    parser.add_argument(
        "--max-abs-shift",
        type=float,
        default=20.0,
        help="Maximum allowed |shift| in pixels; larger values are zeroed (default: 20). "
             "Set to 0 or negative to disable.",
    )
    parser.add_argument(
        "--sigma-z",
        type=float,
        default=2.0,
        help="Gaussian smoothing sigma along z for shift vectors (default: 2.0).",
    )

    args = parser.parse_args()

    roi_dir = Path(args.roi_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    roi_paths = sorted(roi_dir.glob(args.pattern))
    if not roi_paths:
        raise RuntimeError(f"No files matching {args.pattern} in {roi_dir}")

    print(f"Found {len(roi_paths)} ROI stack(s) in {roi_dir}")

    max_abs_shift = args.max_abs_shift
    if max_abs_shift is not None and max_abs_shift <= 0:
        max_abs_shift = None

    for in_path in roi_paths:
        # Preserve base name, add _aligned before extension
        stem = in_path.stem
        suffix = in_path.suffix
        out_path = out_dir / f"{stem}_aligned{suffix}"

        process_roi_stack(
            in_path=in_path,
            out_path=out_path,
            window_radius=args.window_radius,
            blur_sigma=args.blur_sigma,
            upsample_factor=args.upsample_factor,
            max_abs_shift=max_abs_shift,
            sigma_z=args.sigma_z,
        )

    print("\nAll ROI stacks processed.")


if __name__ == "__main__":
    main()
