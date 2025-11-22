#!/usr/bin/env python3
"""
CLAHE for huge EF-SEM BigTIFF volumes.

- Loads volume via memory-mapping
- Applies 2D CLAHE slice-by-slice
- Saves each processed slice as:
      output_dir/slice_XXXX.tif

Supports restricting to a Z-range with --z-start / --z-end
so that Slurm array jobs can process disjoint subsets.
"""

import argparse
from pathlib import Path
import numpy as np
import tifffile
from skimage import exposure


# Fiji-equivalent CLAHE defaults
DEFAULT_KERNEL = 127       # Fiji blocksize
DEFAULT_NBINS  = 256       # Histogram bins
DEFAULT_CLIP   = 3 / 256   # Fiji "maximum=3"


def apply_clahe(img, dtype, kernel, clip, nbins):
    """Apply 2D CLAHE to one slice."""
    out = exposure.equalize_adapthist(
        img,
        kernel_size=kernel,
        clip_limit=clip,
        nbins=nbins,
    )
    if np.issubdtype(dtype, np.integer):
        out = (out * np.iinfo(dtype).max).astype(dtype)
    return out


def clahe_slices(
    input_path: Path,
    output_dir: Path,
    kernel: int,
    clip: float,
    nbins: int,
    z_start: int | None = None,
    z_end: int | None = None,
) -> None:
    """Apply CLAHE to slices [z_start, z_end] (inclusive)."""

    print(f"Loading (memmap) volume from: {input_path}")
    vol = tifffile.imread(str(input_path), out="memmap")   # shape (Z, Y, X)
    dtype = vol.dtype
    Z = vol.shape[0]

    if z_start is None:
        z_start = 0
    if z_end is None:
        z_end = Z - 1

    # Clamp to valid range
    z_start = max(0, z_start)
    z_end   = min(Z - 1, z_end)

    if z_start > z_end:
        print(f"Nothing to do: z_start ({z_start}) > z_end ({z_end})")
        return

    print(f"Volume shape = {vol.shape}, dtype={dtype}")
    print(f"Saving slices {z_start}–{z_end} to: {output_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    for z in range(z_start, z_end + 1):
        print(f"Slice {z+1}/{Z} (z={z})")

        img = np.array(vol[z])        # materialise slice
        out = apply_clahe(img, dtype, kernel, clip, nbins)

        out_path = output_dir / f"slice_{z:04d}.tif"
        tifffile.imwrite(str(out_path), out)

    print("Finished writing assigned slices.")


def main():
    parser = argparse.ArgumentParser(
        description="CLAHE slice-by-slice for huge EF-SEM BigTIFF volumes."
    )
    parser.add_argument("-i", "--input", required=True, help="Input BigTIFF")
    parser.add_argument("-d", "--output-dir", required=True,
                        help="Output directory for slices")

    parser.add_argument("--kernel", type=int, default=DEFAULT_KERNEL)
    parser.add_argument("--nbins",  type=int, default=DEFAULT_NBINS)
    parser.add_argument("--clip",   type=float, default=DEFAULT_CLIP)

    parser.add_argument("--z-start", type=int, default=None,
                        help="First Z slice (0-based, inclusive). Default: 0")
    parser.add_argument("--z-end",   type=int, default=None,
                        help="Last Z slice (0-based, inclusive). Default: last")

    args = parser.parse_args()

    clahe_slices(
        Path(args.input),
        Path(args.output_dir),
        args.kernel,
        args.clip,
        args.nbins,
        args.z_start,
        args.z_end,
    )


if __name__ == "__main__":
    main()
