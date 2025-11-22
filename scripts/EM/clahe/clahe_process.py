#!/usr/bin/env python3
"""
CLAHE for huge EF-SEM BigTIFF volumes.

- Loads volume via memory-mapping
- Applies 2D CLAHE slice-by-slice
- Saves each processed slice as:
      output_dir/slice_XXXX.tif

Much simpler and avoids any TIFF append or metadata issues.
"""

import argparse
from pathlib import Path
import numpy as np
import tifffile
from skimage import exposure


# ============================================================
# Fiji-equivalent CLAHE defaults
# ============================================================

DEFAULT_KERNEL = 127       # Fiji blocksize
DEFAULT_NBINS  = 256       # Histogram bins
DEFAULT_CLIP   = 3 / 256   # Fiji "maximum=3" → clip limit


# ============================================================
# CLAHE helper
# ============================================================

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


# ============================================================
# Main slice-writer
# ============================================================

def clahe_slices(input_path, output_dir, kernel, clip, nbins):
    print(f"Loading (memmap) volume from: {input_path}")
    vol = tifffile.imread(str(input_path), out="memmap")   # shape (Z, Y, X)
    dtype = vol.dtype
    Z = vol.shape[0]

    print(f"Volume shape = {vol.shape}, dtype={dtype}")
    print(f"Saving slices to: {output_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    for z in range(Z):
        print(f"Slice {z+1}/{Z}")

        img = np.array(vol[z])        # materialise slice
        out = apply_clahe(img, dtype, kernel, clip, nbins)

        out_path = output_dir / f"slice_{z:04d}.tif"
        tifffile.imwrite(str(out_path), out)

    print("Finished writing all slices.")


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="CLAHE slice-by-slice for huge EF-SEM BigTIFF volumes."
    )
    parser.add_argument("-i", "--input", required=True, help="Input BigTIFF")
    parser.add_argument("-d", "--output-dir", required=True, help="Output directory for slices")

    parser.add_argument("--kernel", type=int, default=DEFAULT_KERNEL)
    parser.add_argument("--nbins",  type=int, default=DEFAULT_NBINS)
    parser.add_argument("--clip",   type=float, default=DEFAULT_CLIP)

    args = parser.parse_args()

    clahe_slices(
        Path(args.input),
        Path(args.output_dir),
        args.kernel,
        args.clip,
        args.nbins,
    )


if __name__ == "__main__":
    main()
