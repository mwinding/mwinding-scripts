#!/usr/bin/env python3
"""
CLAHE for huge BigTIFF EF-SEM volumes using tifffile's Zarr backend.

- Reads slices lazily: arr[z, :, :]
- Supports TIFF or BDV output
- Uses Fiji-equivalent CLAHE defaults:
    blocksize=127, histogram=256, maximum=3
"""

import argparse
from pathlib import Path
import numpy as np
import tifffile
import h5py
from skimage import exposure


# ============================================================
# Fiji-equivalent defaults
# ============================================================

# Fiji CLAHE:
#   blocksize = 127  → kernel size
#   histogram = 256  → nbins
#   maximum   = 3    → clip limit = 3/256
DEFAULT_KERNEL = 127
DEFAULT_NBINS = 256
DEFAULT_CLIP = 3 / 256

# BDV chunk size (good for large EM volumes)
BDV_CHUNKS = (1, 512, 512)


# ============================================================
# CLAHE processing
# ============================================================

def apply_clahe(img, dtype, kernel, clip, nbins):
    """Apply CLAHE with Fiji-matching parameters."""
    out = exposure.equalize_adapthist(
        img,
        kernel_size=kernel,
        clip_limit=clip,
        nbins=nbins
    )

    # Convert back to original integer range (e.g. uint16)
    if np.issubdtype(dtype, np.integer):
        out = (out * np.iinfo(dtype).max).astype(dtype)

    return out


# ============================================================
# TIFF output
# ============================================================

def clahe_to_tiff(input_path, output_path, kernel, clip, nbins):
    """Apply CLAHE slice-by-slice and write a TIFF stack."""
    with tifffile.TiffFile(input_path) as tif:

        series = tif.series[0]
        arr = series.aszarr()          # Zarr-backed access (lazy)
        dtype = series.dtype
        shape = series.shape           # (Z, Y, X)
        Z = shape[0]

        print(f"BigTIFF detected: shape={shape}, dtype={dtype}")
        print(f"Writing TIFF to {output_path}")

        # First slice → create output file
        img = arr[0, :, :]
        out = apply_clahe(img, dtype, kernel, clip, nbins)
        tifffile.imwrite(output_path, out, imagej=True)

        # Remaining slices → append
        for z in range(1, Z):
            print(f"Slice {z+1}/{Z}")
            img = arr[z, :, :]
            out = apply_clahe(img, dtype, kernel, clip, nbins)
            tifffile.imwrite(output_path, out, append=True)

    print("TIFF CLAHE complete.")


# ============================================================
# BDV output
# ============================================================

def clahe_to_bdv(input_path, bdv_dir, kernel, clip, nbins):
    """Apply CLAHE slice-by-slice and write BDV (XML + HDF5)."""

    bdv_dir.mkdir(parents=True, exist_ok=True)
    xml_path = bdv_dir / "dataset.xml"
    h5_path = bdv_dir / "dataset.h5"

    with tifffile.TiffFile(input_path) as tif, h5py.File(h5_path, "w") as h5:

        series = tif.series[0]
        arr = series.aszarr()
        dtype = series.dtype
        shape = series.shape           # (Z, Y, X)
        Z, Y, X = shape

        print(f"BigTIFF detected: shape={shape}, dtype={dtype}")
        print(f"Writing BDV dataset to {bdv_dir}")

        # Create BDV dataset (single mipmap level s0)
        ds = h5.create_dataset(
            "s0",
            shape=shape,
            dtype=dtype,
            chunks=BDV_CHUNKS,
            compression="gzip",
            compression_opts=1
        )

        # Process each slice
        for z in range(Z):
            print(f"Slice {z+1}/{Z}")
            img = arr[z, :, :]
            out = apply_clahe(img, dtype, kernel, clip, nbins)
            ds[z] = out

    # Minimal BDV XML descriptor
    xml = f"""
<SpimData version="0.2">
  <BasePath type="relative">.</BasePath>
  <SequenceDescription>
    <ImageLoader format="bdv.hdf5">
      <hdf5>{h5_path.name}</hdf5>
    </ImageLoader>
    <ViewSetups>
      <ViewSetup>
        <id>0</id>
        <name>CLAHE</name>
      </ViewSetup>
    </ViewSetups>
    <Timepoints type="range">
      <first>0</first>
      <last>0</last>
    </Timepoints>
  </SequenceDescription>
  <ViewRegistrations>
    <ViewRegistration timepoint="0" setup="0">
      <affine>1 0 0 0   0 1 0 0   0 0 1 0</affine>
    </ViewRegistration>
  </ViewRegistrations>
</SpimData>
"""
    xml_path.write_text(xml)
    print("BDV CLAHE complete.")


# ============================================================
# Main CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="CLAHE over huge BigTIFF stacks via tifffile's Zarr backend."
    )
    parser.add_argument("-i", "--input", required=True, help="Input BigTIFF")
    parser.add_argument("-d", "--output-dir", required=True, help="Output directory")
    parser.add_argument(
        "-o", "--output-format",
        required=True,
        choices=["tiff", "bdv"],
        help="Output format: tiff | bdv"
    )

    parser.add_argument(
        "--kernel",
        type=int,
        default=DEFAULT_KERNEL,
        help="CLAHE kernel size (Fiji blocksize=127)"
    )
    parser.add_argument(
        "--nbins",
        type=int,
        default=DEFAULT_NBINS,
        help="Histogram bins (Fiji histogram=256)"
    )
    parser.add_argument(
        "--clip",
        type=float,
        default=DEFAULT_CLIP,
        help="Clip limit (Fiji maximum=3 → 3/256 ≈ 0.0117)"
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    if args.output_format == "tiff":
        out_tif = output_dir / f"CLAHE_{input_path.name}"
        clahe_to_tiff(input_path, out_tif, args.kernel, args.clip, args.nbins)
    else:
        bdv_subdir = output_dir / "CLAHE_BDV"
        clahe_to_bdv(input_path, bdv_subdir, args.kernel, args.clip, args.nbins)


if __name__ == "__main__":
    main()
