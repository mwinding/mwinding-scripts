#!/usr/bin/env python3
"""
CLAHE for huge EF-SEM BigTIFF volumes.

- Loads the volume as a memory-mapped array using tifffile.imread(..., out="memmap")
- Applies 2D CLAHE slice-by-slice along Z
- Outputs either:
    - a TIFF stack  (-o tiff)
    - a BDV HDF5+XML dataset  (-o bdv)

CLAHE parameters match Fiji defaults:
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

# BDV chunk size (reasonable for large EM volumes)
BDV_CHUNKS = (1, 512, 512)


# ============================================================
# CLAHE processing
# ============================================================

def apply_clahe(img, dtype, kernel, clip, nbins):
    """Apply CLAHE with parameters chosen to match Fiji."""
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
# TIFF output
# ============================================================

def clahe_to_tiff(input_path: Path, output_path: Path,
                  kernel: int, clip: float, nbins: int) -> None:
    """
    Apply CLAHE to a big EF-SEM TIFF and write a TIFF stack.

    Uses tifffile.imread(..., out="memmap") to avoid loading
    the entire volume into RAM at once.
    """
    print(f"Loading volume (memory-mapped) from: {input_path}")
    vol = tifffile.imread(str(input_path), out="memmap")  # shape (Z, Y, X)
    dtype = vol.dtype
    shape = vol.shape
    Z, Y, X = shape

    print(f"Volume shape: {shape}, dtype: {dtype}")
    print(f"Writing TIFF to: {output_path}")

    # First slice → create file
    print(f"Slice 1/{Z}")
    img0 = np.array(vol[0])  # materialise slice
    out0 = apply_clahe(img0, dtype, kernel, clip, nbins)
    tifffile.imwrite(str(output_path), out0, imagej=True)

    # Remaining slices → append
    for z in range(1, Z):
        print(f"Slice {z+1}/{Z}")
        img = np.array(vol[z])  # materialise slice
        out = apply_clahe(img, dtype, kernel, clip, nbins)
        tifffile.imwrite(str(output_path), out, append=True)

    print("TIFF CLAHE complete.")


# ============================================================
# BDV output
# ============================================================

def clahe_to_bdv(input_path: Path, bdv_dir: Path,
                 kernel: int, clip: float, nbins: int) -> None:
    """
    Apply CLAHE to a big EF-SEM TIFF and write a BDV (XML+HDF5) dataset.

    Again uses tifffile.imread(..., out="memmap") to keep RAM usage under control.
    """

    bdv_dir.mkdir(parents=True, exist_ok=True)
    xml_path = bdv_dir / "dataset.xml"
    h5_path = bdv_dir / "dataset.h5"

    print(f"Loading volume (memory-mapped) from: {input_path}")
    vol = tifffile.imread(str(input_path), out="memmap")  # (Z, Y, X)
    dtype = vol.dtype
    shape = vol.shape
    Z, Y, X = shape

    print(f"Volume shape: {shape}, dtype: {dtype}")
    print(f"Writing BDV dataset to: {bdv_dir}")

    with h5py.File(h5_path, "w") as h5:
        ds = h5.create_dataset(
            "s0",
            shape=shape,
            dtype=dtype,
            chunks=BDV_CHUNKS,
            compression="gzip",
            compression_opts=1,
        )

        for z in range(Z):
            print(f"Slice {z+1}/{Z}")
            img = np.array(vol[z])  # materialise slice
            out = apply_clahe(img, dtype, kernel, clip, nbins)
            ds[z] = out

    # Minimal BDV XML descriptor
    xml_content = f"""
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
      <affine>1 0 0 0  0 1 0 0  0 0 1 0</affine>
    </ViewRegistration>
  </ViewRegistrations>
</SpimData>
"""
    xml_path.write_text(xml_content)
    print("BDV CLAHE complete.")


# ============================================================
# Main CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="CLAHE over huge EF-SEM BigTIFF volumes (memory-mapped)."
    )
    parser.add_argument("-i", "--input", required=True,
                        help="Input BigTIFF path")
    parser.add_argument("-d", "--output-dir", required=True,
                        help="Output directory")
    parser.add_argument(
        "-o", "--output-format",
        required=True,
        choices=["tiff", "bdv"],
        help="Output format: tiff | bdv"
    )

    parser.add_argument(
        "--kernel", type=int, default=DEFAULT_KERNEL,
        help="CLAHE kernel size (Fiji blocksize=127)"
    )
    parser.add_argument(
        "--nbins", type=int, default=DEFAULT_NBINS,
        help="Histogram bins (Fiji histogram=256)"
    )
    parser.add_argument(
        "--clip", type=float, default=DEFAULT_CLIP,
        help="Clip limit (Fiji maximum=3 → 3/256 ≈ 0.0117)"
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.output_format == "tiff":
        out_tif = output_dir / f"CLAHE_{input_path.name}"
        clahe_to_tiff(input_path, out_tif, args.kernel, args.clip, args.nbins)
    else:
        bdv_subdir = output_dir / "CLAHE_BDV"
        clahe_to_bdv(input_path, bdv_subdir, args.kernel, args.clip, args.nbins)


if __name__ == "__main__":
    main()
