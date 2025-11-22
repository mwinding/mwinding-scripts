#!/usr/bin/env python3
"""
Apply CLAHE to huge TIFF stacks (>100 GB) using Fiji-equivalent parameters.

Outputs either:
  - a TIFF stack (-o tiff)
  - a BDV HDF5 dataset (-o bdv)

Runs slice-by-slice → tiny memory footprint.
"""

import argparse
import numpy as np
import tifffile
import h5py
from skimage import exposure
from pathlib import Path


# ============================================================
#                   USER-FACING DEFAULTS
# ============================================================

# Fiji CLAHE defaults:
#   blocksize = 127 → kernel size
#   histogram = 256 → nbins
#   maximum   = 3   → clip limit = 3/256
DEFAULT_KERNEL = 127
DEFAULT_NBINS = 256
DEFAULT_CLIP = 3 / 256

# BDV chunk size (good for large EM volumes)
BDV_CHUNKS = (1, 512, 512)


# ============================================================
#                   CORE PROCESSING FUNCTIONS
# ============================================================

def apply_clahe(img, dtype, kernel_size, clip_limit, nbins):
    """CLAHE matched to Fiji settings."""
    out = exposure.equalize_adapthist(
        img,
        kernel_size=kernel_size,
        clip_limit=clip_limit,
        nbins=nbins
    )

    # Convert back to original integer range (e.g. uint16)
    if np.issubdtype(dtype, np.integer):
        out = (out * np.iinfo(dtype).max).astype(dtype)

    return out


def clahe_to_tiff(input_path, output_path, kernel_size, clip_limit, nbins):
    """Write CLAHE-processed TIFF volume."""
    with tifffile.TiffFile(input_path) as tif:

        n = len(tif.pages)
        dtype = tif.pages[0].asarray().dtype

        print(f"Processing {n} slices → TIFF")

        # First slice → create output file
        img = tif.pages[0].asarray()
        out = apply_clahe(img, dtype, kernel_size, clip_limit, nbins)
        tifffile.imwrite(output_path, out, imagej=True)

        # Append remaining slices
        for i in range(1, n):
            print(f"Slice {i+1}/{n}")
            img = tif.pages[i].asarray()
            out = apply_clahe(img, dtype, kernel_size, clip_limit, nbins)
            tifffile.imwrite(output_path, out, append=True)

    print(f"Finished: {output_path}")


def clahe_to_bdv(input_path, bdv_dir, kernel_size, clip_limit, nbins):
    """Write CLAHE-processed BDV (XML + HDF5)."""

    bdv_dir.mkdir(parents=True, exist_ok=True)
    xml_path = bdv_dir / "dataset.xml"
    h5_path = bdv_dir / "dataset.h5"

    with tifffile.TiffFile(input_path) as tif, h5py.File(h5_path, "w") as h5:

        n = len(tif.pages)
        first = tif.pages[0].asarray()
        dtype = first.dtype
        shape = (n, first.shape[0], first.shape[1])

        print(f"Processing {n} slices → BDV ({shape})")

        # Create BDV dataset
        ds = h5.create_dataset(
            "s0",
            shape=shape,
            dtype=dtype,
            chunks=BDV_CHUNKS,
            compression="gzip",
            compression_opts=1
        )

        # Fill dataset slice by slice
        for i in range(n):
            print(f"Slice {i+1}/{n}")
            img = tif.pages[i].asarray()
            out = apply_clahe(img, dtype, kernel_size, clip_limit, nbins)
            ds[i] = out

    # Minimal BDV XML
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
    print(f"Finished BDV export: {bdv_dir}")


# ============================================================
#                           MAIN
# ============================================================

def main():

    # ----------------- CLI -----------------
    parser = argparse.ArgumentParser(
        description="CLAHE processing for huge TIFF stacks (>100 GB)"
    )

    parser.add_argument("-i", "--input", required=True, help="Input TIFF stack")
    parser.add_argument("-d", "--output-dir", required=True, help="Output directory")
    parser.add_argument("-o", "--output-format", required=True,
                        choices=["tiff", "bdv"],
                        help="Output type: tiff or bdv")

    parser.add_argument("--kernel", type=int, default=DEFAULT_KERNEL,
                        help="CLAHE kernel size (Fiji blocksize=127)")
    parser.add_argument("--nbins", type=int, default=DEFAULT_NBINS,
                        help="Histogram bins (Fiji histogram=256)")
    parser.add_argument("--clip", type=float, default=DEFAULT_CLIP,
                        help="Clip limit (Fiji maximum=3 → clip=3/256≈0.0117)")

    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    # ----------------- RUN -----------------
    if args.output_format == "tiff":
        out_path = output_dir / f"CLAHE_{input_path.name}"
        clahe_to_tiff(input_path, out_path, args.kernel, args.clip, args.nbins)

    else:  # BDV
        bdv_dir = output_dir / "CLAHE_BDV"
        clahe_to_bdv(input_path, bdv_dir, args.kernel, args.clip, args.nbins)


if __name__ == "__main__":
    main()
