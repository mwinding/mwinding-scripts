#!/usr/bin/env python3
"""
CLAHE for huge EF-SEM BigTIFF volumes.

Correct slice access method for your file:
    frame = tif.series[0].pages[z]
    img = frame.asarray()

This works for tiled single-image BigTIFFs where Zarr fails.
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

DEFAULT_KERNEL = 127      # blocksize=127
DEFAULT_NBINS = 256       # histogram=256
DEFAULT_CLIP = 3 / 256    # maximum=3 (clip limit)
BDV_CHUNKS = (1, 512, 512)


# ============================================================
# CLAHE processing
# ============================================================

def apply_clahe(img, dtype, kernel, clip, nbins):
    out = exposure.equalize_adapthist(
        img,
        kernel_size=kernel,
        clip_limit=clip,
        nbins=nbins
    )
    if np.issubdtype(dtype, np.integer):
        out = (out * np.iinfo(dtype).max).astype(dtype)
    return out


# ============================================================
# TIFF output
# ============================================================

def clahe_to_tiff(input_path, output_path, kernel, clip, nbins):
    with tifffile.TiffFile(input_path) as tif:

        series = tif.series[0]
        dtype = series.dtype
        shape = series.shape             # (Z, Y, X)
        Z = shape[0]

        print(f"BigTIFF detected: shape={shape}, dtype={dtype}")
        print(f"Writing TIFF to {output_path}")

        # First slice
        frame = series.pages[0]
        img = frame.asarray()
        out = apply_clahe(img, dtype, kernel, clip, nbins)
        tifffile.imwrite(output_path, out, imagej=True)

        # Remaining slices
        for z in range(1, Z):
            print(f"Slice {z+1}/{Z}")
            frame = series.pages[z]
            img = frame.asarray()
            out = apply_clahe(img, dtype, kernel, clip, nbins)
            tifffile.imwrite(output_path, out, append=True)

    print("TIFF CLAHE complete.")


# ============================================================
# BDV output
# ============================================================

def clahe_to_bdv(input_path, bdv_dir, kernel, clip, nbins):

    bdv_dir.mkdir(parents=True, exist_ok=True)
    xml_path = bdv_dir / "dataset.xml"
    h5_path = bdv_dir / "dataset.h5"

    with tifffile.TiffFile(input_path) as tif, h5py.File(h5_path, "w") as h5:

        series = tif.series[0]
        dtype = series.dtype
        shape = series.shape
        Z, Y, X = shape

        print(f"BigTIFF detected: shape={shape}, dtype={dtype}")
        print(f"Writing BDV dataset to {bdv_dir}")

        ds = h5.create_dataset(
            "s0", shape=shape, dtype=dtype,
            chunks=BDV_CHUNKS,
            compression="gzip", compression_opts=1
        )

        for z in range(Z):
            print(f"Slice {z+1}/{Z}")
            frame = series.pages[z]
            img = frame.asarray()
            out = apply_clahe(img, dtype, kernel, clip, nbins)
            ds[z] = out

    xml_content = f"""
<SpimData version="0.2">
  <BasePath type="relative">.</BasePath>
  <SequenceDescription>
    <ImageLoader format="bdv.hdf5">
      <hdf5>{h5_path.name}</hdf5>
    </ImageLoader>
    <ViewSetups>
      <ViewSetup><id>0</id><name>CLAHE</name></ViewSetup>
    </ViewSetups>
    <Timepoints type="range"><first>0</first><last>0</last></Timepoints>
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
    parser = argparse.ArgumentParser(description="CLAHE over huge EF-SEM BigTIFF volumes.")
    parser.add_argument("-i", "--input", required=True)
    parser.add_argument("-d", "--output-dir", required=True)
    parser.add_argument("-o", "--output-format", required=True, choices=["tiff", "bdv"])
    parser.add_argument("--kernel", type=int, default=DEFAULT_KERNEL)
    parser.add_argument("--nbins", type=int, default=DEFAULT_NBINS)
    parser.add_argument("--clip", type=float, default=DEFAULT_CLIP)
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    if args.output_format == "tiff":
        out_tif = output_dir / f"CLAHE_{input_path.name}"
        clahe_to_tiff(input_path, out_tif, args.kernel, args.clip, args.nbins)
    else:
        clahe_to_bdv(input_path, output_dir / "CLAHE_BDV",
                     args.kernel, args.clip, args.nbins)


if __name__ == "__main__":
    main()
