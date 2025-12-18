#!/usr/bin/env python3
"""
Crop small 3D stacks from a TIFF volume using a Fiji ROI CSV.

Assumptions:
- Input volume is a TIFF stack (including BigTIFF), shape (Z, Y, X).
- ROI CSV is exported from Fiji ROI Manager and includes columns:
      Name, X, Y, Width, Height
- Each ROI is a rectangle applied identically to all Z-slices.

For each ROI, this script:
- Reads (Name, X, Y, Width, Height)
- Crops all Z slices (optionally restricted to a Z-range)
- Writes one TIFF stack per ROI:  ROI_<Name>.tif
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile


def load_rois_from_csv(csv_path: Path):
    """Load ROIs from a Fiji ROI Manager CSV export."""
    df = pd.read_csv(csv_path)

    required = ["Name", "X", "Y", "Width", "Height"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' not found in CSV: {csv_path}")

    rois = []
    for _, row in df.iterrows():
        name = str(row["Name"])
        x = int(row["X"])
        y = int(row["Y"])
        w = int(row["Width"])
        h = int(row["Height"])
        rois.append((name, x, y, w, h))
    return rois


def clamp_roi(x, y, w, h, X, Y):
    """Clamp ROI to image bounds. Returns (x0, y0, x1, y1) as ints."""
    x0 = max(0, x)
    y0 = max(0, y)
    x1 = min(X, x + w)
    y1 = min(Y, y + h)
    return x0, y0, x1, y1


def crop_stack_for_roi(vol, roi, out_dir: Path, z_start: int, z_end: int):
    """
    Crop a 3D stack for a single ROI from a memmapped volume.

    vol: memmap-like array, shape (Z, Y, X)
    roi: (name, x, y, w, h)
    z_start, z_end: inclusive bounds (already clamped)
    """
    name, x, y, w, h = roi
    Z, Y, X = vol.shape

    x0, y0, x1, y1 = clamp_roi(x, y, w, h, X, Y)
    if x0 >= x1 or y0 >= y1:
        print(f"  ROI '{name}': empty after clamping to bounds, skipping.")
        return

    out_path = out_dir / f"ROI_{name}.tif"
    print(
        f"  ROI '{name}': x=[{x0},{x1}), y=[{y0},{y1}), z=[{z_start},{z_end}] → {out_path}"
    )

    # Materialise only the cropped region slice-by-slice (keeps memory sane)
    cropped_slices = []
    for z in range(z_start, z_end + 1):
        img = np.array(vol[z])              # materialise slice
        crop = img[y0:y1, x0:x1]            # y, then x
        cropped_slices.append(crop)

    out_vol = np.stack(cropped_slices, axis=0)  # (Z, Y, X) for cropped region
    tifffile.imwrite(str(out_path), out_vol)
    print(f"    wrote stack with shape {out_vol.shape}, dtype={out_vol.dtype}")


def main():
    parser = argparse.ArgumentParser(
        description="Crop 3D stacks from a TIFF volume using a Fiji ROI CSV."
    )
    parser.add_argument(
        "-i", "--input", required=True,
        help="Input TIFF stack (e.g. BigTIFF), expected shape (Z, Y, X)",
    )
    parser.add_argument(
        "-r", "--roi-csv", required=True,
        help="ROI CSV file exported from Fiji ROI Manager",
    )
    parser.add_argument(
        "-o", "--output-dir", required=True,
        help="Directory to write cropped stacks",
    )
    parser.add_argument(
        "--z-start", type=int, default=None,
        help="First Z slice (0-based, inclusive). Default: 0",
    )
    parser.add_argument(
        "--z-end", type=int, default=None,
        help="Last Z slice (0-based, inclusive). Default: last",
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    csv_path = Path(args.roi_csv)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Input stack:      {input_path}")
    print(f"ROI CSV:          {csv_path}")
    print(f"Output dir:       {out_dir}")

    print(f"Loading (memmap) volume from: {input_path}")
    vol = tifffile.imread(str(input_path), out="memmap")  # expect (Z, Y, X)

    if vol.ndim != 3:
        raise ValueError(f"Expected a 3D volume (Z,Y,X). Got shape={vol.shape}")

    Z, Y, X = vol.shape
    print(f"Volume shape = {vol.shape}, dtype={vol.dtype}")

    z_start = 0 if args.z_start is None else args.z_start
    z_end = (Z - 1) if args.z_end is None else args.z_end

    # Clamp Z range
    z_start = max(0, z_start)
    z_end = min(Z - 1, z_end)
    if z_start > z_end:
        raise ValueError(f"Invalid Z range after clamping: z_start={z_start}, z_end={z_end}")

    print(f"Using Z range: {z_start}–{z_end} (inclusive)")

    rois = load_rois_from_csv(csv_path)
    print(f"Found {len(rois)} ROI(s) in CSV")

    for roi in rois:
        crop_stack_for_roi(vol, roi, out_dir, z_start, z_end)

    print("All ROIs processed.")


if __name__ == "__main__":
    main()
