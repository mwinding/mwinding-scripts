#!/usr/bin/env python3
"""
Crop small 3D stacks from CLAHE slices using a Fiji ROI CSV.

Assumptions:
- CLAHE output is a directory of 2D slices:
      slice_0000.tif, slice_0001.tif, ...
- ROI CSV is like M09_D17_10MHz_3nA_8x8x8_20V_rois.csv with columns:
      Name, X, Y, Width, Height, ...
- Each ROI is a rectangle applied identically to all Z-slices.

For each ROI, this script:
- Reads (Name, X, Y, Width, Height)
- Crops all slices
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

    # Defensive: handle possible weird first column like ' '
    # We only care about: Name, X, Y, Width, Height
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


def get_clahe_slices(clahe_dir: Path):
    """Return a sorted list of CLAHE slice paths (slice_*.tif)."""
    slices = sorted(clahe_dir.glob("slice_*.tif"))
    if not slices:
        raise RuntimeError(f"No slice_*.tif files found in {clahe_dir}")
    return slices


def crop_stack_for_roi(slices, roi, out_dir: Path):
    """
    Crop a 3D stack for a single ROI.

    slices: list of slice paths (sorted)
    roi: (name, x, y, w, h)
    out_dir: where to write the TIFF stack
    """
    name, x, y, w, h = roi
    out_path = out_dir / f"ROI_{name}.tif"

    print(f"  ROI '{name}': x=[{x},{x+w}), y=[{y},{y+h}) → {out_path}")

    cropped_slices = []

    for i, sp in enumerate(slices):
        img = tifffile.imread(str(sp))
        crop = img[y:y+h, x:x+w]   # y, then x
        cropped_slices.append(crop)

    vol = np.stack(cropped_slices, axis=0)  # (Z, Y, X)
    tifffile.imwrite(str(out_path), vol)
    print(f"    wrote stack with shape {vol.shape}")


def main():
    parser = argparse.ArgumentParser(
        description="Crop 3D stacks from CLAHE slices using a Fiji ROI CSV."
    )
    parser.add_argument(
        "-c", "--clahe-dir", required=True,
        help="Directory containing CLAHE slices (slice_XXXX.tif)",
    )
    parser.add_argument(
        "-r", "--roi-csv", required=True,
        help="ROI CSV file exported from Fiji ROI Manager",
    )
    parser.add_argument(
        "-o", "--output-dir", required=True,
        help="Directory to write cropped stacks",
    )

    args = parser.parse_args()

    clahe_dir = Path(args.clahe_dir)
    csv_path = Path(args.roi_csv)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"CLAHE slices dir: {clahe_dir}")
    print(f"ROI CSV:          {csv_path}")
    print(f"Output dir:       {out_dir}")

    slices = get_clahe_slices(clahe_dir)
    print(f"Found {len(slices)} slices")

    rois = load_rois_from_csv(csv_path)
    print(f"Found {len(rois)} ROI(s) in CSV")

    for roi in rois:
        crop_stack_for_roi(slices, roi, out_dir)

    print("All ROIs processed.")


if __name__ == "__main__":
    main()
