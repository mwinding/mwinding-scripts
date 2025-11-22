#!/usr/bin/env python3

"""
Crop small 3D stacks from CLAHE slices using Fiji ROI files.

Assumptions:
- CLAHE output is a directory of 2D slices:
      slice_0000.tif, slice_0001.tif, ...
- ROIs are saved from Fiji ROI Manager as .roi files (one per ROI),
  or as multiple .roi files in a folder.
- Each ROI defines an x–y rectangle, applied identically to all slices.

For each ROI, this script:
- Reads the ROI
- Uses its bounding box on every slice
- Writes a cropped TIFF stack: one file per ROI
"""

import argparse
from pathlib import Path

import numpy as np
import tifffile
from roifile import ImagejRoi  # pip install roifile


def load_rois(roi_path: Path):
    """
    Return a list of (name, left, top, right, bottom) from .roi files.

    roi_path:
      - if a single .roi file → just that ROI
      - if a directory       → all *.roi files inside
    """
    if roi_path.is_file() and roi_path.suffix.lower() == ".roi":
        roi_files = [roi_path]
    elif roi_path.is_dir():
        roi_files = sorted(roi_path.glob("*.roi"))
    else:
        raise ValueError(f"{roi_path} is neither a .roi file nor a directory of .roi files")

    if not roi_files:
        raise RuntimeError(f"No .roi files found in {roi_path}")

    rois = []
    for rf in roi_files:
        r = ImagejRoi.fromfile(str(rf))
        # ImageJ coordinates: left, top, right, bottom (pixels)
        left, top, right, bottom = r.left, r.top, r.right, r.bottom
        name = rf.stem
        rois.append((name, left, top, right, bottom))
    return rois


def get_clahe_slices(clahe_dir: Path):
    """Return a sorted list of CLAHE slice paths."""
    # Adjust glob if your naming differs
    slices = sorted(clahe_dir.glob("slice_*.tif"))
    if not slices:
        raise RuntimeError(f"No slice_*.tif files found in {clahe_dir}")
    return slices


def crop_stack_for_roi(slices, roi, out_dir: Path):
    """
    Crop a 3D stack for a single ROI.

    slices: list of slice paths (sorted)
    roi: (name, left, top, right, bottom)
    out_dir: where to write the TIFF stack
    """
    name, left, top, right, bottom = roi
    out_path = out_dir / f"ROI_{name}.tif"

    print(f"  ROI '{name}': x=[{left},{right}), y=[{top},{bottom}) → {out_path}")

    cropped_slices = []

    for i, sp in enumerate(slices):
        img = tifffile.imread(str(sp))
        # y = top:bottom, x = left:right
        crop = img[top:bottom, left:right]
        cropped_slices.append(crop)

    vol = np.stack(cropped_slices, axis=0)  # (Z, Y, X)
    tifffile.imwrite(str(out_path), vol)
    print(f"    wrote stack with shape {vol.shape}")


def main():
    parser = argparse.ArgumentParser(
        description="Crop 3D stacks from CLAHE slices using Fiji .roi files."
    )
    parser.add_argument(
        "-c", "--clahe-dir", required=True,
        help="Directory containing CLAHE slices (slice_XXXX.tif)",
    )
    parser.add_argument(
        "-r", "--rois", required=True,
        help="Single .roi file or directory containing .roi files",
    )
    parser.add_argument(
        "-o", "--output-dir", required=True,
        help="Directory to write cropped stacks",
    )

    args = parser.parse_args()

    clahe_dir = Path(args.clahe_dir)
    roi_path = Path(args.rois)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"CLAHE slices dir: {clahe_dir}")
    print(f"ROI source:       {roi_path}")
    print(f"Output dir:       {out_dir}")

    slices = get_clahe_slices(clahe_dir)
    print(f"Found {len(slices)} slices")

    rois = load_rois(roi_path)
    print(f"Found {len(rois)} ROI(s)")

    for roi in rois:
        crop_stack_for_roi(slices, roi, out_dir)

    print("All ROIs processed.")


if __name__ == "__main__":
    main()
