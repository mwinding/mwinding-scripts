#!/usr/bin/env python3
"""
Create separate projection grids for splits classified as brain-only, vnc-only, or both.

Examples:
  PYTHONNOUSERSITE=1 python make_split_class_grids.py
  PYTHONNOUSERSITE=1 python make_split_class_grids.py --out_dir data
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path
from typing import Dict, Tuple

def norm(s: str) -> str:
    return (s or "").strip()

def load_split_classes(path: Path) -> Dict[str, str]:
    split_class: Dict[str, str] = {}
    with path.open(newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            split = norm(row.get("split", ""))
            cls = norm(row.get("majority_class", ""))
            if split and cls:
                split_class[split] = cls
    return split_class

def load_first_image_per_split(path: Path) -> Dict[str, Tuple[str, Path]]:
    chosen: Dict[str, Tuple[str, Path]] = {}
    with path.open(newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            split = norm(row.get("split", ""))
            ad_dbd = norm(row.get("AD_DBD", ""))
            lp = norm(row.get("local_path", ""))
            if not split or not lp:
                continue
            p = Path(lp)
            if not p.exists():
                continue
            if split not in chosen:
                chosen[split] = (ad_dbd, p)
    return chosen


def write_grid_csv(out_path: Path, rows: list[dict[str, str]]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["split", "AD_DBD", "local_path"])
        w.writeheader()
        w.writerows(rows)


def run_grid(
    csv_path: Path,
    out_path: Path,
    args: argparse.Namespace,
    max_splits: int,
) -> None:
    script_path = Path(__file__).with_name("make_projection_grid.py")
    cmd = [
        sys.executable,
        str(script_path),
        "--csv", str(csv_path),
        "--out", str(out_path),
        "--cols", str(args.cols),
        "--max_splits", str(max_splits),
        "--tile_w", str(args.tile_w),
        "--img_h", str(args.img_h),
        "--label_h", str(args.label_h),
        "--gap", str(args.gap),
        "--margin", str(args.margin),
        "--font_size", str(args.font_size),
    ]
    subprocess.run(cmd, check=True)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Build separate grids for brain-only, vnc-only, and both splits."
    )
    ap.add_argument("--by_split_csv",
                    default="flylight-splits_not-in-lab_brain-vnc_by-split.csv",
                    help="Split-level summary CSV (one row per split).")
    ap.add_argument("--per_image_csv",
                    default="flylight-splits_not-in-lab_brain-vnc_per-image.csv",
                    help="Per-image CSV with local_path.")
    ap.add_argument("--out_dir", default="data",
                    help="Directory to write grid images and inputs.")
    ap.add_argument("--grid_prefix", default="grid_not_in_lab",
                    help="Prefix for output grid image filenames.")
    ap.add_argument("--max_splits", type=int, default=0,
                    help="Maximum number of splits to render per grid (0 = no limit).")

    ap.add_argument("--cols", type=int, default=6)
    ap.add_argument("--tile_w", type=int, default=320)
    ap.add_argument("--img_h", type=int, default=260)
    ap.add_argument("--label_h", type=int, default=60)
    ap.add_argument("--gap", type=int, default=10)
    ap.add_argument("--margin", type=int, default=10)
    ap.add_argument("--font_size", type=int, default=16)
    args = ap.parse_args()

    split_class = load_split_classes(Path(args.by_split_csv))
    chosen = load_first_image_per_split(Path(args.per_image_csv))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    classes = {
        "brain": "brain_only",
        "vnc": "vnc_only",
        "both": "both",
        "none": "none",
    }

    for cls, name in classes.items():
        rows: list[dict[str, str]] = []
        for split, split_cls in split_class.items():
            if split_cls != cls:
                continue
            if split not in chosen:
                continue
            ad_dbd, p = chosen[split]
            rows.append({
                "split": split,
                "AD_DBD": ad_dbd,
                "local_path": str(p),
            })

        if not rows:
            print(f"Skipping {cls}; no usable images found.")
            continue

        if args.max_splits > 0 and len(rows) > args.max_splits:
            chunks = [
                rows[i:i + args.max_splits]
                for i in range(0, len(rows), args.max_splits)
            ]
        else:
            chunks = [rows]

        for idx, chunk in enumerate(chunks, start=1):
            suffix = f"_p{idx:02d}" if len(chunks) > 1 else ""
            grid_csv = out_dir / f"{args.grid_prefix}_{name}{suffix}_input.csv"
            grid_png = out_dir / f"{args.grid_prefix}_{name}{suffix}.png"
            write_grid_csv(grid_csv, chunk)
            run_grid(grid_csv, grid_png, args, max_splits=0)
            print(f"Wrote: {grid_png}")


if __name__ == "__main__":
    main()
