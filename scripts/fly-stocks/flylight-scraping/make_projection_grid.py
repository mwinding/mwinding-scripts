#!/usr/bin/env python3

'''
To run:
PYTHONNOUSERSITE=1 python make_projection_grid.py \
  --csv flylight-splits_not-in-lab_brain-vnc_per-image.csv \
  --out data/grid.png

'''

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Dict, Tuple

from PIL import Image, ImageDraw, ImageFont


def norm(s: str) -> str:
    return (s or "").strip()


def load_font(size: int) -> ImageFont.ImageFont:
    for name in ["DejaVuSans.ttf", "Arial.ttf", "LiberationSans-Regular.ttf"]:
        try:
            return ImageFont.truetype(name, size)
        except Exception:
            pass
    return ImageFont.load_default()


def fit_image(img: Image.Image, w: int, h: int) -> Image.Image:
    img = img.convert("RGB")
    iw, ih = img.size
    scale = min(w / iw, h / ih)
    nw, nh = int(iw * scale), int(ih * scale)
    return img.resize((nw, nh), Image.Resampling.LANCZOS)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Create a grid image of FlyLight projections from a CSV."
    )
    ap.add_argument("--csv", required=True,
                    help="Input CSV containing split, AD_DBD, local_path.")
    ap.add_argument("--out", default="grid.png",
                    help="Output PNG (openable in Preview).")

    ap.add_argument("--cols", type=int, default=6)
    ap.add_argument("--max_splits", type=int, default=60,
                    help="Maximum number of splits to render (0 = no limit).")

    ap.add_argument("--tile_w", type=int, default=320)
    ap.add_argument("--img_h", type=int, default=260)
    ap.add_argument("--label_h", type=int, default=60)
    ap.add_argument("--gap", type=int, default=10)
    ap.add_argument("--margin", type=int, default=10)
    ap.add_argument("--font_size", type=int, default=16)

    args = ap.parse_args()

    rows = []
    with open(args.csv, newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append({k: norm(v) for k, v in row.items()})

    # pick one image per split (first valid local_path)
    chosen: Dict[str, Tuple[str, Path]] = {}
    for r in rows:
        split = r.get("split", "")
        lp = r.get("local_path", "")
        if not split or not lp:
            continue
        p = Path(lp)
        if not p.exists():
            continue
        if split not in chosen:
            chosen[split] = (r.get("AD_DBD", ""), p)

    splits = list(chosen.items())
    if args.max_splits > 0:
        splits = splits[:args.max_splits]

    if not splits:
        raise SystemExit("No usable images found (check local_path column).")

    n = len(splits)
    cols = args.cols
    rows_n = math.ceil(n / cols)

    tile_h = args.img_h + args.label_h
    W = args.margin * 2 + cols * args.tile_w + (cols - 1) * args.gap
    H = args.margin * 2 + rows_n * tile_h + (rows_n - 1) * args.gap

    canvas = Image.new("RGB", (W, H), (30, 30, 30))
    draw = ImageDraw.Draw(canvas)
    font = load_font(args.font_size)

    for i, (split, (ad_dbd, path)) in enumerate(splits):
        r = i // cols
        c = i % cols

        x0 = args.margin + c * (args.tile_w + args.gap)
        y0 = args.margin + r * (tile_h + args.gap)

        # image background
        draw.rectangle([x0, y0, x0 + args.tile_w, y0 + args.img_h], fill=(0, 0, 0))

        try:
            img = Image.open(path)
            fitted = fit_image(img, args.tile_w, args.img_h)
            fw, fh = fitted.size
            canvas.paste(
                fitted,
                (x0 + (args.tile_w - fw) // 2,
                 y0 + (args.img_h - fh) // 2)
            )
        except Exception:
            draw.text((x0 + 5, y0 + 5), "LOAD ERROR", fill=(255, 0, 0), font=font)

        # label background
        ly = y0 + args.img_h
        draw.rectangle([x0, ly, x0 + args.tile_w, ly + args.label_h], fill=(0, 0, 0))

        draw.text((x0 + 5, ly + 2), split, fill=(255, 255, 255), font=font)
        draw.text((x0 + 5, ly + args.label_h // 2),
                  ad_dbd, fill=(200, 200, 200), font=font)

    out = Path(args.out)
    canvas.save(out)
    print(f"Wrote {out} with {n} splits ({cols} × {rows_n}).")


if __name__ == "__main__":
    main()
