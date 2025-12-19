#!/usr/bin/env python3
"""
To run:
PYTHONNOUSERSITE=1 python download-classify_flylight-splits.py \
  --in_csv flylight_splits_not_in_lab.csv \
  --download_dir projections_Gjpg \
  --out_per_image brain_vnc_per_image.csv \
  --out_by_split brain_vnc_by_split.csv \
  --out_annotated flylight_splits_not_in_lab_with_expression.csv
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import numpy as np

try:
    from PIL import Image
except Exception as e:
    raise SystemExit(
        "Missing dependency: pillow.\n"
        "If you are on conda (recommended): conda install -c conda-forge pillow\n"
        "Then run with: PYTHONNOUSERSITE=1 python download_classify_flylight_splits.py ...\n"
        f"Import error was: {e}"
    ) from e


def norm(s: str) -> str:
    return (s or "").strip()


# Matches "lmbjanupload-MB026B-Y" or "lmbjanupload-SS00671-Y"
_SPLIT_FROM_UPLOADDIR = re.compile(r"lmbjanupload-([A-Za-z0-9]+)-Y")


def infer_split(
    split: str,
    page_dir: str,
    projection_url: str,
    tif_url: str,
    basename: str,
) -> str:
    """
    Infer split if missing/inconsistent.

    Priority:
      1) Provided split if non-empty
      2) page_dir like lmbjanupload-MB026B-Y -> MB026B
      3) URLs containing .../lmbjanupload-MB026B-Y/... -> MB026B
      4) basename prefix like SS55554_... -> SS55554 (also MB###X_... if present)
    """
    s = norm(split)
    if s:
        return s

    pd = norm(page_dir)
    m = _SPLIT_FROM_UPLOADDIR.search(pd)
    if m:
        return m.group(1)

    for u in (projection_url, tif_url):
        uu = norm(u)
        m = _SPLIT_FROM_UPLOADDIR.search(uu)
        if m:
            return m.group(1)

    # last-ditch basename prefix
    b = norm(basename)
    m = re.match(r"^(SS\d+|MB\d+[A-Z]?)_", b)
    if m:
        return m.group(1)

    return ""


def safe_filename_from_url(url: str) -> str:
    """
    Make a stable filename from the URL. Prefer the basename, but add a short hash
    to avoid collisions.
    """
    base = url.split("?")[0].rstrip("/").split("/")[-1] or "projection.jpg"
    h = hashlib.sha1(url.encode("utf-8")).hexdigest()[:10]
    if "." in base:
        stem, ext = base.rsplit(".", 1)
        ext = "." + ext
    else:
        stem, ext = base, ""
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", stem)
    return f"{stem}__{h}{ext}"


def download(url: str, out_path: Path, timeout: int = 60) -> tuple[bool, str]:
    """Download URL to out_path. Returns (ok, message)."""
    try:
        req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urlopen(req, timeout=timeout) as r:
            data = r.read()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(data)
        return True, "ok"
    except HTTPError as e:
        return False, f"HTTPError {e.code}"
    except URLError as e:
        return False, f"URLError {e.reason}"
    except Exception as e:
        return False, f"Error {type(e).__name__}: {e}"


def load_green_projection(path: Path) -> np.ndarray:
    """Load *_G.jpg projection. If RGB/RGBA, take green channel."""
    img = Image.open(path)
    arr = np.asarray(img)
    if arr.ndim == 3 and arr.shape[2] >= 3:
        arr = arr[:, :, 1]  # green channel
    return arr


def normalise01(img: np.ndarray) -> np.ndarray:
    img = img.astype(np.float32, copy=False)
    mn = float(np.min(img))
    mx = float(np.max(img))
    if mx <= mn:
        return np.zeros_like(img, dtype=np.float32)
    return (img - mn) / (mx - mn)


def region_stats(img01: np.ndarray, cut_frac: float) -> dict[str, float]:
    """
    Split image into top (brain) and bottom (VNC) by cut_frac of height.
    Compute robust intensity stats to catch sparse bright somata.
    """
    h = img01.shape[0]
    cut = int(round(cut_frac * h))
    cut = max(1, min(h - 1, cut))

    top = img01[:cut, :]
    bot = img01[cut:, :]

    def q(x: np.ndarray, p: float) -> float:
        return float(np.quantile(x, p))

    def frac_above(x: np.ndarray, thr: float) -> float:
        return float(np.mean(x >= thr))

    return {
        "cut_frac": float(cut_frac),
        "h": float(h),
        "cut_px": float(cut),

        "q99_all": q(img01, 0.99),
        "q995_all": q(img01, 0.995),
        "q999_all": q(img01, 0.999),
        "frac90_all": frac_above(img01, 0.90),
        "frac95_all": frac_above(img01, 0.95),

        "q99_top": q(top, 0.99),
        "q995_top": q(top, 0.995),
        "q999_top": q(top, 0.999),
        "frac90_top": frac_above(top, 0.90),
        "frac95_top": frac_above(top, 0.95),

        "q99_bot": q(bot, 0.99),
        "q995_bot": q(bot, 0.995),
        "q999_bot": q(bot, 0.999),
        "frac90_bot": frac_above(bot, 0.90),
        "frac95_bot": frac_above(bot, 0.95),
    }


def classify_from_stats(
    stats: dict[str, float],
    q99_thr: float,
    frac95_thr: float,
) -> tuple[str, bool, bool]:
    """
    Decide region presence using two signals:
      - robust brightness: q99 >= q99_thr
      - sparse brights: frac95 >= frac95_thr
    Then classify: none / brain / vnc / both.
    """
    top_on = (stats["q99_top"] >= q99_thr) or (stats["frac95_top"] >= frac95_thr)
    bot_on = (stats["q99_bot"] >= q99_thr) or (stats["frac95_bot"] >= frac95_thr)

    if top_on and bot_on:
        return "both", True, True
    if top_on:
        return "brain", True, False
    if bot_on:
        return "vnc", False, True
    return "none", False, False


def majority_vote(labels: list[str]) -> tuple[str, float]:
    """Returns (label, agreement_fraction)."""
    if not labels:
        return "unknown", 0.0
    c = Counter(labels)
    best, n = c.most_common(1)[0]
    return best, n / len(labels)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Download FlyLight *_G.jpg projections for missing splits and classify brain/VNC/both/none; "
            "collapse to one call per split; annotate CSV. "
            "Handles missing split values by inferring from page_dir/URL/basename."
        )
    )
    ap.add_argument("--in_csv", default="flylight_splits_not_in_lab.csv",
                    help="Input CSV containing at least AD_DBD and projection_url. split/page_dir/tif_url are optional.")
    ap.add_argument("--download_dir", default="projections_Gjpg",
                    help="Directory to store downloaded *_G.jpg projections.")
    ap.add_argument("--skip_download", action="store_true",
                    help="Do not download; assume files already exist in --download_dir.")
    ap.add_argument("--timeout", type=int, default=60, help="Download timeout (seconds).")

    ap.add_argument("--cut_frac", type=float, default=0.40,
                    help="Top fraction of image treated as brain region (default 0.40).")
    ap.add_argument("--q99_thr", type=float, default=0.20,
                    help="Threshold on region q99 (image normalised to [0,1]) to call signal present.")
    ap.add_argument("--frac95_thr", type=float, default=0.001,
                    help="Threshold on fraction of pixels >= 0.95 (in [0,1]) to call sparse bright signal present.")

    ap.add_argument("--out_per_image", default="brain_vnc_per_image.csv",
                    help="Output per-image classification CSV.")
    ap.add_argument("--out_by_split", default="brain_vnc_by_split.csv",
                    help="Output collapsed per-split classification CSV.")
    ap.add_argument("--out_annotated", default="flylight_splits_not_in_lab_with_expression.csv",
                    help="Output annotated version of --in_csv (adds split-level classification columns).")
    args = ap.parse_args()

    in_path = Path(args.in_csv)
    if not in_path.exists():
        raise SystemExit(f"Input CSV not found: {in_path}")

    dl_dir = Path(args.download_dir)
    dl_dir.mkdir(parents=True, exist_ok=True)

    # Load rows
    rows: list[dict[str, str]] = []
    with in_path.open(newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)

        if not r.fieldnames:
            raise SystemExit("Input CSV appears to have no header.")

        fieldset = set(r.fieldnames)

        # Required minimal columns
        required = {"AD_DBD", "projection_url"}
        if not required.issubset(fieldset):
            raise SystemExit(f"Input CSV must have columns: {sorted(required)} (found: {sorted(fieldset)})")

        # Optional columns used for split inference
        # split, page_dir, tif_url
        for row in r:
            rows.append({k: norm(v) for k, v in row.items() if k is not None})

    # Process each row/image
    per_image_out: list[dict[str, Any]] = []
    by_split_labels: dict[str, list[str]] = defaultdict(list)
    by_split_stats: dict[str, list[dict[str, float]]] = defaultdict(list)

    download_failures = 0
    processed = 0

    for row in rows:
        raw_split = norm(row.get("split", ""))
        basename = norm(row.get("AD_DBD", ""))
        proj_url = norm(row.get("projection_url", ""))
        tif_url = norm(row.get("tif_url", ""))
        page_dir = norm(row.get("page_dir", ""))

        split = infer_split(raw_split, page_dir, proj_url, tif_url, basename)

        if not proj_url:
            download_failures += 1
            per_image_out.append({
                "split": split,
                "AD_DBD": basename,
                "projection_url": proj_url,
                "local_path": "",
                "download_ok": "0",
                "download_msg": "missing_url",
                "classification": "unknown",
            })
            continue

        fname = safe_filename_from_url(proj_url)
        local_path = dl_dir / fname

        ok = True
        msg = "exists"
        if not args.skip_download:
            if not local_path.exists():
                ok, msg = download(proj_url, local_path, timeout=args.timeout)

        if not local_path.exists():
            ok = False
            if msg == "exists":
                msg = "not_found_after_download"

        if not ok:
            download_failures += 1
            per_image_out.append({
                "split": split,
                "AD_DBD": basename,
                "projection_url": proj_url,
                "local_path": str(local_path),
                "download_ok": "0",
                "download_msg": msg,
                "classification": "unknown",
            })
            continue

        # Load + classify
        try:
            img = load_green_projection(local_path)
            img01 = normalise01(img)
            stats = region_stats(img01, cut_frac=args.cut_frac)
            classification, top_on, bot_on = classify_from_stats(
                stats, q99_thr=args.q99_thr, frac95_thr=args.frac95_thr
            )

            outrow: dict[str, Any] = {
                "split": split,
                "AD_DBD": basename,
                "projection_url": proj_url,
                "local_path": str(local_path),
                "download_ok": "1",
                "download_msg": msg,
                "classification": classification,
                "brain_on": int(top_on),
                "vnc_on": int(bot_on),
                "q99_top": stats["q99_top"],
                "q99_bot": stats["q99_bot"],
                "frac95_top": stats["frac95_top"],
                "frac95_bot": stats["frac95_bot"],
                "q99_all": stats["q99_all"],
                "frac95_all": stats["frac95_all"],
                "cut_frac": stats["cut_frac"],
                "cut_px": stats["cut_px"],
            }
            per_image_out.append(outrow)

            if split:
                by_split_labels[split].append(classification)
                by_split_stats[split].append(stats)

            processed += 1

        except Exception as e:
            per_image_out.append({
                "split": split,
                "AD_DBD": basename,
                "projection_url": proj_url,
                "local_path": str(local_path),
                "download_ok": "1",
                "download_msg": msg,
                "classification": "error",
                "error": f"{type(e).__name__}: {e}",
            })

    # Write per-image CSV
    per_image_fields = [
        "split", "AD_DBD", "projection_url", "local_path",
        "download_ok", "download_msg",
        "classification", "brain_on", "vnc_on",
        "q99_top", "q99_bot", "frac95_top", "frac95_bot",
        "q99_all", "frac95_all",
        "cut_frac", "cut_px",
        "error",
    ]
    with open(args.out_per_image, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=per_image_fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(per_image_out)

    # Collapse to split-level
    split_rows: list[dict[str, Any]] = []
    for split, labels in sorted(by_split_labels.items()):
        maj, agree = majority_vote(labels)
        stats_list = by_split_stats.get(split, [])

        def mean_key(k: str) -> float:
            vals = [d[k] for d in stats_list if k in d]
            return float(np.mean(vals)) if vals else float("nan")

        split_rows.append({
            "split": split,
            "n_images": len(labels),
            "majority_class": maj,
            "agreement_frac": agree,
            "labels": "|".join(labels),
            "mean_q99_top": mean_key("q99_top"),
            "mean_q99_bot": mean_key("q99_bot"),
            "mean_frac95_top": mean_key("frac95_top"),
            "mean_frac95_bot": mean_key("frac95_bot"),
            "cut_frac": float(args.cut_frac),
            "q99_thr": float(args.q99_thr),
            "frac95_thr": float(args.frac95_thr),
        })

    split_fields = [
        "split", "n_images", "majority_class", "agreement_frac", "labels",
        "mean_q99_top", "mean_q99_bot", "mean_frac95_top", "mean_frac95_bot",
        "cut_frac", "q99_thr", "frac95_thr",
    ]
    with open(args.out_by_split, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=split_fields)
        w.writeheader()
        w.writerows(split_rows)

    # Annotate original rows with split-level info.
    # Also overwrite/emit a cleaned "split" field (inferred).
    split_to_summary = {r["split"]: r for r in split_rows}

    annotated_rows: list[dict[str, str]] = []
    for row in rows:
        raw_split = norm(row.get("split", ""))
        basename = norm(row.get("AD_DBD", ""))
        proj_url = norm(row.get("projection_url", ""))
        tif_url = norm(row.get("tif_url", ""))
        page_dir = norm(row.get("page_dir", ""))

        split = infer_split(raw_split, page_dir, proj_url, tif_url, basename)
        summ = split_to_summary.get(split)

        out = dict(row)
        out["split"] = split  # write back the fixed split

        if summ:
            out["split_majority_class"] = str(summ["majority_class"])
            out["split_agreement_frac"] = f"{summ['agreement_frac']:.3f}"
            out["split_n_images"] = str(summ["n_images"])
            out["split_mean_q99_top"] = f"{summ['mean_q99_top']:.6f}"
            out["split_mean_q99_bot"] = f"{summ['mean_q99_bot']:.6f}"
            out["split_mean_frac95_top"] = f"{summ['mean_frac95_top']:.6f}"
            out["split_mean_frac95_bot"] = f"{summ['mean_frac95_bot']:.6f}"
        else:
            out["split_majority_class"] = "unknown"
            out["split_agreement_frac"] = ""
            out["split_n_images"] = ""
            out["split_mean_q99_top"] = ""
            out["split_mean_q99_bot"] = ""
            out["split_mean_frac95_top"] = ""
            out["split_mean_frac95_bot"] = ""

        annotated_rows.append(out)

    # Preserve original column order, but ensure 'split' exists and add new cols.
    annotated_fields = list(rows[0].keys())

    if "split" not in annotated_fields:
        annotated_fields.insert(0, "split")

    new_cols = [
        "split_majority_class", "split_agreement_frac", "split_n_images",
        "split_mean_q99_top", "split_mean_q99_bot",
        "split_mean_frac95_top", "split_mean_frac95_bot",
    ]
    for c in new_cols:
        if c not in annotated_fields:
            annotated_fields.append(c)

    with open(args.out_annotated, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=annotated_fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(annotated_rows)

    # Summary
    n_unknown = sum(1 for r in per_image_out if r.get("classification") in {"unknown", "error"})
    print(f"Wrote: {args.out_per_image}  (per-image; processed={processed}, unknown/error={n_unknown})")
    print(f"Wrote: {args.out_by_split}  (per-split; n_splits={len(split_rows)})")
    print(f"Wrote: {args.out_annotated}  (annotated input rows={len(annotated_rows)})")
    if download_failures:
        print(f"WARNING: download/URL failures: {download_failures}")


if __name__ == "__main__":
    main()
