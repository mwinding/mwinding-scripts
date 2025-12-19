#!/usr/bin/env python3

'''
To run:

PYTHONNOUSERSITE=1 python compare_lab_vs_flylight.py \
  --lab_csv winding-lab_fly-stocks.csv \
  --out_not_in_lab flylight_splits_not_in_lab.csv \
  --out_in_lab flylight_splits_in_lab.csv

'''
from __future__ import annotations

import argparse
import csv
import re
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urljoin
from urllib.request import Request, urlopen


DEFAULT_BASE = "https://raw.larval.flylight.virtualflybrain.org"
DEFAULT_PAGE = "/explore/splits"


def norm(s: str) -> str:
    return (s or "").strip()


def is_split_gal4(type_field: str) -> bool:
    """
    Heuristic: keep things labelled as split-GAL4 in the lab sheet.
    Adjust if your Type values differ (e.g. 'Split Gal4', 'split-gal4', etc).
    """
    t = (type_field or "").strip().lower()
    return ("split" in t) and ("gal4" in t)


def fetch_text(url: str, timeout: int = 60) -> str:
    req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(req, timeout=timeout) as r:
        return r.read().decode("utf-8", errors="replace")


def infer_split_from_basename(basename: str) -> str:
    """
    Basename examples:
      76F05_15B01_MB054B_020113A
      25B07_58F01_SS00672_A
      TH_72B05_MB065B_011113A
    """
    m = re.search(r"(MB\d+[A-Z]?|SS\d+)", basename)
    return m.group(1) if m else ""


class FlylightSplitsHTMLParser(HTMLParser):
    """
    Extract projection JPGs from the /explore/splits page.

    We look for <img src="/images/.../<basename>_G.jpg">
    """
    def __init__(self) -> None:
        super().__init__()
        self.proj_srcs: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() != "img":
            return
        d = {k.lower(): (v or "") for k, v in attrs}
        src = d.get("src", "")
        if not src:
            return
        # Only the green projection thumbnails
        if src.endswith("_G.jpg") and "/images/" in src:
            self.proj_srcs.append(src)


def scrape_projection_records(page_html: str, base_url: str) -> list[dict[str, str]]:
    """
    Return records with:
      - split (MBxxxx / SSxxxxx)
      - AD_DBD (basename, exactly as-is)
      - projection_url (absolute *_G.jpg URL)
      - tif_url (absolute .tif URL)
      - folder (the /images/<folder>/ part)
    """
    p = FlylightSplitsHTMLParser()
    p.feed(page_html)

    records: list[dict[str, str]] = []
    seen: set[str] = set()

    # src example: /images/lmbjanupload-MB054B-Y/76F05_15B01_MB054B_020113A_G.jpg
    pat = re.compile(r"^/images/(?P<folder>[^/]+)/(?P<basename>.+)_G\.jpg$")

    for src in p.proj_srcs:
        m = pat.match(src)
        if not m:
            continue
        folder = m.group("folder")
        basename = m.group("basename")

        key = f"{folder}/{basename}"
        if key in seen:
            continue
        seen.add(key)

        split = infer_split_from_basename(basename)
        if not split:
            # keep it, but mark split empty so it won't match lab IDs
            split = ""

        projection_url = urljoin(base_url, src)
        tif_rel = f"/images/{folder}/{basename}.tif"
        tif_url = urljoin(base_url, tif_rel)

        records.append({
            "split": split,
            "AD_DBD": basename,              # you wanted this exact basename string
            "folder": folder,
            "projection_url": projection_url,
            "tif_url": tif_url,
        })

    return records


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Scrape FlyLight larval splits from the website and compare against lab stocks (split-GAL4 only)."
    )
    ap.add_argument("--lab_csv", default="winding-lab_fly-stocks.csv",
                    help="Lab stock CSV with columns including 'ID' and 'Type'.")
    ap.add_argument("--base_url", default=DEFAULT_BASE,
                    help="Base URL for FlyLight raw site.")
    ap.add_argument("--splits_page", default=DEFAULT_PAGE,
                    help="Path to FlyLight splits explore page.")
    ap.add_argument("--out_not_in_lab", default="flylight_splits_not_in_lab.csv",
                    help="Output: FlyLight records whose split is NOT present in lab split-GAL4 IDs.")
    ap.add_argument("--out_in_lab", default="flylight_splits_in_lab.csv",
                    help="Output: FlyLight records whose split IS present in lab split-GAL4 IDs.")
    args = ap.parse_args()

    lab_path = Path(args.lab_csv)
    if not lab_path.exists():
        raise SystemExit(f"Lab CSV not found: {lab_path}")

    # 1) Load lab split-GAL4 IDs
    lab_split_ids: set[str] = set()
    with lab_path.open(newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        if not r.fieldnames:
            raise SystemExit("Lab CSV appears to have no header row.")
        if "ID" not in r.fieldnames or "Type" not in r.fieldnames:
            raise SystemExit("Lab CSV must have columns 'ID' and 'Type'.")
        for row in r:
            if is_split_gal4(row.get("Type", "")):
                lab_split_ids.add(norm(row.get("ID", "")))

    # 2) Scrape FlyLight
    page_url = urljoin(args.base_url, args.splits_page)
    html = fetch_text(page_url)
    fly_recs = scrape_projection_records(html, args.base_url)

    if not fly_recs:
        raise SystemExit(
            f"Did not find any *_G.jpg projections on {page_url}. "
            "Page structure may have changed."
        )

    # 3) Partition into in-lab vs not-in-lab (by split)
    in_lab: list[dict[str, str]] = []
    not_in_lab: list[dict[str, str]] = []
    split_missing_count = 0

    for rec in fly_recs:
        split = norm(rec.get("split", ""))
        if not split:
            # can't match; treat as missing
            not_in_lab.append(rec)
            split_missing_count += 1
            continue
        if split in lab_split_ids:
            in_lab.append(rec)
        else:
            not_in_lab.append(rec)

    # 4) Write outputs
    fieldnames = ["split", "AD_DBD", "projection_url", "tif_url", "folder"]

    with open(args.out_not_in_lab, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(not_in_lab)

    with open(args.out_in_lab, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(in_lab)

    # 5) Summary
    fly_splits = sorted({r["split"] for r in fly_recs if r["split"]})
    not_in_lab_splits = sorted({r["split"] for r in not_in_lab if r["split"]})
    in_lab_splits = sorted({r["split"] for r in in_lab if r["split"]})

    print(f"Scraped page: {page_url}")
    print(f"Total FlyLight projection records (one per *_G.jpg): {len(fly_recs)}")
    print(f"FlyLight unique splits (MB/SS inferred): {len(fly_splits)}")
    print(f"Lab split-GAL4 IDs (filtered by Type): {len(lab_split_ids)}")
    print(f"Records IN lab (by split): {len(in_lab)}  -> {args.out_in_lab}")
    print(f"Records NOT in lab (by split): {len(not_in_lab)} -> {args.out_not_in_lab}")
    if split_missing_count:
        print(f"WARNING: {split_missing_count} records had no inferable MB/SS split token in basename (kept in NOT in lab).")
    print(f"Unique splits IN lab:  {len(in_lab_splits)}")
    print(f"Unique splits NOT in lab: {len(not_in_lab_splits)}")


if __name__ == "__main__":
    main()
