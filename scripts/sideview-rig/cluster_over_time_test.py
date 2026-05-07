#!/usr/bin/env python3
# %%
"""
Plot number of clustered animals over time (binned every 4000 frames)
for GH and SI datasets found in data/sideview_data/test.
"""

from pathlib import Path
import os
import sys
import site

# Avoid importing x86_64 user-site wheels in the conda env.
user_site = site.getusersitepackages()
if user_site in sys.path:
    sys.path.remove(user_site)
os.environ.setdefault("PYTHONNOUSERSITE", "1")

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
TEST_DIR = ROOT / "data" / "sideview_data" / "test"
BIN_SIZE = 4000
OUT_PNG = TEST_DIR / "clustered_animals_over_time_4000bin.png"


def load_cluster_file(path: Path):
    df = pd.read_feather(path)
    # Expected columns: frame, track_id, cluster (at least)
    if "frame" not in df.columns or "cluster" not in df.columns:
        raise ValueError(f"{path} missing required columns")
    df = df[["frame", "cluster"]].copy()
    df["frame"] = df["frame"].astype(int)
    return df


def clustered_counts_by_bin(df: pd.DataFrame):
    # Count clustered instances per frame (cluster != -1)
    clustered = df[df["cluster"].notna() & (df["cluster"].astype(int) != -1)]
    per_frame = clustered.groupby("frame").size()
    if per_frame.empty:
        return pd.DataFrame(columns=["bin", "mean_count"])
    bins = (per_frame.index // BIN_SIZE).astype(int)
    by_bin = per_frame.groupby(bins).mean().reset_index()
    by_bin.columns = ["bin", "mean_count"]
    return by_bin


def main():
    if not TEST_DIR.exists():
        raise SystemExit(f"Missing test dir: {TEST_DIR}")
    files = sorted(TEST_DIR.glob("*DBSCAN-eps-43_cos-0.86.feather"))
    if not files:
        raise SystemExit("No DBSCAN feather files found in test dir.")

    gh_file = next((p for p in files if "GH" in p.name), None)
    si_file = next((p for p in files if "SI" in p.name), None)
    if gh_file is None or si_file is None:
        raise SystemExit("Expected one GH and one SI DBSCAN file in test dir.")

    gh_df = load_cluster_file(gh_file)
    si_df = load_cluster_file(si_file)

    gh_bins = clustered_counts_by_bin(gh_df)
    si_bins = clustered_counts_by_bin(si_df)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    if not gh_bins.empty:
        ax.plot(gh_bins["bin"] * BIN_SIZE, gh_bins["mean_count"], label="GH", color="#4a90e2")
    if not si_bins.empty:
        ax.plot(si_bins["bin"] * BIN_SIZE, si_bins["mean_count"], label="SI", color="#f5a623")

    ax.set_xlabel("Frame (bin start)")
    ax.set_ylabel("Mean clustered animals per frame")
    ax.set_title(f"Clustered animals over time (bin = {BIN_SIZE} frames)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=160)
    print("Saved plot to:", OUT_PNG)


if __name__ == "__main__":
    main()
