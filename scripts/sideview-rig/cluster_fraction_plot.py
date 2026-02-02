#!/usr/bin/env python3
# %%
"""
Compute fraction of time spent in each DBSCAN cluster per manual track and plot.

Usage (from repo root):
  python scripts/sideview-rig/cluster_fraction_plot.py
"""

from pathlib import Path
import os
import sys
import site

# Avoid importing x86_64 user-site wheels (e.g., PIL) in the conda env.
user_site = site.getusersitepackages()
if user_site in sys.path:
    sys.path.remove(user_site)
os.environ.setdefault("PYTHONNOUSERSITE", "1")

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# %% Config
ROOT = Path(__file__).resolve().parents[2]
CLUSTER_FEATHER = ROOT / "data" / "sideview_data" / "proofreading" / (
    "GH2026-01-16_12-39-46_SV35.predictions_DBSCAN-eps-43_cos-0.86.feather"
)
MANUAL_CSV = ROOT / "data" / "sideview_data" / "proofreading" / (
    "4tracks-v2_GH2026-01-16_12-39-46_SV35.manual_track.20260131_131658.csv"
)
OUTPUT_PNG = ROOT / "data" / "sideview_data" / "proofreading" / "cluster_fraction_plot.png"


# %% Helpers
def find_col(columns, candidates):
    lower = {c.lower(): c for c in columns}
    for cand in candidates:
        if cand.lower() in lower:
            return lower[cand.lower()]
    return None


def detect_cluster_column(df):
    cols = list(df.columns)
    # Prefer explicit cluster columns
    for name in cols:
        lname = name.lower()
        if "cluster" in lname or "dbscan" in lname or lname == "label":
            return name
    # Fallback: any integer-like column with small number of unique values
    for name in cols:
        if pd.api.types.is_integer_dtype(df[name]) or pd.api.types.is_float_dtype(df[name]):
            uniq = df[name].dropna().unique()
            if len(uniq) > 0 and len(uniq) < 200:
                return name
    return None


# %% Load data
if not CLUSTER_FEATHER.exists():
    raise SystemExit(f"Missing clustering file: {CLUSTER_FEATHER}")
if not MANUAL_CSV.exists():
    raise SystemExit(f"Missing manual track file: {MANUAL_CSV}")

manual_df = pd.read_csv(MANUAL_CSV)
print("Manual columns:", list(manual_df.columns))

# %% Prepare manual data
if "frame" not in manual_df.columns:
    raise SystemExit("Manual CSV must include 'frame'.")
if "instance_id" not in manual_df.columns:
    raise SystemExit("Manual CSV must include 'instance_id'.")
if "track_id" not in manual_df.columns:
    manual_df["track_id"] = 0

manual_df = manual_df.dropna(subset=["frame", "instance_id"])
manual_df["frame"] = manual_df["frame"].astype(int)
manual_df["instance_id"] = manual_df["instance_id"].astype(int)
manual_df["track_id"] = manual_df["track_id"].astype(int)

# Deduplicate per track/frame (keep first)
manual_df = manual_df.sort_values(["track_id", "frame"])
manual_df = manual_df.drop_duplicates(subset=["track_id", "frame"], keep="first")

frames = set(manual_df["frame"].unique())
instances = set(manual_df["instance_id"].unique())

# %% Load clustering data (filtered)
try:
    import pyarrow.feather as feather

    table = feather.read_table(str(CLUSTER_FEATHER))
    cols = table.schema.names
    frame_col = find_col(cols, ["frame", "frame_idx", "frame_index", "frame_id"])
    inst_col = find_col(cols, ["instance_id", "track_id", "track", "instance", "animal_id", "identity"])
    sample = table.slice(0, min(5000, table.num_rows)).to_pandas()
    cluster_col = detect_cluster_column(sample)
    if frame_col is None or inst_col is None or cluster_col is None:
        raise SystemExit(
            f"Could not detect columns. frame={frame_col}, instance={inst_col}, cluster={cluster_col}"
        )
    table = feather.read_table(str(CLUSTER_FEATHER), columns=[frame_col, inst_col, cluster_col])
    cluster_df = table.to_pandas()
except Exception:
    cluster_df = pd.read_feather(CLUSTER_FEATHER)
    frame_col = find_col(cluster_df.columns, ["frame", "frame_idx", "frame_index", "frame_id"])
    inst_col = find_col(cluster_df.columns, ["instance_id", "track_id", "track", "instance", "animal_id", "identity"])
    cluster_col = detect_cluster_column(cluster_df)
    if frame_col is None or inst_col is None or cluster_col is None:
        raise SystemExit(
            f"Could not detect columns. frame={frame_col}, instance={inst_col}, cluster={cluster_col}"
        )
    cluster_df = cluster_df[[frame_col, inst_col, cluster_col]]

print("Cluster columns:", list(cluster_df.columns))
print("Using columns:", {"frame": frame_col, "instance": inst_col, "cluster": cluster_col})

cluster_df = cluster_df[cluster_df[frame_col].isin(frames) & cluster_df[inst_col].isin(instances)].copy()
cluster_df[frame_col] = cluster_df[frame_col].astype(int)
cluster_df[inst_col] = cluster_df[inst_col].astype(int)

# %% Build cluster map (frame, instance) -> cluster_label
cluster_map = (
    cluster_df.groupby([frame_col, inst_col])[cluster_col]
    .agg(lambda s: s.mode().iloc[0] if not s.mode().empty else s.iloc[0])
)

# %% Join cluster labels
manual_df["cluster"] = [
    cluster_map.get((f, inst), np.nan)
    for f, inst in zip(manual_df["frame"], manual_df["instance_id"])
]

# %% Compute fractions per track (cluster identity ignored)
cluster_series = pd.to_numeric(manual_df["cluster"], errors="coerce")
mask_clustered = cluster_series.notna() & (cluster_series != -1)
summary = (
    manual_df.assign(clustered=mask_clustered)
    .groupby("track_id")
    .agg(total_frames=("frame", "size"), clustered_frames=("clustered", "sum"))
    .reset_index()
)
summary["clustered_fraction"] = summary["clustered_frames"] / summary["total_frames"]

print(summary)

# %% Plot
fig, ax = plt.subplots(figsize=(8, 4.5))
tracks = summary["track_id"].astype(int).tolist()
vals = summary["clustered_fraction"].to_numpy()

bars = ax.bar(tracks, vals, color="#4caf50")
for bar, total in zip(bars, summary["total_frames"].astype(int).tolist()):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.02,
        f"{total} frames",
        ha="center",
        va="bottom",
        fontsize=9,
        rotation=0,
    )

ax.set_ylim(0, 1.1)
ax.set_xlabel("Track ID")
ax.set_xticks(tracks)
ax.set_xticklabels([str(t) for t in tracks])
ax.set_ylabel("Fraction of frames in cluster (non -1)")
ax.set_title("Cluster occupancy per manual track (excluding -1)")
plt.tight_layout()
plt.savefig(OUTPUT_PNG, dpi=160)
print(f"Saved plot to: {OUTPUT_PNG}")
