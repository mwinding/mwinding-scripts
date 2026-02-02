#!/usr/bin/env python3
# %%
"""
Recluster manual tracks using the same DBSCAN logic as Crick-HPC (sleap-track_batch.py),
after filling gaps by linear interpolation of head/tail positions.

Usage (from repo root):
  PYTHONNOUSERSITE=1 /Users/windinm/miniconda3/envs/sleap/bin/python \
    scripts/sideview-rig/recluster_manual_tracks.py
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
from sklearn.cluster import DBSCAN
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# %% Config
ROOT = Path(__file__).resolve().parents[2]
DATASETS = [
    {
        "name": "GH2026-01-16_12-39-46_SV35",
        "manual_csv": ROOT / "data" / "sideview_data" / "proofreading" / (
            "4tracks-v3_GH2026-01-16_12-39-46_SV35.manual_track.20260131_131658.csv"
        ),
        "pred_feather": ROOT / "data" / "sideview_data" / "proofreading" / (
            "GH2026-01-16_12-39-46_SV35.predictions.feather"
        ),
        "label_map": {1: "GH_E", 2: "GH_G", 3: "GH_H"},
    },
    {
        "name": "SI2026-01-16_12-46-58_SV37",
        "manual_csv": ROOT / "data" / "sideview_data" / "proofreading" / (
            "4tracks_SI2026-01-16_12-46-58_SV37.manual_track.20260201_005758.csv"
        ),
        "pred_feather": ROOT / "data" / "sideview_data" / "proofreading" / (
            "SI2026-01-16_12-46-58_SV37.predictions.feather"
        ),
        "label_map": {1: "SI_C", 2: "SI_D", 3: "SI_J"},
    },
]

EPS = 43
COS = 0.86
MIN_SAMPLES = 3


# %% DBSCAN per frame (same as Crick-HPC script)
def custom_distance(A, B, cos_thresh=COS):
    tail_A = A[:2]
    tail_B = B[:2]
    euclidean_tail_dist = np.linalg.norm(tail_A - tail_B)

    vector_A = A[2:]
    vector_B = B[2:]
    magnitude_A = np.linalg.norm(vector_A)
    magnitude_B = np.linalg.norm(vector_B)
    if magnitude_A == 0 or magnitude_B == 0:
        return 1000
    cos_similarity = np.dot(vector_A, vector_B) / (magnitude_A * magnitude_B)
    if cos_similarity > cos_thresh:
        return euclidean_tail_dist
    return 1000


def recluster_dataset(entry):
    name = entry["name"]
    manual_csv = entry["manual_csv"]
    pred_feather = entry["pred_feather"]
    label_map = entry.get("label_map", {})

    if not manual_csv.exists():
        raise SystemExit(f"Missing manual track file: {manual_csv}")
    if not pred_feather.exists():
        raise SystemExit(f"Missing predictions file: {pred_feather}")

    manual_df = pd.read_csv(manual_csv)
    for col in ("frame", "instance_id"):
        if col not in manual_df.columns:
            raise SystemExit(f"Manual CSV must include '{col}'.")
    if "track_id" not in manual_df.columns:
        manual_df["track_id"] = 0

    manual_df = manual_df.dropna(subset=["frame", "instance_id"]).copy()
    manual_df["frame"] = manual_df["frame"].astype(int)
    manual_df["instance_id"] = manual_df["instance_id"].astype(int)
    manual_df["track_id"] = manual_df["track_id"].astype(int)
    manual_df = manual_df.sort_values(["track_id", "frame"]).drop_duplicates(subset=["track_id", "frame"], keep="first")

    pred_df = pd.read_feather(pred_feather)
    required = {"frame", "track_id", "x_head", "y_head", "x_tail", "y_tail"}
    missing = required - set(pred_df.columns)
    if missing:
        raise SystemExit(f"Predictions file missing columns: {sorted(missing)}")
    pred_df = pred_df[["frame", "track_id", "x_head", "y_head", "x_tail", "y_tail"]].copy()
    pred_df["frame"] = pred_df["frame"].astype(int)
    pred_df["track_id"] = pred_df["track_id"].astype(int)

    track_spans = (
        manual_df.groupby("track_id")["frame"]
        .agg(["min", "max"])
        .reset_index()
    )
    frames_union = set()
    for _, row in track_spans.iterrows():
        frames_union.update(range(int(row["min"]), int(row["max"]) + 1))
    pred_df = pred_df[pred_df["frame"].isin(frames_union)].copy()

    map_df = manual_df[["frame", "instance_id", "track_id"]].rename(
        columns={"instance_id": "track_id", "track_id": "manual_track_id"}
    )
    pred_df = pred_df.merge(map_df, on=["frame", "track_id"], how="left")

    filled_rows = []
    for tid, span in track_spans.set_index("track_id").iterrows():
        min_f = int(span["min"])
        max_f = int(span["max"])
        full_idx = pd.RangeIndex(min_f, max_f + 1)
        known = pred_df[pred_df["manual_track_id"] == tid][
            ["frame", "x_head", "y_head", "x_tail", "y_tail"]
        ].drop_duplicates(subset=["frame"])
        if known.empty:
            continue
        known = known.set_index("frame").reindex(full_idx)
        for col in ("x_head", "y_head", "x_tail", "y_tail"):
            known[col] = known[col].interpolate(method="linear", limit_area="inside")
        existing_frames = set(
            pred_df.loc[pred_df["manual_track_id"] == tid, "frame"].astype(int).tolist()
        )
        synth = known.loc[~known.index.isin(existing_frames)].dropna().reset_index()
        if synth.empty:
            continue
        synth = synth.rename(columns={"index": "frame"})
        synth["frame"] = synth["frame"].astype(int)
        synth["manual_track_id"] = tid
        synth["track_id"] = -1000 - int(tid)
        filled_rows.append(synth)

    if filled_rows:
        filled = pd.concat(filled_rows, ignore_index=True)
        pred_df = pd.concat([pred_df, filled], ignore_index=True)

    cluster_input = pred_df.dropna(subset=["x_head", "y_head", "x_tail", "y_tail"]).copy()

    clustered_frames = []
    for frame, group in cluster_input.groupby("frame"):
        coords = group[["x_tail", "y_tail"]].to_numpy(dtype=float)
        vectors = (group[["x_tail", "y_tail"]].to_numpy(dtype=float) -
                   group[["x_head", "y_head"]].to_numpy(dtype=float))
        data = np.concatenate([coords, vectors], axis=1)
        dbscan = DBSCAN(eps=EPS, min_samples=MIN_SAMPLES, metric=lambda A, B: custom_distance(A, B, COS))
        labels = dbscan.fit_predict(data)
        out = group.copy()
        out["cluster"] = labels
        clustered_frames.append(out)

    clustered = pd.concat(clustered_frames, ignore_index=True)
    out_feather = ROOT / "data" / "sideview_data" / "proofreading" / (
        f"{name}.manual_tracks_recluster_DBSCAN-eps-{EPS}_cos-{COS}.feather"
    )
    clustered.to_feather(out_feather)

    cluster_series = pd.to_numeric(clustered["cluster"], errors="coerce")
    mask_clustered = cluster_series.notna() & (cluster_series != -1)
    manual_only = clustered[clustered["manual_track_id"].notna()].copy()
    manual_only["manual_track_id"] = manual_only["manual_track_id"].astype(int)
    summary = (
        manual_only.assign(clustered=mask_clustered.loc[manual_only.index])
        .groupby("manual_track_id")
        .agg(total_frames=("frame", "size"), clustered_frames=("clustered", "sum"))
        .reset_index()
        .rename(columns={"manual_track_id": "track_id"})
    )
    summary["clustered_fraction"] = summary["clustered_frames"] / summary["total_frames"]

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
        )
    ax.set_ylim(0, 1.1)
    ax.set_xlabel("Track ID")
    ax.set_xticks(tracks)
    ax.set_xticklabels([label_map.get(t, str(t)) for t in tracks])
    ax.set_ylabel("Fraction of frames in cluster (non -1)")
    ax.set_title(f"Reclustered manual tracks (DBSCAN, per frame)\\n{name}")
    plt.tight_layout()
    out_png = ROOT / "data" / "sideview_data" / "proofreading" / f"cluster_fraction_recluster_{name}.png"
    plt.savefig(out_png, dpi=160)

    print("Saved reclustered labels to:", out_feather)
    print("Saved plot to:", out_png)
    print(summary)
    return {
        "name": name,
        "summary": summary,
        "label_map": label_map,
        "plot_path": out_png,
    }


results = []
for entry in DATASETS:
    results.append(recluster_dataset(entry))

# Combined plot: GH (blue) on left, SI (orange) on right
if len(results) >= 2:
    gh = results[0]
    si = results[1]
    gh_labels = [gh["label_map"].get(t, str(t)) for t in gh["summary"]["track_id"].astype(int).tolist()]
    si_labels = [si["label_map"].get(t, str(t)) for t in si["summary"]["track_id"].astype(int).tolist()]
    gh_vals = gh["summary"]["clustered_fraction"].to_numpy()
    si_vals = si["summary"]["clustered_fraction"].to_numpy()

    labels = gh_labels + si_labels
    x = np.arange(len(labels))
    gh_x = x[: len(gh_labels)]
    si_x = x[len(gh_labels):]

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.bar(gh_x, gh_vals, color="#4a90e2", label="GH")
    ax.bar(si_x, si_vals, color="#f5a623", label="SI")
    ax.set_ylim(0, 1.1)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Fraction of frames in cluster (non -1)")
    ax.set_title("Reclustered manual tracks (GH vs SI)")
    ax.axvline(len(gh_labels) - 0.5, color="#666666", linewidth=1, alpha=0.6)
    ax.legend()
    plt.tight_layout()
    combined_png = ROOT / "data" / "sideview_data" / "proofreading" / "cluster_fraction_recluster_GH_vs_SI.png"
    plt.savefig(combined_png, dpi=160)
    print("Saved combined plot to:", combined_png)

    # Combined plot with frame counts on bars
    fig2, ax2 = plt.subplots(figsize=(10, 4.5))
    bars_gh = ax2.bar(gh_x, gh_vals, color="#4a90e2", label="GH")
    bars_si = ax2.bar(si_x, si_vals, color="#f5a623", label="SI")
    ax2.set_ylim(0, 1.15)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels)
    ax2.set_ylabel("Fraction of frames in cluster (non -1)")
    ax2.set_title("Reclustered manual tracks (GH vs SI) with frame counts")
    ax2.axvline(len(gh_labels) - 0.5, color="#666666", linewidth=1, alpha=0.6)
    ax2.legend()

    gh_counts = gh["summary"]["total_frames"].astype(int).tolist()
    si_counts = si["summary"]["total_frames"].astype(int).tolist()
    for bar, total in zip(bars_gh, gh_counts):
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{total}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    for bar, total in zip(bars_si, si_counts):
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{total}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    plt.tight_layout()
    combined_counts_png = ROOT / "data" / "sideview_data" / "proofreading" / "cluster_fraction_recluster_GH_vs_SI_with_counts.png"
    plt.savefig(combined_counts_png, dpi=160)
    print("Saved combined plot with counts to:", combined_counts_png)
