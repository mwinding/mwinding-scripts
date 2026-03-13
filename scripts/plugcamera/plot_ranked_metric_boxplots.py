#!/usr/bin/env python3
"""
Rank conditions by median effect vs control and draw box/bar plots for plugcamera data.

Default behavior:
- Loads sensory/split CSVs from data/plugcamera_data (auto-discovery).
- Uses condition_norm with control label "control".
- Plots raw total_holes as boxplots.

Examples (from repo root):
  python scripts/plugcamera/plot_ranked_metric_boxplots.py
  python scripts/plugcamera/plot_ranked_metric_boxplots.py --metric total_holes --use-residual
  python scripts/plugcamera/plot_ranked_metric_boxplots.py --plot-kind bar
  python scripts/plugcamera/plot_ranked_metric_boxplots.py --plot-kind points-ci --use-residual
"""

from __future__ import annotations

from pathlib import Path
import argparse
import re
import warnings

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

try:
    from scipy.stats import mannwhitneyu

    HAS_SCIPY = True
except Exception:
    HAS_SCIPY = False


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = ROOT / "data" / "plugcamera_data"
DEFAULT_OUT_DIR = DEFAULT_DATA_DIR / "plots"
DEFAULT_SPLIT_ANNOTATION_CSV = DEFAULT_DATA_DIR / "student_screenshot_annotations.csv"
DEFAULT_SENSORY_STOCKS_CSV = DEFAULT_DATA_DIR / "stocks-sensory-screen.csv"
CONTROL_COLOR = "#8f8f8f"
NS_COLOR = "#d8d8d8"
UP_COLOR = "#4c72b0"
DOWN_COLOR = "#c44e52"
UP_TREND_COLOR = "#9eb6dc"
DOWN_TREND_COLOR = "#de9fa5"
WIDTH_PER_CONDITION = 0.145
MIN_PANEL_WIDTH = 8.0
PER_HIT_COL_WIDTH = 2.45
PER_HIT_ROW_HEIGHT = 2.4
MIN_PER_HIT_WIDTH = 10.5
MIN_PER_HIT_HEIGHT = 4.8


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Make ranked effect-vs-control boxplots for plugcamera metrics."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help=f"Directory containing plugcamera CSVs (default: {DEFAULT_DATA_DIR})",
    )
    parser.add_argument(
        "--sensory-file",
        type=Path,
        default=None,
        help="Optional explicit sensory CSV path.",
    )
    parser.add_argument(
        "--split-file",
        type=Path,
        default=None,
        help="Optional explicit split CSV path.",
    )
    parser.add_argument(
        "--screen",
        choices=("both", "sensory", "split"),
        default="both",
        help="Which screen(s) to plot (default: both).",
    )
    parser.add_argument(
        "--metric",
        default="total_holes",
        help="Metric column to plot (default: total_holes).",
    )
    parser.add_argument(
        "--use-residual",
        action="store_true",
        help="If set, use <metric>_residual (or metric if already residual).",
    )
    parser.add_argument(
        "--condition-column",
        default="condition_norm",
        help="Condition column to group/rank by (default: condition_norm).",
    )
    parser.add_argument(
        "--control-label",
        default="control",
        help="Control label in the condition column (default: control).",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="Significance threshold for highlighting (default: 0.05).",
    )
    parser.add_argument(
        "--trend-p-threshold",
        type=float,
        default=None,
        help="Optional trend-color threshold. If set (e.g. 0.1), p in [alpha, threshold) are light blue/red by effect direction.",
    )
    parser.add_argument(
        "--min-n",
        type=int,
        default=3,
        help="Minimum non-NaN samples per condition to include (default: 3).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output path (default: data/plugcamera_data/plots/ranked_<plot-kind>plots_<metric>.png).",
    )
    parser.add_argument(
        "--plot-kind",
        choices=("box", "bar", "points-ci", "box-points"),
        default="box",
        help="Plot style for ranked conditions (default: box).",
    )
    parser.add_argument(
        "--control-position",
        choices=("first", "ranked"),
        default="ranked",
        help="Place control first or at its ranked position (default: ranked).",
    )
    parser.add_argument(
        "--center-to-control",
        action="store_true",
        help="Shift plotted values by control median so control is centered at 0.",
    )
    parser.add_argument(
        "--whis",
        type=float,
        default=1.3,
        help="IQR multiplier for boxplot whiskers (default: 1.3; smaller = more outliers shown).",
    )
    parser.add_argument(
        "--ci-level",
        type=float,
        default=95.0,
        help="Confidence interval level for bar plots in percent (default: 95).",
    )
    parser.add_argument(
        "--value-scale",
        choices=("raw", "control-sd"),
        default="control-sd",
        help="Y-value scaling: raw values or control-centered/scaled units (default: control-sd).",
    )
    parser.add_argument(
        "--single-sort-metric",
        default=None,
        help="Optional metric used to order conditions for single-metric plots (e.g. total_holes).",
    )
    parser.add_argument(
        "--split-label-mode",
        choices=("annotated", "id"),
        default="annotated",
        help="Split-screen x-axis labels: 'annotated' uses ID|name, 'id' uses raw SS/MB IDs only.",
    )
    parser.add_argument(
        "--sensory-label-mode",
        choices=("annotated", "id"),
        default="annotated",
        help="Sensory-screen x-axis labels: 'annotated' uses ID|GAL4 line, 'id' uses raw condition IDs only.",
    )
    parser.add_argument(
        "--sensory-stocks-csv",
        type=Path,
        default=None,
        help="Optional sensory stock table CSV. Default: data/plugcamera_data/stocks-sensory-screen.csv (or data/plugcamera-data fallback).",
    )
    parser.add_argument(
        "--composite-residuals",
        action="store_true",
        help="Also write a composite figure with the three residual metrics stacked and a shared condition order.",
    )
    parser.add_argument(
        "--composite-sort-metric",
        default="total_holes",
        help="Metric used to define shared condition order for composite residuals (total_holes, avg_max_size, avg_lifetime, or composite).",
    )
    parser.add_argument(
        "--composite-output",
        type=Path,
        default=None,
        help="Optional output path for composite residual plot.",
    )
    parser.add_argument(
        "--composite-heatmap",
        action="store_true",
        help="Also write a 3xN residual heatmap with shared condition order.",
    )
    parser.add_argument(
        "--composite-heatmap-output",
        type=Path,
        default=None,
        help="Optional output path for composite residual heatmap.",
    )
    parser.add_argument(
        "--skip-main-plot",
        action="store_true",
        help="Skip writing the single-metric main plot (useful for composite-only runs).",
    )
    parser.add_argument(
        "--per-hit-composites",
        action="store_true",
        help="Write one 3-metric boxplot figure per significant hit (Number of Holes, Hole Size, Lifetime).",
    )
    parser.add_argument(
        "--per-hit-source-metric",
        default="total_holes",
        help="Metric used to define significant hits for per-hit composites (default: total_holes).",
    )
    parser.add_argument(
        "--per-hit-output-dir",
        type=Path,
        default=None,
        help="Output directory for per-hit grid figure (default: data/plugcamera_data/plots/per_hit_composites_<screen>).",
    )
    parser.add_argument(
        "--per-hit-grid-ncols",
        type=int,
        default=6,
        help="Number of columns in per-hit grid layout (default: 6).",
    )
    return parser.parse_args()


def discover_dataset_file(data_dir: Path, kind: str) -> Path:
    preferred = data_dir / f"residuals_all_metrics_{kind}-350max-pupae.csv"
    if preferred.exists():
        return preferred

    candidates = sorted(data_dir.glob(f"*{kind}*.csv"))
    if not candidates:
        raise FileNotFoundError(
            f"No CSV found for '{kind}' in {data_dir}. "
            f"Tried preferred file and glob '*{kind}*.csv'."
        )
    if len(candidates) > 1:
        warnings.warn(
            f"Multiple CSVs found for '{kind}'. Using first match: {candidates[0].name}",
            stacklevel=2,
        )
    return candidates[0]


def resolve_metric_column(df: pd.DataFrame, metric: str, use_residual: bool) -> str:
    if use_residual:
        if metric.endswith("_residual"):
            metric_col = metric
        else:
            metric_col = f"{metric}_residual"
    else:
        metric_col = metric

    if metric_col not in df.columns:
        raise ValueError(
            f"Metric column '{metric_col}' not found. Available columns: {list(df.columns)}"
        )
    return metric_col


def y_axis_label(metric_col: str, value_scale: str) -> str:
    metric_key = metric_col.replace("_residual", "")
    pretty = {
        "total_holes": "Number of Holes",
        "avg_max_size": "Hole Size",
        "avg_lifetime": "Lifetime",
    }.get(metric_key, metric_key.replace("_", " ").title())
    if value_scale == "control-sd":
        return f"{pretty} (Relative Effect)"
    if metric_col.endswith("_residual"):
        return f"{pretty} Residual"
    return pretty


def metric_pretty_name(metric_col: str) -> str:
    metric_key = metric_col.replace("_residual", "")
    return {
        "total_holes": "Number of Holes",
        "avg_max_size": "Hole Size",
        "avg_lifetime": "Lifetime",
    }.get(metric_key, metric_key.replace("_", " ").title())


def resolve_residual_metric(metric: str) -> str:
    if metric.lower() == "composite":
        return "composite"
    return metric if metric.endswith("_residual") else f"{metric}_residual"


def metric_base_name(metric_col: str) -> str:
    return metric_col.replace("_residual", "")


def infer_metric_from_output_name(path: Path) -> str | None:
    stem = path.stem
    match = re.search(
        r"^ranked_[a-z0-9_]+plots_(total_holes|avg_max_size|avg_lifetime)(?:[_-]|$)",
        stem,
    )
    if match:
        return str(match.group(1))
    return None


def residual_heatmap_cmap() -> LinearSegmentedColormap:
    # Balanced diverging ramp:
    # - smooth red->white on [-1.5, -0.5]
    # - white plateau on [-0.5, +0.5]
    # - smooth white->blue on [+0.5, +1.5]
    # with vmin=-1.5, vmax=+1.5.
    white_lo = ( -0.5 + 1.5 ) / 3.0  # 0.333...
    white_hi = (  0.5 + 1.5 ) / 3.0  # 0.666...
    return LinearSegmentedColormap.from_list(
        "residual_whiteband",
        [
            (0.0, "#b2182b"),
            (0.18, "#ef8a62"),
            (white_lo, "#ffffff"),
            (white_hi, "#ffffff"),
            (0.82, "#67a9cf"),
            (1.0, "#2166ac"),
        ],
    )


def clean_text(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    text = str(value).strip()
    return "" if text.lower() in {"nan", "none"} else text


def is_valid_student_annotation(text: str) -> bool:
    low = text.lower().strip()
    if not low:
        return False
    junk_phrases = (
        "same lineage",
        "different lineage",
        "search for ",
    )
    return not any(p in low for p in junk_phrases)


def load_split_annotation_map() -> dict[str, str]:
    # Intentionally trust only the student-provided annotation table.
    mapping: dict[str, str] = {}
    if DEFAULT_SPLIT_ANNOTATION_CSV.exists():
        df = pd.read_csv(DEFAULT_SPLIT_ANNOTATION_CSV)
        if {"split_id", "annotation"}.issubset(df.columns):
            for _, row in df.iterrows():
                sid = clean_text(row.get("split_id", "")).upper()
                ann = clean_text(row.get("annotation", ""))
                if sid and ann and is_valid_student_annotation(ann) and sid not in mapping:
                    mapping[sid] = ann
    return mapping


def display_label(cond: str, annotation_map: dict[str, str], annotation_only: bool = False) -> str:
    ann = annotation_map.get(cond.upper(), "")
    if not ann:
        return cond
    return ann if annotation_only else f"{cond} | {ann}"


def normalize_sensory_id(value: object) -> str:
    text = clean_text(value).upper()
    if not text:
        return ""
    text = re.sub(r"\s+", "", text)
    match = re.match(r"^T(\d+)([A-Z])(\d+)$", text)
    if match:
        return f"T{int(match.group(1)):02d}{match.group(2)}{int(match.group(3)):02d}"
    if text.isdigit():
        return str(int(text))
    return text


def resolve_sensory_stocks_csv(path_arg: Path | None) -> Path | None:
    if path_arg is not None:
        p = path_arg.resolve()
        return p if p.exists() else None
    candidates = [
        DEFAULT_SENSORY_STOCKS_CSV,
        ROOT / "data" / "plugcamera-data" / "stocks-sensory-screen.csv",
    ]
    for p in candidates:
        if p.exists():
            return p.resolve()
    return None


def load_sensory_annotation_map(path_arg: Path | None) -> dict[str, str]:
    csv_path = resolve_sensory_stocks_csv(path_arg)
    if csv_path is None:
        return {}
    try:
        df = pd.read_csv(csv_path, dtype=str, encoding="utf-8-sig")
    except Exception:
        return {}

    cols = {str(c).strip().lower(): c for c in df.columns}
    loc_col = cols.get("stock-location") or cols.get("stock location") or cols.get("stock l.ocation")
    line_col = cols.get("lines")
    bloom_col = cols.get("blooming stock")
    if loc_col is None and bloom_col is None:
        return {}

    mapping: dict[str, str] = {}
    for _, row in df.iterrows():
        gal4 = clean_text(row.get(line_col, "")) if line_col is not None else ""
        if not gal4:
            continue
        key_loc = normalize_sensory_id(row.get(loc_col, "")) if loc_col is not None else ""
        key_bloom = normalize_sensory_id(row.get(bloom_col, "")) if bloom_col is not None else ""
        if key_loc and key_loc not in mapping:
            mapping[key_loc] = gal4
        if key_bloom and key_bloom not in mapping:
            mapping[key_bloom] = gal4
    return mapping


def write_per_hit_composites(
    df: pd.DataFrame,
    screen: str,
    args: argparse.Namespace,
    condition_col: str,
    control_label: str,
    metric_labels: dict[str, str],
) -> tuple[int, Path]:
    metric_triplet = ["total_holes", "avg_max_size", "avg_lifetime"]
    metric_cols = [resolve_metric_column(df, m, args.use_residual) for m in metric_triplet]
    source_metric_col = resolve_metric_column(df, args.per_hit_source_metric, args.use_residual)

    ranked_source = build_ranked_data(
        df,
        metric_col=source_metric_col,
        condition_col=condition_col,
        control_label=control_label,
        min_n=args.min_n,
        alpha=args.alpha,
        control_position=args.control_position,
        center_to_control=args.center_to_control,
        value_scale=args.value_scale,
    )
    sig_set = set(ranked_source["sig_conditions"])
    control_name = ranked_source["control_name"]
    # Include all hits significant in any of the three metrics.
    sig_union: set[str] = set()
    hits: list[str] = []

    ranked_by_metric: dict[str, dict] = {}
    for mcol in metric_cols:
        ranked_by_metric[mcol] = build_ranked_data(
            df,
            metric_col=mcol,
            condition_col=condition_col,
            control_label=control_label,
            min_n=args.min_n,
            alpha=args.alpha,
            control_position=args.control_position,
            center_to_control=args.center_to_control,
            value_scale=args.value_scale,
        )
        sig_union.update(ranked_by_metric[mcol]["sig_conditions"])

    hits = [c for c in ranked_source["order"] if c in sig_union and c != control_name]

    out_dir = (
        args.per_hit_output_dir.resolve()
        if args.per_hit_output_dir
        else (DEFAULT_OUT_DIR / f"per_hit_composites_{screen}").resolve()
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    valid_hits: list[tuple[str, list[np.ndarray], list[str]]] = []
    for hit in hits:
        vals: list[np.ndarray] = []
        labels: list[str] = []
        ok = True
        for mcol in metric_cols:
            r = ranked_by_metric[mcol]
            if hit not in r["values_by_cond"]:
                ok = False
                break
            vals.append(r["values_by_cond"][hit])
            labels.append(metric_labels[mcol])
        if ok:
            valid_hits.append((hit, vals, labels))

    if not valid_hits:
        return 0, out_dir

    ncols = max(1, int(args.per_hit_grid_ncols))
    n = len(valid_hits)
    nrows = int(np.ceil(n / ncols))
    fig_w, fig_h = max(MIN_PER_HIT_WIDTH, ncols * PER_HIT_COL_WIDTH), max(MIN_PER_HIT_HEIGHT, nrows * PER_HIT_ROW_HEIGHT)
    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h), constrained_layout=True)
    axes_arr = np.atleast_1d(axes).ravel()
    fig.patch.set_facecolor("white")
    y_label = "Relative Effect" if args.value_scale == "control-sd" else "Residual Value"
    for ax, (hit, vals, labels) in zip(axes_arr, valid_hits):
        ax.set_facecolor("white")
        bp = ax.boxplot(
            vals,
            patch_artist=True,
            widths=0.62,
            showfliers=True,
            whis=args.whis,
            medianprops={"color": "#1a1a1a", "linewidth": 1.2},
            whiskerprops={"color": "#444444", "linewidth": 0.9},
            capprops={"color": "#444444", "linewidth": 0.9},
            boxprops={"edgecolor": "#444444", "linewidth": 0.9},
            flierprops={"marker": "o", "markersize": 1.8, "markerfacecolor": "#333333", "markeredgecolor": "none", "alpha": 0.25},
        )
        # Color each metric box by that metric's significance + effect direction for this hit.
        for patch, mcol in zip(bp["boxes"], metric_cols):
            c = condition_color(hit, ranked_by_metric[mcol], args.alpha, args.trend_p_threshold)
            patch.set_facecolor(c)
            patch.set_alpha(0.9)
        ax.axhline(0.0, color="#666666", linestyle="--", linewidth=0.9)
        ax.set_xticks([1, 2, 3])
        ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=6)
        ax.set_title(hit, fontsize=9)
        ax.tick_params(axis="y", labelsize=7)

    for ax in axes_arr[n:]:
        ax.axis("off")

    # Left-most column keeps y-labels to reduce visual clutter.
    for i, ax in enumerate(axes_arr[:n]):
        if i % ncols == 0:
            ax.set_ylabel(y_label, fontsize=8)
        else:
            ax.set_ylabel("")

    source_slug = source_metric_col.replace("_residual", "")
    trend_slug = ""
    if args.trend_p_threshold is not None:
        trend_slug = f"_trend{str(args.trend_p_threshold).replace('.', 'p')}"
    out_path = out_dir / f"per_hit_metric_grid_{screen}_all-metrics_source-{source_slug}{trend_slug}.png"
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return n, out_path


def build_ranked_data(
    df: pd.DataFrame,
    metric_col: str,
    condition_col: str,
    control_label: str,
    min_n: int,
    alpha: float,
    control_position: str,
    center_to_control: bool,
    value_scale: str,
    allowed_conditions: set[str] | None = None,
    fixed_order: list[str] | None = None,
) -> dict:
    if condition_col not in df.columns:
        raise ValueError(f"Missing condition column '{condition_col}'.")

    work = df[[condition_col, metric_col]].copy()
    work[condition_col] = work[condition_col].astype(str)
    work[metric_col] = pd.to_numeric(work[metric_col], errors="coerce")
    work = work.dropna(subset=[metric_col])

    control_mask = work[condition_col].str.lower() == control_label.lower()
    control_values_raw = work.loc[control_mask, metric_col].to_numpy()
    if control_values_raw.size < min_n:
        raise ValueError(
            f"Control group '{control_label}' has only {control_values_raw.size} valid rows (< min_n={min_n})."
        )

    actual_control_name = (
        work.loc[control_mask, condition_col].iloc[0] if control_mask.any() else control_label
    )
    raw_control_median = float(np.median(control_values_raw))
    control_scale = 1.0
    if value_scale == "control-sd":
        mad = float(np.median(np.abs(control_values_raw - raw_control_median)))
        robust_sd = 1.4826 * mad
        if np.isfinite(robust_sd) and robust_sd > 1e-12:
            control_scale = robust_sd
        else:
            std = float(np.std(control_values_raw, ddof=1)) if control_values_raw.size > 1 else 1.0
            control_scale = std if np.isfinite(std) and std > 1e-12 else 1.0

    control_median = 0.0 if (center_to_control or value_scale == "control-sd") else raw_control_median

    values_by_cond: dict[str, np.ndarray] = {}
    summary_rows = []
    for cond, grp in work.groupby(condition_col):
        if allowed_conditions is not None and cond not in allowed_conditions:
            continue
        vals_raw = grp[metric_col].dropna().to_numpy()
        if vals_raw.size < min_n:
            continue

        if value_scale == "control-sd":
            vals = (vals_raw - raw_control_median) / control_scale
        elif center_to_control:
            vals = vals_raw - raw_control_median
        else:
            vals = vals_raw
        med = float(np.median(vals))
        effect = med - control_median
        pval = np.nan
        if HAS_SCIPY and cond.lower() != control_label.lower():
            try:
                if value_scale == "control-sd":
                    control_vals_for_test = (control_values_raw - raw_control_median) / control_scale
                elif center_to_control:
                    control_vals_for_test = control_values_raw - raw_control_median
                else:
                    control_vals_for_test = control_values_raw
                _, pval = mannwhitneyu(vals, control_vals_for_test, alternative="two-sided")
            except ValueError:
                pval = np.nan

        values_by_cond[cond] = vals
        summary_rows.append(
            {
                "condition": cond,
                "n": int(vals.size),
                "median": med,
                "effect_vs_control": effect,
                "p_value": pval,
            }
        )

    summary = pd.DataFrame(summary_rows)
    if summary.empty:
        raise ValueError(f"No groups found with at least min_n={min_n} for {metric_col}.")

    sig_conditions = set()
    if HAS_SCIPY:
        non_control = summary[summary["condition"].str.lower() != control_label.lower()].copy()
        sig_conditions = set(
            non_control.loc[non_control["p_value"] < alpha, "condition"].astype(str).tolist()
        )

    summary_sorted = summary.sort_values("effect_vs_control", ascending=True)
    if fixed_order is not None:
        order = [c for c in fixed_order if c in set(summary_sorted["condition"])]
        if not order:
            raise ValueError("Fixed order contains no conditions available for this metric.")
        return {
            "order": order,
            "values_by_cond": values_by_cond,
            "summary": summary,
            "effect_by_cond": summary.set_index("condition")["effect_vs_control"].to_dict(),
            "p_by_cond": summary.set_index("condition")["p_value"].to_dict(),
            "control_name": actual_control_name,
            "control_median": control_median,
            "sig_conditions": sig_conditions,
        }
    if control_position == "first":
        non_control = summary_sorted[summary_sorted["condition"].str.lower() != control_label.lower()].copy()
        order = [actual_control_name] + non_control["condition"].tolist()
    else:
        sig_mask = summary_sorted["condition"].isin(sig_conditions)
        sig_down = summary_sorted[sig_mask & (summary_sorted["effect_vs_control"] < 0.0)]
        nonsig = summary_sorted[~sig_mask]  # Includes control; stays sorted by effect.
        sig_up = summary_sorted[sig_mask & (summary_sorted["effect_vs_control"] >= 0.0)]
        order = (
            sig_down["condition"].tolist()
            + nonsig["condition"].tolist()
            + sig_up["condition"].tolist()
        )

    return {
        "order": order,
        "values_by_cond": values_by_cond,
        "summary": summary,
        "effect_by_cond": summary.set_index("condition")["effect_vs_control"].to_dict(),
        "p_by_cond": summary.set_index("condition")["p_value"].to_dict(),
        "control_name": actual_control_name,
        "control_median": control_median,
        "sig_conditions": sig_conditions,
    }


def add_reference_lines(ax: plt.Axes, control_median: float) -> None:
    ax.axhline(0.0, color="#666666", linestyle="--", linewidth=1.0)


def condition_color(cond: str, ranked: dict, alpha: float, trend_p_threshold: float | None = None) -> str:
    if cond == ranked["control_name"]:
        return CONTROL_COLOR
    pval = float(ranked["p_by_cond"].get(cond, np.nan))
    effect = float(ranked["effect_by_cond"].get(cond, 0.0))
    if cond not in ranked["sig_conditions"]:
        if trend_p_threshold is not None and np.isfinite(pval) and alpha <= pval < trend_p_threshold:
            return UP_TREND_COLOR if effect >= 0.0 else DOWN_TREND_COLOR
        return NS_COLOR
    return UP_COLOR if effect >= 0.0 else DOWN_COLOR


def plot_ranked_boxplot(
    ax: plt.Axes,
    ranked: dict,
    dataset_label: str,
    metric_col: str,
    alpha: float,
    whis: float,
    x_labels: list[str] | None = None,
    trend_p_threshold: float | None = None,
) -> None:
    order = ranked["order"]
    values = [ranked["values_by_cond"][c] for c in order]
    control_name = ranked["control_name"]
    control_median = ranked["control_median"]
    sig_conditions = ranked["sig_conditions"]

    bp = ax.boxplot(
        values,
        patch_artist=True,
        widths=0.65,
        showfliers=True,
        whis=whis,
        medianprops={"color": "#1a1a1a", "linewidth": 1.5},
        whiskerprops={"color": "#444444", "linewidth": 1.0},
        capprops={"color": "#444444", "linewidth": 1.0},
        boxprops={"edgecolor": "#444444", "linewidth": 1.0},
        flierprops={
            "marker": "o",
            "markersize": 2.5,
            "markerfacecolor": "#333333",
            "markeredgecolor": "none",
            "alpha": 0.4,
        },
    )

    for patch, cond in zip(bp["boxes"], order):
        patch.set_facecolor(condition_color(cond, ranked, alpha, trend_p_threshold))
        patch.set_alpha(0.95)

    add_reference_lines(ax, control_median)

    ax.set_xticks(range(1, len(order) + 1))
    ax.set_xticklabels(x_labels if x_labels is not None else order, rotation=62, ha="right", fontsize=8)
    ax.set_xlabel("")
    ax.set_ylabel(metric_col)
    ax.set_title(dataset_label, fontsize=14)


def bootstrap_mean_ci(
    values: np.ndarray, ci_level: float, n_boot: int = 2000, seed: int = 0
) -> tuple[float, float, float]:
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return np.nan, np.nan, np.nan
    if vals.size == 1:
        m = float(vals[0])
        return m, m, m
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, vals.size, size=(n_boot, vals.size))
    boot_means = vals[idx].mean(axis=1)
    alpha = (100.0 - ci_level) / 2.0
    lo = float(np.percentile(boot_means, alpha))
    hi = float(np.percentile(boot_means, 100.0 - alpha))
    m = float(vals.mean())
    return m, lo, hi


def plot_ranked_barplot(
    ax: plt.Axes,
    ranked: dict,
    dataset_label: str,
    metric_col: str,
    alpha: float,
    ci_level: float,
    x_labels: list[str] | None = None,
    trend_p_threshold: float | None = None,
) -> None:
    order = ranked["order"]
    control_name = ranked["control_name"]
    control_median = ranked["control_median"]
    sig_conditions = ranked["sig_conditions"]

    means: list[float] = []
    yerr_lo: list[float] = []
    yerr_hi: list[float] = []
    for i, cond in enumerate(order):
        m, lo, hi = bootstrap_mean_ci(
            ranked["values_by_cond"][cond], ci_level=ci_level, n_boot=2000, seed=9173 + i
        )
        means.append(m)
        yerr_lo.append(max(0.0, m - lo))
        yerr_hi.append(max(0.0, hi - m))

    x = np.arange(1, len(order) + 1)
    colors = []
    for cond in order:
        colors.append(condition_color(cond, ranked, alpha, trend_p_threshold))

    ax.bar(
        x,
        means,
        width=0.72,
        color=colors,
        edgecolor="#444444",
        linewidth=1.0,
        yerr=np.vstack([yerr_lo, yerr_hi]),
        capsize=2.5,
        ecolor="#333333",
        error_kw={"elinewidth": 1.0, "capthick": 1.0},
        zorder=2,
    )

    add_reference_lines(ax, control_median)
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels if x_labels is not None else order, rotation=62, ha="right", fontsize=8)
    ax.set_xlabel("")
    ax.set_ylabel(metric_col)
    ax.set_title(dataset_label, fontsize=14)


def plot_ranked_points_ci(
    ax: plt.Axes,
    ranked: dict,
    dataset_label: str,
    metric_col: str,
    alpha: float,
    ci_level: float,
    x_labels: list[str] | None = None,
    trend_p_threshold: float | None = None,
) -> None:
    order = ranked["order"]
    control_name = ranked["control_name"]
    control_median = ranked["control_median"]
    sig_conditions = ranked["sig_conditions"]

    x = np.arange(1, len(order) + 1)
    means: list[float] = []
    yerr_lo: list[float] = []
    yerr_hi: list[float] = []
    for i, cond in enumerate(order):
        vals = ranked["values_by_cond"][cond]
        m, lo, hi = bootstrap_mean_ci(vals, ci_level=ci_level, n_boot=2000, seed=9173 + i)
        means.append(m)
        yerr_lo.append(max(0.0, m - lo))
        yerr_hi.append(max(0.0, hi - m))

        point_color = condition_color(cond, ranked, alpha, trend_p_threshold)
        rng = np.random.default_rng(1500 + i)
        jitter = rng.uniform(-0.15, 0.15, size=vals.size)
        ax.scatter(
            np.full(vals.size, x[i]) + jitter,
            vals,
            s=7,
            color=point_color,
            alpha=0.12,
            linewidths=0,
            zorder=1,
        )

    mean_colors = [condition_color(cond, ranked, alpha, trend_p_threshold) for cond in order]

    ax.errorbar(
        x,
        means,
        yerr=np.vstack([yerr_lo, yerr_hi]),
        fmt="o",
        color="#3d3d3d",
        ecolor="#3d3d3d",
        elinewidth=1.4,
        capsize=2.8,
        markersize=4.8,
        markerfacecolor="white",
        markeredgewidth=1.4,
        zorder=3,
    )
    for xi, yi, c in zip(x, means, mean_colors):
        ax.plot(xi, yi, marker="o", markersize=5.2, markerfacecolor="white", markeredgecolor=c, markeredgewidth=1.6, zorder=4)

    add_reference_lines(ax, control_median)
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels if x_labels is not None else order, rotation=62, ha="right", fontsize=8)
    ax.set_xlabel("")
    ax.set_ylabel(metric_col)
    ax.set_title(dataset_label, fontsize=14)


def plot_ranked_box_points(
    ax: plt.Axes,
    ranked: dict,
    dataset_label: str,
    metric_col: str,
    alpha: float,
    whis: float,
    x_labels: list[str] | None = None,
    trend_p_threshold: float | None = None,
) -> None:
    order = ranked["order"]
    values = [ranked["values_by_cond"][c] for c in order]
    control_name = ranked["control_name"]
    control_median = ranked["control_median"]
    sig_conditions = ranked["sig_conditions"]

    bp = ax.boxplot(
        values,
        patch_artist=True,
        widths=0.62,
        showfliers=False,
        whis=whis,
        medianprops={"color": "#1a1a1a", "linewidth": 1.4},
        whiskerprops={"color": "#555555", "linewidth": 0.9},
        capprops={"color": "#555555", "linewidth": 0.9},
        boxprops={"edgecolor": "#444444", "linewidth": 1.0},
    )
    for patch, cond in zip(bp["boxes"], order):
        patch.set_facecolor(condition_color(cond, ranked, alpha, trend_p_threshold))
        patch.set_alpha(0.88)

    for i, cond in enumerate(order):
        vals = ranked["values_by_cond"][cond]
        point_color = condition_color(cond, ranked, alpha, trend_p_threshold)
        rng = np.random.default_rng(2600 + i)
        jitter = rng.uniform(-0.18, 0.18, size=vals.size)
        ax.scatter(
            np.full(vals.size, i + 1) + jitter,
            vals,
            s=7,
            color=point_color,
            alpha=0.18,
            linewidths=0,
            zorder=1,
        )

    add_reference_lines(ax, control_median)
    ax.set_xticks(range(1, len(order) + 1))
    ax.set_xticklabels(x_labels if x_labels is not None else order, rotation=62, ha="right", fontsize=8)
    ax.set_xlabel("")
    ax.set_ylabel(metric_col)
    ax.set_title(dataset_label, fontsize=14)


def main() -> None:
    args = parse_args()
    if args.skip_main_plot and not (args.composite_residuals or args.composite_heatmap or args.per_hit_composites):
        raise SystemExit("--skip-main-plot requires --composite-residuals, --composite-heatmap, and/or --per-hit-composites.")
    data_dir = args.data_dir.resolve()
    if not data_dir.exists():
        raise SystemExit(f"Data dir does not exist: {data_dir}")

    screen_order = ["sensory", "split"] if args.screen == "both" else [args.screen]
    file_by_screen: dict[str, Path] = {}
    if "sensory" in screen_order:
        sensory_file = args.sensory_file.resolve() if args.sensory_file else discover_dataset_file(data_dir, "sensory")
        if not sensory_file.exists():
            raise SystemExit(f"Sensory file not found: {sensory_file}")
        file_by_screen["sensory"] = sensory_file
    if "split" in screen_order:
        split_file = args.split_file.resolve() if args.split_file else discover_dataset_file(data_dir, "split")
        if not split_file.exists():
            raise SystemExit(f"Split file not found: {split_file}")
        file_by_screen["split"] = split_file

    metric_col: str | None = None
    df_by_screen: dict[str, pd.DataFrame] = {}
    ranked_by_screen: dict[str, dict] = {}
    for screen in screen_order:
        df = pd.read_csv(file_by_screen[screen])
        df_by_screen[screen] = df
        resolved_metric = resolve_metric_column(df, args.metric, args.use_residual)
        if metric_col is None:
            metric_col = resolved_metric
        elif resolved_metric != metric_col:
            raise SystemExit(f"Metric mismatch between screens: {metric_col} vs {resolved_metric}")
        sort_order: list[str] | None = None
        if args.single_sort_metric:
            sort_metric_col = resolve_metric_column(df, args.single_sort_metric, args.use_residual)
            ranked_sort = build_ranked_data(
                df,
                metric_col=sort_metric_col,
                condition_col=args.condition_column,
                control_label=args.control_label,
                min_n=args.min_n,
                alpha=args.alpha,
                control_position=args.control_position,
                center_to_control=args.center_to_control,
                value_scale=args.value_scale,
            )
            sort_order = ranked_sort["order"]
        ranked_by_screen[screen] = build_ranked_data(
            df,
            metric_col=resolved_metric,
            condition_col=args.condition_column,
            control_label=args.control_label,
            min_n=args.min_n,
            alpha=args.alpha,
            control_position=args.control_position,
            center_to_control=args.center_to_control,
            value_scale=args.value_scale,
            fixed_order=sort_order,
        )
    if metric_col is None:
        raise SystemExit("No screen selected to plot.")

    output_path: Path | None = None
    if args.output:
        output_path = args.output.resolve()
        implied_metric = infer_metric_from_output_name(output_path)
        actual_metric = metric_base_name(metric_col)
        if implied_metric is not None and implied_metric != actual_metric and not args.skip_main_plot:
            raise SystemExit(
                f"Output filename suggests metric '{implied_metric}', but --metric resolved to '{actual_metric}'. "
                "Use matching --metric, rename --output, or add --skip-main-plot."
            )
    elif not args.skip_main_plot:
        DEFAULT_OUT_DIR.mkdir(parents=True, exist_ok=True)
        suffix = metric_col.replace("_residual", "-residual")
        plot_kind_slug = args.plot_kind.replace("-", "_")
        screen_suffix = "" if args.screen == "both" else f"_{args.screen}"
        output_path = (DEFAULT_OUT_DIR / f"ranked_{plot_kind_slug}plots_{suffix}{screen_suffix}.png").resolve()

    split_annotation_map = load_split_annotation_map() if args.split_label_mode == "annotated" else {}
    sensory_annotation_map = (
        load_sensory_annotation_map(args.sensory_stocks_csv) if args.sensory_label_mode == "annotated" else {}
    )
    if not args.skip_main_plot:
        max_conditions = max(len(ranked_by_screen[s]["order"]) for s in screen_order)
        fig_width = max(MIN_PANEL_WIDTH, max_conditions * WIDTH_PER_CONDITION)
        fig_height = 5 * len(screen_order)
        fig, axes = plt.subplots(len(screen_order), 1, figsize=(fig_width, fig_height), constrained_layout=True)
        if len(screen_order) == 1:
            axes = [axes]
        fig.patch.set_facecolor("white")
        for ax, screen in zip(axes, screen_order):
            ax.set_facecolor("white")
            ranked = ranked_by_screen[screen]
            if screen == "split":
                labels = [display_label(cond, split_annotation_map) for cond in ranked["order"]]
            elif screen == "sensory":
                labels = [display_label(cond, sensory_annotation_map, annotation_only=True) for cond in ranked["order"]]
            else:
                labels = ranked["order"]
            dataset_label = "Sensory inactivation screen" if screen == "sensory" else "Split-GAL4 inactivation screen"
            y_label = y_axis_label(metric_col, args.value_scale)
            if args.plot_kind == "bar":
                plot_ranked_barplot(ax, ranked, dataset_label, y_label, args.alpha, args.ci_level, labels, args.trend_p_threshold)
            elif args.plot_kind == "points-ci":
                plot_ranked_points_ci(ax, ranked, dataset_label, y_label, args.alpha, args.ci_level, labels, args.trend_p_threshold)
            elif args.plot_kind == "box-points":
                plot_ranked_box_points(ax, ranked, dataset_label, y_label, args.alpha, args.whis, labels, args.trend_p_threshold)
            else:
                plot_ranked_boxplot(ax, ranked, dataset_label, y_label, args.alpha, args.whis, labels, args.trend_p_threshold)

        assert output_path is not None
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=180)
        plt.close(fig)

    print(f"Sensory file: {file_by_screen.get('sensory', 'not used')}")
    print(f"Split file:   {file_by_screen.get('split', 'not used')}")
    print(f"Screen:       {args.screen}")
    print(f"Metric:       {metric_col}")
    print(f"Plot kind:    {args.plot_kind}")
    print(f"Split labels: {args.split_label_mode}")
    print(f"Sensory labels: {args.sensory_label_mode}")
    if args.sensory_label_mode == "annotated":
        print(f"Sensory stocks: {resolve_sensory_stocks_csv(args.sensory_stocks_csv) or 'not found'}")
    print(f"Trend p<:     {args.trend_p_threshold}")
    print(f"Control pos:  {args.control_position}")
    print(f"Centered:     {args.center_to_control}")
    print(f"Value scale:  {args.value_scale}")
    if args.plot_kind in {"box", "box-points"}:
        print(f"Whisker IQR:  {args.whis:.2f}")
    if args.plot_kind == "bar":
        print(f"CI level:     {args.ci_level:.1f}%")
    if output_path is not None:
        print(f"Output:       {output_path}")
    else:
        print("Output:       [skipped main plot]")
    if not HAS_SCIPY:
        print("SciPy not found: significance highlighting disabled.")

    if args.composite_residuals or args.composite_heatmap:
        if args.screen == "both":
            raise SystemExit(
                "--composite-residuals/--composite-heatmap require --screen sensory or --screen split (single screen)."
            )
        screen = screen_order[0]
        df_comp = df_by_screen[screen]
        residual_metrics = [m for m in ("total_holes_residual", "avg_max_size_residual", "avg_lifetime_residual") if m in df_comp.columns]
        if len(residual_metrics) < 2:
            raise SystemExit("Composite residual plot requires at least two residual metric columns.")
        sort_metric_col = resolve_residual_metric(args.composite_sort_metric)
        if sort_metric_col != "composite" and sort_metric_col not in df_comp.columns:
            raise SystemExit(
                f"Composite sort metric '{sort_metric_col}' not found. Available residuals: {', '.join(residual_metrics)}, composite"
            )
        if sort_metric_col != "composite" and sort_metric_col not in residual_metrics:
            residual_metrics = [sort_metric_col] + residual_metrics

        work = df_comp[[args.condition_column] + residual_metrics].copy()
        work[args.condition_column] = work[args.condition_column].astype(str)
        common_conditions = set(work[args.condition_column].unique())
        for mcol in residual_metrics:
            counts = work.dropna(subset=[mcol]).groupby(args.condition_column).size()
            keep = set(counts[counts >= args.min_n].index.astype(str))
            common_conditions &= keep
        if not common_conditions:
            raise SystemExit("No conditions pass min_n across all composite residual metrics.")

        metric_ranked_for_sort: list[tuple[str, dict]] = []
        for mcol in residual_metrics:
            r = build_ranked_data(
                df_comp,
                metric_col=mcol,
                condition_col=args.condition_column,
                control_label=args.control_label,
                min_n=args.min_n,
                alpha=args.alpha,
                control_position=args.control_position,
                center_to_control=args.center_to_control,
                value_scale=args.value_scale,
                allowed_conditions=common_conditions,
            )
            metric_ranked_for_sort.append((mcol, r))

        if sort_metric_col == "composite":
            conds_sorted = sorted(common_conditions)
            score_by_cond = {
                cond: float(np.mean([r["effect_by_cond"].get(cond, np.nan) for _, r in metric_ranked_for_sort]))
                for cond in conds_sorted
            }
            score_by_cond = {k: v for k, v in score_by_cond.items() if np.isfinite(v)}
            if not score_by_cond:
                raise SystemExit("Could not compute composite ordering scores.")
            sig_union = set()
            for _, r in metric_ranked_for_sort:
                sig_union.update(r["sig_conditions"])
            sorted_conds = sorted(score_by_cond, key=lambda c: score_by_cond[c])
            control_name = metric_ranked_for_sort[0][1]["control_name"]
            if args.control_position == "first":
                others = [c for c in sorted_conds if c.lower() != args.control_label.lower()]
                shared_order = [control_name] + others
            else:
                sig_down = [c for c in sorted_conds if c in sig_union and score_by_cond[c] < 0]
                nonsig = [c for c in sorted_conds if c not in sig_union]
                sig_up = [c for c in sorted_conds if c in sig_union and score_by_cond[c] >= 0]
                shared_order = sig_down + nonsig + sig_up
        else:
            ranked_sort = next((r for m, r in metric_ranked_for_sort if m == sort_metric_col), None)
            if ranked_sort is None:
                ranked_sort = build_ranked_data(
                    df_comp,
                    metric_col=sort_metric_col,
                    condition_col=args.condition_column,
                    control_label=args.control_label,
                    min_n=args.min_n,
                    alpha=args.alpha,
                    control_position=args.control_position,
                    center_to_control=args.center_to_control,
                    value_scale=args.value_scale,
                    allowed_conditions=common_conditions,
                )
            shared_order = ranked_sort["order"]

        ranked_metrics: list[tuple[str, dict]] = []
        for mcol in residual_metrics:
            ranked_m = build_ranked_data(
                df_comp,
                metric_col=mcol,
                condition_col=args.condition_column,
                control_label=args.control_label,
                min_n=args.min_n,
                alpha=args.alpha,
                control_position=args.control_position,
                center_to_control=args.center_to_control,
                value_scale=args.value_scale,
                allowed_conditions=common_conditions,
                fixed_order=shared_order,
            )
            ranked_metrics.append((mcol, ranked_m))

        sort_slug = "composite" if sort_metric_col == "composite" else sort_metric_col.replace("_residual", "")
        screen_slug = screen
        panel_title = "Sensory inactivation screen" if screen == "sensory" else "Split-GAL4 inactivation screen"

        if args.composite_residuals:
            comp_fig_width = max(MIN_PANEL_WIDTH, len(shared_order) * WIDTH_PER_CONDITION)
            comp_fig_height = 3.15 * len(ranked_metrics)
            fig_c, axes_c = plt.subplots(len(ranked_metrics), 1, figsize=(comp_fig_width, comp_fig_height), constrained_layout=True, sharex=True)
            if len(ranked_metrics) == 1:
                axes_c = [axes_c]
            fig_c.patch.set_facecolor("white")
            for i, ((mcol, ranked_m), axc) in enumerate(zip(ranked_metrics, axes_c)):
                axc.set_facecolor("white")
                y_label = y_axis_label(mcol, args.value_scale)
                subplot_title = metric_pretty_name(mcol)
                if screen == "split":
                    labels = [display_label(cond, split_annotation_map) for cond in ranked_m["order"]]
                elif screen == "sensory":
                    labels = [display_label(cond, sensory_annotation_map, annotation_only=True) for cond in ranked_m["order"]]
                else:
                    labels = ranked_m["order"]
                if args.plot_kind == "bar":
                    plot_ranked_barplot(axc, ranked_m, subplot_title, y_label, args.alpha, args.ci_level, labels, args.trend_p_threshold)
                elif args.plot_kind == "points-ci":
                    plot_ranked_points_ci(axc, ranked_m, subplot_title, y_label, args.alpha, args.ci_level, labels, args.trend_p_threshold)
                elif args.plot_kind == "box-points":
                    plot_ranked_box_points(axc, ranked_m, subplot_title, y_label, args.alpha, args.whis, labels, args.trend_p_threshold)
                else:
                    plot_ranked_boxplot(axc, ranked_m, subplot_title, y_label, args.alpha, args.whis, labels, args.trend_p_threshold)
                # Move title into the plotting area so stacked subplots fit tighter.
                axc.set_title("")
                axc.text(0.5, 0.985, subplot_title, transform=axc.transAxes, ha="center", va="top", fontsize=12)
                if i < len(ranked_metrics) - 1:
                    axc.tick_params(axis="x", labelbottom=False)
                    axc.set_xlabel("")
            if args.composite_output:
                comp_output = args.composite_output.resolve()
            else:
                kind_slug = args.plot_kind.replace("-", "_")
                comp_output = (DEFAULT_OUT_DIR / f"ranked_{kind_slug}plots_residuals_composite_{screen_slug}_sort-{sort_slug}.png").resolve()
            comp_output.parent.mkdir(parents=True, exist_ok=True)
            fig_c.savefig(comp_output, dpi=180)
            plt.close(fig_c)
            print(f"Composite sort: {sort_metric_col}")
            print(f"Composite out:  {comp_output}")

        if args.composite_heatmap:
            row_names = [metric_pretty_name(mcol) for mcol, _ in ranked_metrics]
            mat = np.array(
                [[ranked_m["effect_by_cond"].get(cond, np.nan) for cond in shared_order] for _, ranked_m in ranked_metrics],
                dtype=float,
            )
            mat_plot = np.clip(mat, -1.5, 1.5)
            cmap = residual_heatmap_cmap()
            hm_width = max(MIN_PANEL_WIDTH, len(shared_order) * WIDTH_PER_CONDITION)
            hm_height = max(2.2, 0.48 * len(ranked_metrics) + 0.8)

            fig_h, ax_h = plt.subplots(
                figsize=(hm_width, hm_height),
                constrained_layout=True,
            )
            fig_h.patch.set_facecolor("white")
            ax_h.set_facecolor("white")
            im = ax_h.imshow(mat_plot, aspect="auto", cmap=cmap, vmin=-1.5, vmax=1.5, interpolation="nearest")
            ax_h.set_xticks(np.arange(len(shared_order)))
            if screen == "split":
                hm_labels = [display_label(cond, split_annotation_map) for cond in shared_order]
            elif screen == "sensory":
                hm_labels = [display_label(cond, sensory_annotation_map, annotation_only=True) for cond in shared_order]
            else:
                hm_labels = shared_order
            ax_h.set_xticklabels(hm_labels, rotation=62, ha="right", fontsize=7)
            ax_h.set_yticks(np.arange(len(row_names)))
            ax_h.set_yticklabels(row_names, fontsize=11)
            ax_h.set_title(f"{panel_title}: residual metrics heatmap", fontsize=14)
            cbar = fig_h.colorbar(im, ax=ax_h, fraction=0.03, pad=0.015)
            cbar.set_label("Relative Effect", fontsize=10)
            cbar.set_ticks([-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5])
            if args.composite_heatmap_output:
                hm_output = args.composite_heatmap_output.resolve()
            else:
                hm_output = (
                    DEFAULT_OUT_DIR
                    / f"ranked_heatmap_residuals_composite_{screen_slug}_sort-{sort_slug}.png"
                ).resolve()
            hm_output.parent.mkdir(parents=True, exist_ok=True)
            fig_h.savefig(hm_output, dpi=180)
            plt.close(fig_h)
            print(f"Heatmap sort:   {sort_metric_col}")
            print(f"Heatmap output: {hm_output}")

    if args.per_hit_composites:
        if args.screen == "both":
            raise SystemExit("--per-hit-composites requires --screen sensory or --screen split (single screen).")
        screen = screen_order[0]
        metric_labels = {
            resolve_metric_column(df_by_screen[screen], "total_holes", args.use_residual): "Number of Holes",
            resolve_metric_column(df_by_screen[screen], "avg_max_size", args.use_residual): "Hole Size",
            resolve_metric_column(df_by_screen[screen], "avg_lifetime", args.use_residual): "Lifetime",
        }
        n_written, out_path = write_per_hit_composites(
            df=df_by_screen[screen],
            screen=screen,
            args=args,
            condition_col=args.condition_column,
            control_label=args.control_label,
            metric_labels=metric_labels,
        )
        print(f"Per-hit source: {resolve_metric_column(df_by_screen[screen], args.per_hit_source_metric, args.use_residual)}")
        print(f"Per-hit count:  {n_written}")
        print(f"Per-hit out:    {out_path}")


if __name__ == "__main__":
    main()
