#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
PLOT_SCRIPT = ROOT / "scripts" / "plugcamera" / "plot_ranked_metric_boxplots.py"
DEFAULT_OUT_DIR = ROOT / "data" / "plugcamera_data" / "tables"
DEFAULT_DATA_DIR = ROOT / "data" / "plugcamera_data"


def load_plot_module():
    spec = importlib.util.spec_from_file_location("plot_ranked_metric_boxplots", PLOT_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load plotting module from {PLOT_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export significant residual hits for split and mechano-split screens."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help=f"Directory containing plugcamera CSVs (default: {DEFAULT_DATA_DIR})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help=f"Directory for exported tables (default: {DEFAULT_OUT_DIR})",
    )
    parser.add_argument(
        "--screen",
        choices=("split", "mechano-split", "both"),
        default="both",
        help="Which screen to export (default: both).",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="Significance threshold (default: 0.05).",
    )
    parser.add_argument(
        "--min-n",
        type=int,
        default=3,
        help="Minimum non-NaN samples per condition to include (default: 3).",
    )
    parser.add_argument(
        "--control-label",
        default="control",
        help="Control label in the condition column (default: control).",
    )
    return parser.parse_args()


def format_mean_sd(values: np.ndarray | None) -> str:
    if values is None or values.size == 0:
        return ""
    mean = float(np.mean(values))
    sd = float(np.std(values, ddof=1)) if values.size > 1 else 0.0
    return f"{mean:.3f} +/- {sd:.3f}"


def summarize_screen(
    plotmod,
    data_dir: Path,
    screen: str,
    alpha: float,
    min_n: int,
    control_label: str,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, str]]:
    csv_path = plotmod.discover_dataset_file(data_dir, "split" if screen == "split" else "mechano-split")
    df = pd.read_csv(csv_path)
    condition_col = "condition_norm"
    if condition_col in df.columns:
        df[condition_col] = df[condition_col].astype(str).str.strip().str.upper()

    if screen == "split":
        annotation_map = plotmod.load_split_annotation_map()
    else:
        annotation_map, _ = plotmod.load_mechano_annotation_maps()

    metric_specs = [
        ("total_holes", "Hole Count"),
        ("avg_max_size", "Hole Size"),
        ("avg_lifetime", "Lifetime"),
    ]
    metric_cols = {
        pretty: plotmod.resolve_metric_column(df, metric, use_residual=True)
        for metric, pretty in metric_specs
    }

    ranked_by_metric: dict[str, dict] = {}
    sig_union: set[str] = set()
    for pretty, metric_col in metric_cols.items():
        ranked = plotmod.build_ranked_data(
            df=df,
            metric_col=metric_col,
            condition_col=condition_col,
            control_label=control_label,
            min_n=min_n,
            alpha=alpha,
            control_position="ranked",
            center_to_control=False,
            value_scale="control-sd",
        )
        ranked_by_metric[pretty] = ranked
        sig_union.update(ranked["sig_conditions"])

    source_ranked = ranked_by_metric["Hole Count"]
    hits = [cond for cond in source_ranked["order"] if cond in sig_union and cond != source_ranked["control_name"]]

    rows: list[dict[str, object]] = []
    for cond in hits:
        neuron_name = annotation_map.get(cond.upper(), "")
        sig_metrics = [pretty for pretty, ranked in ranked_by_metric.items() if cond in ranked["sig_conditions"]]
        row: dict[str, object] = {
            "screen": screen,
            "ss_id": cond,
            "neuron_name": neuron_name,
            "significant_metrics": "; ".join(sig_metrics),
        }
        for pretty, metric_col in metric_cols.items():
            ranked = ranked_by_metric[pretty]
            vals = ranked["values_by_cond"].get(cond)
            summary = ranked["summary"].set_index("condition")
            row[f"{pretty.lower().replace(' ', '_')}_source_column"] = metric_col
            row[f"{pretty.lower().replace(' ', '_')}_n"] = int(vals.size) if vals is not None else 0
            row[f"{pretty.lower().replace(' ', '_')}_mean_relative_effect"] = float(np.mean(vals)) if vals is not None else np.nan
            row[f"{pretty.lower().replace(' ', '_')}_sd_relative_effect"] = (
                float(np.std(vals, ddof=1)) if vals is not None and vals.size > 1 else 0.0 if vals is not None else np.nan
            )
            row[f"{pretty.lower().replace(' ', '_')}_median_relative_effect"] = (
                float(np.median(vals)) if vals is not None else np.nan
            )
            row[f"{pretty.lower().replace(' ', '_')}_p_value"] = float(ranked["p_by_cond"].get(cond, np.nan))
            row[f"{pretty.lower().replace(' ', '_')}_is_significant"] = cond in ranked["sig_conditions"]
            row[f"{pretty.lower().replace(' ', '_')}_display"] = format_mean_sd(vals)
            if cond in summary.index:
                row[f"{pretty.lower().replace(' ', '_')}_effect_vs_control"] = float(
                    summary.loc[cond, "effect_vs_control"]
                )
            else:
                row[f"{pretty.lower().replace(' ', '_')}_effect_vs_control"] = np.nan
        rows.append(row)

    export_df = pd.DataFrame(rows)
    display_df = export_df[
        [
            "screen",
            "ss_id",
            "neuron_name",
            "significant_metrics",
            "hole_count_display",
            "hole_size_display",
            "lifetime_display",
        ]
    ].rename(
        columns={
            "screen": "Screen",
            "ss_id": "SS ID",
            "neuron_name": "Neuron Name",
            "significant_metrics": "Significant Metrics",
            "hole_count_display": "Hole Count (mean +/- SD)",
            "hole_size_display": "Hole Size (mean +/- SD)",
            "lifetime_display": "Lifetime (mean +/- SD)",
        }
    )

    metadata = {
        "screen": screen,
        "input_csv": str(csv_path),
        "condition_column": condition_col,
        "alpha": str(alpha),
        "min_n": str(min_n),
        "value_scale": "control-sd",
        "source_columns": ", ".join(metric_cols.values()),
    }
    return export_df, display_df, metadata


def write_markdown(path: Path, display_df: pd.DataFrame, metadata: dict[str, str]) -> None:
    def escape_cell(value: object) -> str:
        text = "" if pd.isna(value) else str(value)
        return text.replace("|", "\\|").replace("\n", " ")

    lines = [
        f"# Significant Hits: {metadata['screen']}",
        "",
        f"- Input CSV: `{metadata['input_csv']}`",
        f"- Significance: Mann-Whitney U p < {metadata['alpha']}",
        f"- Minimum n per condition: {metadata['min_n']}",
        f"- Reported values: mean +/- SD of plotted relative-effect values",
        f"- Relative-effect source columns: `{metadata['source_columns']}`",
        "",
    ]
    headers = [escape_cell(col) for col in display_df.columns]
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for _, row in display_df.iterrows():
        cells = [escape_cell(row[col]) for col in display_df.columns]
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    path.write_text("\n".join(lines))


def main() -> None:
    args = parse_args()
    plotmod = load_plot_module()
    data_dir = args.data_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    screens = ["split", "mechano-split"] if args.screen == "both" else [args.screen]
    for screen in screens:
        export_df, display_df, metadata = summarize_screen(
            plotmod=plotmod,
            data_dir=data_dir,
            screen=screen,
            alpha=args.alpha,
            min_n=args.min_n,
            control_label=args.control_label,
        )
        stem = f"significant_hits_{screen}_residual_relative_effect"
        csv_path = output_dir / f"{stem}.csv"
        md_path = output_dir / f"{stem}.md"
        export_df.to_csv(csv_path, index=False)
        write_markdown(md_path, display_df, metadata)
        print(f"{screen}: {len(export_df)} hits")
        print(f"CSV: {csv_path}")
        print(f"MD:  {md_path}")


if __name__ == "__main__":
    main()
