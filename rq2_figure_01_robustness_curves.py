#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Plot RQ2 Figure 1: robustness versus extremity level.

The figure contains four panels:
    (a) Hard Constraint Violation Rate
    (b) Soft Constraint Violation Rate
    (c) Daily Cost
    (d) Min-FCSD

Input data:
    rq2/results/{algorithm}_extremity_{level}_s{seed}.json

The plotted values are the mean across seeds.
No error bars are drawn.

Default outputs:
    figures/rq2_figure_01_robustness_curves.png
    figures/rq2_figure_01_robustness_curves.pdf
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import FormatStrFormatter

ALGORITHMS = ["hfg_sac", "sac", "ppo_lagrangian", "cpo", "safety_layer"]
SEEDS = [42, 123, 456, 789, 2024]
EXTREMITY_LEVELS = [0.8, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]

ALGORITHM_LABELS = {
    "hfg_sac": "HFG-SAC",
    "sac": "SAC",
    "ppo_lagrangian": "PPO-Lagrangian",
    "cpo": "CPO",
    "safety_layer": "Safety Layer",
}

STYLES = {
    "hfg_sac": {
        "color": "#d62728",
        "linestyle": "-",
        "marker": "o",
        "linewidth": 2.3,
        "markersize": 4.9,
        "markerfacecolor": "#d62728",
    },
    "sac": {
        "color": "#1f77b4",
        "linestyle": "--",
        "marker": "s",
        "linewidth": 2.0,
        "markersize": 4.7,
        "markerfacecolor": "none",
    },
    "ppo_lagrangian": {
        "color": "#9467bd",
        "linestyle": "-.",
        "marker": "^",
        "linewidth": 2.0,
        "markersize": 4.8,
        "markerfacecolor": "#9467bd",
    },
    "cpo": {
        "color": "#8c564b",
        "linestyle": "-.",
        "marker": "D",
        "linewidth": 2.0,
        "markersize": 4.7,
        "markerfacecolor": "#8c564b",
    },
    "safety_layer": {
        "color": "#7f7f7f",
        "linestyle": ":",
        "marker": "v",
        "linewidth": 1.9,
        "markersize": 4.8,
        "markerfacecolor": "none",
    },
}

PLOT_ORDER = ["sac", "safety_layer", "ppo_lagrangian", "cpo", "hfg_sac"]
LEGEND_ORDER = ["hfg_sac", "sac", "ppo_lagrangian", "cpo", "safety_layer"]

METRICS = [
    {
        "key": "violation_rate",
        "ylabel": "Violation Rate (%)",
        "ylim": (0, 105),
        "yticks": [0, 20, 40, 60, 80, 100],
        "format": None,
    },
    {
        "key": "soft_violation_rate",
        "ylabel": "Violation Rate (%)",
        "ylim": (0, 105),
        "yticks": [0, 20, 40, 60, 80, 100],
        "format": None,
    },
    {
        "key": "cost_per_day",
        "ylabel": "Cost (RMB/day)",
        "ylim": (7000, 18700),
        "yticks": [8000, 10000, 12000, 14000, 16000, 18000],
        "format": "%.0f",
    },
    {
        "key": "min_fcsd",
        "ylabel": "Min-FCSD",
        "ylim": (-0.05, 1.05),
        "yticks": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        "format": "%.1f",
    },
]

PANEL_CAPTIONS = [
    "(a) Hard Constraint Violation Rate",
    "(b) Soft Constraint Violation Rate",
    "(c) Daily Cost",
    "(d) Min-FCSD",
]

# 维持类似 RQ1 的横向画幅
FIG_WIDTH = 12.1
FIG_HEIGHT = 6.9

# 与 RQ1 Fig.1 保持一致的 x 轴标签与子图说明间距
CAPTION_GAP_PT = 4.73


def configure_plot_style() -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size": 11,
        "axes.labelsize": 16,
        "xtick.labelsize": 13,
        "ytick.labelsize": 13,
        "legend.fontsize": 11.5,
        "axes.linewidth": 1.1,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.major.width": 0.9,
        "ytick.major.width": 0.9,
        "xtick.major.size": 3.0,
        "ytick.major.size": 3.0,
        "legend.frameon": True,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
        "text.usetex": False,
        "mathtext.fontset": "stix",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def find_results_dir(explicit_dir: Path | None) -> tuple[Path, Path]:
    if explicit_dir is not None:
        results_dir = explicit_dir.expanduser().resolve()
        if not results_dir.is_dir():
            raise FileNotFoundError(f"RQ2 results directory does not exist: {results_dir}")
        if results_dir.name == "results" and results_dir.parent.name == "rq2":
            return results_dir, results_dir.parent.parent
        return results_dir, results_dir.parent

    script_dir = Path(__file__).resolve().parent
    current_dir = Path.cwd().resolve()
    search_bases = [script_dir, *script_dir.parents, current_dir, *current_dir.parents]

    checked = set()
    for base in search_bases:
        if base in checked:
            continue
        checked.add(base)
        for candidate in [base / "rq2" / "results", base / "outputs" / "rq2" / "results"]:
            if candidate.is_dir():
                return candidate, base

    raise FileNotFoundError(
        "Cannot locate the RQ2 results directory. Expected either:\n"
        "  <project_root>/rq2/results\n"
        "  <project_root>/outputs/rq2/results\n"
        "Please pass --results-dir explicitly if your directory structure differs."
    )


def load_one_result(results_dir: Path, algorithm: str, extremity: float, seed: int) -> dict:
    filename = f"{algorithm}_extremity_{extremity:.1f}_s{seed}.json"
    file_path = results_dir / filename
    if not file_path.is_file():
        raise FileNotFoundError(f"Missing RQ2 result file: {file_path}")

    with file_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_all_metrics(results_dir: Path) -> dict:
    raw = {}
    loaded_files = 0

    for algorithm in ALGORITHMS:
        for extremity in EXTREMITY_LEVELS:
            for metric in METRICS:
                raw[(algorithm, extremity, metric["key"])] = []

            for seed in SEEDS:
                result = load_one_result(results_dir, algorithm, extremity, seed)
                loaded_files += 1

                for metric in METRICS:
                    key = metric["key"]
                    if key not in result:
                        raise KeyError(
                            f"'{key}' is missing in {algorithm}_extremity_{extremity:.1f}_s{seed}.json"
                        )
                    value = float(result[key])
                    if not np.isfinite(value):
                        raise ValueError(
                            f"Non-finite value for '{key}' in "
                            f"{algorithm}_extremity_{extremity:.1f}_s{seed}.json"
                        )
                    raw[(algorithm, extremity, key)].append(value)

    print(f"[Loaded] {loaded_files} JSON files")
    return raw


def summarize_metrics(raw: dict) -> dict:
    summary = {}
    for algorithm in ALGORITHMS:
        for metric in METRICS:
            key = metric["key"]
            means = []
            for extremity in EXTREMITY_LEVELS:
                arr = np.asarray(raw[(algorithm, extremity, key)], dtype=float)
                means.append(float(np.mean(arr)))
            summary[(algorithm, key)] = np.asarray(means, dtype=float)
    return summary


def make_legend_handles() -> list[Line2D]:
    handles = []
    for algorithm in LEGEND_ORDER:
        style = STYLES[algorithm]
        handles.append(
            Line2D(
                [0], [0],
                color=style["color"],
                linestyle=style["linestyle"],
                linewidth=style["linewidth"],
                marker=style["marker"],
                markersize=style["markersize"],
                markerfacecolor=style["markerfacecolor"],
                markeredgecolor=style["color"],
                markeredgewidth=0.8,
                label=ALGORITHM_LABELS[algorithm],
            )
        )
    return handles


def style_axis(ax, metric: dict, show_xlabel: bool = True) -> None:
    ax.set_xlim(0.76, 1.54)
    ax.set_xticks(np.arange(0.8, 1.51, 0.1))
    ax.set_ylim(*metric["ylim"])
    ax.set_yticks(metric["yticks"])

    ax.set_ylabel(metric["ylabel"], fontsize=16, labelpad=6)
    ax.set_xlabel(r"Extremity Multiplier $\xi$" if show_xlabel else "", fontsize=16, labelpad=4)

    if metric["format"] is not None:
        ax.yaxis.set_major_formatter(FormatStrFormatter(metric["format"]))

    ax.grid(
        which="major",
        axis="both",
        linestyle="-",
        linewidth=0.45,
        color="#D9D9D9",
        alpha=0.48,
        zorder=0,
    )
    ax.set_axisbelow(True)

    for spine in ax.spines.values():
        spine.set_linewidth(1.1)
        spine.set_color("black")

    ax.tick_params(
        axis="both",
        which="major",
        direction="in",
        width=0.9,
        length=3.0,
        top=True,
        right=True,
        pad=3,
    )


def add_panel_caption(fig, ax, text: str, renderer, target_gap_fig: float) -> None:
    pos = ax.get_position()
    x_center = (pos.x0 + pos.x1) / 2

    xlabel_bbox = ax.xaxis.label.get_window_extent(renderer=renderer)
    xlabel_bottom_fig = fig.transFigure.inverted().transform((0, xlabel_bbox.y0))[1]
    y_text = xlabel_bottom_fig - target_gap_fig

    fig.text(
        x_center,
        y_text,
        text,
        ha="center",
        va="top",
        fontsize=16,
    )


def plot_metric(ax, summary: dict, metric: dict, show_legend: bool = False) -> None:
    x = np.asarray(EXTREMITY_LEVELS, dtype=float)

    for algorithm in PLOT_ORDER:
        mean = summary[(algorithm, metric["key"])]
        style = STYLES[algorithm]

        ax.plot(
            x, mean,
            color=style["color"],
            linestyle=style["linestyle"],
            marker=style["marker"],
            linewidth=style["linewidth"],
            markersize=style["markersize"],
            markerfacecolor=style["markerfacecolor"],
            markeredgecolor=style["color"],
            markeredgewidth=0.85,
            zorder=7 if algorithm == "hfg_sac" else 4,
        )

    style_axis(ax, metric, show_xlabel=True)

    if show_legend:
        legend = ax.legend(
            handles=make_legend_handles(),
            loc="center",
            bbox_to_anchor=(0.49, 0.50),
            ncol=2,
            frameon=True,
            fancybox=False,
            framealpha=0.96,
            edgecolor="black",
            facecolor="white",
            borderpad=0.30,
            labelspacing=0.25,
            columnspacing=0.72,
            handlelength=1.35,
            handletextpad=0.36,
            fontsize=11.5,
        )
        legend.get_frame().set_linewidth(0.8)


def plot_rq2_figure_01(summary: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(FIG_WIDTH, FIG_HEIGHT))

    plot_metric(axes[0, 0], summary, METRICS[0], show_legend=True)
    plot_metric(axes[0, 1], summary, METRICS[1], show_legend=False)
    plot_metric(axes[1, 0], summary, METRICS[2], show_legend=False)
    plot_metric(axes[1, 1], summary, METRICS[3], show_legend=False)

    fig.subplots_adjust(
        left=0.118,
        right=0.986,
        bottom=0.18,
        top=0.975,
        wspace=0.22,
        hspace=0.37,
    )

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    target_gap_fig = (CAPTION_GAP_PT / 72.0) / fig.get_figheight()

    for ax, caption in zip(axes.flat, PANEL_CAPTIONS):
        add_panel_caption(fig, ax, caption, renderer, target_gap_fig)

    # 保持你现在的输出文件名不变
    png_path = output_dir / "rq2_figure_01_robustness_curves.png"
    pdf_path = output_dir / "rq2_figure_01_robustness_curves.pdf"

    fig.savefig(png_path, dpi=300, bbox_inches="tight", pad_inches=0.05)
    fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.05)

    plt.close(fig)

    print("\n[Saved]")
    print(f"PNG: {png_path}")
    print(f"PDF: {pdf_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot RQ2 Figure 1: robustness versus extremity level."
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=None,
        help="RQ2 JSON directory. Auto-detects rq2/results or outputs/rq2/results."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. Default: <project_root>/figures."
    )
    args = parser.parse_args()

    results_dir, project_root = find_results_dir(args.results_dir)
    output_dir = project_root / "figures" if args.output_dir is None else args.output_dir.expanduser().resolve()

    configure_plot_style()

    print(f"Project root : {project_root}")
    print(f"Results dir  : {results_dir}")
    print(f"Output dir   : {output_dir}")
    print("Layout       : RQ1-matched caption spacing below all panels; x-label shown on all panels; legend only in panel (a)")
    print("Statistics   : mean across seeds; no error bars")

    raw = load_all_metrics(results_dir)
    summary = summarize_metrics(raw)
    plot_rq2_figure_01(summary, output_dir)


if __name__ == "__main__":
    main()