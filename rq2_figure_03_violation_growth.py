#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
RQ2 Figure 03 — Degradation Curve

Show how hard-constraint violation rate changes as the extremity multiplier
xi increases, using five random seeds for each algorithm and extremity level.

Input files:
    {algorithm}_extremity_{xi}_s{seed}.json

Required JSON field:
    violation_rate

Default input directory:
    outputs/rq2/results/  (or rq2/results/)

Output files:
    rq2_figure_03_degradation_curve.png
    rq2_figure_03_degradation_curve.pdf

Statistics:
    Mean +/- sample standard deviation across seeds (shaded band).
    Slope annotations are ordinary least-squares slopes fitted to the seven
    mean violation-rate points, in percentage points per unit xi.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


# =====================================================================
# 1. Experiment configuration
# =====================================================================

ALGORITHMS = [
    "hfg_sac",
    "sac",
    "ppo_lagrangian",
    "cpo",
    "safety_layer",
]

SEEDS = [42, 123, 456, 789, 2024]

EXTREMITY_LEVELS = [0.8, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]

# "ours" is intentionally removed from the legend and annotations.
ALGORITHM_LABELS = {
    "hfg_sac": "HFG-SAC",
    "sac": "SAC",
    "ppo_lagrangian": "PPO-Lagrangian",
    "cpo": "CPO",
    "safety_layer": "Safety Layer",
}

# Colors, line styles, and markers follow the established RQ1/RQ2 palette.
STYLES = {
    "hfg_sac": {
        "color": "#d62728",
        "linestyle": "-",
        "marker": "o",
        "linewidth": 3.1,
        "markersize": 8.0,
        "zorder": 7,
    },
    "sac": {
        "color": "#1f77b4",
        "linestyle": "--",
        "marker": "s",
        "linewidth": 2.5,
        "markersize": 7.6,
        "zorder": 6,
    },
    "ppo_lagrangian": {
        "color": "#9467bd",
        "linestyle": "-.",
        "marker": "^",
        "linewidth": 2.5,
        "markersize": 8.0,
        "zorder": 5,
    },
    "cpo": {
        "color": "#8c564b",
        "linestyle": "-.",
        "marker": "D",
        "linewidth": 2.5,
        "markersize": 7.6,
        "zorder": 4,
    },
    "safety_layer": {
        "color": "#7f7f7f",
        "linestyle": ":",
        "marker": "v",
        "linewidth": 2.5,
        "markersize": 7.8,
        "zorder": 3,
    },
}

# Shaded area: mean +/- sample standard deviation across the five seeds.
BAND_ALPHA = 0.10

# All five slope annotations are centered on the same vertical axis.
# Positions use axes coordinates, so they remain centered if axis limits change.
SLOPE_LABEL_X = 0.50
SLOPE_LABEL_Y = {
    "sac": 0.623,
    "cpo": 0.537,
    "safety_layer": 0.450,
    "hfg_sac": 0.364,
    "ppo_lagrangian": 0.277,
}

# Match the axes-box aspect ratio of each panel in RQ2 Figure 1 (about 2.04:1).
# Keep the existing width and reduce the height to obtain the same wide panel shape.
FIGSIZE = (10.6, 5.66)

OUTPUT_STEM = "rq2_figure_03_violation_growth"


# =====================================================================
# 2. Find project and results directories
# =====================================================================

def find_default_results_dir() -> tuple[Path, Path]:
    """Find <project>/outputs/rq2/results or <project>/rq2/results."""
    search_bases = [
        Path(__file__).resolve().parent,
        *Path(__file__).resolve().parents,
        Path.cwd(),
        *Path.cwd().parents,
    ]

    seen: set[Path] = set()

    for base in search_bases:
        base = base.resolve()
        if base in seen:
            continue
        seen.add(base)

        for relative in (
            Path("outputs/rq2/results"),
            Path("rq2/results"),
        ):
            candidate = base / relative
            if candidate.is_dir():
                return base, candidate

    raise FileNotFoundError(
        "Cannot find the RQ2 results directory. Expected either:\n"
        "  outputs/rq2/results/\n"
        "  rq2/results/\n"
        "Pass --results-dir to specify it manually."
    )


def infer_project_root(results_dir: Path) -> Path:
    """Infer the project root from a standard RQ2 results directory."""
    resolved = results_dir.resolve()

    if resolved.name == "results" and resolved.parent.name == "rq2":
        if resolved.parent.parent.name == "outputs":
            return resolved.parent.parent.parent
        return resolved.parent.parent

    return resolved.parent


# =====================================================================
# 3. Load data and calculate statistics
# =====================================================================

def load_one_run(
    results_dir: Path,
    algorithm: str,
    extremity: float,
    seed: int,
) -> float:
    """Load one seed's hard violation rate (already expressed in percent)."""
    xi_tag = f"{extremity:.1f}"
    filename = f"{algorithm}_extremity_{xi_tag}_s{seed}.json"
    file_path = results_dir / filename

    if not file_path.is_file():
        raise FileNotFoundError(f"Missing experiment file: {file_path}")

    with file_path.open("r", encoding="utf-8") as f:
        result = json.load(f)

    if "violation_rate" not in result:
        raise KeyError(
            f"{filename} does not contain 'violation_rate'. "
            f"Available fields: {list(result.keys())}"
        )

    value = float(result["violation_rate"])
    if not math.isfinite(value):
        raise ValueError(
            f"{filename}: violation_rate is not finite: {value}"
        )

    return value


def load_degradation_data(results_dir: Path) -> dict:
    """Return per-algorithm means and sample standard deviations."""
    summary = {}

    for algorithm in ALGORITHMS:
        seed_values_by_xi = []
        means = []
        stds = []

        for xi in EXTREMITY_LEVELS:
            seed_values = [
                load_one_run(results_dir, algorithm, xi, seed)
                for seed in SEEDS
            ]
            values = np.asarray(seed_values, dtype=float)

            mean = float(np.mean(values))
            std = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0

            seed_values_by_xi.append(values)
            means.append(mean)
            stds.append(std)

        means_array = np.asarray(means, dtype=float)
        stds_array = np.asarray(stds, dtype=float)

        # Linear least-squares slope across the seven xi-level means.
        slope, intercept = np.polyfit(
            np.asarray(EXTREMITY_LEVELS, dtype=float),
            means_array,
            deg=1,
        )

        # Avoid displaying a negative zero caused by floating-point precision.
        if abs(float(slope)) < 1e-10:
            slope = 0.0

        summary[algorithm] = {
            "seed_values": seed_values_by_xi,
            "mean": means_array,
            "std": stds_array,
            "slope": float(slope),
            "intercept": float(intercept),
        }

    return summary


# =====================================================================
# 4. Publication plot style — consistent with RQ1 and RQ2 Figure 2
# =====================================================================

def configure_plot_style() -> None:
    """Use the typography and axis styling established in the RQ figures."""
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size": 18,
        "axes.labelsize": 27,
        "axes.titlesize": 22,
        "xtick.labelsize": 21,
        "ytick.labelsize": 21,
        "legend.fontsize": 19,
        "axes.linewidth": 2.0,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.major.width": 1.0,
        "ytick.major.width": 1.0,
        "xtick.major.size": 3.5,
        "ytick.major.size": 3.5,
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


# =====================================================================
# 5. Draw the degradation curve
# =====================================================================

def plot_degradation_curve(summary: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    configure_plot_style()

    x = np.asarray(EXTREMITY_LEVELS, dtype=float)
    fig, ax = plt.subplots(figsize=FIGSIZE)

    # Draw variability bands first, so all mean curves remain visible.
    plot_order = [
        "cpo",
        "ppo_lagrangian",
        "safety_layer",
        "sac",
        "hfg_sac",
    ]

    for algorithm in plot_order:
        stats = summary[algorithm]
        style = STYLES[algorithm]

        ax.fill_between(
            x,
            stats["mean"] - stats["std"],
            stats["mean"] + stats["std"],
            color=style["color"],
            alpha=BAND_ALPHA,
            linewidth=0,
            zorder=1,
        )

    # Mean curves and markers.
    for algorithm in plot_order:
        stats = summary[algorithm]
        style = STYLES[algorithm]

        ax.plot(
            x,
            stats["mean"],
            color=style["color"],
            linestyle=style["linestyle"],
            linewidth=style["linewidth"],
            marker=style["marker"],
            markersize=style["markersize"],
            markerfacecolor=style["color"],
            markeredgecolor=style["color"],
            markeredgewidth=0.7,
            label=ALGORITHM_LABELS[algorithm],
            zorder=style["zorder"] + 2,
        )

    # No large title: the figure title belongs in the paper caption.
    ax.set_xlabel(
        "Extremity Multiplier ξ",
        labelpad=8,
        fontsize=27,
    )
    ax.set_ylabel(
        "Hard Violation Rate (%)",
        labelpad=8,
        fontsize=27,
    )

    # Tighter x-range: the final data point is at xi=1.5.
    # This keeps a small safety margin while reducing unused space on the right.
    ax.set_xlim(0.76, 1.54)
    ax.set_xticks(np.arange(0.8, 1.51, 0.1))

    # Reduce unused space below zero while retaining the lower confidence band.
    ax.set_ylim(-12, 103.5)
    ax.set_yticks([0, 20, 40, 60, 80, 100])

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
        spine.set_linewidth(2.0)
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

    # -------------------------------------------------------------
    # Legend: two rows, three columns, shifted slightly to the right.
    # "ours" is omitted from the HFG-SAC legend label.
    # -------------------------------------------------------------

    # Column-major order gives a balanced two-row legend:
    # top: HFG-SAC / PPO-Lagrangian / Safety Layer
    # bottom: SAC / CPO
    legend_order = [
        "hfg_sac",
        "sac",
        "ppo_lagrangian",
        "cpo",
        "safety_layer",
    ]

    legend_handles = []
    for algorithm in legend_order:
        style = STYLES[algorithm]
        legend_handles.append(
            Line2D(
                [0], [0],
                color=style["color"],
                linestyle=style["linestyle"],
                linewidth=style["linewidth"],
                marker=style["marker"],
                markersize=style["markersize"],
                markerfacecolor=style["color"],
                markeredgecolor=style["color"],
                markeredgewidth=0.7,
                label=ALGORITHM_LABELS[algorithm],
            )
        )

    legend = ax.legend(
        handles=legend_handles,
        # Place the legend in the empty upper band, safely below the SAC curve
        # and above all low-violation curves and slope annotations.
        loc="upper center",
        bbox_to_anchor=(0.49, 0.91),
        ncol=3,
        frameon=True,
        fancybox=False,
        framealpha=0.96,
        edgecolor="black",
        facecolor="white",
        borderpad=0.48,
        labelspacing=0.38,
        columnspacing=1.25,
        handlelength=1.85,
        handletextpad=0.50,
        fontsize=19,
    )
    legend.get_frame().set_linewidth(0.8)

    # -------------------------------------------------------------
    # Five slope annotations in one vertical column.
    # SAC is no longer isolated at the top of the blue curve.
    # -------------------------------------------------------------

    annotation_order = [
        "sac",
        "cpo",
        "safety_layer",
        "hfg_sac",
        "ppo_lagrangian",
    ]

    for algorithm in annotation_order:
        slope = summary[algorithm]["slope"]
        color = STYLES[algorithm]["color"]
        label = ALGORITHM_LABELS[algorithm]

        ax.text(
            SLOPE_LABEL_X,
            SLOPE_LABEL_Y[algorithm],
            f"{label}: slope={slope:+.1f}%/ξ",
            transform=ax.transAxes,
            color=color,
            fontsize=19,
            ha="center",
            va="center",
            clip_on=True,
            zorder=12,
        )

    # Keep the plotting panel large while preserving room for axis labels.
    # Use the same outer margins as the RQ2 Figure 1 reference script.
    fig.subplots_adjust(
        left=0.118,
        right=0.986,
        bottom=0.18,
        top=0.975,
    )

    png_path = output_dir / f"{OUTPUT_STEM}.png"
    pdf_path = output_dir / f"{OUTPUT_STEM}.pdf"

    fig.savefig(
        png_path,
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.05,
    )
    fig.savefig(
        pdf_path,
        bbox_inches="tight",
        pad_inches=0.05,
    )
    plt.close(fig)

    print("\nFigure saved:")
    print(f"  PNG: {png_path}")
    print(f"  PDF: {pdf_path}")


# =====================================================================
# 6. Main
# =====================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot the RQ2 hard-violation degradation curve."
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=None,
        help="RQ2 JSON results directory; auto-detected if omitted.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Figure output directory; defaults to <project-root>/figures.",
    )
    args = parser.parse_args()

    if args.results_dir is None:
        project_root, results_dir = find_default_results_dir()
    else:
        results_dir = args.results_dir.expanduser().resolve()
        if not results_dir.is_dir():
            raise FileNotFoundError(
                f"Results directory does not exist: {results_dir}"
            )
        project_root = infer_project_root(results_dir)

    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else project_root / "figures"
    )

    print("=" * 68)
    print("RQ2 Figure 03 — Degradation Curve")
    print("=" * 68)
    print(f"Project root : {project_root}")
    print(f"Results dir  : {results_dir}")
    print(f"Output dir   : {output_dir}")
    print(f"Extremity    : {EXTREMITY_LEVELS}")
    print(f"Seeds        : {SEEDS}")
    print("Shading      : Mean ± sample standard deviation")
    print("Slope        : OLS slope fitted to seven mean points")
    print("=" * 68)

    summary = load_degradation_data(results_dir)

    print("\nSummary (mean ± sample std):")
    for algorithm in ALGORITHMS:
        stats = summary[algorithm]
        print(f"\n{ALGORITHM_LABELS[algorithm]}:")
        for xi, mean, std in zip(
            EXTREMITY_LEVELS,
            stats["mean"],
            stats["std"],
        ):
            print(f"  ξ={xi:.1f}: {mean:7.3f} ± {std:7.3f}%")
        print(f"  OLS slope: {stats['slope']:+.3f}%/ξ")

    plot_degradation_curve(summary, output_dir)


if __name__ == "__main__":
    main()
