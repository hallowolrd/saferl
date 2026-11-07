
#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
RQ2 Figure 02 — Pareto View: Safety vs Cost

Input:
    outputs/rq2/results/

File naming:
    {algorithm}_extremity_{xi}_s{seed}.json

Output:
    figures/rq2_figure_02_pareto_safety_vs_cost.png
    figures/rq2_figure_02_pareto_safety_vs_cost.pdf

Statistics:
    Mean ± 95% CI (n=5 seeds)

X-axis:
    Daily cost (RMB/day)

Y-axis:
    Hard violation rate (%)
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# ============================================================
# 1. Experiment configuration
# ============================================================

ALGORITHMS = [
    "hfg_sac",
    "sac",
    "ppo_lagrangian",
    "cpo",
    "safety_layer",
]

SEEDS = [42, 123, 456, 789, 2024]

DEFAULT_EXTREMITY = 1.5

ALGORITHM_LABELS = {
    "hfg_sac": "HFG-SAC",
    "sac": "SAC",
    "ppo_lagrangian": "PPO-Lagrangian",
    "cpo": "CPO",
    "safety_layer": "Safety Layer",
}


# ============================================================
# 2. Algorithm styles
#    Colors and markers follow the previous RQ1 figures.
# ============================================================

STYLES = {
    "hfg_sac": {
        "color": "#d62728",
        "marker": "o",
        "markersize": 8.5,
        "zorder": 6,
    },

    "sac": {
        "color": "#1f77b4",
        "marker": "s",
        "markersize": 8.0,
        "zorder": 5,
    },

    "ppo_lagrangian": {
        "color": "#9467bd",
        "marker": "^",
        "markersize": 8.5,
        "zorder": 5,
    },

    "cpo": {
        "color": "#8c564b",
        "marker": "D",
        "markersize": 8.0,
        "zorder": 4,
    },

    "safety_layer": {
        "color": "#7f7f7f",
        "marker": "v",
        "markersize": 8.0,
        "zorder": 4,
    },
}


# Annotation positions: (horizontal offset, vertical offset, alignment)
# SAC is placed below-right; PPO-Lagrangian is placed above-left
# to keep both labels inside the axes.
LABEL_OFFSETS = {
    "hfg_sac": (12, 13, "left"),
    # Center SAC directly underneath its blue square marker.
    "sac": (0, -14, "center"),
    # Lift PPO-Lagrangian slightly to create more clearance below the marker.
    "ppo_lagrangian": (0, 21, "center"),
    "cpo": (12, 10, "left"),
    # Shift Safety Layer left slightly to avoid the HFG-SAC vertical error bar.
    "safety_layer": (4, 10, "left"),
}


# Hard violation safety threshold (%)
SAFETY_THRESHOLD = 5.0


# 95% two-sided Student-t critical value, n=5, df=4
T_CRITICAL_95 = 2.7764451051977987


# Figure size
FIG_WIDTH = 18.5 / 2.54
FIG_HEIGHT = 14.5 / 2.54


OUTPUT_STEM = "rq2_figure_02_pareto_safety_cost"


# ============================================================
# 3. Project path helpers
# ============================================================

def find_default_results_dir() -> tuple[Path, Path]:

    search_bases = [
        Path(__file__).resolve().parent,
        *Path(__file__).resolve().parents,
        Path.cwd(),
        *Path.cwd().parents,
    ]

    seen = set()

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
        "Cannot find the RQ2 results directory.\n"
        "Expected either:\n"
        "  outputs/rq2/results/\n"
        "  rq2/results/\n"
        "Please pass --results-dir explicitly if necessary."
    )


def infer_project_root(results_dir: Path) -> Path:

    resolved = results_dir.resolve()

    if resolved.name == "results" and resolved.parent.name == "rq2":

        if resolved.parent.parent.name == "outputs":
            return resolved.parent.parent.parent

        return resolved.parent.parent

    return resolved.parent


# ============================================================
# 4. Data loading
# ============================================================

def load_one_run(
    results_dir: Path,
    algorithm: str,
    extremity_tag: str,
    seed: int,
) -> dict:

    file_path = (
        results_dir
        / f"{algorithm}_extremity_{extremity_tag}_s{seed}.json"
    )

    if not file_path.is_file():
        raise FileNotFoundError(
            f"Missing result file: {file_path}"
        )

    with file_path.open("r", encoding="utf-8") as f:
        result = json.load(f)

    required_fields = (
        "cost_per_day",
        "violation_rate",
    )

    missing = [
        field
        for field in required_fields
        if field not in result
    ]

    if missing:
        raise KeyError(
            f"{file_path.name} is missing fields {missing}."
        )

    values = {}

    for field in required_fields:

        value = float(result[field])

        if not math.isfinite(value):
            raise ValueError(
                f"{file_path.name}: {field} is not finite."
            )

        values[field] = value

    return values


# ============================================================
# 5. Statistics
#    Mean ± 95% confidence interval
# ============================================================

def mean_and_95ci(
    values: list[float],
) -> tuple[float, float]:

    arr = np.asarray(values, dtype=float)

    if arr.size != len(SEEDS):
        raise ValueError(
            f"Expected {len(SEEDS)} seeds, "
            f"but received {arr.size}."
        )

    mean = float(np.mean(arr))

    sample_std = float(
        np.std(arr, ddof=1)
    )

    ci_half_width = (
        T_CRITICAL_95
        * sample_std
        / math.sqrt(arr.size)
    )

    return mean, ci_half_width


def load_summary(
    results_dir: Path,
    extremity_tag: str,
) -> dict:

    summary = {}

    for algorithm in ALGORITHMS:

        runs = [
            load_one_run(
                results_dir,
                algorithm,
                extremity_tag,
                seed,
            )
            for seed in SEEDS
        ]

        costs = [
            run["cost_per_day"]
            for run in runs
        ]

        violations = [
            run["violation_rate"]
            for run in runs
        ]

        cost_mean, cost_ci = mean_and_95ci(costs)

        violation_mean, violation_ci = mean_and_95ci(
            violations
        )

        summary[algorithm] = {
            "cost_mean": cost_mean,
            "cost_ci": cost_ci,
            "violation_mean": violation_mean,
            "violation_ci": violation_ci,
        }

        print(
            f"[Summary] {algorithm:16s} | "
            f"Daily cost = {cost_mean:,.2f} ± {cost_ci:,.2f} | "
            f"Hard violation = "
            f"{violation_mean:.3f} ± {violation_ci:.3f}%"
        )

    return summary


# ============================================================
# 6. Publication plot style
#    Same typography as the final RQ1 figures.
# ============================================================

def configure_plot_style() -> None:

    plt.rcParams.update({

        "font.family": "serif",

        "font.serif": [
            "Times New Roman",
            "Times",
            "DejaVu Serif",
        ],

        "font.size": 12,

        "axes.labelsize": 18,
        "axes.titlesize": 18,

        "xtick.labelsize": 15,
        "ytick.labelsize": 15,

        "legend.fontsize": 13,

        "axes.linewidth": 1.1,

        "xtick.direction": "in",
        "ytick.direction": "in",

        "xtick.major.width": 0.9,
        "ytick.major.width": 0.9,

        "xtick.major.size": 3.0,
        "ytick.major.size": 3.0,

        "figure.dpi": 150,
        "savefig.dpi": 300,

        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,

        "text.usetex": False,
        "mathtext.fontset": "stix",

        "pdf.fonttype": 42,
        "ps.fonttype": 42,

    })


# ============================================================
# 7. Main plotting function
# ============================================================

def plot_pareto_view(
    summary: dict,
    output_dir: Path,
    extremity: float,
) -> None:

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    configure_plot_style()

    fig, ax = plt.subplots(
        figsize=(FIG_WIDTH, FIG_HEIGHT)
    )

    # --------------------------------------------------------
    # 7.1 Safety threshold
    # --------------------------------------------------------

    ax.axhline(
        y=SAFETY_THRESHOLD,
        color=STYLES["hfg_sac"]["color"],
        linestyle="--",
        linewidth=1.2,
        alpha=0.65,
        zorder=1,
    )

    # --------------------------------------------------------
    # 7.2 Plot algorithms with 95% CI error bars
    # --------------------------------------------------------

    for algorithm in ALGORITHMS:

        stats = summary[algorithm]

        style = STYLES[algorithm]

        color = style["color"]

        x = stats["cost_mean"]
        y = stats["violation_mean"]

        x_ci = stats["cost_ci"]
        y_ci = stats["violation_ci"]

        # Error bars and markers

        ax.errorbar(
            x,
            y,

            xerr=x_ci,
            yerr=y_ci,

            fmt=style["marker"],

            color=color,
            ecolor=color,

            markersize=style["markersize"],

            markerfacecolor=color,
            markeredgecolor=color,
            markeredgewidth=0.8,

            elinewidth=1.25,

            capsize=4.0,
            capthick=1.15,

            linestyle="none",

            zorder=style["zorder"],
        )

        # Direct algorithm annotations

        dx, dy, ha = LABEL_OFFSETS[algorithm]

        ax.annotate(
            ALGORITHM_LABELS[algorithm],

            xy=(x, y),

            xytext=(dx, dy),

            textcoords="offset points",

            ha=ha,
            va="center",

            fontsize=13,

            color=color,

            clip_on=True,

            zorder=8,
        )

    # --------------------------------------------------------
    # 7.3 Axis labels
    # The figure title is intentionally omitted; its context belongs in the paper caption.
    # --------------------------------------------------------

    ax.set_xlabel(
        "Daily Cost (RMB/day)",
        fontsize=18,
        labelpad=7,
    )

    ax.set_ylabel(
        "Hard Violation Rate (%)",
        fontsize=18,
        labelpad=7,
    )

    # --------------------------------------------------------
    # 7.4 X-axis
    # --------------------------------------------------------

    ax.set_xlim(7100, 18700)

    ax.set_xticks([
        8000,
        10000,
        12000,
        14000,
        16000,
        18000,
    ])

    # --------------------------------------------------------
    # 7.5 Y-axis: symmetric logarithmic scale
    # --------------------------------------------------------

    ax.set_yscale(
        "symlog",
        base=10,
        linthresh=1.0,
        linscale=1.0,
    )

    ax.set_ylim(-20, 180)

    ax.set_yticks([
        -10,
        -1,
        0,
        1,
        10,
        100,
    ])

    # --------------------------------------------------------
    # 7.6 Grid
    # --------------------------------------------------------

    ax.grid(
        which="major",
        axis="both",

        visible=True,

        linestyle="-",

        linewidth=0.55,

        color="#D9D9D9",

        alpha=0.65,

        zorder=0,
    )

    ax.set_axisbelow(True)

    # --------------------------------------------------------
    # 7.7 Axes borders
    # --------------------------------------------------------

    for spine in ax.spines.values():

        spine.set_linewidth(1.1)

        spine.set_color("black")

    # --------------------------------------------------------
    # 7.8 Tick parameters
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # 7.9 Layout
    # --------------------------------------------------------

    fig.subplots_adjust(
        left=0.17,
        right=0.97,
        bottom=0.15,
        top=0.96,
    )

    # --------------------------------------------------------
    # 7.10 Save PNG and PDF
    # --------------------------------------------------------

    png_path = (
        output_dir
        / f"{OUTPUT_STEM}.png"
    )

    pdf_path = (
        output_dir
        / f"{OUTPUT_STEM}.pdf"
    )

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

    print()
    print("[Saved] PNG:", png_path)
    print("[Saved] PDF:", pdf_path)


# ============================================================
# 8. Main
# ============================================================

def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "Plot RQ2 Pareto view: "
            "hard violation rate vs daily cost."
        )
    )

    parser.add_argument(
        "--results-dir",
        type=Path,
        default=None,
        help="RQ2 JSON results directory.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Figure output directory.",
    )

    parser.add_argument(
        "--extremity",
        type=float,
        default=DEFAULT_EXTREMITY,
        help="Extremity multiplier. Default: 1.5.",
    )

    args = parser.parse_args()

    extremity_tag = f"{args.extremity:.1f}"

    # Locate project and data directories

    if args.results_dir is None:

        project_root, results_dir = (
            find_default_results_dir()
        )

    else:

        results_dir = (
            args.results_dir
            .expanduser()
            .resolve()
        )

        if not results_dir.is_dir():
            raise FileNotFoundError(
                f"Results directory does not exist: {results_dir}"
            )

        project_root = infer_project_root(
            results_dir
        )

    # Output directory

    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else project_root / "figures"
    )

    if not results_dir.is_dir():
        raise FileNotFoundError(
            f"Results directory does not exist: {results_dir}"
        )

    # Print configuration

    print("=" * 65)
    print("RQ2 Figure 02 — Pareto View")
    print("=" * 65)

    print(f"Project root : {project_root}")
    print(f"Results dir  : {results_dir}")
    print(f"Output dir   : {output_dir}")

    print(f"Extremity    : {extremity_tag}")
    print(f"Seeds        : {SEEDS}")

    print("Statistics   : Mean ± 95% CI")
    print("Y scale      : Symmetric logarithmic")
    print("Threshold    : 5%")

    print("=" * 65)
    print()

    # Load data

    summary = load_summary(
        results_dir,
        extremity_tag,
    )

    # Draw figure

    plot_pareto_view(
        summary=summary,
        output_dir=output_dir,
        extremity=args.extremity,
    )


if __name__ == "__main__":
    main()
