#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Plot RQ1 Figure 3: Feasibility Constraint Satisfaction Degree (FCSD).

The plotting style is synchronized with the confirmed final RQ1 Fig.1 script:
- Times New Roman (with serif fallbacks)
- 29.0 cm x 8.4 cm, two side-by-side panels
- no overall title and no titles above individual panels
- legends inside BOTH panels, lower-right, 3 columns x 2 rows
- subfigure captions below the axes
- 16 pt axis labels and subfigure captions; 13 pt tick labels
- 11.5 pt legends; 1.1 pt axes frames
- 7-episode moving average; mean +/- sample standard deviation across seeds
- Safety Layer shown as a horizontal baseline averaged over episodes 5-15

Default input directory is detected as either:
    outputs/rq1/results/
    rq1/results/

Default outputs:
    figures/rq1_figure_03_fcsd_curves.png
    figures/rq1_figure_03_fcsd_curves.pdf

Expected JSON filename pattern:
    {algorithm}_{scenario}_s{seed}.json
Expected JSON field:
    episode_fcsds
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import FormatStrFormatter


# ---------------------------------------------------------------------
# 1. Experiment configuration
# ---------------------------------------------------------------------

ALGORITHMS = [
    "hfg_sac",
    "sac",
    "ppo",
    "ppo_lagrangian",
    "cpo",
    "safety_layer",
]

SEEDS = [42, 123, 456, 789, 2024]

SCENARIOS = {
    "S3_island_normal": "(a) Islanded Normal",
    "S4_island_extreme": "(b) Islanded Extreme",
}

ALGORITHM_LABELS = {
    "hfg_sac": "HFG-SAC",
    "sac": "SAC",
    "ppo": "PPO",
    "ppo_lagrangian": "PPO-Lagrangian",
    "cpo": "CPO",
    "safety_layer": "Safety Layer",
}

STYLES = {
    "hfg_sac": {"color": "#d62728", "linestyle": "-",  "linewidth": 2.7},
    "sac": {"color": "#1f77b4", "linestyle": "--", "linewidth": 1.8},
    "ppo": {"color": "#2ca02c", "linestyle": "--", "linewidth": 1.8},
    "ppo_lagrangian": {"color": "#9467bd", "linestyle": "-.", "linewidth": 1.8},
    "cpo": {"color": "#8c564b", "linestyle": "-.", "linewidth": 1.8},
    "safety_layer": {"color": "#7f7f7f", "linestyle": ":", "linewidth": 1.8},
}

# Matplotlib arranges ncol=3 legend handles column-wise. This order produces:
# Row 1: HFG-SAC | PPO | CPO
# Row 2: SAC | PPO-Lagrangian | Safety Layer
LEGEND_ORDER = [
    "hfg_sac",
    "sac",
    "ppo",
    "ppo_lagrangian",
    "cpo",
    "safety_layer",
]

PLOT_ORDER = [
    "sac",
    "ppo",
    "ppo_lagrangian",
    "cpo",
    "hfg_sac",
    "safety_layer",
]

SMOOTH_WINDOW = 7
BAND_ALPHA = 0.10
SAFETY_LAYER_AS_BASELINE = True
SAFETY_BASELINE_EPISODES = (5, 15)

# Exact figure dimensions from the confirmed final RQ1 Fig.1 script.
FIG_WIDTH = 29.0 / 2.54   # 29.0 cm
FIG_HEIGHT = 8.4 / 2.54   # 8.4 cm
CAPTION_GAP_PT = 5.0      # physical gap between the x-axis label and subfigure caption

# FCSD is dimensionless and bounded in [0, 1].
Y_LIMITS = (0.0, 1.05)
Y_TICKS = np.arange(0.0, 1.01, 0.2)


# ---------------------------------------------------------------------
# 2. Project and data paths
# ---------------------------------------------------------------------

def locate_project_and_results(script_dir: Path) -> tuple[Path, Path]:
    """Find the project root and RQ1 results directory."""
    for candidate in [script_dir, *script_dir.parents]:
        possible_dirs = [
            candidate / "outputs" / "rq1" / "results",
            candidate / "rq1" / "results",
        ]
        for results_dir in possible_dirs:
            if results_dir.is_dir():
                return candidate, results_dir

    raise FileNotFoundError(
        "Cannot find RQ1 results. Expected either 'outputs/rq1/results' "
        "or 'rq1/results' under the project directory. "
        "You can also pass --results-dir explicitly."
    )


def infer_project_root_from_results(results_dir: Path) -> Path:
    """Infer project root when --results-dir is explicitly provided."""
    resolved = results_dir.resolve()
    if resolved.name == "results" and resolved.parent.name == "rq1":
        if resolved.parent.parent.name == "outputs":
            return resolved.parent.parent.parent
        return resolved.parent.parent
    return resolved.parent


# ---------------------------------------------------------------------
# 3. Data loading
# ---------------------------------------------------------------------

def load_one_run(
    results_dir: Path,
    algorithm: str,
    scenario: str,
    seed: int,
) -> np.ndarray | None:
    file_path = results_dir / f"{algorithm}_{scenario}_s{seed}.json"

    if not file_path.is_file():
        warnings.warn(f"Missing result file: {file_path}")
        return None

    with file_path.open("r", encoding="utf-8") as f:
        result = json.load(f)

    if "episode_fcsds" not in result:
        raise KeyError(
            f"{file_path.name} does not contain 'episode_fcsds'. "
            f"Available keys: {list(result.keys())}"
        )

    values = np.asarray(result["episode_fcsds"], dtype=float)

    if values.ndim != 1 or values.size == 0:
        raise ValueError(
            f"{file_path.name}: episode_fcsds must be a non-empty 1-D array."
        )
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{file_path.name}: episode_fcsds contains NaN/Inf.")
    if np.any(values < 0.0) or np.any(values > 1.0):
        warnings.warn(
            f"{file_path.name}: FCSD values outside [0, 1] were found. "
            "The data will not be clipped automatically."
        )

    return values


def load_all_curves(results_dir: Path) -> dict:
    curves = {}

    for scenario in SCENARIOS:
        for algorithm in ALGORITHMS:
            seed_curves = []

            for seed in SEEDS:
                values = load_one_run(results_dir, algorithm, scenario, seed)
                if values is not None:
                    seed_curves.append(values)

            if not seed_curves:
                warnings.warn(f"No valid FCSD curves found for {algorithm} / {scenario}.")
                continue

            min_len = min(len(x) for x in seed_curves)
            aligned = np.stack([x[:min_len] for x in seed_curves], axis=0)
            curves[(algorithm, scenario)] = aligned

            print(
                f"[Loaded] {algorithm:16s} | {scenario:20s} | "
                f"seeds={len(seed_curves)}/{len(SEEDS)} | episodes={min_len}"
            )

    return curves


# ---------------------------------------------------------------------
# 4. Statistics and smoothing (same method as RQ1 Fig.1)
# ---------------------------------------------------------------------

def smooth_curves(curves: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return curves.copy()
    if window > curves.shape[1]:
        raise ValueError(
            f"Smoothing window {window} exceeds episode length {curves.shape[1]}."
        )

    left = (window - 1) // 2
    right = window - 1 - left
    padded = np.pad(curves, ((0, 0), (left, right)), mode="edge")
    kernel = np.ones(window, dtype=float) / window

    return np.stack(
        [np.convolve(row, kernel, mode="valid") for row in padded],
        axis=0,
    )


def calculate_mean_std(seed_curves: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = seed_curves.mean(axis=0)
    if seed_curves.shape[0] > 1:
        std = seed_curves.std(axis=0, ddof=1)
    else:
        std = np.zeros_like(mean)
    return mean, std


# ---------------------------------------------------------------------
# 5. Plot style (copied from the confirmed final RQ1 Fig.1 settings)
# ---------------------------------------------------------------------

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


def make_legend_handles():
    return [
        Line2D(
            [0], [0],
            color=STYLES[algorithm]["color"],
            linestyle=STYLES[algorithm]["linestyle"],
            linewidth=STYLES[algorithm]["linewidth"],
            label=ALGORITHM_LABELS[algorithm],
        )
        for algorithm in LEGEND_ORDER
    ]


# ---------------------------------------------------------------------
# 6. Plot one panel
# ---------------------------------------------------------------------

def plot_one_axis(
    ax,
    curves: dict,
    scenario: str,
    smooth_window: int,
    show_legend: bool = True,
) -> None:
    for algorithm in PLOT_ORDER:
        key = (algorithm, scenario)
        if key not in curves:
            continue

        seed_curves = curves[key]
        style = STYLES[algorithm]

        # Safety Layer is a horizontal reference baseline, as in Fig.1/Fig.2.
        if algorithm == "safety_layer" and SAFETY_LAYER_AS_BASELINE:
            start_ep, end_ep = SAFETY_BASELINE_EPISODES
            if start_ep < 1 or end_ep < start_ep or end_ep > seed_curves.shape[1]:
                raise ValueError(
                    f"Safety Layer baseline interval {start_ep}-{end_ep} "
                    f"exceeds valid episode range 1-{seed_curves.shape[1]}."
                )

            baseline = float(seed_curves[:, start_ep - 1:end_ep].mean())
            ax.axhline(
                y=baseline,
                color=style["color"],
                linestyle=style["linestyle"],
                linewidth=style["linewidth"],
                zorder=3,
            )
            print(
                f"[Baseline] {scenario} | Safety Layer | "
                f"episodes={start_ep}-{end_ep} | FCSD={baseline:.4f}"
            )
            continue

        smoothed = smooth_curves(seed_curves, smooth_window)
        mean, std = calculate_mean_std(smoothed)
        episodes = np.arange(len(mean))

        ax.plot(
            episodes,
            mean,
            color=style["color"],
            linestyle=style["linestyle"],
            linewidth=style["linewidth"],
            zorder=5 if algorithm == "hfg_sac" else 4,
        )
        ax.fill_between(
            episodes,
            mean - std,
            mean + std,
            color=style["color"],
            alpha=BAND_ALPHA,
            linewidth=0,
            zorder=2,
        )

    # Axes: same typography and tick density as the final Fig.1 script.
    ax.set_xlabel("Episode", fontsize=16, labelpad=4)
    ax.set_ylabel("FCSD", fontsize=16, labelpad=2)
    ax.set_xlim(0, 200)
    ax.set_xticks([0, 50, 100, 150, 200])
    ax.set_ylim(*Y_LIMITS)
    ax.set_yticks(Y_TICKS)
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))

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

    if show_legend:
        legend = ax.legend(
            handles=make_legend_handles(),
            loc="lower right",
            ncol=3,
            frameon=True,
            fancybox=False,
            framealpha=0.96,
            edgecolor="black",
            facecolor="white",
            borderpad=0.22,
            labelspacing=0.16,
            columnspacing=0.65,
            handlelength=1.35,
            handletextpad=0.25,
            fontsize=11.5,
        )
        legend.get_frame().set_linewidth(0.8)


# ---------------------------------------------------------------------
# 7. Combined figure
# ---------------------------------------------------------------------

def plot_combined_figure(
    curves: dict,
    output_dir: Path,
    smooth_window: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(FIG_WIDTH, FIG_HEIGHT))

    plot_one_axis(
        axes[0], curves, "S3_island_normal",
        smooth_window=smooth_window, show_legend=True,
    )
    plot_one_axis(
        axes[1], curves, "S4_island_extreme",
        smooth_window=smooth_window, show_legend=True,
    )

    # Keep the panel layout consistent with RQ1 Fig.1/Fig.2.
    fig.subplots_adjust(
        left=0.075,
        right=0.992,
        top=0.985,
        bottom=0.28,
        wspace=0.22,
    )

    # Place each subfigure caption relative to the ACTUAL rendered x-axis label
    # bounding box. A fixed figure-coordinate offset (e.g. pos.y0 - 0.135) can
    # cause the caption to touch/overlap "Episode" after layout or font changes.
    # This keeps a consistent physical gap of CAPTION_GAP_PT points.
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()

    for ax, scenario in zip(axes, SCENARIOS):
        pos = ax.get_position()
        x_center = (pos.x0 + pos.x1) / 2

        xlabel_bbox = ax.xaxis.label.get_window_extent(renderer=renderer)
        caption_y_display = (
            xlabel_bbox.y0 - CAPTION_GAP_PT * fig.dpi / 72.0
        )
        caption_y = fig.transFigure.inverted().transform(
            (0, caption_y_display)
        )[1]

        fig.text(
            x_center,
            caption_y,
            SCENARIOS[scenario],
            ha="center",
            va="top",
            fontsize=16,
        )

    png_path = output_dir / "rq1_figure_03_fcsd_curves.png"
    pdf_path = output_dir / "rq1_figure_03_fcsd_curves.pdf"

    fig.savefig(png_path, dpi=300, bbox_inches="tight", pad_inches=0.05)
    fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)

    print(f"[Saved] PNG: {png_path}")
    print(f"[Saved] PDF: {pdf_path}")


# ---------------------------------------------------------------------
# 8. Main
# ---------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot RQ1 Fig.3 FCSD curves in the same style as final Fig.1/Fig.2."
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=None,
        help="RQ1 JSON directory. Auto-detects outputs/rq1/results or rq1/results.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Figure output directory. Default: project_root/figures.",
    )
    parser.add_argument(
        "--smooth",
        type=int,
        default=SMOOTH_WINDOW,
        help="Moving-average window. Default: 7; use 1 to disable smoothing.",
    )
    args = parser.parse_args()

    if args.smooth < 1:
        parser.error("--smooth must be >= 1")

    if args.results_dir is None:
        project_root, results_dir = locate_project_and_results(Path(__file__).resolve().parent)
    else:
        results_dir = args.results_dir.resolve()
        if not results_dir.is_dir():
            raise FileNotFoundError(f"Results directory does not exist: {results_dir}")
        try:
            project_root, _ = locate_project_and_results(Path(__file__).resolve().parent)
        except FileNotFoundError:
            project_root = infer_project_root_from_results(results_dir)

    output_dir = (
        project_root / "figures"
        if args.output_dir is None
        else args.output_dir.resolve()
    )

    if not results_dir.is_dir():
        raise FileNotFoundError(f"Results directory does not exist: {results_dir}")

    print(f"Project root : {project_root}")
    print(f"Results dir  : {results_dir}")
    print(f"Output dir   : {output_dir}")
    print(f"Smoothing    : {args.smooth}")
    print("Figure size  : 29.0 cm x 8.4 cm")
    print("Font         : Times New Roman")
    print("Axis labels  : 16 pt; ticks: 13 pt; legend: 11.5 pt; captions: 16 pt")
    print("Legend       : inside both panels, lower-right, 3 columns x 2 rows")
    print(f"Caption gap  : {CAPTION_GAP_PT:.1f} pt below each x-axis label")
    print("Line widths  : HFG-SAC 2.7 pt; other algorithms 1.8 pt")
    print("Band alpha   : 0.10")
    print("S-L baseline : mean FCSD over episodes 5-15")
    print("Output       : rq1_figure_03_fcsd_curves.png / .pdf")

    configure_plot_style()
    curves = load_all_curves(results_dir)
    if not curves:
        raise RuntimeError("No valid FCSD curves loaded. Check filenames and data fields.")

    plot_combined_figure(curves, output_dir, smooth_window=args.smooth)


if __name__ == "__main__":
    main()
