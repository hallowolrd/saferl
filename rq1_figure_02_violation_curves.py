#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot RQ1 Figure 2: Hard Violation Rate per Episode.

Style note
----------
This version follows the finalized RQ1 Figure 1 layout/style:
1. Two scenarios are plotted side-by-side in ONE combined figure.
2. No overall title and no subplot title are drawn inside the axes.
3. Each subplot keeps a compact legend INSIDE the upper-middle blank region.
4. Subfigure captions are placed BELOW each subplot:
      (a) Islanded Normal
      (b) Islanded Extreme
5. Typography, line styles, spacing, and legend layout are aligned with
   the accepted RQ1 Figure 1 format.

Default input:
    outputs/rq1/results/

Expected JSON filename:
    {algorithm}_{scenario}_s{seed}.json

Required JSON field:
    episode_violation_rates

Default outputs:
    figures/rq1_figure_02_hard_violation_rate.png
    figures/rq1_figure_02_hard_violation_rate.pdf
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

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
    "S3_island_normal": {
        "caption": "(a) Islanded Normal",
        "y_limits": (-2, 105),
        "y_ticks": [0, 20, 40, 60, 80, 100],
    },
    "S4_island_extreme": {
        "caption": "(b) Islanded Extreme",
        "y_limits": (-2, 105),
        "y_ticks": [0, 20, 40, 60, 80, 100],
    },
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

METRIC_FIELD = "episode_violation_rates"
SMOOTH_WINDOW = 7
BAND_ALPHA = 0.10
SAFETY_LAYER_AS_BASELINE = True
SAFETY_BASELINE_EPISODES = (5, 15)
FIGURE_STEM = "rq1_figure_02_violation_curves"

# Match the finalized Figure 1 size.
FIG_WIDTH = 29.0 / 2.54
FIG_HEIGHT = 9.0 / 2.54

# ---------------------------------------------------------------------
# 2. Project paths
# ---------------------------------------------------------------------
def find_project_root() -> Path:
    script_dir = Path(__file__).resolve().parent
    candidates = [script_dir, *script_dir.parents, Path.cwd(), *Path.cwd().parents]
    for candidate in candidates:
        if (candidate / "outputs" / "rq1" / "results").is_dir():
            return candidate
    raise FileNotFoundError(
        "Cannot find outputs/rq1/results. Place this script inside the project, "
        "run it from the project root, or pass --results-dir explicitly."
    )


def infer_project_root(results_dir: Path) -> Path:
    if results_dir.name == "results" and results_dir.parent.name == "rq1":
        return results_dir.parent.parent.parent
    return results_dir.parent

# ---------------------------------------------------------------------
# 3. Data loading
# ---------------------------------------------------------------------
def result_filename(algorithm: str, scenario: str, seed: int) -> str:
    return f"{algorithm}_{scenario}_s{seed}.json"


def load_one_run(results_dir: Path, algorithm: str, scenario: str, seed: int) -> np.ndarray | None:
    file_path = results_dir / result_filename(algorithm, scenario, seed)
    if not file_path.is_file():
        warnings.warn(f"Missing experiment file: {file_path}")
        return None

    with file_path.open("r", encoding="utf-8") as f:
        result = json.load(f)

    if METRIC_FIELD not in result:
        raise KeyError(
            f"{file_path.name} does not contain '{METRIC_FIELD}'. "
            f"Available fields: {list(result.keys())}"
        )

    values = np.asarray(result[METRIC_FIELD], dtype=float)
    if values.ndim != 1 or values.size == 0:
        raise ValueError(f"{file_path.name}: {METRIC_FIELD} must be a non-empty 1-D array.")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{file_path.name}: {METRIC_FIELD} contains NaN or Inf.")
    return values


def load_all_curves(results_dir: Path) -> Dict[Tuple[str, str], np.ndarray]:
    curves: Dict[Tuple[str, str], np.ndarray] = {}
    for scenario in SCENARIOS:
        for algorithm in ALGORITHMS:
            seed_curves: List[np.ndarray] = []
            for seed in SEEDS:
                values = load_one_run(results_dir, algorithm, scenario, seed)
                if values is not None:
                    seed_curves.append(values)

            if not seed_curves:
                warnings.warn(f"No valid data: {algorithm} / {scenario}")
                continue

            min_episodes = min(len(values) for values in seed_curves)
            aligned = np.stack([values[:min_episodes] for values in seed_curves], axis=0)
            curves[(algorithm, scenario)] = aligned

            print(
                f"[Loaded] {algorithm:16s} | {scenario:20s} | "
                f"seeds={len(seed_curves)} | episodes={min_episodes}"
            )
    return curves

# ---------------------------------------------------------------------
# 4. Smoothing and statistics
# ---------------------------------------------------------------------
def smooth_curves(seed_curves: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return seed_curves.copy()
    if window > seed_curves.shape[1]:
        raise ValueError(
            f"Smoothing window {window} exceeds episode count {seed_curves.shape[1]}."
        )
    left = (window - 1) // 2
    right = window - 1 - left
    padded = np.pad(seed_curves, ((0, 0), (left, right)), mode="edge")
    kernel = np.ones(window, dtype=float) / window
    return np.stack([np.convolve(row, kernel, mode="valid") for row in padded], axis=0)


def calculate_mean_std(seed_curves: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mean = seed_curves.mean(axis=0)
    if seed_curves.shape[0] > 1:
        std = seed_curves.std(axis=0, ddof=1)
    else:
        std = np.zeros_like(mean)
    return mean, std


def calculate_safety_baseline(seed_curves: np.ndarray) -> float:
    start_ep, end_ep = SAFETY_BASELINE_EPISODES
    if start_ep < 1 or end_ep < start_ep or end_ep > seed_curves.shape[1]:
        raise ValueError(
            f"Safety baseline Episodes {start_ep}-{end_ep} are outside the "
            f"available range 1-{seed_curves.shape[1]}."
        )
    return float(seed_curves[:, start_ep - 1:end_ep].mean())

# ---------------------------------------------------------------------
# 5. Plot style (aligned with RQ1 Fig.1)
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
            color=STYLES[alg]["color"],
            linestyle=STYLES[alg]["linestyle"],
            linewidth=STYLES[alg]["linewidth"],
            label=ALGORITHM_LABELS[alg],
        )
        for alg in LEGEND_ORDER
    ]


def plot_one_axis(ax, curves: Dict[Tuple[str, str], np.ndarray], scenario: str, smooth_window: int) -> None:
    cfg = SCENARIOS[scenario]

    for algorithm in PLOT_ORDER:
        key = (algorithm, scenario)
        if key not in curves:
            continue

        seed_curves = curves[key]
        style = STYLES[algorithm]

        if algorithm == "safety_layer" and SAFETY_LAYER_AS_BASELINE:
            baseline = calculate_safety_baseline(seed_curves)
            ax.axhline(
                y=baseline,
                color=style["color"],
                linestyle=style["linestyle"],
                linewidth=style["linewidth"],
                zorder=3,
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
            solid_capstyle="round",
            dash_capstyle="round",
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

    ax.set_xlabel("Episode", fontsize=16, labelpad=4)
    ax.set_ylabel("Hard Violation Rate (%)", fontsize=16, labelpad=2)
    ax.set_xlim(0, 200)
    ax.set_xticks([0, 50, 100, 150, 200])
    ax.set_ylim(*cfg["y_limits"])
    ax.set_yticks(cfg["y_ticks"])

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

    # The legend is shared by the two panels and is placed above the axes.
    # This avoids covering the descending PPO-Lagrangian and CPO curves.


def plot_figure(curves: Dict[Tuple[str, str], np.ndarray], output_dir: Path, smooth_window: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    configure_plot_style()

    fig, axes = plt.subplots(1, 2, figsize=(FIG_WIDTH, FIG_HEIGHT), sharex=True, sharey=False)
    plot_one_axis(axes[0], curves, "S3_island_normal", smooth_window)
    plot_one_axis(axes[1], curves, "S4_island_extreme", smooth_window)

    # Keep y-axes independent (but identically scaled) so both panels
    # render their own numeric y tick labels reliably.
    for ax in axes:
        ax.tick_params(axis="y", which="major", labelleft=True, left=True)
        for tick_label in ax.get_yticklabels():
            tick_label.set_visible(True)

    # Put the legend inside each subplot, in the upper-middle blank area
    # (matching the marked red-box region) to avoid covering the main curves.
    for ax in axes:
        legend = ax.legend(
            handles=make_legend_handles(),
            loc="upper center",
            bbox_to_anchor=(0.64, 0.86),
            ncol=3,
            frameon=True,
            fancybox=False,
            framealpha=0.95,
            edgecolor="black",
            facecolor="white",
            borderpad=0.22,
            labelspacing=0.16,
            columnspacing=0.48,
            handlelength=1.15,
            handletextpad=0.22,
            fontsize=10.8,
        )
        legend.get_frame().set_linewidth(0.8)

    # Bring the panels closer to their captions, restoring the compact
    # bottom layout used in the earlier accepted figure.
    fig.subplots_adjust(left=0.075, right=0.992, top=0.985, bottom=0.225, wspace=0.22)

    for ax, scenario in zip(axes, SCENARIOS):
        pos = ax.get_position()
        x_center = (pos.x0 + pos.x1) / 2
        y_text = 0.068
        fig.text(
            x_center,
            y_text,
            SCENARIOS[scenario]["caption"],
            ha="center",
            va="top",
            fontsize=16,
        )

    png_path = output_dir / f"{FIGURE_STEM}.png"
    pdf_path = output_dir / f"{FIGURE_STEM}.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight", pad_inches=0.05)
    fig.savefig(pdf_path, format="pdf", bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)

    print(f"[Saved] PNG: {png_path}")
    print(f"[Saved] PDF: {pdf_path}")

# ---------------------------------------------------------------------
# 6. Main
# ---------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot RQ1 Figure 2 in the same style as the finalized RQ1 Figure 1."
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=None,
        help="RQ1 JSON directory; default: <project>/outputs/rq1/results",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory; default: <project>/figures",
    )
    parser.add_argument(
        "--smooth",
        type=int,
        default=SMOOTH_WINDOW,
        help="Moving-average window; default 7; set to 1 to disable smoothing.",
    )
    args = parser.parse_args()

    if args.smooth < 1:
        parser.error("--smooth must be >= 1")

    if args.results_dir is None:
        project_root = find_project_root()
        results_dir = project_root / "outputs" / "rq1" / "results"
    else:
        results_dir = args.results_dir.expanduser().resolve()
        try:
            project_root = find_project_root()
        except FileNotFoundError:
            project_root = infer_project_root(results_dir)

    output_dir = project_root / "figures" if args.output_dir is None else args.output_dir.expanduser().resolve()

    if not results_dir.is_dir():
        raise FileNotFoundError(f"Experiment data directory does not exist: {results_dir}")

    expected = results_dir / result_filename("hfg_sac", "S3_island_normal", SEEDS[0])
    if not expected.is_file():
        raise FileNotFoundError(
            f"Cannot find expected RQ1 training file: {expected}\n"
            "This figure requires files such as hfg_sac_S3_island_normal_s42.json "
            f"with the {METRIC_FIELD} field."
        )

    print(f"Project root : {project_root}")
    print(f"Results dir  : {results_dir}")
    print(f"Output dir   : {output_dir}")
    print(f"Smooth window: {args.smooth}")
    print(f"Metric field : {METRIC_FIELD}")
    print("Layout       : one combined two-panel figure")
    print("Legend       : centered inside BOTH panels, compact 3 columns × 2 rows; 10.8 pt")
    print("Y ticks      : numeric labels explicitly enabled on BOTH panels")
    print("Titles       : removed inside axes; subfigure captions below panels; 16 pt")
    print("Figure size  : 29.0 cm × 9.0 cm")
    print("Ticks        : X every 50 episodes; Y every 20%")
    print("Style        : Times New Roman; 16 pt axis labels; 13 pt ticks; 1.1 pt frame")
    print(f"Outputs      : {FIGURE_STEM}.png / .pdf")

    curves = load_all_curves(results_dir)
    if not curves:
        raise RuntimeError("No valid RQ1 training curves were loaded.")

    plot_figure(curves, output_dir, args.smooth)


if __name__ == "__main__":
    main()
