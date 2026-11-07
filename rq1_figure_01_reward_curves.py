#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Plot RQ1 reward curves as ONE combined two-panel figure.

Default input:
    outputs/rq1/results/

Default outputs:
    figures/rq1_figure_01_reward_curves.png
    figures/rq1_figure_01_reward_curves.pdf

Design choices
--------------
1. Two scenarios are plotted side-by-side in a single figure.
2. No overall title and no subplot title are drawn inside the axes.
3. A single legend is shown INSIDE the left subplot, at the lower-right corner.
4. Subfigure captions are placed BELOW each subplot:
      (a) Islanded Normal
      (b) Islanded Extreme
5. Scientific notation is unified as ×10^6.
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import ScalarFormatter

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
        "y_limits": (-9.2e5, 2.5e4),
        "y_ticks": [0, -0.2e6, -0.4e6, -0.6e6, -0.8e6],
    },
    "S4_island_extreme": {
        "caption": "(b) Islanded Extreme",
        "y_limits": (-1.55e6, 1.0e5),
        "y_ticks": [0, -0.5e6, -1.0e6, -1.5e6],
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

SMOOTH_WINDOW = 7
BAND_ALPHA = 0.10
SAFETY_LAYER_AS_BASELINE = True
SAFETY_BASELINE_EPISODES = (5, 15)

FIG_WIDTH = 29.0 / 2.54   # 29.0 cm
FIG_HEIGHT = 8.4 / 2.54   # 8.4 cm


def find_project_root() -> Path:
    script_dir = Path(__file__).resolve().parent
    for candidate in [script_dir, *script_dir.parents]:
        if (candidate / "outputs" / "rq1" / "results").is_dir():
            return candidate
    raise FileNotFoundError(
        "Cannot find outputs/rq1/results. Please place the script under the project "
        "directory or pass --results-dir explicitly."
    )


def load_one_run(results_dir: Path, algorithm: str, scenario: str, seed: int) -> np.ndarray | None:
    file_path = results_dir / f"{algorithm}_{scenario}_s{seed}.json"
    if not file_path.is_file():
        warnings.warn(f"Missing result file: {file_path}")
        return None

    with file_path.open("r", encoding="utf-8") as f:
        result = json.load(f)

    if "episode_rewards" not in result:
        raise KeyError(
            f"{file_path.name} does not contain 'episode_rewards'. "
            f"Available keys: {list(result.keys())}"
        )

    rewards = np.asarray(result["episode_rewards"], dtype=float)
    if rewards.ndim != 1 or rewards.size == 0:
        raise ValueError(f"{file_path.name}: episode_rewards must be a non-empty 1-D array.")
    if not np.all(np.isfinite(rewards)):
        raise ValueError(f"{file_path.name}: episode_rewards contains NaN/Inf.")
    return rewards


def load_all_curves(results_dir: Path) -> dict:
    curves = {}
    for scenario in SCENARIOS:
        for algorithm in ALGORITHMS:
            seed_curves = []
            for seed in SEEDS:
                arr = load_one_run(results_dir, algorithm, scenario, seed)
                if arr is not None:
                    seed_curves.append(arr)

            if not seed_curves:
                warnings.warn(f"No valid curves found for {algorithm} / {scenario}.")
                continue

            min_len = min(len(x) for x in seed_curves)
            aligned = np.stack([x[:min_len] for x in seed_curves], axis=0)
            curves[(algorithm, scenario)] = aligned

            print(
                f"[Loaded] {algorithm:16s} | {scenario:20s} | "
                f"seeds={len(seed_curves)} | episodes={min_len}"
            )
    return curves


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
    return np.stack([np.convolve(row, kernel, mode="valid") for row in padded], axis=0)


def calculate_mean_std(seed_curves: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = seed_curves.mean(axis=0)
    if seed_curves.shape[0] > 1:
        std = seed_curves.std(axis=0, ddof=1)
    else:
        std = np.zeros_like(mean)
    return mean, std


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


def apply_scientific_format(ax) -> None:
    formatter = ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((6, 6))
    formatter.set_useOffset(False)
    ax.yaxis.set_major_formatter(formatter)
    ax.yaxis.get_offset_text().set_fontsize(13)


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


def plot_one_axis(ax, curves: dict, scenario: str, smooth_window: int, show_legend: bool = False) -> None:
    cfg = SCENARIOS[scenario]

    for algorithm in PLOT_ORDER:
        key = (algorithm, scenario)
        if key not in curves:
            continue

        seed_curves = curves[key]
        style = STYLES[algorithm]

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
            continue

        smoothed = smooth_curves(seed_curves, smooth_window)
        mean, std = calculate_mean_std(smoothed)
        episodes = np.arange(len(mean))

        ax.plot(
            episodes, mean,
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

    ax.set_xlabel("Episode", fontsize=16, labelpad=4)
    ax.set_ylabel("Episode Return", fontsize=16, labelpad=2)
    ax.set_xlim(0, 200)
    ax.set_xticks([0, 50, 100, 150, 200])
    ax.set_ylim(*cfg["y_limits"])
    ax.set_yticks(cfg["y_ticks"])
    apply_scientific_format(ax)

    ax.grid(
        which="major", axis="both", visible=True,
        linestyle="-", linewidth=0.55, color="#D9D9D9", alpha=0.65, zorder=0
    )
    ax.set_axisbelow(True)

    for spine in ax.spines.values():
        spine.set_linewidth(1.1)
        spine.set_color("black")

    ax.tick_params(
        axis="both", which="major",
        direction="in", width=0.9, length=3.0, top=True, right=True, pad=3
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


def plot_combined_figure(curves: dict, output_dir: Path, smooth_window: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(FIG_WIDTH, FIG_HEIGHT))
    plot_one_axis(axes[0], curves, "S3_island_normal", smooth_window=smooth_window, show_legend=True)
    plot_one_axis(axes[1], curves, "S4_island_extreme", smooth_window=smooth_window, show_legend=True)

    fig.subplots_adjust(left=0.075, right=0.992, top=0.985, bottom=0.28, wspace=0.22)

    # Match the caption-to-x-label spacing used in RQ1 Fig.2.
    # The target is the measured 4.73 pt (approximately 1.67 mm).
    # Measure the rendered Episode label so the spacing remains stable
    # even if the font metrics or figure dimensions change.
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    target_gap_pt = 4.73
    target_gap_fig = (target_gap_pt / 72.0) / fig.get_figheight()

    for ax, scenario in zip(axes, SCENARIOS):
        pos = ax.get_position()
        x_center = (pos.x0 + pos.x1) / 2

        xlabel_bbox = ax.xaxis.label.get_window_extent(renderer=renderer)
        xlabel_bottom_fig = fig.transFigure.inverted().transform(
            (0, xlabel_bbox.y0)
        )[1]
        y_text = xlabel_bottom_fig - target_gap_fig

        fig.text(
            x_center, y_text,
            SCENARIOS[scenario]["caption"],
            ha="center", va="top", fontsize=16,
        )

    png_path = output_dir / "rq1_figure_01_reward_curves.png"
    pdf_path = output_dir / "rq1_figure_01_reward_curves.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight", pad_inches=0.05)
    fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)

    print(f"[Saved] PNG: {png_path}")
    print(f"[Saved] PDF: {pdf_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot RQ1 reward curves as one combined two-panel figure."
    )
    parser.add_argument("--results-dir", type=Path, default=None,
                        help="Directory containing RQ1 JSON results. Default: outputs/rq1/results")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Directory for saving figures. Default: figures/")
    parser.add_argument("--smooth", type=int, default=SMOOTH_WINDOW,
                        help="Moving-average window. Default: 7. Use 1 to disable smoothing.")
    args = parser.parse_args()

    if args.smooth < 1:
        parser.error("--smooth must be >= 1")

    if args.results_dir is None:
        project_root = find_project_root()
        results_dir = project_root / "outputs" / "rq1" / "results"
    else:
        results_dir = args.results_dir.resolve()
        try:
            project_root = find_project_root()
        except FileNotFoundError:
            project_root = results_dir.parent.parent.parent

    output_dir = project_root / "figures" if args.output_dir is None else args.output_dir.resolve()
    if not results_dir.is_dir():
        raise FileNotFoundError(f"Results directory does not exist: {results_dir}")

    print(f"Project root: {project_root}")
    print(f"Results dir : {results_dir}")
    print(f"Output dir  : {output_dir}")
    print(f"Smoothing   : {args.smooth}")
    print("Layout      : one combined two-panel figure")
    print("Legend      : inside BOTH subplots, lower-right corner, 3 columns × 2 rows; 11.5 pt")
    print("Titles      : removed inside axes; subfigure captions below panels; 16 pt")
    print("Figure size : 29.0 cm × 8.4 cm")
    print("Ticks       : X every 50 episodes; Y 4–5 major ticks")
    print("Style       : Times New Roman; 16 pt axis labels; 13 pt ticks; 1.1 pt frame")
    print("Notation    : scientific notation with ×10^6 style")
    print("Outputs     : rq1_figure_01_reward_curves.png / .pdf")

    configure_plot_style()
    curves = load_all_curves(results_dir)
    if not curves:
        raise RuntimeError("No valid curves loaded. Please check your result files.")

    plot_combined_figure(curves, output_dir, smooth_window=args.smooth)


if __name__ == "__main__":
    main()
