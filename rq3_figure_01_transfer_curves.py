#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""RQ3 Figure 01: T4 islanded extreme-shift transfer curves.

Plots the cross-seed mean of episode_rewards and episode_violation_rates
for four transfer/baseline methods. The no-M1 and no-M4 ablations are excluded.

Expected input files under rq3/results (or outputs/rq3/results):
    hfg_sac_T4_island_shifted_hfg_transfer_s{seed}.json
    hfg_sac_T4_island_shifted_naive_transfer_s{seed}.json
    hfg_sac_T4_island_shifted_from_scratch_s{seed}.json
    ppo_lagrangian_T4_island_shifted_ppo_lagrangian_scratch_s{seed}.json

Default output:
    figures/rq3_figure_01_t4_transfer_curves.png
    figures/rq3_figure_01_t4_transfer_curves.pdf

Run:
    python plot_rq3_figure_01.py
    python plot_rq3_figure_01.py --smooth 1
    python plot_rq3_figure_01.py --results-dir outputs/rq3/results
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


# ---------------------------------------------------------------------
# 1. Experiment configuration
# ---------------------------------------------------------------------

SEEDS = [42, 123, 456, 789, 2024]
EPISODE_COUNT = 200

# A 2-episode moving average reproduces the light smoothing in the reference.
# Set --smooth 1 to plot the unsmoothed cross-seed mean.
SMOOTH_WINDOW = 2

METHODS = {
    "hfg_transfer": {
        "prefix": "hfg_sac_T4_island_shifted_hfg_transfer",
        "label": "HFG-SAC transfer",
        "color": "#d62728",
        "linestyle": "-",
        "linewidth": 2.7,
        "zorder": 7,
    },
    "naive_transfer": {
        "prefix": "hfg_sac_T4_island_shifted_naive_transfer",
        "label": "Naive transfer",
        "color": "#7f7f7f",
        "linestyle": ":",
        "linewidth": 1.9,
        "zorder": 3,
    },
    "from_scratch": {
        "prefix": "hfg_sac_T4_island_shifted_from_scratch",
        "label": "From scratch",
        "color": "#1f77b4",
        "linestyle": "-.",
        "linewidth": 1.9,
        "zorder": 3,
    },
    "ppo_lagrangian_scratch": {
        "prefix": "ppo_lagrangian_T4_island_shifted_ppo_lagrangian_scratch",
        "label": "PPO-Lagrangian scratch",
        "color": "#2ca02c",
        "linestyle": "-.",
        "linewidth": 1.9,
        "zorder": 3,
    },
}

# Plot the three baselines first so the HFG-SAC transfer curve remains visible.
PLOT_ORDER = [
    "naive_transfer",
    "from_scratch",
    "ppo_lagrangian_scratch",
    "hfg_transfer",
]

LEGEND_ORDER = [
    "hfg_transfer",
    "naive_transfer",
    "from_scratch",
    "ppo_lagrangian_scratch",
]

FIG_WIDTH = 29.0 / 2.54   # 29.0 cm, identical to RQ1
FIG_HEIGHT = 8.4 / 2.54   # 8.4 cm, identical to RQ1
OUTPUT_STEM = "rq3_figure_01_transfer_curves"


# ---------------------------------------------------------------------
# 2. Locate project/results directories
# ---------------------------------------------------------------------

def find_default_results_dir() -> tuple[Path, Path]:
    """Find either <root>/outputs/rq3/results or <root>/rq3/results."""
    script_dir = Path(__file__).resolve().parent
    search_bases = [script_dir, *script_dir.parents, Path.cwd(), *Path.cwd().parents]
    seen: set[Path] = set()

    for base in search_bases:
        base = base.resolve()
        if base in seen:
            continue
        seen.add(base)

        for relative in (Path("outputs/rq3/results"), Path("rq3/results")):
            candidate = base / relative
            if candidate.is_dir():
                return base, candidate

    raise FileNotFoundError(
        "未找到 RQ3 数据目录。预期目录为 outputs/rq3/results 或 rq3/results。\n"
        "请将脚本放在项目目录中，或通过 --results-dir 显式指定。"
    )


def infer_project_root(results_dir: Path) -> Path:
    resolved = results_dir.resolve()
    if resolved.name == "results" and resolved.parent.name == "rq3":
        if resolved.parent.parent.name == "outputs":
            return resolved.parent.parent.parent
        return resolved.parent.parent
    return resolved.parent


# ---------------------------------------------------------------------
# 3. Read and validate episode-level data
# ---------------------------------------------------------------------

def read_episode_series(
    results_dir: Path,
    prefix: str,
    seed: int,
    field: str,
) -> np.ndarray:
    file_path = results_dir / f"{prefix}_s{seed}.json"

    if not file_path.is_file():
        raise FileNotFoundError(f"缺少实验文件：{file_path}")

    with file_path.open("r", encoding="utf-8") as f:
        result = json.load(f)

    if field not in result:
        raise KeyError(
            f"{file_path.name} 中没有字段 {field!r}；"
            f"实际字段：{list(result.keys())}"
        )

    values = np.asarray(result[field], dtype=float)

    if values.ndim != 1 or values.size == 0:
        raise ValueError(
            f"{file_path.name} 的 {field} 必须是一维非空数组。"
        )

    if not np.all(np.isfinite(values)):
        raise ValueError(
            f"{file_path.name} 的 {field} 包含 NaN 或 Inf。"
        )

    return values


def load_all_curves(results_dir: Path) -> dict[str, dict[str, np.ndarray]]:
    """Load all seeds, align each method to its shortest available length."""
    curves: dict[str, dict[str, np.ndarray]] = {}

    for method_key, config in METHODS.items():
        reward_runs = []
        violation_runs = []

        for seed in SEEDS:
            reward_runs.append(
                read_episode_series(
                    results_dir,
                    config["prefix"],
                    seed,
                    "episode_rewards",
                )
            )
            violation_runs.append(
                read_episode_series(
                    results_dir,
                    config["prefix"],
                    seed,
                    "episode_violation_rates",
                )
            )

        lengths = [
            len(x) for x in reward_runs + violation_runs
        ]
        min_length = min(lengths)

        if min_length < EPISODE_COUNT:
            warnings.warn(
                f"{method_key} 的数据长度为 {min_length}，"
                f"少于参考图的 {EPISODE_COUNT} episodes；将按实际长度绘制。"
            )

        reward_array = np.stack(
            [x[:min_length] for x in reward_runs],
            axis=0,
        )
        violation_array = np.stack(
            [x[:min_length] for x in violation_runs],
            axis=0,
        )

        curves[method_key] = {
            "rewards": reward_array,
            "violations": violation_array,
        }

        print(
            f"[读取成功] {method_key:24s} | "
            f"seeds={len(reward_runs)} | episodes={min_length}"
        )

    return curves


# ---------------------------------------------------------------------
# 4. Smoothing and aggregation
# ---------------------------------------------------------------------

def smooth_curves(seed_curves: np.ndarray, window: int) -> np.ndarray:
    """Smooth each seed separately, preserving the original episode count."""
    if window <= 1:
        return seed_curves.copy()

    if window > seed_curves.shape[1]:
        raise ValueError(
            f"平滑窗口 {window} 大于 Episode 数量 {seed_curves.shape[1]}。"
        )

    left = (window - 1) // 2
    right = window - 1 - left
    padded = np.pad(
        seed_curves,
        pad_width=((0, 0), (left, right)),
        mode="edge",
    )
    kernel = np.ones(window, dtype=float) / window

    return np.stack(
        [np.convolve(row, kernel, mode="valid") for row in padded],
        axis=0,
    )


def calculate_mean_curves(
    curves: dict[str, dict[str, np.ndarray]],
    smooth_window: int,
) -> dict[str, dict[str, np.ndarray]]:
    """Return per-episode cross-seed means; no std band is plotted."""
    means: dict[str, dict[str, np.ndarray]] = {}

    for method_key, data in curves.items():
        rewards = smooth_curves(data["rewards"], smooth_window)
        violations = smooth_curves(data["violations"], smooth_window)

        means[method_key] = {
            "rewards": rewards.mean(axis=0),
            "violations": violations.mean(axis=0),
        }

    return means


# ---------------------------------------------------------------------
# 5. Plot style — consistent with the final RQ1 figures
# ---------------------------------------------------------------------

def configure_plot_style() -> None:
    """Use the exact typography and axis styling defined in the RQ1 script."""
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


# ---------------------------------------------------------------------
# 6. Draw RQ3 Figure 01
# ---------------------------------------------------------------------

def plot_rq3_figure_01(
    mean_curves: dict[str, dict[str, np.ndarray]],
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    configure_plot_style()

    # Match the wide, two-panel composition used for the final RQ1 figures.
    fig, axes = plt.subplots(
        nrows=1,
        ncols=2,
        figsize=(FIG_WIDTH, FIG_HEIGHT),
        sharex=True,
        sharey=False,
    )
    ax_return, ax_violation = axes

    for method_key in PLOT_ORDER:
        config = METHODS[method_key]
        reward_mean = mean_curves[method_key]["rewards"]
        violation_mean = mean_curves[method_key]["violations"]

        episodes_reward = np.arange(1, len(reward_mean) + 1)
        episodes_violation = np.arange(1, len(violation_mean) + 1)

        common_style = {
            "color": config["color"],
            "linestyle": config["linestyle"],
            "linewidth": 2.7 if method_key == "hfg_transfer" else 1.8,
            "zorder": config["zorder"],
            "solid_capstyle": "round",
            "dash_capstyle": "butt",
        }
        ax_return.plot(episodes_reward, reward_mean, **common_style)
        ax_violation.plot(episodes_violation, violation_mean, **common_style)

    # Axis labels and ranges. Subfigure descriptions are placed below the axes,
    # as in the final RQ1 figures, rather than above the plotting area.
    ax_return.set_xlabel("Episode", fontsize=16, labelpad=4)
    ax_return.set_ylabel("Episode Return", fontsize=16, labelpad=2)
    ax_return.set_xlim(0, 200)
    ax_return.set_xticks([0, 50, 100, 150, 200])
    ax_return.set_ylim(-1.06e6, -0.12e6)
    ax_return.set_yticks([-1.0e6, -0.8e6, -0.6e6, -0.4e6, -0.2e6])

    formatter = ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((6, 6))
    ax_return.yaxis.set_major_formatter(formatter)

    ax_violation.set_xlabel("Episode", fontsize=16, labelpad=4)
    ax_violation.set_ylabel("Hard Violation Rate (%)", fontsize=16, labelpad=2)
    ax_violation.set_xlim(0, 200)
    ax_violation.set_xticks([0, 50, 100, 150, 200])
    ax_violation.set_ylim(0, 105)
    ax_violation.set_yticks(np.arange(0, 101, 20))

    # Axis frame, inward ticks and subtle grid: consistent across both panels.
    for ax in axes:
        ax.set_axisbelow(True)
        ax.grid(
            True,
            which="major",
            axis="both",
            linestyle="-",
            linewidth=0.55,
            color="#D9D9D9",
            alpha=0.65,
            zorder=0,
        )
        for spine in ax.spines.values():
            spine.set_linewidth(1.1)
            spine.set_color("black")
        ax.tick_params(
            axis="both",
            which="major",
            direction="in",
            top=True,
            right=True,
            width=0.9,
            length=3.0,
            pad=3,
        )

    # Make the scientific-notation offset use the same serif font and scale.
    offset = ax_return.yaxis.get_offset_text()
    offset.set_fontsize(13)

    # Use a separate in-axes legend for each panel.
    # Panel (a): lower-right region is clear of the late-episode reward curves.
    # Panel (b): upper-right region is clear of the violation-rate trajectories.
    legend_handles = [
        Line2D(
            [0], [0],
            color=METHODS[key]["color"],
            linestyle=METHODS[key]["linestyle"],
            linewidth=2.7 if key == "hfg_transfer" else 1.8,
            label=METHODS[key]["label"],
        )
        for key in LEGEND_ORDER
    ]

    legend_style = {
        "handles": legend_handles,
        "ncol": 1,
        "frameon": True,
        "framealpha": 0.94,
        "edgecolor": "black",
        "facecolor": "white",
        "handlelength": 1.35,
        "handletextpad": 0.35,
        "labelspacing": 0.18,
        "borderpad": 0.25,
        "borderaxespad": 0.35,
        "fontsize": 11.5,
    }

    legend_a = ax_return.legend(loc="lower right", **legend_style)
    legend_b = ax_violation.legend(loc="upper right", **legend_style)

    legend_a.get_frame().set_linewidth(0.8)
    legend_b.get_frame().set_linewidth(0.8)

    # Match RQ1's exact canvas margins and panel spacing.
    fig.subplots_adjust(
        left=0.075,
        right=0.992,
        top=0.985,
        bottom=0.28,
        wspace=0.22,
    )

    # Place captions at the same measured 4.73 pt gap below each x-axis label,
    # rather than using a fixed figure y-coordinate.
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    target_gap_pt = 4.73
    target_gap_fig = (target_gap_pt / 72.0) / fig.get_figheight()

    panel_captions = [
        "(a) Learning Performance",
        "(b) Safety Performance",
    ]
    for ax, caption in zip(axes, panel_captions):
        pos = ax.get_position()
        center_x = (pos.x0 + pos.x1) / 2
        xlabel_bbox = ax.xaxis.label.get_window_extent(renderer=renderer)
        xlabel_bottom_fig = fig.transFigure.inverted().transform(
            (0, xlabel_bbox.y0)
        )[1]
        caption_y = xlabel_bottom_fig - target_gap_fig
        fig.text(
            center_x,
            caption_y,
            caption,
            ha="center",
            va="top",
            fontsize=16,
        )

    png_path = output_dir / f"{OUTPUT_STEM}.png"
    pdf_path = output_dir / f"{OUTPUT_STEM}.pdf"

    fig.savefig(png_path, dpi=300, bbox_inches="tight", pad_inches=0.05)
    fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)

    print("\n绘图完成：")
    print("字体：Times New Roman（按 RQ1 字体回退规则）")
    print(f"PNG: {png_path}")
    print(f"PDF: {pdf_path}")


# ---------------------------------------------------------------------
# 7. Main
# ---------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot RQ3 Figure 01: islanded extreme-shift transfer."
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=None,
        help="RQ3 JSON 数据目录；默认自动寻找 rq3/results 或 outputs/rq3/results。",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="图片输出目录；默认项目根目录下 figures/。",
    )
    parser.add_argument(
        "--smooth",
        type=int,
        default=SMOOTH_WINDOW,
        help="移动平均窗口；默认 2，设为 1 可关闭平滑。",
    )

    args = parser.parse_args()

    if args.smooth < 1:
        parser.error("--smooth 必须大于或等于 1。")

    if args.results_dir is None:
        project_root, results_dir = find_default_results_dir()
    else:
        results_dir = args.results_dir.expanduser().resolve()
        if not results_dir.is_dir():
            raise FileNotFoundError(f"实验数据目录不存在：{results_dir}")
        project_root = infer_project_root(results_dir)

    output_dir = (
        project_root / "figures"
        if args.output_dir is None
        else args.output_dir.expanduser().resolve()
    )

    print("=" * 68)
    print("RQ3 Figure 01 — Islanded Extreme-Shift Transfer")
    print("=" * 68)
    print(f"项目根目录：{project_root}")
    print(f"实验数据目录：{results_dir}")
    print(f"图片输出目录：{output_dir}")
    print(f"随机种子：{SEEDS}")
    print(f"平滑窗口：{args.smooth}")
    print("绘图方法：HFG-SAC transfer、Naive transfer、From scratch、PPO-Lagrangian scratch")
    print("统计方式：逐 Episode 跨 seed Mean（不绘制 Std 阴影）")
    print("=" * 68)

    raw_curves = load_all_curves(results_dir)
    mean_curves = calculate_mean_curves(
        raw_curves,
        smooth_window=args.smooth,
    )

    plot_rq3_figure_01(
        mean_curves=mean_curves,
        output_dir=output_dir,
    )


if __name__ == "__main__":
    main()
