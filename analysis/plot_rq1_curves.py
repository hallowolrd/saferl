"""Plot RQ1 training curves: episode cost, violation rate, FCSD across 4 scenarios."""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 140,
    "savefig.dpi": 220,
    "savefig.bbox": "tight",
})

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "outputs" / "experiments" / "results"
FIG = ROOT / "analysis" / "output" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

ALG_ORDER = ["sac", "ppo", "ppo_lagrangian", "cpo", "safety_layer", "hfg_sac"]
ALG_LABEL = {
    "sac": "SAC", "ppo": "PPO", "ppo_lagrangian": "PPO-Lag",
    "cpo": "CPO", "safety_layer": "SL", "hfg_sac": "HFG-SAC (ours)",
}
ALG_COLOR = {
    "sac": "#9e9e9e", "ppo": "#64b5f6", "ppo_lagrangian": "#ffb74d",
    "cpo": "#e57373", "safety_layer": "#ba68c8", "hfg_sac": "#d32f2f",
}
SEEDS = [42, 123, 456, 789, 2024]
SCEN = ["S1_grid_normal", "S2_grid_extreme", "S3_island_normal", "S4_island_extreme"]
SC_LABEL = {
    "S1_grid_normal": "S1 Grid normal",
    "S2_grid_extreme": "S2 Grid extreme",
    "S3_island_normal": "S3 Island normal",
    "S4_island_extreme": "S4 Island extreme",
}


def load_curves(alg: str, scen: str, field: str) -> np.ndarray | None:
    """Return (n_seeds, n_episodes) array for a given alg/scen/field, or None."""
    runs = []
    for seed in SEEDS:
        p = RESULTS / f"{alg}_{scen}_s{seed}.json"
        if not p.exists():
            continue
        with open(p, encoding="utf-8") as f:
            d = json.load(f)
        if field in d and isinstance(d[field], list) and len(d[field]) > 10:
            runs.append(d[field])
    if not runs:
        return None
    # align to min length
    min_len = min(len(r) for r in runs)
    return np.array([r[:min_len] for r in runs])


def plot_metric(field: str, ylabel: str, title: str, fname: str,
                smooth: int = 10, y_max: float | None = None) -> None:
    """4×6 grid: rows = scenarios, columns = metrics? Actually: 4 rows × 1 col = 4 scenarios, all algs."""
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), sharex=True)
    axes = axes.flatten()
    eps_axis = None
    for j, sc in enumerate(SCEN):
        ax = axes[j]
        for alg in ALG_ORDER:
            arr = load_curves(alg, sc, field)
            if arr is None:
                continue
            # moving average for readability
            if smooth > 1:
                kernel = np.ones(smooth) / smooth
                arr_smooth = np.array([np.convolve(r, kernel, mode="valid") for r in arr])
            else:
                arr_smooth = arr
            mean = arr_smooth.mean(axis=0)
            std = arr_smooth.std(axis=0, ddof=1) if len(arr_smooth) > 1 else np.zeros_like(mean)
            eps = np.arange(smooth, smooth + len(mean)) if smooth > 1 else np.arange(1, len(mean) + 1)
            lw = 2.0 if alg == "hfg_sac" else 1.1
            ax.plot(eps, mean, label=ALG_LABEL[alg], color=ALG_COLOR[alg], linewidth=lw)
            ax.fill_between(eps, mean - std, mean + std, color=ALG_COLOR[alg], alpha=0.12)
        ax.set_title(SC_LABEL[sc])
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3, linestyle="--")
        if y_max is not None:
            ax.set_ylim(top=y_max)
        if j == 0:
            ax.legend(frameon=False, fontsize=8, ncol=2, loc="upper right")
    for ax in axes[2:]:
        ax.set_xlabel("Episode")
    fig.suptitle(title, y=1.00, fontsize=12)
    fig.tight_layout()
    p = FIG / fname
    fig.savefig(p)
    plt.close(fig)
    print(f"Saved {p.name}")


def main() -> None:
    # Episode cost (lower better)
    plot_metric(
        "episode_costs",
        "Episode cost (CNY)",
        "RQ1 Training: episode cost across scenarios (mean ± std over 5 seeds, 10-episode moving avg)",
        "fig06_rq1_train_cost.png",
        smooth=10,
    )
    # Episode violation rate (lower better)
    plot_metric(
        "episode_violation_rates",
        "Episode violation rate (%)",
        "RQ1 Training: episode violation rate across scenarios (10-episode moving avg)",
        "fig07_rq1_train_violation.png",
        smooth=10,
        y_max=100,
    )
    # Episode FCSD (higher better)
    plot_metric(
        "episode_fcsds",
        "Episode FCSD",
        "RQ1 Training: episode FCSD across scenarios (10-episode moving avg)",
        "fig08_rq1_train_fcsd.png",
        smooth=10,
    )


if __name__ == "__main__":
    main()
