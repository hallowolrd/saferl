"""Strict RQ1 & RQ2 analysis: all 5 seeds, descriptive + inferential + figures."""
from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

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
OUT = ROOT / "analysis" / "output"
FIG = OUT / "figures"
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
RQ1_SCEN = ["S1_grid_normal", "S2_grid_extreme", "S3_island_normal", "S4_island_extreme"]
RQ1_LABEL = {"S1_grid_normal": "S1\nGrid normal", "S2_grid_extreme": "S2\nGrid extreme",
             "S3_island_normal": "S3\nIsland normal", "S4_island_normal": "S4\nIsland extreme"}
RQ1_LABEL["S4_island_extreme"] = "S4\nIsland extreme"
RQ2_LEVELS = [0.8, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]
RQ2_SCEN = [f"extremity_{x:.1f}" for x in RQ2_LEVELS]


def load_all() -> dict:
    data = defaultdict(lambda: defaultdict(list))
    for alg in ALG_ORDER:
        for sc in RQ1_SCEN + RQ2_SCEN:
            for seed in SEEDS:
                p = RESULTS / f"{alg}_{sc}_s{seed}.json"
                if not p.exists():
                    continue
                with open(p, encoding="utf-8") as f:
                    d = json.load(f)
                data[alg][sc].append(d)
    return data


def ci95(xs: list[float]) -> tuple[float, float, float]:
    a = np.asarray(xs, dtype=float)
    n = len(a)
    m = a.mean()
    s = a.std(ddof=1) if n > 1 else 0.0
    h = stats.t.ppf(0.975, n - 1) * s / math.sqrt(n) if n > 1 else 0.0
    return float(m), float(h), float(s)


def cohens_d(a: list[float], b: list[float]) -> float:
    a, b = np.asarray(a), np.asarray(b)
    n1, n2 = len(a), len(b)
    if n1 < 2 or n2 < 2:
        return float("nan")
    sp = math.sqrt(((n1 - 1) * a.var(ddof=1) + (n2 - 1) * b.var(ddof=1)) / (n1 + n2 - 2))
    return float((a.mean() - b.mean()) / sp) if sp > 0 else float("nan")


# ---------- tables ----------
def descriptive_md(data: dict, scenarios: list[str], label: str) -> str:
    lines = [f"## {label} descriptive (mean ± 95% CI, n=5)", "",
             "| Algorithm | Scenario | n | Cost/day | Violation % | FCSD | Train s |",
             "|---|---|---:|---:|---:|---:|---:|"]
    for sc in scenarios:
        for alg in ALG_ORDER:
            runs = data[alg][sc]
            if not runs:
                continue
            cm, ch, _ = ci95([r["cost_per_day"] for r in runs])
            # violation_rate is already stored in percent units (e.g., 89.43 = 89.43%)
            vm, _, _ = ci95([r["violation_rate"] for r in runs])
            fm, _, _ = ci95([r["avg_fcsd"] for r in runs])
            tm, _, _ = ci95([r["training_time_s"] for r in runs])
            lines.append(
                f"| {ALG_LABEL[alg]} | {sc} | {len(runs)} | "
                f"{cm:,.0f} ± {ch:,.0f} | {vm:.2f} | {fm:.3f} | {tm:,.0f} |"
            )
    return "\n".join(lines)


def inferential_md(data: dict, scenarios: list[str], label: str) -> str:
    lines = [f"## {label} inferential: HFG-SAC vs each baseline (cost/day)", "",
             "Welch's two-sided t-test, n=5 vs n=5. Cohen's d = (HFG − baseline)/pooled SD.",
             "Positive Δ means HFG is costlier; positive d = HFG costs more.", "",
             "| Scenario | Baseline | HFG cost | Base cost | Δ | t | p | Cohen's d |",
             "|---|---|---:|---:|---:|---:|---:|---:|"]
    for sc in scenarios:
        hfg = [r["cost_per_day"] for r in data["hfg_sac"][sc]]
        if not hfg:
            continue
        for base in ALG_ORDER:
            if base == "hfg_sac":
                continue
            bl = [r["cost_per_day"] for r in data[base][sc]]
            if not bl:
                continue
            t, p = stats.ttest_ind(hfg, bl, equal_var=False)
            d = cohens_d(hfg, bl)
            delta = np.mean(hfg) - np.mean(bl)
            lines.append(
                f"| {sc} | {ALG_LABEL[base]} | {np.mean(hfg):,.0f} | {np.mean(bl):,.0f} | "
                f"{delta:+,.0f} | {t:+.2f} | {p:.4f} | {d:+.2f} |"
            )
    return "\n".join(lines)


# ---------- figures ----------
def fig_rq1_cost(data: dict) -> Path:
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    x = np.arange(len(RQ1_SCEN))
    w = 0.13
    for i, alg in enumerate(ALG_ORDER):
        means, errs = [], []
        for sc in RQ1_SCEN:
            vals = [r["cost_per_day"] for r in data[alg][sc]]
            m, h, _ = ci95(vals)
            means.append(m); errs.append(h)
        ax.bar(x + (i - 2.5) * w, means, w, yerr=errs, capsize=2.5,
               label=ALG_LABEL[alg], color=ALG_COLOR[alg],
               edgecolor="black", linewidth=0.4)
    ax.set_xticks(x)
    ax.set_xticklabels([RQ1_LABEL[s] for s in RQ1_SCEN])
    ax.set_ylabel("Cost per day (CNY)")
    ax.set_title("RQ1: Cost/day across 4 operating scenarios (mean ± 95% CI, n=5)")
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.legend(ncol=3, frameon=False, fontsize=8, loc="upper left")
    fig.tight_layout()
    p = FIG / "fig01_rq1_cost.png"
    fig.savefig(p); plt.close(fig)
    return p


def fig_rq1_violation(data: dict) -> Path:
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    x = np.arange(len(RQ1_SCEN))
    w = 0.13
    for i, alg in enumerate(ALG_ORDER):
        means, errs = [], []
        for sc in RQ1_SCEN:
            vals = [r["violation_rate"] for r in data[alg][sc]]
            m, h, _ = ci95(vals)
            means.append(m); errs.append(h)
        ax.bar(x + (i - 2.5) * w, means, w, yerr=errs, capsize=2.5,
               label=ALG_LABEL[alg], color=ALG_COLOR[alg],
               edgecolor="black", linewidth=0.4)
    ax.axhline(5.0, color="red", linestyle=":", linewidth=1, alpha=0.7)
    ax.text(3.4, 5.5, "5% safety threshold", color="red", fontsize=8, ha="right")
    ax.set_xticks(x)
    ax.set_xticklabels([RQ1_LABEL[s] for s in RQ1_SCEN])
    ax.set_ylabel("Hard constraint violation rate (%)")
    ax.set_title("RQ1: Violation rate across scenarios (mean ± 95% CI, n=5)")
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.legend(ncol=3, frameon=False, fontsize=8, loc="upper right")
    fig.tight_layout()
    p = FIG / "fig02_rq1_violation.png"
    fig.savefig(p); plt.close(fig)
    return p


def fig_pareto(data: dict) -> Path:
    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    markers = {"S1_grid_normal": "o", "S2_grid_extreme": "s",
               "S3_island_normal": "^", "S4_island_extreme": "D"}
    for alg in ALG_ORDER:
        for sc in RQ1_SCEN:
            runs = data[alg][sc]
            if not runs:
                continue
            cm = np.mean([r["cost_per_day"] for r in runs])
            vm = np.mean([r["violation_rate"] for r in runs])
            ax.scatter(cm, vm, s=90 if alg == "hfg_sac" else 60,
                       color=ALG_COLOR[alg], marker=markers[sc],
                       edgecolor="black", linewidth=0.5,
                       label=ALG_LABEL[alg] if sc == RQ1_SCEN[0] else None,
                       zorder=3 if alg == "hfg_sac" else 2)
    ax.axhline(5.0, color="red", linestyle=":", linewidth=1, alpha=0.6)
    ax.set_xlabel("Cost per day (CNY)")
    ax.set_ylabel("Violation rate (%)")
    ax.set_title("RQ1: Cost–safety Pareto (marker shape = scenario)")
    ax.grid(alpha=0.3, linestyle="--")
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    # annotate HFG points
    for sc in RQ1_SCEN:
        runs = data["hfg_sac"][sc]
        cm = np.mean([r["cost_per_day"] for r in runs])
        vm = np.mean([r["violation_rate"] for r in runs])
        ax.annotate(sc.split("_")[0], (cm, vm),
                    textcoords="offset points", xytext=(7, 6), fontsize=8,
                    color="#d32f2f", weight="bold")
    fig.tight_layout()
    p = FIG / "fig03_rq1_pareto.png"
    fig.savefig(p); plt.close(fig)
    return p


def fig_rq2_curves(data: dict) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    plot_algs = ["sac", "ppo_lagrangian", "cpo", "safety_layer", "hfg_sac"]
    for ax, metric, ylabel, title in [
        (axes[0], "cost_per_day", "Cost per day (CNY)", "RQ2: Cost vs demand extremity"),
        (axes[1], "violation_rate", "Violation rate (%)", "RQ2: Violation vs demand extremity"),
    ]:
        for alg in plot_algs:
            means, his = [], []
            for sc in RQ2_SCEN:
                # violation_rate is already in percent units
                vals = [r[metric] for r in data[alg][sc]]
                m, h, _ = ci95(vals)
                means.append(m); his.append(h)
            lw = 2.2 if alg == "hfg_sac" else 1.3
            ax.plot(RQ2_LEVELS, means, marker="o", label=ALG_LABEL[alg],
                    color=ALG_COLOR[alg], linewidth=lw)
            ax.fill_between(RQ2_LEVELS,
                            np.array(means) - np.array(his),
                            np.array(means) + np.array(his),
                            alpha=0.15, color=ALG_COLOR[alg])
        ax.set_xlabel("Extremity level λ")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(alpha=0.3, linestyle="--")
        ax.legend(frameon=False, fontsize=8)
    axes[1].axhline(5.0, color="red", linestyle=":", linewidth=1, alpha=0.6)
    fig.tight_layout()
    p = FIG / "fig04_rq2_curves.png"
    fig.savefig(p); plt.close(fig)
    return p


def fig_rq2_fcsd_drift(data: dict) -> Path:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    plot_algs = ["sac", "ppo_lagrangian", "cpo", "safety_layer", "hfg_sac"]
    x = np.arange(len(plot_algs))
    w = 0.35
    f_lo, f_hi = [], []
    for alg in plot_algs:
        f_lo.append(np.mean([r["avg_fcsd"] for r in data[alg]["extremity_0.8"]]))
        f_hi.append(np.mean([r["avg_fcsd"] for r in data[alg]["extremity_1.5"]]))
    ax.bar(x - w/2, f_lo, w, label="λ=0.8 (mild)", color="#64b5f6", edgecolor="black", linewidth=0.4)
    ax.bar(x + w/2, f_hi, w, label="λ=1.5 (extreme)", color="#d32f2f", edgecolor="black", linewidth=0.4)
    for i, (lo, hi) in enumerate(zip(f_lo, f_hi)):
        ax.text(i - w/2, lo + 0.005, f"{lo:.3f}", ha="center", fontsize=8)
        ax.text(i + w/2, hi + 0.005, f"{hi:.3f}", ha="center", fontsize=8)
        ax.text(i, 0.72, f"Δ={lo-hi:+.3f}", ha="center", fontsize=8, color="#444")
    ax.set_xticks(x)
    ax.set_xticklabels([ALG_LABEL[a] for a in plot_algs])
    ax.set_ylabel("Mean FCSD")
    ax.set_ylim(0.7, 1.03)
    ax.set_title("RQ2: FCSD drift from mild (λ=0.8) to extreme (λ=1.5)")
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.legend(frameon=False)
    fig.tight_layout()
    p = FIG / "fig05_rq2_fcsd_drift.png"
    fig.savefig(p); plt.close(fig)
    return p


# ---------- main ----------
def main() -> None:
    data = load_all()
    n_runs = sum(len(data[a][s]) for a in ALG_ORDER for s in RQ1_SCEN + RQ2_SCEN)
    print(f"Loaded {n_runs} runs across {len(ALG_ORDER)} algorithms.")

    desc1 = descriptive_md(data, RQ1_SCEN, "RQ1")
    desc2 = descriptive_md(data, RQ2_SCEN, "RQ2")
    inf1 = inferential_md(data, RQ1_SCEN, "RQ1")
    inf2 = inferential_md(data, RQ2_SCEN, "RQ2")

    p1 = fig_rq1_cost(data)
    p2 = fig_rq1_violation(data)
    p3 = fig_pareto(data)
    p4 = fig_rq2_curves(data)
    p5 = fig_rq2_fcsd_drift(data)
    print(f"Figures: {[p.name for p in [p1,p2,p3,p4,p5]]}")

    (OUT / "stats-appendix.md").write_text(
        "# Stats Appendix — RQ1 & RQ2\n\n"
        f"- n per cell: 5 seeds (42, 123, 456, 789, 2024)\n"
        f"- Cost/day: CNY, lower better\n"
        f"- Violation rate: %, lower better\n"
        f"- FCSD: fuzzy constraint satisfaction degree, higher better\n"
        f"- CI: 95% t-CI, df = 4\n"
        f"- Test: Welch's two-sample t-test (unequal variance)\n"
        f"- Effect size: Cohen's d with pooled SD\n\n"
        f"{desc1}\n\n{desc2}\n\n{inf1}\n\n{inf2}\n",
        encoding="utf-8",
    )
    print(f"Wrote {OUT / 'stats-appendix.md'}")


if __name__ == "__main__":
    main()
