"""Strict RQ4 ablation analysis: descriptive stats + inferential tests + figures.

Outputs:
  analysis/rq4-w150-output/summary_table.csv
  analysis/rq4-w150-output/stats.json
  analysis/rq4-w150-output/contrasts.csv
  analysis/rq4-w150-output/figures/figure-01-main-comparison.pdf
  analysis/rq4-w150-output/figures/figure-02-ablation-pairs.pdf
  analysis/rq4-w150-output/figures/figure-03-training-curves.pdf
"""
from __future__ import annotations

import json
import statistics as st
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "outputs" / "experiments" / "results" / "rq4_ablation"
OUT = ROOT / "analysis" / "rq4-w150-output"
FIG = OUT / "figures"
OUT.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)

SEEDS = [42, 123, 456, 789, 2024]
VARIANTS = [
    "hfg_sac_baseline",
    "hfg_sac_no_fuzzyknow",
    "hfg_sac_no_fuzzycon",
    "hfg_sac_full",
]
V_LABEL = {
    "hfg_sac_baseline": "Baseline (SAC)",
    "hfg_sac_no_fuzzyknow": "SAC + FuzzyConstraint",
    "hfg_sac_no_fuzzycon": "SAC + FuzzyKnowledge",
    "hfg_sac_full": "HFG-SAC (full)",
}
SCENARIOS = [
    "S1_grid_normal",
    "S2_grid_extreme",
    "S3_island_normal",
    "S4_island_extreme",
]
S_LABEL = {
    "S1_grid_normal": "S1 grid-normal",
    "S2_grid_extreme": "S2 grid-extreme",
    "S3_island_normal": "S3 island-normal",
    "S4_island_extreme": "S4 island-extreme",
}


def load(variant: str, scenario: str) -> list[dict]:
    tag = scenario.split("_")[0]
    rows = []
    for seed in SEEDS:
        f = RESULTS / f"{variant}_{tag}_s{seed}.json"
        rows.append(json.loads(f.read_text()))
    return rows


# ---------- 1. Build long-form dataframe ----------
records = []
for sc in SCENARIOS:
    for v in VARIANTS:
        for seed, r in zip(SEEDS, load(v, sc)):
            records.append({
                "variant": v,
                "variant_label": V_LABEL[v],
                "scenario": sc,
                "scenario_label": S_LABEL[sc],
                "seed": seed,
                "cost_per_day": r["cost_per_day"],
                "violation_pct": r["violation_rate"],  # already in 0-100 units
                "avg_fcsd": r["avg_fcsd"],
            })
df = pd.DataFrame(records)
df.to_csv(OUT / "long_table.csv", index=False)


# ---------- 2. Descriptive summary ----------
def desc(vals):
    a = np.array(vals, dtype=float)
    n = len(a)
    m = a.mean()
    s = a.std(ddof=1) if n > 1 else 0.0
    se = s / np.sqrt(n) if n > 1 else 0.0
    ci95 = 1.96 * se  # n=5 small -> use t; keep t for correctness
    if n > 1:
        ci95 = stats.t.ppf(0.975, df=n - 1) * se
    return n, m, s, se, ci95


summary_rows = []
for sc in SCENARIOS:
    for v in VARIANTS:
        sub = df[(df.scenario == sc) & (df.variant == v)]
        for metric, key in [
            ("cost_per_day", "cost_per_day"),
            ("violation_pct", "violation_pct"),
            ("avg_fcsd", "avg_fcsd"),
        ]:
            n, m, s, se, ci = desc(sub[key].values)
            summary_rows.append({
                "scenario": sc, "variant": v, "metric": metric,
                "n": n, "mean": m, "std": s, "se": se, "ci95": ci,
            })
summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(OUT / "summary_table.csv", index=False)


# ---------- 3. Inferential tests (paired by seed) ----------
# Primary contrasts per scenario:
#   C1: full vs baseline            -> total HFG effect
#   C2: full vs no_fuzzycon        -> +fuzzy_constraint (knowledge held constant)
#   C3: full vs no_fuzzyknow       -> +fuzzy_knowledge (constraint held constant)
#   C4: no_fuzzycon vs baseline    -> fuzzy_knowledge alone
#   C5: no_fuzzyknow vs baseline   -> fuzzy_constraint alone
CONTRASTS = [
    ("full_vs_baseline",       "hfg_sac_full",         "hfg_sac_baseline"),
    ("constraint_ablation",    "hfg_sac_full",         "hfg_sac_no_fuzzycon"),
    ("knowledge_ablation",    "hfg_sac_full",         "hfg_sac_no_fuzzyknow"),
    ("knowledge_vs_baseline",  "hfg_sac_no_fuzzycon",  "hfg_sac_baseline"),
    ("constraint_vs_baseline", "hfg_sac_no_fuzzyknow", "hfg_sac_baseline"),
]


def cohens_d_paired(a, b):
    """Paired Cohen's d (z / sd_z) where z = a - b."""
    z = np.asarray(a) - np.asarray(b)
    return float(z.mean() / z.std(ddof=1)) if z.std(ddof=1) > 0 else 0.0


contrast_rows = []
for sc in SCENARIOS:
    for cname, vA, vB in CONTRASTS:
        a_df = df[(df.scenario == sc) & (df.variant == vA)].sort_values("seed")
        b_df = df[(df.scenario == sc) & (df.variant == vB)].sort_values("seed")
        for metric in ["cost_per_day", "violation_pct", "avg_fcsd"]:
            a = a_df[metric].values.astype(float)
            b = b_df[metric].values.astype(float)
            diff = a - b
            n = len(a)
            # Paired t
            tstat, p_t = stats.ttest_rel(a, b) if n > 1 else (np.nan, np.nan)
            # Wilcoxon signed-rank
            try:
                wstat, p_w = stats.wilcoxon(a, b, zero_method="wilcox")
            except ValueError:
                wstat, p_w = np.nan, np.nan
            d = cohens_d_paired(a, b)
            contrast_rows.append({
                "scenario": sc,
                "contrast": cname,
                "variant_A": vA, "variant_B": vB,
                "metric": metric, "n": n,
                "mean_A": a.mean(), "mean_B": b.mean(),
                "mean_diff": diff.mean(),
                "t": tstat, "p_t": p_t,
                "wilcoxon_W": wstat, "p_wilcoxon": p_w,
                "cohens_d": d,
            })
contrasts_df = pd.DataFrame(contrast_rows)

# Multiple-comparison correction: BH-FDR per (scenario, metric) family
def bh_fdr(pvals):
    """Benjamini-Hochberg FDR. Returns adjusted p-values."""
    p = np.asarray(pvals, dtype=float)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order]
    adj = ranked * n / (np.arange(n) + 1)
    # enforce monotonicity (from the largest down)
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    adj = np.clip(adj, 0, 1)
    out = np.empty(n)
    out[order] = adj
    return out


def bh(group):
    mask = group["p_t"].notna()
    p = group.loc[mask, "p_t"].values
    if len(p) == 0:
        group.loc[mask, "p_t_bh"] = np.nan
        group.loc[mask, "sig_bh"] = False
        return group
    p_bh = bh_fdr(p)
    group.loc[mask, "p_t_bh"] = p_bh
    group.loc[mask, "sig_bh"] = p_bh < 0.05
    return group

contrasts_df = contrasts_df.groupby(["scenario", "metric"], group_keys=False)[contrasts_df.columns].apply(bh)
contrasts_df = contrasts_df.reset_index(drop=True)
contrasts_df.to_csv(OUT / "contrasts.csv", index=False)


# ---------- 4. Figures ----------
plt.rcParams.update({
    "font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10,
    "legend.fontsize": 8, "pdf.fonttype": 42, "ps.fonttype": 42,
})

PALETTE = {
    "hfg_sac_baseline":       "#9e9e9e",
    "hfg_sac_no_fuzzyknow":   "#2c7fb8",
    "hfg_sac_no_fuzzycon":    "#fdae61",
    "hfg_sac_full":           "#d7301f",
}

# --- Figure 1: main comparison (cost + violation), grouped by scenario ---
fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
x = np.arange(len(SCENARIOS))
width = 0.2
for i, v in enumerate(VARIANTS):
    means_c = [df[(df.scenario == sc) & (df.variant == v)]["cost_per_day"].mean() for sc in SCENARIOS]
    sds_c = [df[(df.scenario == sc) & (df.variant == v)]["cost_per_day"].std(ddof=1) for sc in SCENARIOS]
    axes[0].bar(x + (i - 1.5) * width, means_c, width, yerr=sds_c,
                label=V_LABEL[v], color=PALETTE[v], capsize=2, edgecolor="black", linewidth=0.5)
axes[0].set_xticks(x); axes[0].set_xticklabels([S_LABEL[s] for s in SCENARIOS], rotation=15)
axes[0].set_ylabel("Operating cost (RMB / day)")
axes[0].set_title("(a) Cost per day  (mean ± std, n=5 seeds)")
axes[0].legend(loc="upper right", frameon=False)

for i, v in enumerate(VARIANTS):
    means_v = [df[(df.scenario == sc) & (df.variant == v)]["violation_pct"].mean() for sc in SCENARIOS]
    sds_v = [df[(df.scenario == sc) & (df.variant == v)]["violation_pct"].std(ddof=1) for sc in SCENARIOS]
    axes[1].bar(x + (i - 1.5) * width, means_v, width, yerr=sds_v,
                color=PALETTE[v], capsize=2, edgecolor="black", linewidth=0.5)
axes[1].set_xticks(x); axes[1].set_xticklabels([S_LABEL[s] for s in SCENARIOS], rotation=15)
axes[1].set_ylabel("Violation rate (%)")
axes[1].set_yscale("symlog", linthresh=1.0)
axes[1].set_title("(b) Violation rate  (sym-log scale)")
fig.tight_layout()
fig.savefig(FIG / "figure-01-main-comparison.pdf")
fig.savefig(FIG / "figure-01-main-comparison.png", dpi=160)
plt.close(fig)

# --- Figure 2: ablation pairs on S3 + S4 island (the only scenarios where violation > 0) ---
fig, axes = plt.subplots(1, 2, figsize=(9, 3.8), sharey=False)
for ax, sc in zip(axes, ["S3_island_normal", "S4_island_extreme"]):
    v_order = ["hfg_sac_baseline", "hfg_sac_no_fuzzycon", "hfg_sac_no_fuzzyknow", "hfg_sac_full"]
    means_v = []; sds_v = []
    for v in v_order:
        sub = df[(df.scenario == sc) & (df.variant == v)]["violation_pct"]
        means_v.append(sub.mean()); sds_v.append(sub.std(ddof=1))
    bars = ax.bar([V_LABEL[v].replace(" (", "\n(") for v in v_order],
                  means_v, yerr=sds_v, capsize=3,
                  color=[PALETTE[v] for v in v_order], edgecolor="black", linewidth=0.5)
    ax.set_ylabel("Violation rate (%)")
    ax.set_title(S_LABEL[sc])
    ax.tick_params(axis="x", labelrotation=20)
fig.suptitle("Ablation effect on safety (islanded scenarios)")
fig.tight_layout()
fig.savefig(FIG / "figure-02-ablation-pairs.pdf")
fig.savefig(FIG / "figure-02-ablation-pairs.png", dpi=160)
plt.close(fig)

# --- Figure 3: training curves (reward + violation) on S4_island_extreme, all variants ---
fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
for v in VARIANTS:
    curves_r = []
    curves_v = []
    for seed in SEEDS:
        tag = "S4"
        f = RESULTS / f"{v}_{tag}_s{seed}.json"
        d = json.loads(f.read_text())
        curves_r.append(d["episode_rewards"])
        curves_v.append(d["episode_violation_rates"])
    L = min(len(c) for c in curves_r)
    R = np.array([c[:L] for c in curves_r])
    V = np.array([c[:L] for c in curves_v])
    ep = np.arange(1, L + 1)
    axes[0].plot(ep, R.mean(0), color=PALETTE[v], label=V_LABEL[v], linewidth=1.4)
    axes[0].fill_between(ep, R.mean(0) - R.std(0), R.mean(0) + R.std(0),
                         color=PALETTE[v], alpha=0.15)
    axes[1].plot(ep, V.mean(0), color=PALETTE[v], linewidth=1.4)
    axes[1].fill_between(ep,
                         np.clip(V.mean(0) - V.std(0), 0, None),
                         (V.mean(0) + V.std(0)),
                         color=PALETTE[v], alpha=0.15)
axes[0].set_xlabel("Episode"); axes[0].set_ylabel("Episode reward")
axes[0].set_title("(a) Learning curve (S4 island-extreme)")
axes[0].legend(loc="lower right", frameon=False)
axes[1].set_xlabel("Episode"); axes[1].set_ylabel("Violation rate (%)")
axes[1].set_title("(b) Violation rate over training")
fig.tight_layout()
fig.savefig(FIG / "figure-03-training-curves.pdf")
fig.savefig(FIG / "figure-03-training-curves.png", dpi=160)
plt.close(fig)


# ---------- 5. Console digest ----------
print("== Summary (mean ± std, n=5) ==")
piv = df.groupby(["scenario", "variant"])[["cost_per_day", "violation_pct", "avg_fcsd"]].agg(["mean", "std"])
print(piv.round(2))

print("\n== Key contrasts (BH-FDR adjusted p) ==")
focus = contrasts_df[
    (contrasts_df["metric"] == "violation_pct") &
    (contrasts_df["scenario"].isin(["S3_island_normal", "S4_island_extreme"]))
]
print(focus[["scenario", "contrast", "mean_A", "mean_B", "mean_diff",
            "t", "p_t", "p_t_bh", "cohens_d"]].round(3).to_string(index=False))

print(f"\nArtifacts written to: {OUT}")
