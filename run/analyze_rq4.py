"""Aggregate RQ4 ablation results: 4 variants x 4 scenarios x 5 seeds."""
from __future__ import annotations

import json
import statistics as st
from pathlib import Path

RESULTS = Path("outputs/experiments/results/rq4_ablation")
VARIANTS = [
    "hfg_sac_baseline",
    "hfg_sac_no_fuzzyknow",
    "hfg_sac_no_fuzzycon",
    "hfg_sac_full",
]
SCENARIOS = ["S1_grid_normal", "S2_grid_extreme", "S3_island_normal", "S4_island_extreme"]


def load(variant: str, scenario: str):
    tag = scenario.split("_")[0]
    out = []
    for seed in [42, 123, 456, 789, 2024]:
        f = RESULTS / f"{variant}_{tag}_s{seed}.json"
        if not f.exists():
            return None
        out.append(json.loads(f.read_text()))
    return out


def ms(vals):
    m = st.mean(vals)
    s = st.stdev(vals) if len(vals) > 1 else 0.0
    return m, s


def main():
    print(f"{'variant':<28} {'scenario':<20} {'cost/day':>16} {'viol%':>14} {'FCSD':>12}")
    print("-" * 95)
    for sc in SCENARIOS:
        for v in VARIANTS:
            runs = load(v, sc)
            if not runs:
                continue
            costs = [r["cost_per_day"] for r in runs]
            viols = [r["violation_rate"] * 100 for r in runs]
            fcsds = [r["avg_fcsd"] for r in runs]
            cm, cs = ms(costs)
            vm, vs = ms(viols)
            fm, fs = ms(fcsds)
            print(f"{v:<28} {sc:<20} {cm:>9.0f}±{cs:<5.0f} "
                  f"{vm:>7.2f}±{vs:<5.2f} {fm:>6.3f}±{fs:<.3f}")
        print()


if __name__ == "__main__":
    main()
