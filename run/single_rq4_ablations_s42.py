"""RQ4 S4 ablation: full / no_fuzzycon / no_fuzzyknow / baseline, seed=42, 200 ep.

Runs on S4_island_extreme with new config (500 kW, fcsd_target=0.90).

Run: HFG_NUM_THREADS=4 uv run python run/single_rq4_ablations_s42.py
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "run"))

os.environ.setdefault("HFG_NUM_THREADS", "4")

from experiment_runner import train_agent, SCENARIOS  # noqa: E402

SEED = 42
N_EPISODES = 200
DEVICE = "cpu"
OUT_DIR = REPO_ROOT / "outputs" / "experiments" / "single_rq4_val"
OUT_DIR.mkdir(parents=True, exist_ok=True)

VARIANTS = [
    "hfg_sac_full",
    "hfg_sac_no_fuzzycon",
    "hfg_sac_no_fuzzyknow",
    "hfg_sac_baseline",
]

S4_CFG = SCENARIOS["S4_island_extreme"]

results = []
for algo in VARIANTS:
    tag = f"{algo}_S4_s42"
    print(f"\n[{tag}] start...", flush=True)
    t0 = time.time()
    r = train_agent(
        algo, tag, S4_CFG, SEED,
        n_episodes=N_EPISODES, device=DEVICE, output_dir=OUT_DIR,
    )
    elapsed = time.time() - t0
    row = {
        "variant": algo,
        "scenario": "S4_island_extreme",
        "seed": 42,
        "cost_per_day": round(float(r.cost_per_day), 2),
        "violation_pct": round(float(r.violation_rate), 2),
        "soft_violation_pct": round(float(r.soft_violation_rate), 2),
        "min_fcsd": round(float(r.min_fcsd), 3),
        "convergence_ep": int(r.convergence_episode),
        "wall_min": round(elapsed / 60, 1),
    }
    print(json.dumps(row, indent=2), flush=True)
    results.append(row)

print(f"\n{'='*70}")
print("RQ4 ABLATION SUMMARY (S4_island_extreme, seed=42)")
print(f"{'='*70}")
print(f"{'variant':<25} {'cost/d':>10} {'hard%':>7} {'soft%':>7} {'min_fcsd':>9} {'conv_ep':>8} {'wall_min':>9}")
print("-" * 88)
for r in results:
    print(f"{r['variant']:<25} {r['cost_per_day']:>10.2f} {r['violation_pct']:>7.2f} "
          f"{r['soft_violation_pct']:>7.2f} {r['min_fcsd']:>9.3f} {r['convergence_ep']:>8d} "
          f"{r['wall_min']:>9.1f}")

with open(OUT_DIR / "rq4_ablations_s42.json", "w", encoding="utf-8") as f:
    json.dump({"ablations": results}, f, indent=2)
print(f"\nSaved: {OUT_DIR / 'rq4_ablations_s42.json'}")
