"""RQ3 T4 ablation probe: no_M1 and no_M4, seed=42, 200 ep.

T4: islanded normal -> islanded extreme, band narrowed [0.3,0.8] -> [0.4,0.7].
Tests M4 conservative scaling when safety band shifts (M1 soc sim ~0.6-0.8).

Run: HFG_NUM_THREADS=4 uv run python run/single_t4_ablations_s42.py
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

from experiment_runner import _run_transfer  # noqa: E402

SOURCE = {
    "env_mode": "islanded",
    "scenario_name": "typical_week_summer",
    "extreme_multiplier": None,
    "interruptible_load_kw": 500.0,
    "freq_penalty_slope": 1500.0,
    "freq_volt_shield_safety_ratio": 0.15,
    "load_shed_cost": 1.0,
}
TARGET = {
    "env_mode": "islanded",
    "scenario_name": "summer_extreme",
    "extreme_multiplier": {"pv": 0.5, "load": 1.25, "wt": 0.8},
    "interruptible_load_kw": 500.0,
    "freq_penalty_slope": 1500.0,
    "freq_volt_shield_safety_ratio": 0.15,
    "load_shed_cost": 1.0,
    "soc_optimal_min": 0.4,
    "soc_optimal_max": 0.7,
}

OUT_DIR = REPO_ROOT / "outputs" / "experiments" / "single_t4_probe"
OUT_DIR.mkdir(parents=True, exist_ok=True)

ARMS = [
    ("full",   {"conservative": True,  "use_M1": True,  "use_M3": True, "use_M4": True}),
    ("no_M1",  {"conservative": True,  "use_M1": False, "use_M3": True, "use_M4": True}),
    ("no_M4",  {"conservative": True,  "use_M1": True,  "use_M3": True, "use_M4": False}),
]

results = []
for tag, kwargs in ARMS:
    print(f"\n{'='*60}")
    print(f"[probe] T4 {tag} seed=42 n_ep=200 start...")
    print(f"{'='*60}", flush=True)
    t0 = time.time()
    r = _run_transfer(
        SOURCE, TARGET, 42,
        n_episodes=200, device="cpu", output_dir=OUT_DIR,
        **kwargs,
    )
    elapsed = time.time() - t0
    row = {
        "arm": f"hfg_transfer_{tag}",
        "task": "T4_island_shifted",
        "seed": 42,
        "cost_per_day": round(float(r.cost_per_day), 2),
        "violation_pct": round(float(r.violation_rate), 2),
        "soft_violation_pct": round(float(r.soft_violation_rate), 2),
        "min_fcsd": round(float(r.min_fcsd), 3),
        "convergence_ep": int(r.convergence_episode),
        "training_time_s": round(float(r.training_time_s), 1),
        "wall_clock_s": round(elapsed, 1),
    }
    print(json.dumps(row, indent=2), flush=True)
    results.append(row)

print(f"\n{'='*60}")
print("ABLATION PROBE SUMMARY (T4 island shifted, seed=42)")
print(f"{'='*60}")
print(f"{'arm':<30} {'cost/d':>10} {'hard%':>7} {'soft%':>7} {'min_fcsd':>9} {'conv_ep':>8} {'wall_min':>9}")
print("-" * 88)
for r in results:
    print(f"{r['arm']:<30} {r['cost_per_day']:>10.2f} {r['violation_pct']:>7.2f} "
          f"{r['soft_violation_pct']:>7.2f} {r['min_fcsd']:>9.3f} {r['convergence_ep']:>8d} "
          f"{r['wall_clock_s']/60:>9.1f}")

with open(OUT_DIR / "t4_ablations_s42.json", "w", encoding="utf-8") as f:
    json.dump({"ablations": results}, f, indent=2)
print(f"\nSaved: {OUT_DIR / 't4_ablations_s42.json'}")
