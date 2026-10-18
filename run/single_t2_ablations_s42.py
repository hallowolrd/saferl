"""RQ3 T2 ablation probe: no_M1 and no_M4, seed=42, 200 ep.

Compares against full hfg_transfer baseline:
  cost_per_day=65209.92, violation=0.0%, min_fcsd=0.994, convergence_ep=174

Run: HFG_NUM_THREADS=4 uv run python run/single_t2_ablations_s42.py
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
    "env_mode": "grid_connected",
    "scenario_name": "typical_week_summer",
    "extreme_multiplier": None,
}
TARGET = {
    "env_mode": "grid_connected",
    "scenario_name": "summer_extreme",
    "extreme_multiplier": {"pv": 1.2, "load": 1.3, "wt": 0.6},
    "soc_optimal_min": 0.4,
    "soc_optimal_max": 0.7,
}

OUT_DIR = REPO_ROOT / "outputs" / "experiments" / "single_t2_probe"
OUT_DIR.mkdir(parents=True, exist_ok=True)

ARMS = [
    ("no_M1", {"conservative": True, "use_M1": False, "use_M3": True, "use_M4": True}),
    ("no_M4", {"conservative": True, "use_M1": True, "use_M3": True, "use_M4": False}),
]

results = []
for tag, kwargs in ARMS:
    print(f"\n{'='*60}")
    print(f"[probe] T2 {tag} seed=42 n_ep=200 start...")
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
        "task": "T2_grid_shifted",
        "seed": 42,
        "cost_per_day": round(float(r.cost_per_day), 2),
        "violation_pct": round(float(r.violation_rate), 2),
        "min_fcsd": round(float(r.min_fcsd), 3),
        "convergence_ep": int(r.convergence_episode),
        "training_time_s": round(float(r.training_time_s), 1),
        "wall_clock_s": round(elapsed, 1),
    }
    print(json.dumps(row, indent=2), flush=True)
    results.append(row)

print(f"\n{'='*60}")
print("ABLATION PROBE SUMMARY (T2, seed=42)")
print(f"{'='*60}")
print(f"{'arm':<30} {'cost/d':>10} {'viol%':>7} {'min_fcsd':>9} {'conv_ep':>8} {'wall_min':>9}")
print("-" * 80)
print(f"{'full_hfg_transfer':<30} {'65209.92':>10} {'0.00':>7} {'0.994':>9} {'174':>8} {'20.0':>9}")
for r in results:
    print(f"{r['arm']:<30} {r['cost_per_day']:>10.2f} {r['violation_pct']:>7.2f} "
          f"{r['min_fcsd']:>9.3f} {r['convergence_ep']:>8d} {r['wall_clock_s']/60:>9.1f}")

with open(OUT_DIR / "t2_ablations_s42.json", "w", encoding="utf-8") as f:
    json.dump({"baseline_full": {"cost_per_day": 65209.92, "violation_pct": 0.0,
                                  "min_fcsd": 0.994, "convergence_ep": 174},
               "ablations": results}, f, indent=2)
print(f"\nSaved: {OUT_DIR / 't2_ablations_s42.json'}")
