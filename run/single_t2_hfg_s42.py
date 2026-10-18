"""Single T2 hfg_transfer (M1+M3+M4) seed=42, 200 ep — perf probe.

Run: HFG_NUM_THREADS=4 uv run python run/single_t2_hfg_s42.py
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

t0 = time.time()
print(f"[probe] HFG_NUM_THREADS={os.environ['HFG_NUM_THREADS']}")
print("[probe] T2 hfg_transfer seed=42 n_ep=200 start...", flush=True)

r = _run_transfer(
    SOURCE, TARGET, 42,
    n_episodes=200, device="cpu", output_dir=OUT_DIR,
    conservative=True, use_M1=True, use_M3=True, use_M4=True,
)

elapsed = time.time() - t0
out = {
    "arm": "hfg_transfer",
    "task": "T2_grid_shifted",
    "seed": 42,
    "cost_per_day": round(float(r.cost_per_day), 2),
    "violation_pct": round(float(r.violation_rate), 2),
    "min_fcsd": round(float(r.min_fcsd), 3),
    "convergence_ep": int(r.convergence_episode),
    "training_time_s": round(float(r.training_time_s), 1),
    "wall_clock_s": round(elapsed, 1),
    "hfg_num_threads": int(os.environ["HFG_NUM_THREADS"]),
}
print(json.dumps(out, indent=2), flush=True)
with open(OUT_DIR / "single_t2_hfg_s42.json", "w", encoding="utf-8") as f:
    json.dump(out, f, indent=2)
print(f"[probe] done in {elapsed:.1f}s ({elapsed/60:.1f} min)", flush=True)
