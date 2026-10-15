"""Smoke test: T2 from_scratch, 50 episodes, seed=42.

Verify update_skip=16 speedup and basic policy quality.
Run: D:/ProgramData/Anacondaenv/envs/lerobot/python run/smoke_t2_scratch.py
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

os.environ.setdefault("HFG_NUM_THREADS", str(max(1, (os.cpu_count() or 4) // 2)))

from experiment_runner import train_agent  # noqa: E402

TARGET = {
    "env_mode": "grid_connected",
    "scenario_name": "summer_extreme",
    "extreme_multiplier": {"pv": 1.2, "load": 1.3, "wt": 0.6},
    "soc_optimal_min": 0.4,
    "soc_optimal_max": 0.7,
}

SEED = 42
N_EPISODES = 50  # smoke test only
DEVICE = "cpu"
OUT_DIR = REPO_ROOT / "outputs" / "experiments" / "smoke_t2"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def main() -> None:
    print(f"[Smoke] T2 from_scratch, seed={SEED}, n_ep={N_EPISODES}")
    t0 = time.time()
    r = train_agent(
        "hfg_sac", "smoke_T2_scratch", TARGET, SEED,
        n_episodes=N_EPISODES, device=DEVICE, output_dir=OUT_DIR,
    )
    elapsed = time.time() - t0

    summary = {
        "cost_per_day": round(float(r.cost_per_day), 2),
        "violation_pct": round(float(r.violation_rate), 2),
        "min_fcsd": round(float(r.min_fcsd), 3),
        "avg_fcsd": round(float(r.avg_fcsd), 3),
        "convergence_ep": int(r.convergence_episode),
        "train_time_s": round(elapsed, 1),
        "time_per_episode": round(elapsed / N_EPISODES, 2),
    }
    print("\n" + "=" * 50)
    print("SMOKE RESULT")
    print("=" * 50)
    print(json.dumps(summary, indent=2))

    out_json = OUT_DIR / "smoke_summary.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved: {out_json}")


if __name__ == "__main__":
    main()
