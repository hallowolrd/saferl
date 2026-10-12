"""RQ3 T1 pilot: grid summer normal -> grid summer extreme, same battery band.

T1 tests M3 weight transfer under identical safety specs (M1 sim=1.0) with
only the weather distribution shifting. 3 arms x 1 seed x 200 episodes:

  A. from_scratch HFG-SAC on target
  B. naive transfer (M3 weight copy, no M4 conservative scaling)
  C. conservative transfer (M1+M3+M4 full pipeline)

Run: uv run python run/pilot_rq3_t1.py
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

from experiment_runner import train_agent, _run_transfer  # noqa: E402

# T1: grid normal -> grid extreme, same band [0.3, 0.8]
SOURCE = {
    "env_mode": "grid_connected",
    "scenario_name": "typical_week_summer",
    "extreme_multiplier": None,
}
TARGET = {
    "env_mode": "grid_connected",
    "scenario_name": "summer_extreme",
    "extreme_multiplier": {"pv": 1.2, "load": 1.3, "wt": 0.6},
}

SEED = 42
N_EPISODES = 200
DEVICE = "cpu"
OUT_DIR = REPO_ROOT / "outputs" / "experiments" / "pilot_rq3_t1"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def summarise(tag: str, result) -> dict:
    row = {
        "arm": tag,
        "cost_per_day": round(float(result.cost_per_day), 2),
        "violation_pct": round(float(result.violation_rate), 2),
        "min_fcsd": round(float(result.min_fcsd), 3),
        "convergence_ep": int(result.convergence_episode),
        "train_time_s": round(float(result.training_time_s), 1),
    }
    if getattr(result, "metadata", None):
        for k, v in result.metadata.items():
            if isinstance(v, dict):
                row[k] = {kk: round(vv, 3) if isinstance(vv, float) else vv
                          for kk, vv in v.items()}
            else:
                row[k] = v
    return row


def main() -> None:
    rows = []

    print("\n[A] HFG-SAC from scratch on T1 target...")
    t0 = time.time()
    r_a = train_agent(
        "hfg_sac", "T1_pilot_scratch", TARGET, SEED,
        n_episodes=N_EPISODES, device=DEVICE, output_dir=OUT_DIR,
    )
    rows.append(summarise("from_scratch", r_a))
    print(f"    done in {time.time() - t0:.1f}s")

    print("\n[B] Naive transfer (M3 weights, no M4)...")
    t0 = time.time()
    r_b = _run_transfer(
        SOURCE, TARGET, SEED,
        n_episodes=N_EPISODES, device=DEVICE, output_dir=OUT_DIR,
        conservative=False,
    )
    rows.append(summarise("naive_transfer", r_b))
    print(f"    done in {time.time() - t0:.1f}s")

    print("\n[C] Conservative transfer (M1+M3+M4 full)...")
    t0 = time.time()
    r_c = _run_transfer(
        SOURCE, TARGET, SEED,
        n_episodes=N_EPISODES, device=DEVICE, output_dir=OUT_DIR,
        conservative=True, use_M1=True, use_M3=True, use_M4=True,
    )
    rows.append(summarise("conservative_transfer", r_c))
    print(f"    done in {time.time() - t0:.1f}s")

    print("\n" + "=" * 64)
    print("PILOT RQ3 T1 SUMMARY (grid normal->extreme, seed=42, n_episodes=200)")
    print("=" * 64)
    for r in rows:
        print(json.dumps(r, indent=2))

    out_json = OUT_DIR / "pilot_summary.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({"rows": rows, "config": {
            "task": "T1_grid_same", "seed": SEED, "n_episodes": N_EPISODES,
        }}, f, indent=2)
    print(f"\nSaved: {out_json}")


if __name__ == "__main__":
    main()
