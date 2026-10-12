"""RQ3 T4 pilot: island summer normal -> island summer extreme, band narrowed [0.4, 0.7].

T4 tests M4 conservative scaling on islanded mode with safety band shift
(M1 soc sim ~0.64-0.77, freq sim=1.0).
3 arms x 1 seed x 200 episodes.

Run: uv run python run/pilot_rq3_t4.py
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

ISLAND_OVERRIDES = {
    "interruptible_load_kw": 600.0,
    "freq_penalty_slope": 1500.0,
    "freq_volt_shield_safety_ratio": 0.15,
    "load_shed_cost": 1.0,
}

SOURCE = {
    "env_mode": "islanded",
    "scenario_name": "typical_week_summer",
    "extreme_multiplier": None,
    **ISLAND_OVERRIDES,
}
TARGET = {
    "env_mode": "islanded",
    "scenario_name": "summer_extreme",
    "extreme_multiplier": {"pv": 0.5, "load": 1.25, "wt": 0.8},
    **ISLAND_OVERRIDES,
    "soc_optimal_min": 0.4,
    "soc_optimal_max": 0.7,
}

SEED = 42
N_EPISODES = 200
DEVICE = "cpu"
OUT_DIR = REPO_ROOT / "outputs" / "experiments" / "pilot_rq3_t4"
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

    print("\n[A] from_scratch on T4 target...")
    t0 = time.time()
    r_a = train_agent("hfg_sac", "T4_pilot_scratch", TARGET, SEED,
                      n_episodes=N_EPISODES, device=DEVICE, output_dir=OUT_DIR)
    rows.append(summarise("from_scratch", r_a))
    print(f"    done in {time.time() - t0:.1f}s")

    print("\n[B] naive transfer...")
    t0 = time.time()
    r_b = _run_transfer(SOURCE, TARGET, SEED, n_episodes=N_EPISODES,
                        device=DEVICE, output_dir=OUT_DIR, conservative=False)
    rows.append(summarise("naive_transfer", r_b))
    print(f"    done in {time.time() - t0:.1f}s")

    print("\n[C] conservative transfer (M1+M3+M4)...")
    t0 = time.time()
    r_c = _run_transfer(SOURCE, TARGET, SEED, n_episodes=N_EPISODES,
                        device=DEVICE, output_dir=OUT_DIR,
                        conservative=True, use_M1=True, use_M3=True, use_M4=True)
    rows.append(summarise("conservative_transfer", r_c))
    print(f"    done in {time.time() - t0:.1f}s")

    print("\n" + "=" * 64)
    print("PILOT RQ3 T4 (island extreme with band [0.4,0.7], seed=42, n_ep=200)")
    print("=" * 64)
    for r in rows:
        print(json.dumps(r, indent=2))

    out_json = OUT_DIR / "pilot_summary.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({"rows": rows, "config": {
            "task": "T4_island_shifted", "seed": SEED, "n_episodes": N_EPISODES,
        }}, f, indent=2)
    print(f"\nSaved: {out_json}")


if __name__ == "__main__":
    main()
