"""Single-seed hfg_sac validation for RQ1 + RQ2 after param tuning.

RQ1: hfg_sac on S1/S2/S3/S4 (4 scenarios)
RQ2: hfg_sac on extremity 0.8/1.0/1.1/1.2/1.3/1.4/1.5 (7 levels)

Total: 11 runs x 200 ep, seed=42, cpu, 4 threads.

Run: HFG_NUM_THREADS=4 uv run python run/single_hfg_rq1_rq2_s42.py
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
OUT_DIR = REPO_ROOT / "outputs" / "experiments" / "single_hfg_val"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _extremity_cfg(level: float) -> dict:
    """Mirror _extremity_cfg in run_all_parallel.py."""
    return {
        "env_mode": "islanded",
        "scenario_name": "summer_extreme",
        "extreme_multiplier": {
            "pv": max(0.2, 1.0 - level * 0.3),
            "load": 1.0 + level * 0.2,
            "wt": 1.0 - level * 0.15,
        },
        "interruptible_load_kw": 500.0,
        "freq_penalty_slope": 1500.0,
        "freq_volt_shield_safety_ratio": 0.15,
        "load_shed_cost": 1.0,
        "fcsd_target": 0.90,
    }


def run_job(tag: str, algo: str, scenario_cfg: dict) -> dict:
    t0 = time.time()
    print(f"\n[{tag}] start...", flush=True)
    r = train_agent(
        algo, tag, scenario_cfg, SEED,
        n_episodes=N_EPISODES, device=DEVICE, output_dir=OUT_DIR,
    )
    elapsed = time.time() - t0
    row = {
        "tag": tag,
        "cost_per_day": round(float(r.cost_per_day), 2),
        "violation_pct": round(float(r.violation_rate), 2),
        "soft_violation_pct": round(float(r.soft_violation_rate), 2),
        "min_fcsd": round(float(r.min_fcsd), 3),
        "convergence_ep": int(r.convergence_episode),
        "wall_min": round(elapsed / 60, 1),
    }
    print(f"  -> {json.dumps(row)}", flush=True)
    return row


jobs = []

# RQ1: 4 scenarios
for scenario_name, scenario_cfg in SCENARIOS.items():
    jobs.append((f"hfg_sac_{scenario_name}", "hfg_sac", scenario_cfg))

# RQ2: 7 extremity levels on islanded
for level in [0.8, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]:
    jobs.append((f"hfg_sac_extremity_{level}", "hfg_sac", _extremity_cfg(level)))

print(f"Total jobs: {len(jobs)}", flush=True)

results = []
for tag, algo, cfg in jobs:
    results.append(run_job(tag, algo, cfg))

print(f"\n{'='*70}")
print("RQ1+RQ2 SINGLE-SEED VALIDATION SUMMARY (hfg_sac, seed=42)")
print(f"{'='*70}")
print(f"{'tag':<35} {'cost/d':>10} {'hard%':>7} {'soft%':>7} {'min_fcsd':>9} {'conv_ep':>8} {'wall_min':>9}")
print("-" * 88)
for r in results:
    print(f"{r['tag']:<35} {r['cost_per_day']:>10.2f} {r['violation_pct']:>7.2f} "
          f"{r['soft_violation_pct']:>7.2f} {r['min_fcsd']:>9.3f} {r['convergence_ep']:>8d} "
          f"{r['wall_min']:>9.1f}")

with open(OUT_DIR / "rq1_rq2_s42_summary.json", "w", encoding="utf-8") as f:
    json.dump({"results": results}, f, indent=2)
print(f"\nSaved: {OUT_DIR / 'rq1_rq2_s42_summary.json'}")
