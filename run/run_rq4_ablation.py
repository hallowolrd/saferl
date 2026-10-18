"""Run full RQ4 ablation study: 4 variants x 5 seeds x 4 scenarios.

Variants:
  - hfg_sac_full           (both layers)
  - hfg_sac_no_fuzzycon    (no fuzzy constraint)
  - hfg_sac_no_fuzzyknow   (no fuzzy knowledge / shaping)
  - hfg_sac_baseline       (plain SAC, no fuzzy layers)

Scenarios: S1_grid_normal, S2_grid_extreme, S3_island_normal, S4_island_extreme.

Outputs per-run JSON to outputs/experiments/results/rq4_ablation/.
"""
from __future__ import annotations

import json
import logging
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

if sys.platform == "win32":
    mp.set_start_method("spawn", force=True)

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "run"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("rq4_ablation")

OUTPUT_DIR = ROOT / "outputs" / "experiments" / "results" / "rq4_ablation"
SEEDS = [42, 123, 456, 789, 2024]
VARIANTS = [
    "hfg_sac_full",
    "hfg_sac_no_fuzzycon",
    "hfg_sac_no_fuzzyknow",
    "hfg_sac_baseline",
]
SCENARIOS_TO_RUN = [
    "S1_grid_normal",
    "S2_grid_extreme",
    "S3_island_normal",
    "S4_island_extreme",
]
N_EPISODES = 200
N_WORKERS = 4


def run_one(args):
    variant, scenario_name, seed = args
    os.environ["HFG_NUM_THREADS"] = "4"

    from experiment_runner import train_agent, SCENARIOS

    scenario_cfg = SCENARIOS[scenario_name]

    t0 = time.time()
    r = train_agent(
        variant, scenario_name, scenario_cfg, seed,
        n_episodes=N_EPISODES, device="cpu", output_dir=None,
    )
    elapsed = time.time() - t0

    result = {
        "algorithm": variant,
        "scenario": scenario_name,
        "seed": seed,
        "total_cost": r.total_cost,
        "cost_per_day": r.cost_per_day,
        "violation_rate": r.violation_rate,
        "avg_fcsd": r.avg_fcsd,
        "min_fcsd": r.min_fcsd,
        "convergence_episode": r.convergence_episode,
        "training_time_s": r.training_time_s,
        "final_reward": r.final_reward,
        "episode_rewards": r.episode_rewards,
        "episode_costs": r.episode_costs,
        "episode_violation_rates": r.episode_violation_rates,
        "episode_fcsds": r.episode_fcsds,
    }

    tag = scenario_name.split("_")[0]  # S1/S2/S3/S4
    out_file = OUTPUT_DIR / f"{variant}_{tag}_s{seed}.json"
    with open(out_file, "w") as f:
        json.dump(result, f, indent=2)

    logger.info(
        f"DONE {variant} {scenario_name} seed={seed}: cost/day={r.cost_per_day:.0f}  "
        f"viol={r.violation_rate * 100:.2f}%  FCSD={r.avg_fcsd:.4f}  "
        f"time={elapsed / 60:.1f}min"
    )
    return variant, scenario_name, seed


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    tasks = [
        (v, s, seed)
        for v in VARIANTS
        for s in SCENARIOS_TO_RUN
        for seed in SEEDS
    ]
    logger.info(
        f"RQ4 Ablation: {len(tasks)} runs ({len(VARIANTS)} variants x "
        f"{len(SCENARIOS_TO_RUN)} scenarios x {len(SEEDS)} seeds), "
        f"{N_EPISODES} ep each, {N_WORKERS} parallel workers"
    )

    start = time.time()
    with mp.Pool(N_WORKERS) as pool:
        for _ in pool.imap_unordered(run_one, tasks):
            pass
    total = time.time() - start

    logger.info(f"All done in {total / 60:.1f} min. Results in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
