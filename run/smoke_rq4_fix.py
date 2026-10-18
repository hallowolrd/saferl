"""Smoke test for RQ4 after Fix #1 (num_rules=8) and Fix #2 (grid-mode rule skip).

Runs S3_island_normal with hfg_sac_full vs hfg_sac_no_fuzzyknow, 100 episodes,
seed=42, single seed. Compares cost / violation / FCSD to previous smoke
(before fixes: full cost=6147, viol=1.49%; no_fuzzyknow cost=6332, viol=1.19%).
"""
from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

# Ensure project root on path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "run"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("smoke_rq4_fix")

from experiment_runner import SCENARIOS, train_agent  # noqa: E402


def main() -> None:
    scenario_name = "S3_island_normal"
    scenario_cfg = SCENARIOS[scenario_name]
    seed = 42
    n_episodes = 100

    out_dir = PROJECT_ROOT / "outputs" / "experiments" / "smoke_s3_postfix"
    out_dir.mkdir(parents=True, exist_ok=True)

    variants = ["hfg_sac_full", "hfg_sac_no_fuzzyknow"]
    results_summary = {}

    for var in variants:
        logger.info(f"\n>>> Running variant: {var} on {scenario_name}, seed={seed}")
        t0 = time.time()
        result = train_agent(
            algorithm_name=var,
            scenario_name=scenario_name,
            scenario_cfg=scenario_cfg,
            seed=seed,
            n_episodes=n_episodes,
            device="cpu",
            output_dir=out_dir,
        )
        dt = time.time() - t0
        results_summary[var] = {
            "total_cost": result.total_cost,
            "violation_rate": result.violation_rate,
            "avg_fcsd": result.avg_fcsd,
            "convergence_episode": result.convergence_episode,
            "runtime_seconds": dt,
        }
        logger.info(
            f"{var}: cost={result.total_cost:.1f}/day, "
            f"viol={result.violation_rate * 100:.2f}%, "
            f"FCSD={result.avg_fcsd:.4f}, "
            f"conv_ep={result.convergence_episode}, "
            f"time={dt / 60:.1f} min"
        )

    with open(out_dir / "smoke_summary.json", "w") as f:
        json.dump(results_summary, f, indent=2)

    print("\n" + "=" * 60)
    print("Smoke test summary (post-fix)")
    print("=" * 60)
    for var, r in results_summary.items():
        print(f"  {var:30s}  cost={r['total_cost']:7.1f}  "
              f"viol={r['violation_rate'] * 100:5.2f}%  "
              f"FCSD={r['avg_fcsd']:.4f}")

    # Compare to pre-fix baseline
    print("\nPre-fix baseline (S3_island, 100 ep):")
    print("  hfg_sac_full:            cost=6147.0  viol=1.49%  FCSD=0.994")
    print("  hfg_sac_no_fuzzyknow:    cost=6332.0  viol=1.19%  FCSD=0.995")


if __name__ == "__main__":
    main()
