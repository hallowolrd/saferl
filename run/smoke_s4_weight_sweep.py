"""Quick S4 smoke: weight sweep for economic-only shaping.

Tests weight_init=150 vs current 300 on S4_island_extreme, hfg_sac_full only,
100 episodes, seed=42. Goal: find weight where full violation <= no_fuzzyknow (2.80%).
"""
from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "run"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger("s4_smoke")

from experiment_runner import SCENARIOS, train_agent, FuzzyConfig  # noqa


def run_with_weight(weight_init: float, weight_final: float) -> dict:
    """Run hfg_sac_full on S4 with custom shaping weight."""
    # Temporarily patch FuzzyConfig defaults
    import run.experiment_runner as er

    scenario_name = "S4_island_extreme"
    scenario_cfg = SCENARIOS[scenario_name]
    seed = 42

    t0 = time.time()
    result = train_agent(
        algorithm_name="hfg_sac_full",
        scenario_name=scenario_name,
        scenario_cfg=scenario_cfg,
        seed=seed,
        n_episodes=100,
        device="cpu",
        output_dir=None,
    )
    dt = time.time() - t0
    return {
        "weight_init": weight_init,
        "cost_per_day": result.cost_per_day,
        "violation_rate": result.violation_rate,
        "avg_fcsd": result.avg_fcsd,
        "time_s": dt,
    }


# We need to actually modify the config. Let's do it differently.
# The train_agent reads hardcoded values. Instead, let's just run with
# current config (weight=300) and compare to no_fuzzyknow already known.
# Then manually change weight and re-run.

if __name__ == "__main__":
    print("This script needs manual weight override.")
    print("Run directly: python -c ...")
