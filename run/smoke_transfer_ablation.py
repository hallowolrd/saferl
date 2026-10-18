"""Minimal transfer ablation smoke: S3_island_normal -> S4_island_extreme.

Question: does fuzzy knowledge (PBRS shaping) help transfer fine-tuning?

Arms (seed=42, 100 ep source + 100 ep target fine-tune):
  A. full HFG-SAC transfer      (knowledge + constraint, warm-start from S3)
  B. no_fuzzyknow transfer      (constraint only, no shaping, warm-start from S3)
  C. full HFG-SAC from scratch  (no transfer, 100 ep on S4)

Run: .venv/Scripts/python.exe run/smoke_transfer_ablation.py
"""
from __future__ import annotations

import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "run"))
os.environ.setdefault("HFG_NUM_THREADS", "4")

logging.basicConfig(level=logging.WARNING)

from experiment_runner import SCENARIOS, create_env_config  # noqa: E402
from hfg_srl.utils.config import HFGConfig, FuzzyConfig, AlgorithmConfig  # noqa: E402
from hfg_srl.utils.seed import set_seed  # noqa: E402
from hfg_srl.env import make_env  # noqa: E402
from hfg_srl.algorithms import make_agent  # noqa: E402
from hfg_srl.buffers.replay_buffer import ReplayBuffer  # noqa: E402

SOURCE = SCENARIOS["S3_island_normal"]
TARGET = SCENARIOS["S4_island_extreme"]
SEED = 42
N_EP = 100
DEVICE = "cpu"
STEPS_PER_EP = 96 * 7


def build_agent(variant: str, env_cfg, data_loader, scenario_cfg: dict):
    """Build agent for a given ablation variant."""
    if variant == "hfg_sac_full":
        algo_name = "hfg_sac"
        use_rules = True
        w_init = 150.0
        w_final = 8.0
    elif variant == "hfg_sac_no_fuzzyknow":
        algo_name = "hfg_sac"
        use_rules = False
        w_init = 0.0
        w_final = 0.0
    else:
        raise ValueError(f"Unknown variant: {variant}")

    fcsd_target = 0.75 if scenario_cfg.get("extreme_multiplier") else 0.85
    shield_ratio = scenario_cfg.get("freq_volt_shield_safety_ratio", 0.175)

    fuzzy_cfg = FuzzyConfig(
        num_rules=8, num_inputs=9, fcsd_target=fcsd_target,
        use_expert_rules=use_rules,
        reward_shaping_weight_init=w_init,
        reward_shaping_weight_final=w_final,
        reward_shaping_decay_steps=10000,
        lambda_init=2.0, lambda_lr=0.05, lambda_max=50.0,
        freq_volt_shield_safety_ratio=shield_ratio,
    )
    algo_cfg = AlgorithmConfig(
        name=algo_name, hidden_dims=[256, 256], batch_size=256,
        buffer_size=int(1e5), warmup_steps=500, eval_interval=2000, log_interval=500,
    )
    hfg_cfg = HFGConfig(env=env_cfg, algorithm=algo_cfg, fuzzy=fuzzy_cfg,
                        seed=SEED, device=DEVICE)
    env = make_env("islanded", env_cfg, data_loader=data_loader)
    env.seed(SEED)
    agent = make_agent(algo_name, hfg_cfg,
                       env.observation_space_shape[0],
                       env.action_space_shape[0],
                       env.action_high, DEVICE)
    return agent, env


def train_loop(agent, env, n_ep: int, warm_start_from=None):
    """Run training loop. If warm_start_from agent, copy weights first."""
    if warm_start_from is not None:
        agent.actor.load_state_dict(warm_start_from.actor.state_dict())
        agent.critic.load_state_dict(warm_start_from.critic.state_dict())
        agent.critic_target.load_state_dict(warm_start_from.critic.state_dict())
        import torch
        with torch.no_grad():
            if hasattr(agent.actor, "log_std_head"):
                agent.actor.log_std_head.bias.fill_(-1.5)

    agent.train()
    buffer = ReplayBuffer(agent.obs_dim, agent.act_dim, int(1e5), DEVICE)
    total_steps = 0
    total_cost = 0.0
    total_viol = 0
    total_steps_count = 0
    fcsd_sum = 0.0

    for ep in range(n_ep):
        obs, info = env.reset()
        ep_cost = 0.0
        ep_viol = 0
        for _ in range(STEPS_PER_EP):
            if total_steps < 500 and warm_start_from is None:
                action = np.random.uniform(env.action_low, env.action_high,
                                          size=agent.act_dim).astype(np.float32)
            else:
                action = agent.select_action(obs, deterministic=False)
            action = np.clip(action, env.action_low, env.action_high)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            buffer.add(obs=obs, action=action, reward=reward, next_obs=next_obs,
                       done=done, fcsd=info.get("fcsd", 1.0),
                       fcsd_min=info.get("fcsd_min", 1.0),
                       constraint_cost=info.get("constraint_violation", 0.0))
            if total_steps >= 500:
                agent.update(buffer.sample(256), total_steps)
            obs = next_obs
            total_steps += 1
            total_steps_count += 1
            ep_cost += -reward
            ep_viol += int(info.get("constraint_violation", 0) > 0)
            fcsd_sum += info.get("fcsd", 1.0)
            if done:
                break
        total_cost += ep_cost
        total_viol += ep_viol
        if (ep + 1) % 25 == 0:
            print(f"  ep {ep+1}/{n_ep}  cost={ep_cost/STEPS_PER_EP:.1f}/step  "
                  f"viol={ep_viol/STEPS_PER_EP*100:.1f}%")

    cost_per_day = total_cost / total_steps_count * 96 * 7
    viol_pct = total_viol / total_steps_count * 100
    avg_fcsd = fcsd_sum / total_steps_count
    return {
        "cost_per_day": round(cost_per_day, 0),
        "violation_pct": round(viol_pct, 2),
        "avg_fcsd": round(avg_fcsd, 4),
    }


def run_arm(arm_name: str, variant: str, do_transfer: bool):
    print(f"\n{'='*60}")
    print(f"ARM: {arm_name}  (variant={variant}, transfer={do_transfer})")
    print(f"{'='*60}")
    t0 = time.time()

    if do_transfer:
        # Build source env + agent, train on S3
        src_env_cfg, src_loader = create_env_config("islanded", SOURCE, seed=SEED, num_days=7)
        src_agent, src_env = build_agent(variant, src_env_cfg, src_loader, SOURCE)
        print("\n[1/2] Training source on S3_island_normal...")
        train_loop(src_agent, src_env, N_EP)

        # Build target env + agent, warm-start from source
        tgt_env_cfg, tgt_loader = create_env_config("islanded", TARGET, seed=SEED, num_days=7)
        tgt_agent, tgt_env = build_agent(variant, tgt_env_cfg, tgt_loader, TARGET)
        print("\n[2/2] Fine-tuning on S4_island_extreme (warm-started)...")
        result = train_loop(tgt_agent, tgt_env, N_EP, warm_start_from=src_agent)
    else:
        # From scratch on S4 directly
        tgt_env_cfg, tgt_loader = create_env_config("islanded", TARGET, seed=SEED, num_days=7)
        tgt_agent, tgt_env = build_agent(variant, tgt_env_cfg, tgt_loader, TARGET)
        print("\nTraining from scratch on S4_island_extreme...")
        result = train_loop(tgt_agent, tgt_env, N_EP)

    elapsed = (time.time() - t0) / 60
    result["arm"] = arm_name
    result["time_min"] = round(elapsed, 1)
    print(f"\n  RESULT: cost/day={result['cost_per_day']:.0f}  "
          f"viol={result['violation_pct']:.2f}%  FCSD={result['avg_fcsd']:.4f}  "
          f"time={elapsed:.1f}min")
    return result


def main():
    results = []

    # Arm A: full HFG-SAC transfer
    results.append(run_arm("A: full transfer", "hfg_sac_full", do_transfer=True))

    # Arm B: no_fuzzyknow transfer (constraint only, no shaping)
    results.append(run_arm("B: no_fuzzyknow transfer", "hfg_sac_no_fuzzyknow", do_transfer=True))

    # Arm C: full HFG-SAC from scratch
    results.append(run_arm("C: full from scratch", "hfg_sac_full", do_transfer=False))

    # Summary
    print("\n" + "=" * 80)
    print("TRANSFER ABLATION SMOKE SUMMARY")
    print("S3_island_normal -> S4_island_extreme, seed=42, 100+100 ep")
    print("=" * 80)
    print(f"{'arm':<30} {'cost/day':>12} {'viol%':>10} {'FCSD':>10} {'time/min':>10}")
    print("-" * 75)
    for r in results:
        print(f"{r['arm']:<30} {r['cost_per_day']:>12.0f} {r['violation_pct']:>10.2f} "
              f"{r['avg_fcsd']:>10.4f} {r['time_min']:>10.1f}")

    # Key comparison
    a = results[0]
    b = results[1]
    c = results[2]
    print(f"\n--- Key comparisons ---")
    print(f"Transfer benefit (A vs C): viol {a['violation_pct']:.2f}% vs {c['violation_pct']:.2f}%  "
          f"({'+' if a['violation_pct']>c['violation_pct'] else '-'}{abs(a['violation_pct']-c['violation_pct']):.2f}%)")
    print(f"Knowledge benefit (A vs B): viol {a['violation_pct']:.2f}% vs {b['violation_pct']:.2f}%  "
          f"({'+' if a['violation_pct']>b['violation_pct'] else '-'}{abs(a['violation_pct']-b['violation_pct']):.2f}%)")

    out = Path("outputs/experiments/results/transfer_ablation_smoke.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
