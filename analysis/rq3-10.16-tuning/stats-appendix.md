# Stats Appendix — RQ3 10.16 timing analysis

## 已完成 run 的描述统计（n=6，仅 T1 scratch）

| group | n | cost/d mean | cost/d std | train_time mean | train_time range |
|---|---:|---:|---:|---:|---:|
| hfg_sac from_scratch | 3 | 30,453 | 3,527 | 3,524s | 2,255–4,179 |
| ppo_lagrangian scratch | 3 | 24,507 | 7,375 | 3,593s | 2,445–4,845 |

n=3 不足以做 t-test；仅用于时间预算估算。

## Update 次数核算

每个 run = 200 ep × 672 step/ep = 134,400 env steps。

| 循环 | update 频率 | update 数 / run | 备注 |
|---|---|---:|---|
| scratch (`train_agent`) | 每 16 步 | ~8,400 | line 396 |
| transfer source (`_run_transfer`) | 每步 | ~133,900 | line 1328 |
| transfer target (`_run_transfer`) | 每步 | ~134,200 | line 1522 |
| **transfer 合计** | | **~268,000** | 是 scratch 的 32× |

## 时间预算分解（CPU，单次 update ≈ 19 ms，单次 env step ≈ 25 ms，含 forward pass）

| 项 | scratch | transfer 当前 | transfer 改 A 后 |
|---|---:|---:|---:|
| env 步进（134K step × 25ms） | 3,360s | 6,720s（source+target） | 6,720s |
| gradient updates | 8.4K × 19ms = 160s | 268K × 19ms = 5,092s | 16.75K × 19ms = 318s |
| final eval（10 ep × 672 × 25ms） | 168s | 168s | 168s |
| **wall time / job** | **~3,700s** | **~11,980s（3.3 h）** | **~7,200s（2.0 h）** |

## 剩余 84 jobs 总 CPU-s 估算

| 方案 | scratch jobs (40) | transfer jobs (44) | CPU-hours | wall time @6 workers |
|---|---:|---:|---:|---:|
| 当前 | 40 × 3,700 = 41 h | 44 × 11,980 = 146 h | **187 h** | **31 h** |
| A 后 | 41 h | 44 × 7,200 = 88 h | 129 h | 21 h |
| A+B 后（source 缓存，省 70 次冗余 source） | 41 h | target 44 × 3,700 + unique source 20 × 3,700 = 65 h | 106 h | 18 h |
| A+B+C（10 workers） | | | 106 h | **11 h** |

注：B 方案的 source 训练仍按 3,700s 计（A 之后）；unique source 数 = 4 tasks × 5 seeds = 20。

## 假设与局限

- 单次 update = 19 ms 来自 `experiment_runner.py:394` 代码注释；未在本次 profiling 中实测。
- env step = 25 ms 由 (scratch wall 3,700s − update 160s − eval 168s) / 134K 反推。
- 未考虑 `torch.set_num_threads` 在 6 workers 下的争用；实际 worker 数从 6→10 时单 worker 速度会下降 ~10–20%，已在 C 方案收益中打折。
