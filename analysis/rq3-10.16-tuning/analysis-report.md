# RQ3 实验结果与时间瓶颈分析（2026-10-16）

## 1. 已完成结果盘点

`实验结果/rq3 10.16/results/` 下共 6 个 JSON，全部为 **T1_grid_same** 的 scratch 臂（3 seeds × 2 arms）：

| algo | seed | cost/d (¥) | viol | fcsd | train_time |
|---|---|---:|---:|---:|---:|
| hfg_sac (from_scratch) | 123 | 33,933 | 0.00% | 0.932 | 2255s |
| hfg_sac (from_scratch) | 42  | 30,543 | 0.00% | 0.996 | 4179s |
| hfg_sac (from_scratch) | 456 | 26,883 | 0.00% | 0.996 | 4137s |
| ppo_lagrangian (scratch) | 123 | 32,990 | 0.00% | 0.931 | 2445s |
| ppo_lagrangian (scratch) | 42  | 20,439 | 0.00% | 0.921 | 4845s |
| ppo_lagrangian (scratch) | 456 | 20,093 | 0.00% | 0.924 | 3490s |

**初步观察（仅 3 seeds，统计上不充分）**：
- 所有 run violation=0%，安全约束已满足；fcsd 0.92–1.00。
- cost/d 在 seed 间方差极大（hfg_sac: 27k–34k；ppo_lagrangian: 20k–33k），说明 5 seeds 仍是必须的，不能减 seeds。
- 训练时间 2255–4845s，**单 run 中位数 ~3500s（≈58 min）**。

## 2. 剩余工作量

`run_all_parallel.py` 中 RQ3 共 90 jobs（log 头 [1/90] 已确认）：

| task | arms / seed | seeds | jobs | 已完成 | 剩余 |
|---|---|---|---:|---:|---:|
| T1_grid_same | 4 (scratch, ppo_scratch, naive, hfg) | 5 | 20 | 6 | **14** |
| T2_grid_shifted | 6 (上面 4 + no_M1, no_M4) | 5 | 30 | 0 | **30** |
| T3_island_same | 4 | 5 | 20 | 0 | **20** |
| T4_island_shifted | 4 | 5 | 20 | 0 | **20** |
| **合计** | | | **90** | **6** | **84** |

按当前 6 workers、单 run 3500s 估算：84 × 3500 / 6 ≈ **13.7 wall hours**（仅 scratch 等价负载）。
但 transfer 臂实际远慢于 scratch（见下节），预计实际 **>30 wall hours**。

## 3. 时间根因（已定位）

### 3.1 主瓶颈：transfer 循环每步都做 gradient update（16× 冗余）

`run/experiment_runner.py`：

- **scratch 循环（line 396）**：`if total_steps >= warmup and total_steps % 16 == 0: agent.update(...)` — 每 16 步更新一次，200 ep × 672 step ≈ **8.4K updates / run**。代码注释明确说明这是为了 4× 加速。
- **transfer source 循环（line 1328–1330）**：`if total_steps >= 500: source_buffer.sample(128); source_agent.update(...)` — **每步都 update**，200 ep ≈ **134K updates**。
- **transfer target 循环（line 1522–1524）**：同样每步 update，200 ep ≈ **134K updates**。

即每个 transfer job 做 **~268K updates**，是 scratch（8.4K）的 **32×**。按单次 update ≈19 ms 估算：
- scratch update 耗时：8.4K × 19ms ≈ 160s（占 3500s 总时间的 ~5%）
- transfer update 耗时：268K × 19ms ≈ 5,100s（占 transfer 总时间的 ~45%）

### 3.2 次瓶颈：source policy 对每个 arm × seed 重复训练

`_run_transfer` 中 source 训练（line 1310–1334）**不依赖**任何 arm flag（`use_M1 / use_M3 / use_M4 / conservative`），但代码对每个 transfer job 都从头训练一次 source：

- T2: 6 arms × 5 seeds = **30 次 source 训练**，但实际只需要 **5 次**（每个 seed 一次）。冗余 25 次。
- T1/T3/T4: 4 arms × 5 seeds = **20 次 source 训练**，实际只需 5 次。冗余 15 次/task。
- 全 RQ3 冗余 source 训练：**70 次** × ~3500s（update/skip=16 后）≈ **68 CPU-hours 纯浪费**。

### 3.3 次要差异

- transfer 用 `batch_size=128`，scratch 用 `256`；与 update_skip=16 不匹配。
- `N_WORKERS=6` 在 20 核 CPU 上只用了 ~18 线程（每 worker `cpu_count//6 ≈ 3` 线程），尚有 headroom。
- 最终 eval 固定 10 ep × 672 step ≈ 67s / job，不能省（论文报告指标的权威来源）。

## 4. 加速方案（按收益排序）

| # | 改动 | 文件:行 | 预期收益 | 统计影响 |
|---|---|---|---|---|
| **A** | transfer source/target 循环加 `if total_steps % 16 == 0` 才 update；同时 batch_size 128→256 | `run/experiment_runner.py:1328-1330, 1522-1524` | 每个 transfer job 从 ~11,800s 降到 ~7,000s（**1.7×**） | 无。scratch 已经用相同 schedule，transfer 与 scratch 的对比仍公平 |
| **B** | source policy 按 `(task, seed)` 缓存：训练一次，actor/critic state_dict 存到 dict/torch file；同一 seed 的 naive / hfg / no_M1 / no_M4 共用 | `run/experiment_runner.py:_run_transfer` 拆为 `_train_source_cached` + `_finetune_target` | 省 70 次冗余 source 训练 ≈ **68 CPU-hours**；整体 wall time 再降 ~40% | 无。source 与 arm 无关，缓存只是去重 |
| **C** | `N_WORKERS` 从 6 提到 10（每 worker 2 线程） | `run/run_all_parallel.py:57` | wall time 再降 ~1.6× | 无（不同 worker 跑不同 seed，无数据依赖） |
| **D** | （可选）source 训练从 200 ep 降到 150 ep | `experiment_runner.py:1308 source_episodes=200` | 省 ~25% source 时间 | **有风险**：注释说 100 ep 曾导致 "half-baked source"；建议先 A+B 后用 1–2 seeds 冒烟测 150 ep 是否收敛 |

### 组合预期

| 方案 | 剩余 84 jobs 的 wall time（6 workers） |
|---|---:|
| 当前（不改） | ~30–35 h |
| A 单独 | ~18–20 h |
| A + B | ~10–12 h |
| A + B + C（10 workers） | **~6–7 h** |

## 5. 不建议动的东西

- **N_EPISODES=200**：3 seeds 间 cost 方差已经很大（27k–34k），减 episode 会放大 seed 噪声，破坏 5-seed t-test 的效力。
- **seeds=5**：journal 标准，不能减。
- **最终 eval 10 ep**：论文表报告指标的唯一权威来源。
- **GPU**：当前 torch 是 `+cpu` 构建，且模型仅 256×256 MLP、obs 8–9 维，GPU kernel launch 开销会占主导，切 GPU 不会更快（代码注释 line 63–65 已说明）。

## 6. 建议下一步

1. 立即实施 **A**（10 行改动，零统计风险）。
2. 实施 **B**（需要把 `_run_transfer` 拆成 source/finetune 两段，约 50 行改动）。
3. 用 1 个 task × 1 个 seed 冒烟（~30 min）确认 A+B 后 transfer 结果与之前 pilot 一致。
4. 然后把 `N_WORKERS` 提到 10，跑完整 RQ3。

## 7. Blocker / 局限

- 当前只有 6 个 scratch job，**没有 transfer job 完成过**，无法直接对比 transfer vs scratch 的 cost。B 方案的 source 收敛性只能靠代码审查 + 冒烟测试确认。
- cost/d 在 seed 间方差 ~25%，5 seeds 的 95% CI 仍会较宽；但这是实验设计问题，不是时间优化问题。
