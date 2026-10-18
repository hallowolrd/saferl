# 失败备忘：DEA-ITSAC 机制在 HFG-SAC transfer 上的引入尝试

日期：2026-10-16
分支：ll_feat
状态：**已回滚**（代码恢复到 HEAD）

## 背景

RQ3 transfer 任务（T1 grid_same，seed=42，200 ep）baseline cost/d ≈ 30k。
论文（终稿杨爱惠）第三章 DEA-ITSAC 提出三个机制平滑 offline→online：
- RI：混合奖励退火 `(1-λ)·r_env + λ·r_imitation`，λ 从 1.0 退火到 0
- BI：双经验回放（专家 buffer + 在线 buffer 分离）
- AR-SAC 二分更新：用 fcsd/VF 连续 gate 策略更新

基于此实现了三个改动：

## 改动清单（已全部回滚）

1. **方案① BI**：`HFGSAC.update()` 加 `bc_batch` / `bc_weight` 参数；target fine-tune 时从 source_buffer 采样，actor loss 加 `bc_weight · MSE(actor(s_bc), a_bc)`，bc_weight 从 0.5 线性退火到 0 跨 50 ep。
2. **方案② RI**：shaping weight_init 0.3→0.5，final 0.01→0，decay_steps 10k→67k。
3. **方案③ fcsd gate M4**：M4 从硬越界触发改为 `fcsd < 0.95` 连续触发，beta ∝ (0.95 - fcsd)。

## 实验结果

| 配置 | cost/d | vs baseline | 耗时 |
|---|---:|---:|---:|
| baseline（全关，source=20ep, target=20ep） | 50,589 | 1.0× | 411s |
| ①+② 无③（20ep 短测） | 115,505 | **2.3× 更差** | 622s |
| ①+②+③ 全开（200ep 完整） | 96,774 | **1.9× 更差** | 188 min |

violation 始终 0%，但 cost 暴涨 2–3 倍。convergence_ep=10 表明策略很早就锁定到一个坏解。

## 根因分析

论文 RI/BI 成立的**两个隐含前提**在当前 RQ3 框架下不满足：

1. **source policy 必须是 GAIL 学出来的 expert**。
   - 论文：source 是 Gurobi 全局最优解 → GAIL 判别器 → SAC 生成器。判别器输出 imitation reward 与 env reward 同尺度。
   - 当前：source 只是 normal weather 下普通 SAC 训出来的 policy。没有专家数据，没有判别器。
   - 直接 BC 一个"普通学生"的动作，相当于把非专家行为强灌给 target，锁死对 extreme 工况的适应。

2. **fcsd 阈值要按 env 实际分布设**。
   - grid-connected 场景 fcsd 典型 0.92–1.00（6 个 scratch run 都在这个区间）。
   - 阈值 0.95 意味着 ~60% 步数触发 M4，每步都推 ESS，完全扭曲调度。
   - 原来硬越界触发（SOC<0.3 或 >0.8）频率 <5%，是合理的。

3. **shaping weight 0.5 + 67k decay**：fuzzy expert rules 占主导时间过长，压过 env reward 学习信号。

## 经验教训

- **不要直接套论文章节里的机制名**，要先检查机制成立的前提（expert data? same reward scale?）。
- **BC 正则只对 expert 数据有效**。对普通 SAC source 做 BC = 破坏性正则。
- **fcsd gate 阈值要按 env 分布 percentile 设**，不要拍脑袋 0.95。grid 场景应该设 0.85 或更低。
- **shaping weight 起点 0.3 / decay 10k** 是经过 RQ1/RQ2 多个 scenario 调出来的，不要为了"更激进"乱改。

## 正确路径（如果未来要做 transfer 改进）

1. 先用 **Gurobi 解出 expert 轨迹**（论文 §3.4.1 做法）作为 source 的训练数据。
2. 引入 GAIL 判别器，输出 imitation reward。
3. 再考虑 RI 退火（此时 λ(t) 是 discriminator reward 的权重）。
4. 这是独立实验，不是 RQ3 transfer 的微调。

## 回滚操作

```bash
git checkout HEAD -- run/experiment_runner.py src/hfg_srl/algorithms/hfg_sac.py
```

保留的工作区改动（非本次引入）：
- `num_rules=16→8`
- `reward_shaping_weight_init=0.3→150`, `final=0.01→8.0`（RQ4 调参）

## 相关产物

- `analysis/rq3-10.16-tuning/analysis-report.md` — 时间瓶颈分析
- `analysis/rq3-10.16-tuning/thesis_offline_online_transfer_gap.md` — 论文机制对照
- `outputs/experiments/rq3_t1_seed42_full.log` — 完整 200ep 失败 run
- `outputs/experiments/rq3_abl_A.log` / `rq3_abl_C.log` — 短测对照
