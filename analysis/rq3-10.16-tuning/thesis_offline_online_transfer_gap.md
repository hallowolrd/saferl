# 论文 offline→online 机制 vs 当前 HFG-SAC transfer：可引入项分析

来源：`paper/终稿（杨爱惠）（5.16) .docx` 第三、四章。
对照实现：`run/experiment_runner.py:_run_transfer`（line 1250–1573）。

---

## 1. 论文中两套 offline→online 框架的核心机制

### 1.1 DEA-ITSAC（并网，第三章）—— 平滑过渡四件套

| 机制 | 论文位置 | 作用 |
|---|---|---|
| **CDDPM 数据增强** | §3.2 | 条件扩散模型生成极端工况专家轨迹，覆盖长尾样本 |
| **GAIL 离线预训练** | §3.3.2 | 用判别器输出的模仿奖励做离线 warm-start |
| **混合奖励退火 RI** | §3.3.2, 式 3.x | `r = (1-λ(t)) · r_env + λ(t) · r_disc`，λ 从 1.0 指数退火到 0，**相变拐点设在训练中点**（论文设为总回合中点）。这是论文明确点名"解决离线→在线梯度震荡"的关键机制 |
| **双经验回放 BI** | §3.4.3 | 离线专家 buffer 与在线交互 buffer 分离；动态采样（前期多采专家样本，后期多采在线样本）。消融 RI+BI 是 DEA-ITSAC(on) 相对其去掉版本的两个独立贡献 |

论文原话（§3.4.3 消融）："DEA-ITSAC(off)-R 只在离线热启动后使用 RI；DEA-ITSAC(off)-RB 则同时使用 RI 和 BI。" 即 RI=混合奖励退火，BI=双经验回放。

### 1.2 AR-SAC（孤岛，第四章）—— 绝对安全四件套

| 机制 | 作用 |
|---|---|
| **集成残差模型 + 虚拟经验池** | N 个 NN 学习物理残差，生成虚拟样本，减少 85% 真实交互 |
| **可行性函数 VF(s,π)** | 折扣"最坏情况下能撑多少步不违约束"，作为连续安全分数 |
| **安全探索** | 动作执行前用集成模型预测最坏下一状态，VF 低于阈值就拒绝 |
| **二分更新** | 安全态：SAC + 安全屏障损失；不安全态：只做恢复 |

---

## 2. 当前 HFG-SAC transfer 已实现了什么

| 论文机制 | 当前 `_run_transfer` | 状态 |
|---|---|---|
| GAIL 离线预训练 | 无。source policy 是直接在 source 环境用 SAC 训的（line 1310–1334），无判别器、无专家数据 | ❌ |
| CDDPM 数据增强 | 无 | ❌ |
| 混合奖励退火 RI | 部分。`FuzzyRewardShaper` 已经是 potential-based shaping，weight_init=0.3 → final=0.01，decay=10k 步。**概念上等价于 λ(t) 退火**，但起点低、衰减快、且没有"相变拐点"设计 | ⚠️ |
| 双经验回放 BI | 无。source_buffer 在 source 训完后被丢弃（line 1305），target 从零建 buffer（line 1454） | ❌ |
| M1 约束相似度（Jaccard） | 已实现（line 1342–1358） | ✅ |
| M3 warm-start（actor/critic 权重拷贝） | 已实现（line 1408–1417），**但没有 λ 退火的目标网络同步逻辑** | ✅ 但薄 |
| M4 violation-triggered bias | 已实现（line 1483–1509），**启发式 exp-decay**，SOC/freq 硬越界才触发 | ⚠️ 粗 |
| 可行性函数 VF | 无。但 `fcsd`（fuzzy constraint satisfaction degree）每步都算出来了（env info），**这就是一个连续可行性分数**，只是没被用来 gate 策略更新 | ⚠️ 有信号没用 |
| 集成残差模型 | 无 | ❌ |
| 安全探索（动作执行前风险评估） | 无。`_apply_shield` 只做动作空间硬裁剪，不预测下一状态 | ❌ |

---

## 3. 可引入项（按 ROI 排序）

### 🟢 高 ROI、低改动、直接对症 transfer 震荡

#### ① 双经验回放 BI（论文 §3.4.3）—— **最高优先级**

**问题诊断**：当前 `_run_transfer` 在 target fine-tune 时，target_buffer 从零开始。前 200 步 warmup 随机，之后 SAC update 完全靠 target 环境的新数据。这意味着 M3 warm-start 的 source 知识**只存在于网络权重里**，target 分布一旦偏移，critic 立刻因为没见过 target transition 而给出错误 Q，actor 被错误 Q 带偏——这正是论文说的"offline→online 梯度震荡"。

**改动**：
- 把 `source_buffer` 保留下来（当前 line 1305 创建的 `source_buffer` 本来就在内存里）。
- target update 时，batch = α(t) · source_buffer.sample + (1-α(t)) · target_buffer.sample。
- α(t) 从 0.5 线性退火到 0，跨越前 50 episodes（≈33K steps）。
- 注意：source transition 的 reward 是 source env 的 cost，**不能直接混进 target Q**。两种处理：
  - **方案 A（简单）**：source buffer 只用来做 **behavioral-policy regularization**（BC loss on source state-action pairs），不进 target critic。即 actor loss = SAC_loss + β · MSE(actor(s_source), a_source)。这是论文 GAIL 的简化版。
  - **方案 B（真 BI）**：把 source transition 的 reward 改写为 fuzzy shaping reward（`FuzzyRewardShaper.shape_reward`），这样 source transition 也能进 target Q。
- **推荐方案 A**：改动 ~30 行，不引入 reward 尺度问题。

**预期收益**：消融显示 RI+BI 比单独 RI 在论文中 cost 再降 15–20%，violation 再降 30%。在 RQ3 这种"target 是 extreme 场景"的迁移任务上，source buffer 提供的是"安全状态下该怎么调度"的先验，正好补 target 早期数据缺失。

#### ② 混合奖励退火 RI 的"相变拐点"调整

**问题诊断**：当前 `reward_shaping_weight_init=0.3, final=0.01, decay=10000 steps`（line 1285–1287）。target fine-tune 总共 200 ep × 672 step = 134K steps。10K 步 ≈ 15 episodes 就把 shaping 降到 0.01，之后 185 episodes 完全靠 raw reward。但 M3 warm-start 的策略在 target 早期最脆弱，shaping 反而应该**在这个阶段最强**。

**改动**（参考论文 §3.4.1 表 3.10）：
- `reward_shaping_weight_init`：0.3 → **0.5**
- `reward_shaping_weight_final`：0.01 → **0.0**（论文退火到 0）
- `reward_shaping_decay_steps`：10000 → **67000**（≈100 episodes，即训练中点）
- 这与论文"λ 相变拐点设在总回合中点"的设计一致。

**改动量**：3 行常量。
**预期收益**：减少 target fine-tune 前 50 episodes 的 cost 震荡；与①叠加。

### 🟡 中 ROI、中等改动、锦上添花

#### ③ 用 fcsd 作为可行性分数 gate M4（AR-SAC 二分更新简化版）

**问题诊断**：当前 M4 只在 SOC 硬越界或 |freq_norm|>0.85 时触发（line 1486–1509）。但 env 每步已经算出 `fcsd ∈ [0,1]`（fuzzy constraint satisfaction degree），这其实就是论文 AR-SAC 的 VF 分数。当 fcsd=0.7 但还没硬越界时，M4 什么都不做，等到硬越界已经晚了。

**改动**：
- 在 M4 触发条件里加：`if fcsd < 0.9 (or m4_soc_min ≤ soc ≤ m4_soc_max but fcsd low)`。
- M4 beta 大小改为 `beta_init · (1 - fcsd) · decay(t)`，而不是只在 breach 时乘 breach depth。
- 这把 M4 从"被动越界纠正"变成"主动风险抑制"，对应 AR-SAC 的二分更新。

**改动量**：~15 行。
**预期收益**：T3/T4（孤岛 extreme）violation 从当前 ~0% 进一步压低（虽然已经是 0%，但 target 未完成，可能 fine-tune 早期有 spike）。

#### ④ 目标网络（critic_target）warm-start 时同步 source 的 target 参数

**问题诊断**：当前 line 1413 已经把 `source_agent.critic.state_dict()` 同时拷给 `target_agent.critic` 和 `target_agent.critic_target`。✅ 这一项其实已经做了。无需改动。

### 🔴 低 ROI、大改动、不建议在 RQ3 引入

| 机制 | 为什么不建议 |
|---|---|
| GAIL 判别器 | RQ3 是 transfer（source 是 SAC 训的，不是专家轨迹），没有专家数据可用。引入 GAIL 需要重新构造 source 训练，属于另一个实验（RQ1 主对比）。 |
| CDDPM 扩散模型 | 纯数据侧增强，与 transfer 算法无关。 |
| Transformer Q-network | 当前 env 是 fully observed 9 维，7 天 horizon；论文用 Transformer 是因为 POMDP + 长时序。我们 obs 已经包含 enough history；换 Transformer 是 ~300 行改动，预期收益小。 |
| 集成残差模型 + 虚拟经验池 | AR-SAC 的核心，但工程量大（N 个 NN + 多样性正则 + 训练循环）。RQ3 的 T3/T4 是 island transfer，理论上能用，但当前 timeline 不允许。可以作为 RQ6 / 下一篇文章。 |
| 安全探索（动作执行前预测下一状态） | 需要 dynamics model，依赖集成模型。 |

---

## 4. 推荐落地顺序

**Phase 1（本次 RQ3 就能上，< 1 天工作量）**：
1. 双经验回放 BI 的方案 A（BC regularization on source state-action pairs）
2. RI 退火参数调整（3 行）
3. M4 用 fcsd 连续 gate

**Phase 2（如果 Phase 1 冒烟有收益，再上）**：
4. BI 方案 B（source transition reward 改写成 shaping reward，进 target replay）

**不做**：GAIL、CDDPM、Transformer、集成模型。

---

## 5. 对实验设计的影响

引入 ①②③ 后，需要在 RQ3 里加一组消融：
- `hfg_transfer_baseline`（当前的 M1+M3+M4）
- `hfg_transfer_+BI`（加 source BC reg）
- `hfg_transfer_+RI`（退火参数调整）
- `hfg_transfer_+BI+RI+fcsd_gate`（完整）

但这会把 T2 的 arm 数从 6 增到 10，jobs 从 90 增到 ~110。**建议先在 T1 上跑 1 seed 冒烟**，确认完整组合比 baseline 好，再决定是否扩展为正式 ablation。

---

## 6. 关键引用（论文章节）

- DEA-ITSAC 混合奖励退火：§3.3.2（式 3.x 复合奖励与 λ 衰减函数）、§3.4.1 表 3.10（超参：相变拐点设在总回合中点，温度系数 10）
- 双经验回放 BI：§3.4.3 消融分析（图 3.23–3.25）
- AR-SAC 可行性函数 VF：§4.2.2（式 4.x）、二分更新（式 4.x VF 阈值指示变量）
- AR-SAC 安全探索：§4.2.2（3）候选动作风险预估
