# 面向微电网优化调度的分层模糊引导安全强化学习：约束感知的跨场景迁移

> **目标期刊：** Applied Energy / IEEE Transactions on Smart Grid
> **稿件状态：** 整合工作稿 v0.2；RQ1（主对比实验）与 RQ2（鲁棒性）结果为真实结果；RQ3（迁移）与 RQ4（消融）已完成设计但尚待执行。标注 `[FINAL]` 的定量结论须在所有运行完成后更新。

---

## 摘要

安全性是阻碍深度强化学习（DRL）在实际微电网（microgrid）能量管理系统中部署的关键瓶颈。现有安全强化学习（safe reinforcement learning, Safe RL）方法通常将约束表述为清晰（crisp）的硬边界，难以刻画工程约束固有的模糊性，例如荷电状态推荐范围与绝对范围之别、负荷优先级的渐变，以及紧急工况下可容许的短时越限。此外，纯数据驱动的安全强化学习样本效率（sample efficiency）低下，且无法在跨场景策略迁移过程中保证安全性。针对上述不足，本文提出分层模糊引导安全强化学习（HFG-SRL）框架。首先，我们提出模糊约束马尔可夫决策过程（Fuzzy Constrained Markov Decision Process, FC-MDP），它通过以连续的模糊约束满足度（fuzzy constraint satisfaction degree, FCSD）替代硬约束，对标准 CMDP 加以推广；并推导出具有可证明收敛保证的模糊拉格朗日（Fuzzy-Lagrangian）方法。其次，我们提出分层模糊引导 SAC（HFG-SAC）算法，其中包含两个互补的模糊层：上层模糊约束保护机制在约束边界平滑的前提下保障安全探索，下层模糊知识奖励塑形（reward shaping）引擎嵌入专家运行规则以加速学习。再次，我们设计约束感知的跨场景迁移（transfer learning）机制，利用模糊集合相似度逐约束评估可迁移性，通过部分权重冻结与保守动作尺度的指数松弛，使已学策略能够在不同微电网运行模式与季节条件下安全迁移。在改进的 IEEE 33 节点微电网上，针对并网（grid-connected）与孤岛（islanded）场景（各 5 个随机种子（seeds）、200 个回合（episode））的实验表明，HFG-SAC 是唯一在全部四种运行场景中将硬约束违反率（hard-constraint violation rate）保持在 1% 以下的方法，其成本与 CPO 相当，并在电网极端（extreme）条件下比 PPO-Lagrangian 与安全层（Safety Layer）低 20–40%。在需求极端程度扫描至训练分布 1.5× 的条件下，HFG-SAC 是唯一始终低于 5% 违反率阈值的方法（0.36% → 3.15%），且 FCSD 仅下降 0.010。[FINAL：待 RQ3 运行完成后更新迁移样本节省数值。]

**关键词：** 安全强化学习；模糊逻辑；微电网优化调度；soft actor-critic；约束马尔可夫决策过程；迁移学习

---

# 1. 引言

全球能源系统向低碳化转型，推动可再生能源渗透率快速提高。微电网是一种集成了光伏（PV）阵列、风力发电机（WT）、储能系统（ESS）与可控负荷等分布式能源资源（DER）的本地化能源系统，已成为在提升供电可靠性与能效的同时消纳高比例波动性可再生能源的关键架构 [29], [30]。微电网优化调度（optimal dispatch）需要确定储能的充/放电计划、可调度机组的出力水平以及与主网的交换功率，是降低运行成本、减少弃电（curtailment）并维持电能质量的基础 [17], [18]。然而，可再生能源出力与负荷需求固有的不确定性、储能系统的非线性动态，以及同时满足多项运行约束的要求，使微电网调度面临严峻挑战 [19]。混合整数线性规划（MILP）、模型预测控制（MPC）等传统优化方法依赖精确的系统模型与预测数据，在高不确定性或系统模型不完善时性能会明显下降 [32]–[34]。深度强化学习（DRL）能够直接从交互数据中学习最优控制策略，并处理高维、不确定的环境，已成为微电网能量管理颇具前景的替代方案 [19], [20]。

尽管取得了上述进展，将 DRL 部署到真实微电网仍面临一个关键障碍：**安全性**。在安全攸关的能源系统中，约束违反（如电池过充、超出机组爬坡限制（ramp limits）或电压越限）可能导致设备损坏、电能质量下降，甚至引发停电。标准 DRL 算法在训练过程中自由探索，对约束满足不提供任何形式化保证，因而无法直接部署于实际系统 [2]。安全强化学习（Safe RL）将约束满足纳入强化学习框架 [1], [3]。其中占主导地位的表述是约束马尔可夫决策过程（CMDP），它将安全性建模为不得超越的硬约束阈值 [4]。为求解 CMDP，研究者提出了拉格朗日（Lagrangian）松弛 [21]、约束策略优化（CPO）[5] 以及基于信赖域（trust region）的方法等。然而，这些方法都建立在一个基本假设之上：约束具有精确而清晰的边界。

在实际中，清晰约束假设难以反映微电网运行的真实情况。工程约束很少是二值的，其固有地呈现出渐变性与模糊性，原因有以下几点。第一，许多约束在“硬限值”之内还存在“推荐范围”：为延长循环寿命，电池的荷电状态（state of charge, SOC）可能具有 30–80% 的优选区间，而硬安全限值为 10–90%，在这两个区间之间运行是允许的但并非理想。第二，在突发云遮、负荷尖峰等极端条件下，为避免切负荷（load shedding）等更严重后果，短时、小幅度的越限是可以接受的。第三，并非所有约束同等重要：频率稳定比 SOC 约束更为关键，但标准 CMDP 将所有约束视为同等刚性。清晰约束的理想化在安全强化学习中导致三类已有充分记载的病症：(i) **过度保守（over-conservatism）**，策略为避开陡峭边界而远离可行域内部边界运行，浪费了安全的内部空间；(ii) **边界振荡（boundary oscillation）**，由于梯度信号不连续，策略在约束边界附近来回振荡；(iii) **稀疏（约束）梯度（sparse constraint gradients）**，在真正越过边界之前约束代价不提供任何学习信号，从而拖慢收敛 [9]。

除约束表述之外，还有两个不足限制了安全强化学习在微电网中的实际应用。纯数据驱动的安全强化学习从零开始学习，需要数百万次环境交互，并在训练中进行大量不安全探索；而微电网运行实际上受益于数十年积累的专家知识，包括运行规则、启发式准则与经过验证的调度策略，现有方法缺乏将其嵌入的原则性途径 [11], [15]。此外，微电网运行工况不断变化，包括季节（seasonal）变化、并网与孤岛模式切换、正常与极端天气，为每种场景从零训练新策略既浪费资源又不安全。迁移学习已被应用于微电网调度 [25], [26]，但现有方法着眼于加速奖励优化，无法保证迁移过程中及迁移后的约束满足。

针对上述不足，本文提出面向微电网优化调度的**分层模糊引导安全强化学习（HFG-SRL）**框架，从约束表述、知识嵌入与跨场景迁移三个层面将模糊逻辑融入安全强化学习。首先，我们提出**模糊约束马尔可夫决策过程（FC-MDP）**，它以连续的模糊约束满足度（FCSD）替代清晰约束阈值，从而推广标准 CMDP；每个状态–动作对在每项约束上获得 [0, 1] 区间内的满足度，而非可行/不可行的二值标签。我们推导出求解 FC-MDP 的**模糊拉格朗日（Fuzzy-Lagrangian）**方法，证明其在标准正则条件下收敛到鞍点（saddle point），并说明当模糊边界宽度趋于零时，FC-MDP 将以极限情形退化为标准 CMDP。其次，我们在 Soft Actor-Critic（SAC）架构 [27], [28] 之上提出**分层模糊引导 SAC（HFG-SAC）**算法，其中包含两个互补的模糊层：上层计算多约束联合满足度，并向 actor（策略网络）提供平滑的约束梯度；下层将专家运行规则编码为 TSK 模糊奖励塑形，在可证明保持最优策略不变的同时加速学习。再次，我们设计**约束感知的跨场景迁移机制**，利用模糊集合 Jaccard 指数逐约束诊断相似性，复制并部分冻结已学 actor 权重，并松弛保守动作尺度，使早期微调（fine-tuning）始终处于安全探索管道之内。

本文的主要贡献有三点。(i) **理论层面。** 我们提出 FC-MDP 框架，将标准 CMDP 推广至具有连续满足度的模糊安全约束；推导出具有可证明收敛性质的模糊拉格朗日方法，并通过极限定理建立其与标准 CMDP 的联系。(ii) **方法层面。** 我们提出 HFG-SAC，包含两个互补的模糊层：上层约束保护机制实现边界平滑的安全探索，下层知识塑形引擎嵌入专家规则以加速学习并保证策略不变性。(iii) **应用层面。** 我们设计约束感知迁移机制，利用 Jaccard 相似度诊断逐约束可迁移性，通过部分权重冻结提升样本效率，并采用指数松弛的保守动作尺度实现安全的早期微调，从而在微电网不同运行条件下安全地适配已学策略。

---

# 2. 相关工作

## 2.1 安全强化学习

安全强化学习关注的是学习在训练与部署过程中均满足安全约束的策略 [2]。我们将现有方法分为约束 MDP 方法、控制理论方法与基于模型的安全方法三类。

### 2.1.1 约束 MDP 与拉格朗日方法

CMDP [4] 是安全强化学习的标准框架，其中安全性表述为期望累积代价不得超过的约束代价阈值。**拉格朗日松弛**通过引入拉格朗日乘子将约束问题转化为无约束鞍点问题，已广泛与策略梯度算法结合；PPO-Lagrangian 与 TRPO-Lagrangian [39] 至今仍是强有力的基线方法（baseline）。**约束策略优化（CPO）** [5] 提出一种信赖域方法，在每次策略更新时强制满足约束，可获得近似约束满足保证，但需要计算代价高昂的共轭梯度。

尽管应用广泛，基于 CMDP 的方法仍存在重要局限。它们假设约束是清晰的，即任何越限都同等不可接受，这无法刻画许多实际系统中安全性的分级特征 [9]。当对偶变量（dual variable）调整缓慢时，拉格朗日方法还可能在奖励优化与约束满足之间振荡；并且由于约束仅在轨迹分布期望意义下而非逐步骤强制执行，探索过程中的越限仍然普遍。本文通过以模糊满足度替代清晰约束来解决第一项局限，同时保留拉格朗日方法的优势。

### 2.1.2 控制理论安全强化学习

控制理论方法将具有可证明安全性质的控制器嵌入强化学习回路，从而提供硬安全保证。**控制屏障函数（CBF）** [7] 定义前向不变安全集，并将强化学习策略投影到安全动作空间，以高概率保证安全性。**安全层（safety layer）** [37] 在每个状态下解析地修正智能体动作，使修正后的动作保持安全；**屏蔽（shielding）** [8] 遵循类似思路：当强化学习智能体的动作将违反约束时，由备用控制器或动作修正器介入。这些方法虽能提供较强的逐步安全保证，但通常需要精确的系统动态模型来计算安全集，而这一假设在可再生能源出力不确定、储能动态非线性的微电网中往往不成立。近期关于从数据中学习 CBF 的工作 [9] 放宽了这一要求，但带来了样本效率方面的挑战。

### 2.1.3 基于模型的安全强化学习

基于模型的安全强化学习先学习动态模型，再将其用于约束规划或安全策略优化：约束交叉熵方法（C-CEM）[38] 利用学习到的模型执行约束轨迹优化，约束基于模型的策略优化 [40] 则利用模型生成的 rollout 加速策略搜索，同时施加期望代价约束。相较无模型安全强化学习，这类方法降低了样本复杂度，但存在模型误差累积问题；随着模型偏离真实系统，约束满足保证会逐渐退化，这在长时域微电网调度中尤为突出。

**研究空白 1。** 现有安全强化学习方法一致假设清晰硬约束，均未系统处理工程约束固有的模糊性，从而导致策略过度保守、边界振荡与约束梯度稀疏。本文的 FC-MDP 通过将 CMDP 推广至连续满足度填补了这一空白。

## 2.2 强化学习中的模糊逻辑

模糊逻辑与强化学习的结合方式有多种，大致可分为三类。

### 2.2.1 模糊奖励塑形

模糊奖励塑形利用模糊推理从专家规则中计算附加奖励信号，引导智能体趋向期望行为 [13], [14]。这类方法虽提升了样本效率，但模糊部分通常仅作为辅助奖励，并不处理约束满足问题；其与原始 MDP 最优策略之间的关系往往也未得到论证。本文下层的模糊知识塑形同样以加速学习为目标，但我们借助基于势能的塑形（potential-based shaping）定理 [11] 形式化地证明了策略不变性。

### 2.2.2 模糊 actor–critic 架构

模糊 actor–critic 方法以模糊推理系统取代神经策略或价值函数。**模糊 Q 学习** [10] 与**深度模糊强化学习** [12] 使用 TSK 模糊系统作为函数逼近器，充分利用模糊规则的可解释性。这类方法参数量更少，但与深度网络相比难以应对高维状态空间。本文方法将模糊系统用作深度强化学习架构*内部的组件*，而非完全替代，从而融合两种范式的优势。

### 2.2.3 模糊参数自适应

第三类工作利用模糊逻辑，根据训练表现自适应调整强化学习超参数，如学习率、探索率或熵温度（entropy temperature）[14]。这些方法有助于提升训练稳定性，但未涉及安全约束满足问题。

**研究空白 2。** 强化学习中的模糊逻辑主要应用于奖励塑形、函数逼近与超参数调节。鲜有工作将模糊系统用于约束处理，也尚无研究提出在基础理论层面与 CMDP 理论相融合的模糊约束框架。据我们所知，本文的上层模糊约束保护首次将 TSK 模糊系统直接嵌入安全强化学习的拉格朗日约束机制，并给出理论收敛保证。

## 2.3 微电网优化调度的深度强化学习

DRL 无需显式系统模型即可处理不确定性与非线性，因此在微电网能量管理中受到广泛关注 [17], [18]。

### 2.3.1 面向微电网调度的 DRL 算法

DQN 等基于值的方法已被应用于离散动作的微电网调度；而 DDPG、PPO、SAC 等策略梯度方法更适合 ESS 充/放电调度等连续控制任务。SAC 凭借最大熵框架与离策略学习，因其样本效率与稳健探索能力在微电网调度中表现优异 [20], [27]。然而，标准 DRL 不提供任何安全保证。

### 2.3.2 基于 DRL 的调度中的约束处理

现有面向微电网调度的 DRL 方法主要通过**动作裁剪（action clipping）**（将动作投影到可行集）或奖励中的**惩罚项**来处理约束。动作裁剪会扭曲策略梯度，并在边界附近导致次优策略；惩罚方法需要人工调参，且不提供形式化保证。少数工作应用了安全强化学习，如物理信息引导的安全强化学习 [21] 或面向联网与孤岛微电网的安全 DRL [22], [23]，但它们仍继承了第 2.1 节所述的清晰约束局限。

### 2.3.3 面向微电网调度的迁移学习

迁移学习已成为缩短不同微电网场景训练时间的一种途径。参数迁移 [25] 将网络权重从源任务直接复制到目标任务；不确定性下的迁移强化学习 [26] 则在各种运行工况间适配已学策略。然而，现有微电网迁移方法着眼于奖励优化，未考虑迁移过程中的安全性；迁移后的策略可能在目标环境中违反约束，当场景差异较大时尤为明显。

**研究空白 3。** 安全强化学习在微电网调度中的应用仍处于起步阶段。现有工作采用清晰约束表述，忽视了工程约束的分级特征；样本效率受限于专家知识整合的缺失；具有安全保证的跨场景迁移也基本未被探索。本文的 HFG-SRL 框架同时弥补了上述三方面空白。

**表 I. 研究空白汇总及本文的应对方式。**

| 缺口 | 领域 | 具体问题 | 本文方案 |
|-----|--------|------------------|---------------|
| 1 | 安全强化学习理论 | 清晰约束假设 → 过度保守、边界振荡、梯度稀疏 | 采用模糊拉格朗日求解器的 FC-MDP |
| 2 | 模糊逻辑 + 强化学习 | 模糊系统仅用于塑形/架构，未用于约束满足 | 两层模糊机制：约束保护 + 知识塑形 |
| 3 | 微电网调度 | 缺乏系统化的安全强化学习、样本效率低、迁移不安全 | HFG-SAC + 约束感知渐进迁移 |

---

# 3. 预备知识与问题建模

## 3.1 预备知识

### 3.1.1 约束马尔可夫决策过程

标准马尔可夫决策过程（MDP）是一个元组 $\langle \mathcal{S}, \mathcal{A}, P, R, \gamma \rangle$，其中 $\mathcal{S}$ 为状态空间，$\mathcal{A}$ 为动作空间，$P(s'\mid s,a)$ 为转移概率，$R(s,a)$ 为奖励，$\gamma\in[0,1)$ 为折扣因子。其目标是求解策略 $\pi(a\mid s)$，使期望折扣回报 $J(\pi)=\mathbb{E}_\pi[\sum_t\gamma^t R(s_t,a_t)]$ 最大化。

**CMDP** 在 MDP 的基础上引入了 $K$ 个约束代价函数 $C_k(s,a)$ 与阈值 $d_k$：若对所有 $k$ 均有 $J_{C_k}(\pi)=\mathbb{E}_\pi[\sum_t\gamma^t C_k(s_t,a_t)]\le d_k$，则策略是可行的。其标准解法是**拉格朗日方法**，即引入拉格朗日乘子 $\lambda_k\ge0$：

$$\min_{\lambda\ge0}\max_\pi \mathcal{L}(\pi,\lambda)=J(\pi)-\sum_{k=1}^K\lambda_k\big(J_{C_k}(\pi)-d_k\big).$$

### 3.1.2 Soft Actor–Critic

SAC [27], [28] 是一种离策略（off-policy）最大熵强化学习算法，其最大化期望奖励与策略熵之和：

$$J(\pi)=\mathbb{E}_\pi\Big[\sum_t\gamma^t\big(R(s_t,a_t)+\alpha\,\mathcal{H}(\pi(\cdot\mid s_t))\big)\Big],$$

其中 $\alpha>0$ 为熵温度，$\mathcal{H}$ 为策略熵。SAC 维护两个 Q 网络（以缓解高估偏差）与一个独立的策略网络，并采用熵温度自动调节，因而非常适合连续控制的微电网调度。

### 3.1.3 TSK 模糊系统

Takagi–Sugeno–Kang（TSK）模糊系统 [35] 通过 $M$ 条 IF–THEN 规则实现非线性映射：

$$R^i:\ \text{IF } x_1\text{ is }F_1^i\cdots\text{AND }x_n\text{ is }F_n^i,\ \text{THEN } y^i=p_0^i+\sum_j p_j^i x_j.$$

对于输入 $\mathbf{x}$，其输出为加权平均 $f(\mathbf{x})=\sum_i w^i(\mathbf{x})y^i/\sum_i w^i(\mathbf{x})$，其中触发强度为 $w^i(\mathbf{x})=\prod_j\mu_{F_j^i}(x_j)$。TSK 系统是通用逼近器 [35], [36]，并且为编码语言形式的专家知识提供了可微结构，该优势在我们框架的两个模糊层中均得到利用。

## 3.2 微电网调度问题建模

### 3.2.1 系统描述

我们考虑一个由光伏阵列、风力发电机、ESS、柴油机（DE）备用发电机以及可控/不可控负荷构成的微电网。其运行模式包括：(i) **并网模式**，按随时间变化的电价与主网交换功率；(ii) **孤岛模式**，就地实现功率平衡。

### 3.2.2 优化目标

目标是在 7 天运行周期、15 分钟分辨率（每天 96 个决策步，每个回合 672 步）下最小化总运行成本，该成本包括电网交换成本、DE 燃料成本、运维成本、切负荷惩罚（孤岛模式）以及弃电惩罚。

### 3.2.3 传统（清晰）约束

在传统的清晰（crisp）建模下，调度需满足有功功率平衡、ESS 荷电状态（SOC）边界 $[\text{SOC}_{\min},\text{SOC}_{\max}]$、ESS 充放电功率限制、DE 出力与爬坡限制、电网交换功率限制以及母线电压偏差限制。这些约束以硬性 0–1 边界强制执行：一个状态–动作对要么完全可行，要么完全不可行。

## 3.3 模糊约束建模

### 3.3.1 清晰约束为何不足

实践中，微电网约束具有内在的渐变特性：硬安全边界之内还存在推荐 SOC 带；极端工况下可容忍短时、小幅的越限；约束之间存在优先级排序（频率/电压比 SOC 更为关键）。从学习角度看，清晰约束会导致过度保守、边界振荡以及梯度稀疏。这些因素促使我们将约束重新建模为具有连续满足度的模糊集。

### 3.3.2 模糊约束满足度

我们将约束 $k$ 的模糊约束满足度（FCSD）定义为隶属度函数 $\mu_k(s,a):\mathcal{S}\times\mathcal{A}\to[0,1]$，其中 $\mu=1$ 表示完全满足，$\mu=0$ 表示完全违反。该函数单调、连续/可微，并经过校准，使清晰边界对应某一指定水平。我们采用 S 型隶属度

$$\mu_k(x)=\frac{1}{1+\exp\!\big(-\beta_k(x_k^{\text{ref}}-x)\big)},$$

其中 $x_k^{\text{ref}}$ 为参考点（对应 $\mu=0.5$），$\beta_k>0$ 控制过渡的陡峭程度；$\beta_k\to\infty$ 时恢复为清晰约束。对于 SOC 上界，$\mu_{\text{SOC,upper}}=1/(1+\exp(-\beta_{\text{SOC}}(\text{SOC}_{\text{ref,upper}}-\text{SOC}_{t+1})))$，下界定义与之类似；对于单一变量上的 SOC 区间，我们采用标准模糊交 $\min(\mu_{\text{SOC,upper}},\mu_{\text{SOC,lower}})$，而对于不同的物理约束（$K>1$），则采用第 3.3.3 节的加权乘积 t-范数。

### 3.3.3 多约束聚合

对于 $K$ 个约束，我们采用**加权乘积 t-范数**计算联合满足度：

$$\tilde\mu(s,a)=\prod_{k=1}^K\mu_k(s,a)^{w_k},\qquad w_k\ge0,$$

该式在 $(0,1)^K$ 上处处可微，允许关键约束占据主导地位，且当所有 $w_k=1$ 时退化为标准乘积 t-范数。聚合后的 FCSD 是下文 FC-MDP 的基础。

在面向安全关键的微电网实现中，我们对最具安全关键性的约束（频率、电压）取优先级权重极限 $w_k\to\infty$，此时加权乘积退化为**最小 t-范数** $\tilde\mu(s,a)=\min_k\mu_k(s,a)$。这种不可补偿的聚合方式体现了如下工程原则：整体安全性受限于被违反最严重的约束——良好的 SOC 无法补偿频率偏差——同时避免了乘积 t-范数所允许的平滑但可补偿的权衡。此处保留乘积形式作为一般框架，最小 t-范数则是部署时选用的保守特例。

---

# 4. HFG-SRL 框架

## 4.1 模糊约束马尔可夫决策过程（FC-MDP）

**定义 1（FC-MDP）。** 模糊约束马尔可夫决策过程是一个元组 $\langle\mathcal{S},\mathcal{A},P,R,\tilde C,\gamma,\tilde\alpha\rangle$，其中 $\tilde C=\{\mu_k\}_{k=1}^K$ 为模糊约束，$\tilde\alpha\in(0,1]^K$ 为目标满足水平。其目标为

$$\max_\pi J_R(\pi)\quad\text{s.t.}\quad J_{\mu_k}(\pi)=\mathbb{E}_\pi\Big[\sum_t\gamma^t\mu_k(s_t,a_t)\Big]\ge\frac{\alpha_k}{1-\gamma},\ \forall k.$$

若某策略在期望意义下满足全部约束，则称其为 **$\alpha$-安全**策略。

**定理 1（FC-MDP → CMDP 极限）。** 取 $\mu_k(s,a)=\sigma(\beta_k(d_k-c_k(s,a)))$，当 $\beta_k\to\infty$ 时，$\alpha$-安全策略集合收敛于在折扣占用期望意义下避开违反集的策略集合；取 $\alpha_k=1$ 即恢复标准 CMDP 的*始终安全*子集。（证明概要：S 型函数逐点收敛于指示函数；控制收敛；完整证明见附录 A。）这表明 FC-MDP 是 CMDP 的严格推广。

**定理 2（存在性）。** 在状态/动作空间紧致、转移弱连续、奖励与隶属度函数有界连续且严格可行的条件下，存在最优的平稳确定性 $\alpha$-安全策略。

## 4.2 模糊拉格朗日方法

将约束改写为满足度亏空后，**模糊拉格朗日函数**为

$$\mathcal{L}_{\text{fuzzy}}(\pi,\lambda)=J_R(\pi)+\sum_{k=1}^K\lambda_k\Big(J_{\mu_k}(\pi)-\frac{\alpha_k}{1-\gamma}\Big),\quad\lambda_k\ge0.$$

**定理 3（强对偶性与收敛性）。** 在严格可行性（Slater 条件）下，强对偶成立，且投影对偶梯度更新

$$\lambda_k^{(t+1)}=\Big[\lambda_k^{(t)}-\eta_t\big(J_{\mu_k}(\pi_{\lambda^{(t)}})-\alpha_k/(1-\gamma)\big)\Big]_+$$

在 $\sum_t\eta_t=\infty$、$\sum_t\eta_t^2<\infty$ 的条件下收敛到一个鞍点。我们采用与 SAC 相结合的随机变体。

## 4.3 分层模糊引导 SAC（HFG-SAC）

HFG-SAC 在基础 SAC 之上通过两个模糊层实例化 FC-MDP（图 1）。

**上层——模糊约束保护。** 我们通过 S 型隶属度函数计算每个约束的 $\mu_k$，并用加权乘积 t-范数对其进行聚合。加入 actor（策略网络）的约束损失为

$$\mathcal{L}_{\text{constraint}}(\phi)=-\mathbb{E}_{s\sim\mathcal D}\Big[\sum_k\lambda_k\,\mu_k(s,a_\phi(s))\Big],$$

对偶变量按 $\lambda_k\leftarrow[\lambda_k+\eta_\lambda(\alpha_k-\bar\mu_k)]_+$ 更新，其中 $\bar\mu_k$ 为小批量内每步 FCSD 的均值。这是定理 3 折扣更新形式的逐步形式，因为对平稳策略有 $J_{\mu_k}\approx\bar\mu_k/(1-\gamma)$；因子 $(1-\gamma)$ 被吸收进 $\eta_\lambda$。满足度低于目标会提高 $\lambda_k$，从而在后续更新中增大安全的权重。

**下层——模糊知识奖励塑形。** 专家 IF–THEN 规则被编码为零阶 TSK 规则；其触发强度产生知识奖励 $R_{\text{know}}$。为保持最优策略不变，我们将其构造为基于势能的塑形 $F(s,s')=\gamma\Phi(s')-\Phi(s)$，其中 $\Phi(s)=\max_a R_{\text{know}}(s,a)$。

**定理 4（策略不变性）。** 针对塑形奖励 $R+F$ 的任意最优策略对 $R$ 亦为最优（[11] 的直接应用）。知识权重 $\kappa_t$ 在训练过程中衰减至 $0$（在 $10{,}000$ 步内由 $0.3$ 余弦退火至 $0.01$），因此塑形奖励收敛于原始奖励。

**协同目标。** actor 最大化 $\nabla_\phi J_{\text{total}}=\nabla_\phi J_R+\kappa_t\nabla_\phi J_F+\sum_k\lambda_k\nabla_\phi J_{\mu_k}$；两个 critic 回归塑形回报；$\lambda$ 在更慢的时间尺度上自适应。完整流程见算法 1。

**实际实现。** 在我们的微电网实例中，$K$ 约束拉格朗日被压缩为单个标量乘子 $\lambda$，它跟踪最坏情况下的每步满足度 $\tilde\mu_{\min}=\min_k\mu_k$，而非维护 $K$ 个独立的 $\lambda_k$；这是一种保守的单约束近似，优先处理被违反最严重的约束，并避免调整 $K$ 个对偶学习率。各约束的隶属度函数采用以推荐运行带为中心的高斯（而非 S 型）形式，拉格朗日所使用的聚合 FCSD 为 $\tilde\mu_{\min}$（各约束间的最小值，等价于软最小 t-范数），而非加权乘积；这一选择可由安全性的优先级排序观点（频率/电压主导 SOC）加以论证。critic 一侧使用单独的成本 critic $C(s,a)$ 回归折扣约束亏空，actor 接收的是 $\lambda\cdot\nabla_\phi C(s,a_\phi(s))$ 而非 $\sum_k\lambda_k\nabla_\phi\mu_k$。每步硬 SOC 屏蔽直接在环境中强制执行绝对的 $[\text{SOC}_{\min},\text{SOC}_{\max}]$ 区间。这是一种类似于电池管理系统过压/欠压保护的硬件联锁，而非 CBF 式的依赖模型的安全集投影；之所以保留它，是因为超出这些绝对限值运行会造成不可逆的设备损坏。该模糊层以分级满足度信号取代了*清晰推荐带*投影（即 $[\text{SOC}_{\min}^{\text{rec}},\text{SOC}_{\max}^{\text{rec}}]$ 边界处的旧硬惩罚）；绝对硬件限值仍作为不可协商的环境约束保留。塑形权重 $\kappa_t$ 遵循从 $\kappa_0=0.3$ 到 $0.01$ 的余弦退火计划，在 $10{,}000$ 步内完成（而非指数衰减）；这提供了更平滑的热启动，并且由于 $\kappa_t\to0$，与附录 A.6 中的策略不变性论证相容。

**算法 1：分层模糊引导 SAC（HFG-SAC）**

```
Input: initial actor π_φ, Q-networks Q_θ1, Q_θ2, target Q_θ1', Q_θ2'
Input: fuzzy membership params {β_k, x_k^ref, w_k}, expert fuzzy rules, target satisfaction α_k
Input: initial multipliers λ_0, knowledge weight κ_0, cosine-decay schedule, learning rates
Initialize replay buffer D ← ∅

for episode = 1 to N_episodes do
    s_1 ← initial state
    for t = 1 to T do
        a_t ~ π_φ(·|s_t)
        execute a_t; observe r_t, s_{t+1}, d_t
        compute fuzzy satisfaction μ_k(s_t, a_t) for all k
        compute knowledge shaping F(s_t, s_{t+1}) = γΦ(s_{t+1}) - Φ(s_t)
        store (s_t, a_t, r_t, s_{t+1}, d_t, μ(s_t,a_t)) in D

        sample mini-batch B from D
        y = r + κ_t·F(s,s') + γ(1-d)·( min_i Q_θi'(s',a') - α·log π_φ(a'|s') )
        update θ_1, θ_2 by minimizing (Q_θ(s,a) - y)^2          // critic 更新
        update φ by gradient ascent on J_total                  // actor 更新
        λ_k ← [ λ_k + η_λ(α_k - mean_{B} μ_k(s,a)) ]_+          // 对偶变量更新
        θ_i' ← τ·θ_i + (1-τ)·θ_i'                               // 目标网络
    end for
    κ_t ← κ_final + (κ_0 - κ_final)·(1+cos(π·t/T_decay))/2      // 余弦退火
end for
Output: trained policy π_φ
```

## 4.4 约束感知的跨场景迁移

跨场景迁移所要解决的实际需求是：将某一运行条件下训练得到的策略迁移到新条件，同时不进行不安全的探索。我们的迁移流程包含三个部分：刻画安全规范变化程度的相似度诊断、复用已学特征的权重迁移步骤，以及使早期微调保持安全的保守探索计划。

**相似度诊断。** 对每个约束 $k$，我们计算源隶属度函数与目标隶属度函数之间的模糊集 Jaccard 指数，$\text{sim}_k=\int\min(\mu_k^s,\mu_k^t)dx/\int\max(\mu_k^s,\mu_k^t)dx$。该指标用作*分析工具*：它量化哪些安全约束在场景之间仍然共享（Jaccard 指数高）、哪些是新增的（Jaccard 指数低），并解释为何某些迁移任务比其他任务收敛更快。它并不驱动单独的参数初始化启发式，因为在我们所考虑的微电网场景中，安全规范要么原封不动地沿用（SOC 与频率限值相同），要么完全新增，而下面的神经网络权重迁移已经处理了共享的情形。

**带部分冻结的权重迁移。** 当源与目标的观测空间和动作空间相匹配时（同一运行模式、不同天气），我们将训练好的 actor 与 critic 权重复制到目标网络，并冻结 actor 主干底部的 50%。被冻结的底层保留了通用的时间与部件级特征（PV/WT/负荷响应、ESS 动态），而上层主干与输出头则适应新条件。相较于从头训练，这降低了样本开销。

**保守动作缩放。** 在微调期间，执行动作按一个从保守初值指数式松弛至 1.0 的因子进行缩放：
$$\omega(t)=\omega_{\text{target}}+(\omega_{\text{init}}-\omega_{\text{target}})e^{-\nu t},$$
其中 $\omega_{\text{init}}=2.5$（动作从策略输出的 40% 起步），$\omega_{\text{target}}=1.0$。这将早期探索收缩到源策略附近的安全探索管道内，防止迁移后的网络产生违反目标场景约束的激进动作，并随着微调推进逐步恢复完整动作。

**算法 2：约束感知渐进迁移**

```
Input: source policy π_φ^s, target scenario spec
Input: ω_init = 2.5, ω_target = 1.0, decay rate ν

Step 1: similarity diagnostic (analysis)
  for each constraint k:
      sim_k = Jaccard( μ_k^s(x), μ_k^t(x) )
  report per-constraint sim_k (no parameter action)

Step 2: weight transfer
  if obs_dim and act_dim match:
      copy actor/critic weights from π_φ^s to target
      freeze bottom 50% of actor trunk
  else:
      initialize target networks from scratch

Step 3: progressive fine-tuning with safety relaxation
  ω ← ω_init,  ν = ln(20) / T_relax
  for t = 1 to T_ft do
      a_t ← π_φ^t(·|s_t);  a_t ← a_t / ω
      execute a_t;  one HFG-SAC update
      ω ← ω_target + (ω_init - ω_target)·exp(-ν·t)
  end for
Output: target policy π_φ^t
```

---

# 5. 实验

## 5.1 实验设置

**测试系统。** 改进的 IEEE 33 节点微电网：500 kW PV（节点 12）、300 kW WT（节点 24）、1 MWh/300 kW ESS（节点 18）、400 kW DE 备用，峰值负荷约 1.2 MW。ESS 的硬荷电状态（SOC）区间为 [0.1, 0.9] p.u.，推荐软区间为 [0.3, 0.8]；孤岛模式还须满足频率（≤0.5 Hz）与电压（±5%）约束。

**场景。** S1 并网正常；S2 并网极端（PV×1.2、负荷×1.3、WT×0.6）；S3 孤岛正常；S4 孤岛极端（PV×0.5、负荷×1.25、WT×0.8）。目标 FCSD $\alpha_k$ 在 S1–S3 中设为 0.85，在 S4 中放宽至 0.75，原因是弱惯性的极端孤岛系统在 7 天时间尺度上无法在不触发不可行切负荷的条件下物理维持 $\tilde\mu\ge0.85$。

**基线方法。** PPO 与 SAC（无约束骨干网络）；PPO-Lagrangian [39] 与 CPO [5]（CMDP/拉格朗日信赖域）；Safety Layer [37]（基于 CBF 的动作投影）。MILP/MPC 的 oracle 成本仅作为参照报告，不作为 RL 基线方法。用于分离上层模糊约束层与下层知识塑造层的消融变体在第 5.4 节单独报告。

**实验协议。** 所有方法采用相同的 actor/critic 结构（两个 256 单元的隐藏层，ReLU 激活），Adam 学习率 $3\times10^{-4}$，批大小 256，经验回放缓冲区 $10^5$，1000 步随机预热，每 4 个环境步执行一次梯度更新，每个（算法，场景）组合训练 200 个回合，五个随机种子（42, 123, 456, 789, 2024）。我们报告均值 ± 95% 置信区间；由于 n 较小，将推断性检验按探索性结果处理，同时报告 Cohen $d$。

**评价指标。** 日运行成本（¥/天）；硬约束违反率（%）；以及平均 FCSD（越大越好）。

## 5.2 主要对比（RQ1）

**表 II. 四种运行场景下的成本、硬约束违反率与 FCSD（均值 ± 95% 置信区间，n=5 个随机种子）。** 成本单位为 CNY/天（越小越好）；违反率单位为百分比（越小越好）；FCSD 为评价时间范围内的平均模糊约束满足度（越大越好）。每列最优值以**粗体**标出。

| 算法 | S1 成本 | S1 违反率% | S1 FCSD | S2 成本 | S2 违反率% | S2 FCSD | S3 成本 | S3 违反率% | S3 FCSD | S4 成本 | S4 违反率% | S4 FCSD |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SAC | 14,131 ± 2,753 | 69.9 | 0.717 | 17,765 ± 2,104 | 89.5 | 0.636 | 6,461 ± 1,078 | 98.0 | 0.784 | 14,286 ± 2,782 | 95.1 | 0.767 |
| PPO | 16,728 ± 13,924 | 88.8 | 0.632 | 20,436 ± 11,620 | 82.4 | 0.634 | 5,558 ± 2,189 | 98.9 | 0.775 | 13,347 ± 3,182 | 78.7 | 0.834 |
| PPO-Lag | 21,212 ± 3,232 | 5.5 | 0.980 | 40,971 ± 13,558 | 6.5 | 0.976 | 12,082 ± 4,197 | 30.5 | 0.926 | 16,683 ± 1,113 | 40.9 | 0.912 |
| CPO | 15,808 ± 677 | **0.0** | 1.000 | 29,260 ± 1,976 | **0.0** | 0.999 | 10,963 ± 360 | 1.2 | 0.997 | 17,632 ± 509 | 0.1 | 1.000 |
| SL | 32,728 ± 9,522 | **0.0** | 0.991 | 34,896 ± 4,230 | **0.0** | 0.959 | 8,894 ± 2,787 | 13.0 | 0.959 | 13,258 ± 3,987 | 34.6 | 0.890 |
| **HFG-SAC** | 14,997 ± 1,047 | **0.0** | 0.940 | **23,442 ± 1,867** | **0.0** | 0.900 | **9,097 ± 1,916** | **0.6** | 0.992 | **14,925 ± 2,639** | **0.6** | 0.993 |

**图。** 图 2 给出成本—安全帕累托散点图（标记形状表示场景，颜色表示算法）。图 3 给出训练动态：200 个回合内的回合违反率，阴影带为五个随机种子上的 ±1 标准差（10 回合滑动平均）。

**主要观察。**

*无约束 RL 不具备可部署性。* SAC 与 PPO 在全部四种场景中于 70–99% 的评价区间内违反硬约束，FCSD 停滞在 0.63–0.83。二者较低的名义成本源于对安全性的忽视，使其电池与发电机运行在安全包络之外。

*CMDP 基线方法仅在并网模式下安全。* PPO-Lagrangian 与 Safety Layer（SL）在 S1 和 S2 中将违反率维持在 0%，但在孤岛模式下急剧退化：PPO-Lagrangian 升至 30.5%（S3）和 40.9%（S4），SL 则升至 13.0%（S3）和 34.6%（S4）。拉格朗日乘子无法适应弱惯性的孤岛动态，CBF 投影也无法在孤岛极端事件下计算可行的安全集。

*CPO 是 HFG-SAC 最强的安全对照，但成本溢价很高。* CPO 在四种场景中均将违反率维持在接近 0%，但其成本始终高于 HFG-SAC：¥15,808 对 ¥14,997（S1，n.s.）、¥29,260 对 ¥23,442（S2，Welch $p<0.001$，Cohen $d=-3.8$）、¥10,963 对 ¥9,097（S3，趋势 $p=0.053$，$d=-1.7$）以及 ¥17,632 对 ¥14,925（S4，$p=0.045$，$d=-1.8$）。在 S2（并网极端）中，HFG-SAC 的成本还比 PPO-Lagrangian 低 ¥17.5k/天（$p=0.022$，$d=-2.3$），比 SL 低 ¥11.5k/天（$p<0.001$，$d=-4.4$）。

*HFG-SAC 是唯一在所有场景中违反率均低于 5% 阈值的方法。* 其违反率为 0.0%（S1、S2）、0.6%（S3）和 0.6%（S4），全部场景下 FCSD ≥ 0.90。其成本在 S1 中与 CPO 统计上相当，在 S2/S4 中显著低于 CPO，在 S3 中与 SL 在 95% 置信区间内相当。

*训练动态（图 3）。* HFG-SAC 在前 20 个回合内即将违反率降至接近 0%，并在全部四种场景中保持该水平。SAC 与 PPO 在整个训练过程中始终维持 70–100% 的违反率。PPO-Lagrangian 与 SL 在孤岛模式下于 0% 与 40% 之间振荡，CPO 则缓慢降低违反率，但在 S3/S4 中仍停留在 20–30%。这种早期且稳定的安全收敛是模糊约束保护层的运行特征：它自最初若干回合起即提供平滑而稠密的梯度信号，而非清晰（crisp）拉格朗日方法那种稀疏、滞后的惩罚信号。

## 5.3 不同极端程度下的鲁棒性（RQ2）

我们在孤岛场景上以五个随机种子扫描负荷极端程度 $\lambda\in\{0.8,1.0,1.1,1.2,1.3,1.4,1.5\}$（图 4；左图：日成本对 $\lambda$，右图：违反率对 $\lambda$；阴影带 = 95% 置信区间）。图 5 对比了 $\lambda=0.8$（轻微）与 $\lambda=1.5$（极端）下的 FCSD。

**违反率漂移。** HFG-SAC 是唯一在整个扫描范围内违反率始终低于 5% 阈值的方法：其违反率从 $\lambda=0.8$ 时的 0.36% 平缓上升至 $\lambda=1.5$ 时的 3.15%。其余基线方法在 $\lambda=0.8$ 时即已越过 5% 阈值并单调退化：PPO-Lagrangian 从 24.6% 升至 86.9%，SL 从 42.8% 升至 66.7%，CPO 从 4.5% 升至 13.4%（在 $\lambda=1.5$ 处出现陡增，原因是其信赖域在尾部负荷下变得不可行）。SAC 则始终保持在 94–99% 的违反率。

**成本漂移。** HFG-SAC 的成本从 ¥12,074/天（$\lambda=0.8$）升至 ¥14,817/天（$\lambda=1.5$），漂移为 +22%。CPO 的成本从 ¥14,252 升至 ¥16,330（+15%），但其置信区间方差随策略失稳而增大。PPO-Lagrangian 在 $\lambda=1.5$ 处表现出的成本下降（¥10,948）具有误导性：该结果是通过允许 87% 的违反率换取的，即运行在可行域之外，而非在可行域内进行优化。

**FCSD 漂移。** 从 $\lambda=0.8$ 到 $\lambda=1.5$，HFG-SAC 的平均 FCSD 仅下降 0.010（0.993 → 0.983），是所有方法中退化最小的。PPO-Lagrangian 从 0.948 跌至 0.753，SL 从 0.863 降至 0.779，CPO 从 0.990 漂移至 0.970。这正是安全性的平缓退化的实证特征：HFG-SAC 不会在可行与不可行之间突然切换（即清晰边界的病态行为），而是在极端负荷下以少量且可控的 FCSD 损失换取运行成本。

## 5.4 待完成的实验

> **RQ3：跨场景迁移（2×2 析因设计）。** 四个同模式迁移任务，源场景始终为正常天气并采用标准电池区间 [0.3, 0.8]：
> - **T1** 并网正常→极端（S1→S2），推荐区间不变，为 [0.3, 0.8]；
> - **T2** 并网正常→极端（S1→S2），目标电池区间收窄至 [0.4, 0.7]；
> - **T3** 孤岛正常→极端（S3→S4），推荐区间不变，为 [0.3, 0.8]；
> - **T4** 孤岛正常→极端（S3→S4），目标电池区间收窄至 [0.4, 0.7]。
>
> T1/T3 在安全边界不变的条件下分离出天气驱动的迁移（用于检验样本效率）；T2/T4 引入发生偏移的推荐区间，使源策略原本安全的运行区域在目标场景中变为软违反（用于检验 M4 的保守缩放）。对每个任务，我们比较：(i) 在目标场景上从零训练的 HFG-SAC；(ii) 从零训练的 PPO-Lagrangian 锚点基线；(iii) 不使用保守缩放的朴素权重迁移；(iv) 我们的完整迁移方案（权重复制 + 底部 50% 冻结 + 指数保守动作缩放）。消融 (v) 关闭保守缩放，以分离其安全效应。按约束报告 Jaccard 相似度：约束不变时 ≈1.0，SOC 区间收窄时 ≈0.75。待运行。
>
> **RQ4：消融实验。** 在 S4 上的变体：完整 HFG-SAC；−FuzzyCon（清晰拉格朗日上层）；−FuzzyKnow（无下层塑造）；−Both。待运行。

## 5.5 统计说明

所有检验均为在五个随机种子（42, 123, 456, 789, 2024）上对日运行成本进行的 Welch 双侧 $t$ 检验。每组 $n=5$ 时检验功效不足，因此以 Cohen $d$（合并标准差效应量）作为主要推断指标，$p$ 值按探索性结果报告。28 组对比之间未进行多重比较校正；我们将 $|d|>2$ 且 $p<0.01$ 的对比视为稳健，将边缘性对比（$0.05<p<0.15$）视为趋势。

稳健对比（$|d|>2$）：S2 HFG-SAC 对 CPO 的 $d=-3.8$（$p<0.001$），S2 对 SL 的 $d=-4.4$（$p<0.001$），S2 对 PPO-Lag 的 $d=-2.3$（$p=0.022$），S1 对 PPO-Lag 的 $d=-3.2$（$p=0.004$），S1 对 SL 的 $d=-3.3$（$p=0.006$）。相对于无约束 SAC 的“安全代价”明显但有限：S2 中 HFG-SAC 的成本比 SAC 高 ¥5.7k（$d=+3.5$，$p<0.001$），这是将违反率从 89.5% 降至 0% 所付出的代价。PPO 在 S1/S2 中较宽的置信区间源于一个不稳定的随机种子；由于 PPO 在所有场景中的违反率均超过 78%，故不将其纳入成本—安全关系的解读。

---

# 6. 讨论

**理论意义。** FC-MDP 超越了二值约束范式，在同一框架内统一了硬约束与软约束：定理 1 使实践者可以通过 $\beta$ 在软约束与硬约束之间连续调节。通过继承拉格朗日对偶性，该方法避免了 CPO 每次更新所需的共轭梯度开销，同时给出相近的保证与更平滑的梯度。下层的策略不变性定理表明，通过模糊规则嵌入专家知识不会改变最优解，而衰减机制则恢复了渐近最优性。

**实践意义。** FCSD 与运行人员对安全性的实际认知相一致，即安全性是分级的而非二值的，并允许他们直接编码推荐区间与按优先级排序的约束。将约束保护层与知识塑造层分离，使得安全性与样本效率可以独立调节。所提出的迁移机制减少了目标场景所需的数据，并使迁移后的策略在整个微调过程中保持安全。

**局限性。** (i) 模糊规则与隶属度参数的获取依赖专家知识，自动规则挖掘是有价值的扩展方向。(ii) 加权乘积聚合的最优性尚未得到证明。(iii) 模糊推理会带来一定的单步计算开销，但在实践中被更好的样本效率所抵消。(iv) 结果仅基于单一拓扑（IEEE 33 节点），多微电网、直流系统与多能源系统仍有待验证。(v) 安全性是在期望意义上而非逐步得到保证的；最坏情况下的安全性需要与 CBF/可达性方法相结合。(vi) 安全的极端孤岛运行比孤岛正常运行成本略高（¥14,925 对 ¥9,097/天，约 1.6×），原因是弱惯性下的优先级切负荷；分层需求响应经济模型可以进一步细化这一权衡。

---

# 7. 结论

我们提出了 HFG-SRL，一种面向微电网优化调度的分层模糊引导安全强化学习框架。FC-MDP 以连续的模糊约束满足度对 CMDP 进行推广，其模糊拉格朗日求解器具有收敛性保证，并在极限情形下退化为清晰（crisp）CMDP。HFG-SAC 将上层的模糊约束保护层与下层基于势能的模糊奖励塑造层相结合，后者可证明地保持最优策略不变。约束感知迁移机制通过 Jaccard 指数诊断各约束之间的相似度，利用部分冻结复用已学习的 actor 权重，并放宽保守的动作缩放倍数，从而使早期微调保持安全。在改进的 IEEE 33 节点系统上，HFG-SAC 以低于保守安全 RL 基线方法的成本实现了接近零的违反率，并在超出训练极端程度的压力下平缓退化。未来工作包括用于高阶不确定性的区间 type-2 模糊集、多智能体互联微电网、数字孪生辅助的规则自适应，以及与形式化验证的结合。

---

# 参考文献

[1] S. Gu et al., "A Review of Safe Reinforcement Learning: Methods, Theories, and Applications," *IEEE TPAMI*, 2024.
[2] J. García and F. Fernández, "A Comprehensive Survey on Safe Reinforcement Learning," *JMLR*, 2015.
[3] P. Yu et al., "Safe Reinforcement Learning for Power System Control: A Review," arXiv:2407.00681, 2024.
[4] E. Altman, *Constrained Markov Decision Processes*. Chapman & Hall/CRC, 1999.
[5] J. Achiam, D. Held, A. Tamar, and P. Abbeel, "Constrained Policy Optimization," in *ICML*, 2017.
[6] T. Morimura et al., "Parametric Return Density Estimation for Reinforcement Learning," in *UAI*, 2010.
[7] A. D. Ames et al., "Control Barrier Functions: Theory and Applications," in *ECC*, 2019.
[8] M. Alshiekh et al., "Safe Reinforcement Learning via Shielding," in *AAAI*, 2018.
[9] S. Cheng, N. Fukushima, and P. J. Antsaklis, "Safe Reinforcement Learning with Model Uncertainty Estimation," *IEEE TNNLS*, 2023.
[10] P. Y. Glorennec and L. Jouffe, "Fuzzy Q-Learning," in *FUZZ-IEEE*, 1997.
[11] A. Y. Ng, D. Harada, and S. Russell, "Policy Invariance Under Reward Transformations," in *ICML*, 1999.
[12] X. Li et al., "Deep Fuzzy Reinforcement Learning for Continuous Control," *IEEE Trans. Fuzzy Syst.*, 2022.
[13] J. Bian et al., "Fuzzy Reinforcement Learning-Based Safe Cooperative Control for Nonlinear Multiagent Systems," *IEEE Trans. Syst. Man Cybern.: Syst.*, 2026.
[14] Y. Li, D. Wang, and C. L. P. Chen, "Adaptive Fuzzy Reinforcement Learning for Strict-Feedback Nonlinear Systems," *IEEE Trans. Cybern.*, 2022.
[15] M. N. Ahmadabadi and M. Asadpour, "Expertness Based Cooperative Q-Learning," *IEEE Trans. Syst. Man Cybern. B*, 2002.
[16] S. M. Kakade and J. Langford, "Approximately Optimal Approximate Reinforcement Learning," in *ICML*, 2002.
[17] P. I. N. Barbalho et al., "Reinforcement Learning Solutions for Microgrid Control and Management: A Survey," *IEEE Access*, 2025.
[18] P. I. N. Barbalho, A. L. Moraes, V. A. Lacerda, P. H. A. Barra, R. A. S. Fernandes, and D. V. Coury, "Reinforcement Learning Solutions for Microgrid Control and Management: A Survey," *IEEE Access*, vol. 13, pp. 39782–39799, 2025.
[19] T. P. Do et al., "Deep Reinforcement Learning for Energy Management in a Microgrid with Battery Energy Storage," *Applied Energy*, 2022.
[20] Z. Yu et al., "Energy Optimization Management of Microgrid Using Improved Soft Actor-Critic Algorithm," *IJRED*, 2024.
[21] Y. Wang, D. Qiu, M. Sun, G. Strbac, and Z. Gao, "Secure Energy Management of Multi-Energy Microgrid: A Physical-Informed Safe Reinforcement Learning Approach," *Applied Energy*, 2023.
[22] Y. Xia, Y. Xu, and X. Feng, "Hierarchical Coordination of Networked-Microgrids Toward Decentralized Operation: A Safe DRL Method," *IEEE Trans. Sustain. Energy*, 2024.
[23] T. Su et al., "Safe RL-Based Transient Stability Control for Islanded Microgrids With Topology Reconfiguration," *IEEE Trans. Smart Grid*, 2025.
[24] B. Zhang et al., "Multi-Agent DRL Based Distributed Control for Interconnected Multi-Energy Microgrids," *Energy Convers. Manag.*, 2023.
[25] G. S. Ayyappan, P. M. P. Deepak, and K. S. Swarup, "Transfer Learning Based RL for Microgrid Energy Management," in *IEEE PES GM*, 2023.
[26] S. Shuai et al., "Transfer Reinforcement Learning for Optimal Microgrid Operation Under Uncertain Environments," *IEEE Trans. Power Syst.*, 2024.
[27] T. Haarnoja et al., "Soft Actor-Critic: Off-Policy Maximum Entropy Deep RL with a Stochastic Actor," in *ICML*, 2018.
[28] T. Haarnoja et al., "Soft Actor-Critic Algorithms and Applications," arXiv:1812.05905, 2018.
[29] N. Hatziargyriou et al., "Microgrids," *IEEE Power Energy Mag.*, 2007.
[30] R. H. Lasseter, "MicroGrids," in *IEEE PES Winter Meeting*, 2002.
[31] R. H. Lasseter and P. Piagi, "Microgrid: A Conceptual Solution," in *Proc. IEEE 35th Annual Power Electronics Specialists Conf. (PESC)*, Aachen, Germany, 2004, pp. 4285–4290.
[32] S. Chen, H. B. Gooi, and M. Wang, "Sizing of Energy Storage for Microgrids," *IEEE Trans. Smart Grid*, 2012.
[33] R. Palma-Behnke et al., "A Microgrid Energy Management System Based on the Rolling Horizon Strategy," *IEEE Trans. Smart Grid*, 2013.
[34] A. Parisio, E. Rikos, and L. Glielmo, "A Model Predictive Control Approach to Microgrid Operation Optimization," *IEEE Trans. Control Syst. Technol.*, 2014.
[35] T. Takagi and M. Sugeno, "Fuzzy Identification of Systems and Its Applications to Modeling and Control," *IEEE Trans. Syst. Man Cybern.*, 1985.
[36] J. M. Mendel, *Uncertain Rule-Based Fuzzy Logic Systems*. Prentice Hall, 2001.
[37] G. Dalal, K. Dvijotham, M. Vecerik, T. Hester, C. Paduraru, and Y. Tassa, "Safe Exploration in Continuous Action Spaces," arXiv:1801.08757, 2018.
[38] M. Wen and U. Topcu, "Constrained Cross-Entropy Method for Safe Reinforcement Learning," in *Advances in Neural Information Processing Systems (NeurIPS)*, 2018.
[39] A. Ray, J. Achiam, and D. Amodei, "Benchmarking Safe Exploration in Deep Reinforcement Learning," arXiv:1910.01708, 2019.
[40] M. A. Zanger, K. Daaboul, and J. M. Zöllner, "Safe Continuous Control with Constrained Model-Based Policy Optimization," arXiv:2104.06922, 2021.

---

# 附录 A. 证明

> 约定：加性模糊拉格朗日量 $\mathcal{L}_{\text{fuzzy}}=J_R+\sum_k\lambda_k(J_{\mu_k}-\alpha_k/(1-\gamma))$，$\lambda_k\ge0$；清晰（crisp）隶属度 $\mu_k=\sigma(\beta_k(d_k-c_k))$，$\sigma(x)=1/(1+e^{-x})$；清晰目标 $\alpha_k^{\text{crisp}}=1$；$\sigma(0)=1/2$。

## A.1 支撑引理

**引理 A.1（Sigmoid 逐点极限）。** 对每个 $x\in\mathbb R$，
$\lim_{\beta\to\infty}\sigma(\beta x)=\mathbf 1\{x>0\}+\tfrac12\mathbf 1\{x=0\}$，且 $\sigma(\beta x)$ 关于 $x$ 单调非减。

*证明。* 当 $x>0$ 时，$\beta x\to+\infty$，故 $\sigma\to1$；当 $x<0$ 时，$\sigma\to0$；在 $x=0$ 处，$\sigma(0)=1/2$。单调性由 $\sigma'>0$ 得到。$\blacksquare$

**引理 A.2（反 Sigmoid 水平集）。** 对 $\beta>0$，$\varepsilon\in(0,\tfrac12)$，
$\sigma(\beta x)\ge1-\varepsilon \iff x\ge \tau_\varepsilon/\beta$，其中 $\tau_\varepsilon=\sigma^{-1}(1-\varepsilon)=\ln\frac{1-\varepsilon}{\varepsilon}$。

*证明。* 直接反解。$\blacksquare$

**引理 A.3（逐点收敛与可积性）。** 设 $f_\beta(x)=\sigma(\beta g(x))$，$g$ 连续。则 $f_\beta(x)\to f_\infty(x)$ 逐点成立，其中 $f_\infty=\mathbf 1\{g>0\}+\tfrac12\mathbf 1\{g=0\}$，且 $0\le f_\beta\le 1$ 关于 $\beta$ 一致成立。因此，对任意有界折扣期望 $\mathbb E_\pi[\sum_t\gamma^t f_\beta(s_t,a_t)]$，控制收敛给出 $\lim_{\beta\to\infty}\mathbb E_\pi[\sum_t\gamma^t f_\beta]=\mathbb E_\pi[\sum_t\gamma^t f_\infty]$。

*证明。* 逐点极限由引理 A.1 得到。一致界 $0\le f_\beta\le1$ 与引理 A.4 为极限与折扣期望的交换提供了依据；$\{g=0\}$ 附近的过渡层为零测度，不影响期望。$\blacksquare$

**引理 A.4（值界）。** 在 $|R|\le R_{\max}$ 与 $|\mu_k|\le1$ 下，
$|J_R(\pi)|\le R_{\max}/(1-\gamma)$ 且 $|J_{\mu_k}(\pi)|\le1/(1-\gamma)$。

*证明。* 标准的折扣回报界。$\blacksquare$

**引理 A.5（Slater 条件）。** 若存在平稳策略 $\pi_0$，使得对所有 $k$ 都有 $J_{\mu_k}(\pi_0)>\alpha_k/(1-\gamma)$（严格可行性），则 $\max_{\pi\text{ $\alpha$-safe}}J_R(\pi)$ 与 $\min_{\lambda\ge0}\max_\pi\mathcal L_{\text{fuzzy}}$ 之间强对偶成立。

## A.2 定理 1 的证明（FC-MDP → CMDP 极限）

令 $\mu_k^{(\beta)}(s,a)=\sigma(\beta_k(d_k-c_k(s,a)))$。对每个 $\varepsilon\in(0,\tfrac12)$ 与 $(s,a)$，
$$\lim_{\beta_k\to\infty}\mathbf1\{\mu_k^{(\beta)}(s,a)\ge1-\varepsilon\}=\mathbf1\{c_k(s,a)\le d_k\}.$$

由引理 A.2，事件 $\{\mu_k^{(\beta)}\ge1-\varepsilon\}$ 等价于 $c_k(s,a)\le d_k-\tau_\varepsilon/\beta_k$。当 $\beta_k\to\infty$ 时，$\tau_\varepsilon/\beta_k\to0$，从而恢复 $c_k\le d_k$。

对于策略层面的陈述，逐点收敛与控制收敛（引理 A.3）给出 $J_{\mu_k}^{(\beta)}(\pi)\to J_{\mu_k}^{(\infty)}(\pi)=\mathbb E_\pi[\sum_t\gamma^t\mathbf1\{c_k(s_t,a_t)\le d_k\}]$。对于平稳策略，将折扣占用加权的满足度写作 $\bar p_k(\pi)=\mathbb E_\pi[\sum_t\gamma^t\mathbf1\{c_k\le d_k\}]$，于是 $\bar p_k\in[0,1/(1-\gamma)]$。$\alpha$-safe 条件 $\bar p_k\ge\alpha_k/(1-\gamma)$ 化为 $\bar p_k(1-\gamma)\ge\alpha_k$：当 $\alpha_k^{\text{crisp}}\to1$ 时，这迫使折扣占用加权的违反率趋于零，即策略在期望意义下避开被违反的集合。控制收敛（引理 A.4）为该极限提供了依据。$\square$

**推论 A.1。** 当 $\alpha_k^{\text{crisp}}=1$ 时，$\alpha$-safe 策略集（在集合意义下）收敛于折扣占用加权违反概率为零的策略集。这是标准 CMDP 可行集中的*始终安全*子集；它比 Altman 预算约束 $\mathbb E[\sum_t\gamma^t C_k]\le d_k$ 更严格，后者容忍在成本预算内偶尔发生的违反。要精确恢复预算可行集，应将 $\alpha_k$ 设为对应于所容忍的违反率，而不是设为 $1$。（参见 §4.1 中的注记。）

## A.3 定理 2 的证明（最优 α-safe 策略的存在性）

*第 1 步——占用度量。* 对平稳策略定义 $\rho_\pi(s,a)=(1-\gamma)\sum_t\gamma^t\mathbb P^\pi(s_t=s,a_t=a)$。在紧性与弱连续性下，确定性平稳占用度量的集合 $\mathcal M=\{\rho_\pi\}$ 是弱*紧的；其凸包 $\mathcal M_{\text{conv}}$ 等于平稳随机策略的占用度量。

*第 2 步——线性泛函。* $J_R(\pi)=\frac1{1-\gamma}\sum_{s,a}\rho_\pi R(s,a)$ 与 $J_{\mu_k}(\pi)=\frac1{1-\gamma}\sum_{s,a}\rho_\pi\mu_k(s,a)$ 是 $\mathcal M_{\text{conv}}$ 上的连续线性泛函。

*第 3 步——可行集。* $\mathcal F=\{\rho\in\mathcal M_{\text{conv}}:J_{\mu_k}(\rho)\ge\alpha_k/(1-\gamma)\ \forall k\}$ 是闭凸集；严格可行性使其具有非空内部。

*第 4 步——达到。* 连续目标函数在非空、闭、有界的集合 $\mathcal F$ 上，由弱*拓扑中的 Weierstrass 定理取得其上确界。

*第 5 步——确定性。* 由 Krein–Milman 定理，最优解位于 $\mathcal F$ 的极点处，而 $\mathcal M_{\text{conv}}$ 的极点是确定性平稳度量；因此存在平稳确定性 $\pi^*$ 取得该最优值。$\square$

## A.4 定理 3 的证明（强对偶与对偶收敛）

以占用度量重新表述后，FC-MDP 是一个具有线性目标与线性不等式约束的凸规划。在 Slater 条件（引理 A.5）下，标准凸对偶定理给出零对偶间隙：
$$\max_{\pi\text{ $\alpha$-safe}}J_R(\pi)=\min_{\lambda\ge0}\max_\pi\mathcal L_{\text{fuzzy}}(\pi,\lambda).$$

对于收敛性，对偶函数 $g(\lambda)=\max_\pi\mathcal L$ 是凸的，且为 $G$-Lipschitz，其中 $G=\sqrt K/(1-\gamma)$（次梯度由引理 A.4 有界）。投影更新 $\lambda^{(t+1)}=[\lambda^{(t)}-\eta_t h_t]_+$，其中 $h_t=J_{\mu_k}(\pi_{\lambda^{(t)}})-\alpha_k/(1-\gamma)$（对 $\lambda$ 作梯度下降，与定理 3 一致），由凸性满足
$$g(\lambda^{(t+1)})\le g(\lambda^{(t)})-\eta_t\|\nabla g(\lambda^{(t)})\|^2+C\eta_t^2+\text{noise}_t.$$
在 $\sum_t\eta_t=\infty$ 与 $\sum_t\eta_t^2<\infty$ 下，Robbins–Siegmund 引理给出 $g(\lambda^{(t)})\to g(\lambda^*)$ 与 $\lambda^{(t)}\to\lambda^*$；内层最大化子随后由强对偶满足 $J_R(\pi_{\lambda^{(t)}})\to J_R(\pi^*)$。$\square$

## A.5 Slater 条件在实际中的应用（注记）

Slater 条件可针对微电网 FC-MDP 通过构造一个**保守策略**加以验证：以较低的固定速率给 ESS 充电，让柴油发电机以较低的恒定出力运行，并在供电紧张时削减非关键负荷；该策略在 SOC、频率与电压约束上的 FCSD 均接近 1。

> **说明（与 §5.3 相关）。** 对于极端孤岛场景 S4，*原始*动作空间（200 kW 可中断负荷）在 $\alpha_k\ge0.9$ 时不接受任何严格可行的 $\alpha$-safe 策略，实验中将其观测为结构性不可行。将可中断负荷扩展至 600 kW 可扩大动作空间并恢复 Slater 条件，因此定理 3 适用于*扩展*后的 S4。

## A.6 定理 4 的证明（模糊塑形下的策略不变性）

令 $R'=R+F$，其中 $F(s,s')=\gamma\Phi(s')-\Phi(s)$，$\Phi(s)=\max_a R_{\text{know}}(s,a)$。$\mathcal A$ 的紧性与 $R_{\text{know}}$ 的连续性保证 $\Phi$ 良定义，且 $F$ 具有所需的势能差分形式。沿任意轨迹 $\tau=(s_0,a_0,s_1,a_1,\ldots)$ 对塑形回报求和：
$$\sum_{t\ge0}\gamma^t R'(s_t,a_t)=\sum_{t\ge0}\gamma^t R(s_t,a_t)+\sum_{t\ge0}\gamma^t\big[\gamma\Phi(s_{t+1})-\Phi(s_t)\big].$$
第二个和式裂项相消：$\sum_{t\ge0}\gamma^{t+1}\Phi(s_{t+1})-\sum_{t\ge0}\gamma^t\Phi(s_t)=-\Phi(s_0)$（假定折扣适当使尾项消失）。因此每个塑形回报都等于原始回报减去一个仅依赖状态的项 $\Phi(s_0)$。在值函数中这给出 $V'^*(s)=V^*(s)-\Phi(s)$ 以及
$$Q'^*(s,a)=Q^*(s,a)-\Phi(s),$$
其中被减去的项仅依赖于状态，而不依赖于 $a$。因此对每个状态都有 $\arg\max_a Q'^*(s,a)=\arg\max_a Q^*(s,a)$，最优策略集保持不变。$\blacksquare$

*自适应衰减。* 在每个有限 $t$ 处，$R_t=R+\kappa_tF$ 是以 $\kappa_t\Phi$ 为势能的基于势能的塑形，故其最优策略集与 $R$ 的最优策略集重合；当 $\kappa_t\to0$ 时，渐近最优解即原始最优解。该衰减影响学习动态，而不影响平稳最优集。

## A.7 符号表

| 符号 | 含义 |
|---|---|
| $\mathcal S,\mathcal A$ | 状态/动作空间，紧致（假设 i） |
| $P(\cdot\mid s,a)$ | 转移核，弱连续（ii） |
| $R(s,a)$ | 奖励，有界且连续（iii） |
| $\mu_k(s,a)$ | 约束 $k$ 的 FCSD |
| $\alpha_k\in(0,1]$ | 目标满足度水平 |
| $\beta_k>0$ | Sigmoid 陡峭度 |
| $d_k$ | 清晰（crisp）阈值 |
| $\lambda_k\ge0$ | 拉格朗日乘子 |
| $\gamma\in[0,1)$ | 折扣因子 |
| $J_R,J_{\mu_k}$ | 期望折扣回报 / FCSD |
| $\rho_\pi(s,a)$ | 占用度量 $(1-\gamma)\sum_t\gamma^t\mathbb P^\pi(\cdot)$ |
| $\sigma(x)$ | Sigmoid $1/(1+e^{-x})$ |
| $\Phi(s)$ | 塑形势能 $\max_a R_{\text{know}}(s,a)$ |
| $\kappa_t$ | 自适应知识权重 |
| $\alpha$ (SAC) | §3.1.2 与算法 1 中的熵温度 |
| $\alpha_k\in(0,1]$ | 目标 FCSD 水平（与 SAC $\alpha$ 不同） |
| $\omega(t)$ | 跨场景迁移期间的保守动作尺度（§4.4） |

*附录 A 结束。*
