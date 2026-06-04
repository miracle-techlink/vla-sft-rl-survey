---
title: "VLA 真机后训练：四篇代表文章解读"
subtitle: "PLD · RECAP · VLA-OPD · RL-for-VLA-Generalization"
author: "刘越（同济大学计算机学院 空间智能课题组）"
date: "2026-06"
documentclass: article
classoption: [a4paper]
geometry: [top=18mm, bottom=20mm, left=20mm, right=20mm]
CJKmainfont: "Noto Serif CJK SC"
mainfont: "Noto Serif CJK SC"
sansfont: "Noto Sans CJK SC"
monofont: "Noto Sans CJK SC"
fontsize: 11pt
linkcolor: "RoyalBlue"
urlcolor: "RoyalBlue"
colorlinks: true
numbersections: true
---

# 导读

VLA（Vision-Language-Action）真机后训练的核心矛盾是：真机一条 rollout 30–60 秒且需人工监管，经验极贵，而直接对 VLA 跑 RL 会触发灾难性遗忘（千倍量级的梯度更新 + 稀疏奖励高方差 + 无教师锚定，使视觉/语言通用能力崩塌）。本文挑四篇代表文章，各自独立解读它们"如何把 RL 变得可行"：

- **PLD** 与 **RECAP** 是两条**无 teacher 的自改进（self-improving）**主线——一个让 RL 产物是数据，一个让 RL 信号走 conditioning 通道；
- **VLA-OPD** 是**有 teacher 的蒸馏**路线，用一个机制（有界 mode-seeking 的 Reverse-KL）同时拿到 SFT 的稠密与 RL 的闭环；
- **RL for VLA Generalization** 是一篇**训练动力学实证**，回答"SFT 与 RL 在 VLA 的不同泛化维度上各自负责什么"。

四篇放在一起，正好覆盖"自采数据 / 自算信号 / 借教师 / 机制实证"四个面。

---

# PLD：用残差 RL 做"部署对齐的数据生成"

> *Self-Improving VLA Models with Data Generation via Residual RL*（NVIDIA/CMU/UCB/UT-Austin, ICLR 2026, arXiv:2511.00091）

**一句话定位。** 冻结 VLA 主干，用 off-policy 残差 RL 训一个轻量专家去接管 VLA 的失败状态，用"基策略探针 + 残差救场"生成一批贴着 VLA 部署分布、含纠错行为的数据，再用普通 SFT 蒸馏回 VLA。**RL 的产物是数据，专家训完即弃。**

**核心机制。** 执行动作是冻结 VLA 与受限残差之和 $\bar a = a_b + a_\delta,\ a_b\sim\pi_b,\ a_\delta\in[-\xi,\xi]$。残差是结构独立的高斯策略（约 10M 参数，5 GB 显存可跑），**之所以用残差而非 LoRA**，是因为残差可以用任意 off-policy 算法训练（维护 replay buffer、老 rollout 反复用），且冻结主干从结构上根除遗忘。三阶段：

- **Stage 1（Learn）**：用 **Cal-QL** 训 critic（保守 + 校准，避免离线 RL 的 OOD Q 高估）+ RLPD 双 buffer（offline 50% 锚定"成功长什么样" + online 50% 提供新鲜信号 + LayerNorm 封顶 Q 外推 + 高 UTD）。一条真机 rollout 进 buffer 后参与约 250k 次梯度更新——**off-policy 复用是真机可行的根本**。
- **Stage 2（Probe，灵魂）**：前 $\alpha T$ 步**只让 VLA 自己走**（漂进它常翻车的状态），后段**残差接管救回**。消融 $\alpha=0.6$ 最优。
- **Stage 3（Distill）**：把数据喂普通 SFT，$\mathcal{L}_{\text{SFT}}=-\mathbb{E}_{(s,a^*)\sim\mathcal{D}^{\text{PLD}}}[\log\pi_{\text{VLA}}(a^*|s)]$。

**精髓——防遗忘机制在 Stage 2 而非 Stage 3。** SFT 的梯度幅度正比于"数据分布与当前策略分布的偏离"。probing 让数据起点落在 VLA 自己的访问分布上，于是目标 $a^*$ 与 $\pi_b(\cdot|s)$ 差异小、BC loss 初值小、梯度小、更新温和——用 KL 语言说，PLD 数据 biased toward base ⟶ SFT 诱导的 $D_{\mathrm{KL}}(\pi'\Vert\pi_b)$ 小 ⟶ 遗忘少。这与"事后冻结参数"（如 MIFO）本质不同：**PLD 从数据侧根除遗忘**。蒸馏后部署零额外开销（不再需要残差）。

**实验。** LIBERO 上 OpenVLA 91.8→99.2、$\pi_0$ 93.4→97.2；SimplerEnv 平均 71.8→96.6；真机 peg insertion 30/30，cluttered 未见物体 PLD 28/30 vs 人类数据 12/30（PLD 探针到角落状态并生成 recovery，人类/RL 数据从没访问过这些状态故卡死）。

**局限。** base 必须够强（>80% 可冲 99%，<50% 救不回）；只用稀疏二元奖励；**per-task 残差，跨任务无 skill 共享**；闭环残差蒸馏进开环 chunk，高动态任务会失效；reward 仅 episode 末尾 0/1，无 step 级信用分配。

---

# RECAP / π\*0.6：把 advantage 从 loss 通道挪到 conditioning 通道

> Physical Intelligence, *π\*0.6: a VLA That Learns From Experience*（arXiv:2511.14759，**未开源**）。RECAP = RL with Experience and Corrections via Advantage-conditioned Policies。

**一句话定位。** 一个**迭代式 offline RL** 配方：用分布式 value function 把 episode 级 0/1 成功标签升级成 step 级 advantage，把 advantage 当作**条件输入（conditioning）而非损失项（loss）**注入策略，用 advantage 加权的纯监督学习实现自改进，**完全绕开 PPO**。

**核心机制。** 三阶段 PRETRAIN→SPECIALIZE→ITERATE，每轮采数据→重训 value→重训策略。

- **value**：每条 episode 只标 1 bit，导出 step reward（成功末步 0 / 失败末步 $-C_{\text{fail}}$ / 其余 $-1$），无折扣下成功轨迹的 MC return $=-(\text{到成功的剩余步数})$，**value 学的就是"到成功的负剩余步数"**。用 **201-bin softmax + 交叉熵**（分布式回归）而非 MSE——因为 return 多峰、交叉熵梯度有界对失败极端负值鲁棒；用 **Monte-Carlo 而非 TD**——躲开大模型 + 稀疏奖励下的 TD 发散（value 只用来算 advantage 做条件、不进反向传播，故相对信号足矣）。
- **advantage → 条件**：$I_t=\mathbb{1}[A>\epsilon_\ell]$（后训练 advantage 用 $N=50$ 步 lookahead，阈值 per-task 校准）。策略目标是 CFG-style 双支 $-\log\pi_\theta(a|o,\ell)-\alpha\log\pi_\theta(a|I_t,o,\ell)$，advantage 实现为文本 prefix "Advantage: positive/negative"（**不改架构**），训练时 **random drop $I_t$ 30%** 联合训 conditional/unconditional 两支（省一遍 forward）。推理 $\beta=1$ 时直接用 conditional 支（单次 forward），$\beta>1$ 时做 classifier-free guidance 锐化。

**精髓。** (1) **每轮从 anchor $\pi_{\text{pre}}$ 重训**（而非增量更新）以避免多轮 drift——这是"流程侧"防遗忘，与 PLD 的"数据侧"防遗忘对照。(2) 失败 episode 不丢——既训 value（失败负样本）又以 $I_t$=False 教策略避错；demo/correction 强制 $I_t$=True（信任人类）。(3) **持续改进的引擎**：旧数据不变，但新 value 重算 advantage 使同一 step 的 $I_t$ 在不同轮**翻转**——策略不断把自己的水平当新及格线，advantage 永远衡量"相对当前的进步"。

**实验。** 制作意式咖啡**连续 13 小时**不中断、折陌生衣物 2 小时+；最难任务 throughput 翻倍以上、failure 约减半。同数据下显著优于 AWR 与 PPO。

**局限。** 未开源；单池只增不减、每轮从头重训，靠 PI 的集群 brute-force；**episode 级 advantage** 对"前 80% 子任务都对、最后装歪致整体失败"的长 episode，会把局部正确的段一并拉低。

---

# VLA-OPD：用有界 mode-seeking 蒸馏桥接 SFT 与 RL

> *Bridging Offline SFT and Online RL for VLA via On-Policy Distillation*（HKUST-GZ, arXiv:2603.26666）

**一句话定位。** 把对齐重述为"**在学生自生成轨迹上的稠密监督**"：在学生自己 rollout 的 on-policy 状态上，用一个冻结强 teacher 的 token 级 logits 做**有界 mode-seeking 的 Reverse-KL 蒸馏**，同时拿到 SFT 的稠密快收敛与 RL 的闭环纠错、抗遗忘。

**核心机制。** 三阶段闭环：学生 on-policy 采样（状态来自学生诱导分布，把失败状态纳入训练）→ 冻结 teacher 对每个时刻标注目标分布 $q_t=\pi_{tea}(a|s_t)$（不执行、只取 logits，**无需稀疏奖励**）→ 最大化负 Reverse-KL

$$\max_\theta\ \mathbb{E}_{s\sim\pi_\theta}\big[-D_{\mathrm{KL}}(\pi_\theta(\cdot|s)\,\Vert\,\pi_{tea}(\cdot|s))\big],$$

落到 token 级即用内禀奖励 $r_t=-\log\frac{\pi_\theta(a_t|s_t)}{\pi_{tea}(a_t|s_t)}$ 做 group policy gradient（学生 $\log\pi_\theta$ 项 stop-gradient，直接把 Reverse-KL 当 advantage）。

**精髓——为什么选 Reverse-KL（熵动力学）。** KL 方向决定 OOD 状态下的熵行为：**Forward-KL**（mode-covering，连 teacher 在 OOD 的高熵犹豫一起模仿）→ 熵爆炸、策略弥散；**Hard-CE**（只追 teacher top-1）→ 过早熵坍缩、丧失动作多样性；**Reverse-KL**（zero-forcing）→ 只承诺 teacher 最自信的主 mode、自动过滤 teacher 的 OOD 长尾不确定性，得到**有界熵**。实测 Forward-KL actor 熵冲到 ~2.0、Hard-CE 跌到 ~0.5、Reverse-KL 维持 ~1.2 的健康有界熵，对应最高最稳的成功率。

**实验。** LIBERO 上把 1-traj 极弱学生（Avg 48.9%）纯蒸馏抬到 87.4%、叠加 GRPO 达 93.4%（几乎复现 teacher 的 93.9%），收敛比 GRPO 基线快约 3×；on-policy 监督显著保住 unseen 任务能力（离线 SFT 的 unseen 性能崩溃）。

**局限（与 PLD/RECAP 的根本区别）。** VLA-OPD **必须有一个现成的强 teacher**——PLD 无 teacher（自采数据）、RECAP 无 teacher（advantage 自算），而 VLA-OPD 的样本效率与抗遗忘是"买"来的，前提是已存在比学生强的 expert。在多数真机场景这样的强 expert 并不存在，这是它相对无 teacher 路线最实质的劣势（论文也把"减少 teacher 依赖"列为 future work）。

---

# RL for VLA Generalization：SFT 记忆、RL 泛化在 VLA 上的实证

> *What Can RL Bring to VLA Generalization? An Empirical Study*（清华, NeurIPS 2025, arXiv:2505.19789）

**一句话定位。** 一篇训练动力学实证：把 VLA 泛化拆成 **视觉 / 语义 / 执行** 三个正交维度各构造 OOD 测试，系统比较 SFT（行为克隆）与 RL 微调在每个维度上的泛化收益。Backbone 用 OpenVLA，环境 ManiSkill pick-and-place，稀疏奖励。

**核心发现——三维度的"分工"格局**（用相对下降 $P=(\text{OOD}-\text{IND})/\text{IND}$ 度量，drop 越小越泛化）：

| 维度 | RL vs SFT | 证据 |
|---|---|---|
| **视觉** | **基本持平** | 两者都不会诱导出超过训练随机化范围的视觉鲁棒性 |
| **语义** | **RL 明显更优** | SFT 各子项 drop 约 −20%~−44%（未见物体 −23.1%、多容器 −44.2%），RL 普遍更轻 |
| **执行** | **RL 显著更优** | 差距最大：SFT 物体位姿 OOD drop −60.9%、机器人位姿 −63.5%，RL 仅 −13%~−20% 量级 |

关键锚点：RL 收敛时在**训练分布上与最强 SFT 相当**，但在**未见物体/桌面上高 42.6%**——RL 不靠拟合训练分布取胜，而是在 OOD 上拉开差距。这是"SFT 记忆训练分布、RL 泛化到 OOD"在 VLA 上的实证。

**机制解释。** (1) **执行维度——轨迹分布覆盖更广**：SFT 轨迹紧贴运动规划器的单一模式成簇，RL 轨迹张成更大工作空间；遇到初始位姿偏移或中途物体重置（demo 中从未出现）时，SFT "照常推进"而失败，RL 见过更广状态分布、能从失败抓取中恢复重对准。(2) **语义维度——技能与物体解耦**：反复试错让"抓取"动作不再依赖具体物体类型，迁移到未见物体。(3) **视觉维度**：鲁棒性主要由数据增强范围决定，而非优化范式。

**实践建议。** 先少量 SFT/warm-up 让 checkpoint 达到可优化的初始成功率，再 **PPO**——且发现 **VLA 上 PPO 稳定优于 GRPO/DPO**（机器人 POMDP 的非平稳性破坏 GRPO 的组相对优势估计，稀疏奖励使 DPO 难分优劣）。**分工：SFT 求训练分布内的基本能力，RL 专门负责语义接地与执行鲁棒，视觉鲁棒靠训练随机化解决。**

**与 LLM 上结论的异同。** 一致：SFT 记忆、RL 泛化的核心命题在具身 VLA 上成立。VLA 特有修正：① 泛化收益**维度依赖**（视觉维度持平，非全面碾压）；② 机制是机器人特有的"轨迹覆盖 + 技能解耦 + 失败恢复"，而非 LLM 的规则/推理层面；③ LLM 当红的 GRPO 在 VLA 反而不稳，**后训练经验不能直接平移**。

---

# 小结

| 文章 | 范式 | 把 RL 变可行的方式 | 防遗忘 | 关键代价 |
|---|---|---|---|---|
| **PLD** | 无 teacher 自改进 | RL 产物是**数据**（probing 生成对齐数据再蒸馏） | 数据侧（probing→小 KL） | base 须够强；per-task 残差 |
| **RECAP** | 无 teacher 自改进 | advantage 走 **conditioning** 通道，监督学习消费 | 流程侧（每轮回 anchor 重训） | 未开源；大算力；episode 级 advantage 太粗 |
| **VLA-OPD** | 有 teacher 蒸馏 | 学生 on-policy 状态上做**有界 Reverse-KL 蒸馏** | on-policy 锚定当前分布 | **必须有现成强 teacher** |
| **RL-VLA-Gen** | 实证研究 | （结论）RL 提语义/执行泛化，SFT 保训练分布内 | — | 仅仿真、仅 pick-and-place |

一条贯穿四篇的主线：**SFT 负责把策略带进训练分布（求准），RL 负责在 OOD 上锐化与纠错（求泛化）**；真机的约束（rollout 极贵）逼出了 off-policy/蒸馏/conditioning 这些"不直接跑 PPO"的迂回设计，而它们共同未解决的是**失败 rollout 的 step/段级利用**与**长 horizon 信用分配**。

> 说明：RECAP、VLA-OPD 等未开源或预印本，精确常数以论文为准；arXiv 26xx.xxxxx 为整理时预印本编号，引用前请核对实际发表信息。更深的机制推导见仓库 `技术报告_VLA后训练_PLD与RECAP机制解读` 与 `docs/A1–A3`。
