---
title: "VLA 真机 SFT+RL 混合训练：经验采样、存储与利用"
subtitle: "PLD / RECAP 两条 self-improving 路线诊断 + HEARS-Buffer 数据层方案 + RL/SFT 训练动力学机制解读"
author: "研究者：刘越（同济大学计算机学院 空间智能课题组）"
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
toc: true
toc-depth: 2
numbersections: true
---

\newpage

# 摘要

本报告聚焦 **VLA（Vision-Language-Action）真机后训练**这一核心工程瓶颈：真机 RL 一条 rollout 需 30–60 秒并需人工监管，经验数据极贵，"怎么采、怎么存、怎么用"决定了 self-improving VLA 能否落地。报告分三部分：

1. **诊断（第 1–5 章）**：论证为什么 VLA 真机必须 SFT+RL 混合训练（灾难性遗忘的 4 因素叠加机制），对比四种后训练范式，并用「9 条件框架」量化"何时必须存经验"，进而深读两条 self-improving 主线——**PLD**（数据驱动）与 **RECAP / π\*0.6**（信号驱动）。
2. **方案（第 6–8 章）**：用三个独立视角（学术审稿 / 工业顾问 / 9 条件框架）评价 PLD 与 RECAP，归纳 8 条 Buffer 设计原则，提出 **HEARS-Buffer**——self-improving VLA 训练范式的"第 4 个组件：数据层基础设施"，其核心创新是**段级 SFT+RL 混训（Pillar C）**，并给出 6 个月验证路径。
3. **机制（第 9 章）**：独立的 **RL/SFT 训练动力学机制解读**，综合 30+ 篇调研与 2025–2026 最新研究，从分布、梯度、熵、遗忘四个层面回答"SFT 与 RL 在物理上到底改变了什么"，并迁移到 VLA。

**一句话结论**：VLA 后训练应是「SFT 求覆盖（求准）→ 在 OOD 峰附近切 RL 求锐化与修复（求泛化），全程显式管熵、用 off-policy replay 缓解 rollout 成本」；而当前 PLD/RECAP 在"失败 rollout 段级利用、跨任务 skill 复用、Buffer 主动管理、运行时检索"四处留有结构性空白，正是 HEARS-Buffer 的切入点。

---

# 第一部分 · 诊断

# 为什么 VLA 真机必须 SFT+RL 混合训练

**核心论点**：SFT 单独不够、RL 单独更糟。VLA 真机直接 RL 会触发"灾难性遗忘"断崖——通用能力 95% → 50%。LLM/VLM 已用 RLHF+KL 锚定解决，但 VLA 中尚无共识方案，这正是后续四范式都"绕开直接 RL VLA"的根本原因。

## SFT 单独 — 2 大本质缺陷

- **分布漂移 / 误差累积（Compounding Error）**：训练只见专家轨迹的 state；部署偶然偏离 → 进入未见 state → 错误累积爆炸。**加数据无效**——示教覆盖不到"机器人卡桌角"这类 OOD 状态。
- **模仿天花板（Imitation Ceiling）**：70% 示教训出 70% 模型，**永远不超过示教者**；平均拟合所有示教（含失败轨迹），**不能利用失败数据**，浪费一半信息。

## RL 单独 — 3 大工程灾难

- **显存 + 真机样本不可承受**：直接 fine-tune VLA ≈ **62 GB 显存**；真机 rollout 30–60 秒/条 + 人工 reset，纯 RL 需百万 rollout ≈ **86 天 24/7 真机**，工程不可行。
- **Cold-Start 困境**：初期策略全失败（reward = 0），梯度信号为零；PPO/GRPO 漂移 25k 步仍学不到有效行为。经验判据：SFT 成功率 < 20% 时，PPO/GRPO 基本学不动。
- **灾难性遗忘 ★（VLA 关键顾虑）**：多步梯度更新洗掉 VLA 的视觉/语言通用能力 → "酸奶专精白痴"。LLM/VLM 用 PPO+KL 锚定（RLHF/DPO）已广泛验证可控，但 VLA 既要保视觉又要保语言，KL 系数定不下来——**这是个 open problem**。

## 为什么是"灾难性"——4 因素叠加（而非单次梯度大）

| # | 对比维度 | SFT | RL（真机 VLA） |
|---|---|---|---|
| 1 | 梯度更新次数 | 150 步（50 示教×3 epoch） | **250,000 步 → 1,600×** |
| 2 | 梯度方向准确性 | 精准（loss 直指 $a^*$） | **高方差**（reward 延迟，50 步动作同等"强化"） |
| 3 | 早期学习信号 | 每步都有 | **前 99 条全失败 = 0 信号**；第 100 条成功 → 大梯度爆炸 |
| 4 | 是否有教师锚定 | ✓ $a^*$ 锚定 | ✗ 无标准答案（最大化 reward → reward hacking） |

**量化效果**（纯 RL 1000 rollout 后）：识别红色物体 95% → **50%**；理解"往左挪一点"90% → **30%**；任务成功 60% → 99%（任务学好了，但通用能力没了）。

> **本章结论**：SFT 负责下限 + RL 负责上限。4 因素叠加下，VLA 直接 RL 是 open problem——下一章看四范式如何各自应对。

---

# 四种 VLA 后训练范式对比

PLD / RPD / VLA-OPD / RECAP 四个工作的 teacher、loss、关键 trick 完全不同，但都在"绕开直接 RL VLA"：PLD 冻结+残差；RPD 训新小模型；VLA-OPD KL 锚定；RECAP 把 RL 信号塞 conditioning 通道。

| 维度 | **PLD** (ICLR 2026) | **RPD** (IROS 2026) | **VLA-OPD** (HKUST GZ) | **RECAP / π\*0.6** (PI, 2025.11 NEW) |
|---|---|---|---|---|
| 目的 | VLA self-improving | VLA → 轻量 student | 强 expert → 升级 VLA | VLA 从经验学习 |
| Teacher | 无（自己采） | VLA 自己 | 必须外部强 expert | 无（advantage 自己算） |
| 被训对象 | 残差 MLP(10M)+蒸馏回 VLA | 全新小模型 100M | VLA backbone | VLA backbone |
| 是否需 reward | ✓ 0/1 sparse | ✓ PPO 需要 | ✗ 不需要 | ✓ 0/1 episode 标签（最弱信号） |
| RL 算法 | Cal-QL (off-policy) | PPO (on-policy) | Pure PG (on-policy) | **无 PPO/REINFORCE**（二值 advantage conditioning） |
| Loss 核心 | Cal-QL Q + actor | PPO + MSE distill | Reverse-KL distill | CFG-style 双支 BC |
| 推理时 | VLA forward | 小 student forward | VLA forward | VLA double forward + CFG |
| Replay buffer | ✓ 双 buffer | ✗ | ✗ | ✗（单池 BC 形式） |
| 显存需求 | **5 GB（RTX 4090）** | 中等 | 大（VLA 全 FT） | 大（multi-GPU 集群必需） |
| 灾难性遗忘风险 | 低（冻结 VLA） | 无（新模型，但失多任务） | 中（KL 锚定） | 低（每轮 from $\pi_{\text{pre}}$ 重训） |
| 真机战绩 | Peg insertion 100%，4-stage 1h | ~95% LIBERO，50Hz | ~95% LIBERO | 折衣 2h+，espresso **13h 连续**，failure 50%↓ |

**一句话定位**：PLD = 用残差 RL 生成"分布对齐"数据再 SFT 蒸馏；RPD = 用 VLA 当 teacher，PPO+MSE 训轻量 student；VLA-OPD = 用强 RL expert 当 teacher 升级 VLA；RECAP = success/failure 二值标签 → advantage 二值化 → 推理时 CFG 引导。

> **本章结论**：PLD 和 RECAP 是当前 self-improving 双主线；**4 范式都未解决**长 horizon 端到端 / 失败 rollout 段级利用 / 跨任务 skill share / Buffer 主动管理。

---

# Buffer 必要性：rollout 成本 + 9 条件框架

## rollout 成本决定 Buffer 形态

**是否需要 buffer** 由 rollout 成本决定，**如何用 buffer** 由真机数据稀缺度决定。

- 仿真 **1 秒/rollout** → 重采便宜，不需存（**RPD**）。
- 真机但有强 expert → expert 替你采，也不需自己存（**VLA-OPD**）。
- 真机无 teacher → 必须存（**PLD + RECAP**）。真机 60 秒/rollout，相对仿真是 **3,600×**。

只有 PLD 落在最严苛假设（真机 + 无 teacher + 要求样本效率），所以用**双 buffer 50/50 对称采样 + Cal-QL UTD=20，把每条 rollout 榨干约 250×**。RECAP 同样真机但 PI 有大算力，故用单池只增不减（粗放但简单）。

## 9 条件框架：何时需要存 Episode

把"是否需要存 Episode"形式化为 9 条件，三层结构，作为后续评 PLD/RECAP 的**统一标尺**：

- **🔴 层 1 · 必要条件（不满足则完全不需要存）**
  - **#1 Rollout 成本高**：rollout 成本 ÷ 梯度更新成本 > 100。真机机器人 ✓，LLM 数学题 ✗。
- **🔵 层 2 · 充分条件（任一满足即值得存）**
  - **#2 Long horizon**（单 episode > 50 步 + 跨步依赖）｜**#3 Continual learning**（部署后持续学习 > 1 周）｜**#4 环境非平稳**（光照/物体/任务会变）｜**#5 非马尔可夫**（当前决策依赖过去状态）
- **🟡 层 3 · 放大条件（决定 buffer 设计复杂度）**
  - **#6 任务集合开放**｜**#7 多任务**｜**#8 端测容量受限**（设备存储 < 100 GB）｜**#9 检索影响 decision**（检索 vs 不检索性能差 > 5%）

> **本章结论**：后续将用这 9 条件量化检验 PLD（**1/9**）和 RECAP（**1–2/9**），证明二者的结构性空白并非单一视角 bias。

---

# 第二部分 · PLD 与 RECAP 深读

# PLD 深度解读：三阶段 + 双 Buffer + ICLR 审稿

PLD（Probe-Learn-Distill, ICLR 2026, arXiv:2511.00091）= 三阶段整合。起点 60%、终点 99%、暖身 50 条。**Stage 2 Probing 是 PLD 的灵魂**，防遗忘根本机制就在这里。

## 三阶段流水线

```
π_VLA (60%) + 50 条示教
   → Stage 1 [Probe+Learn]：冻结 VLA + 残差 RL
   → Stage 2 ★ [Probing]：让 VLA 先掉坑再救（采集分布对齐数据）
   → Stage 3 [Distill]：普通 SFT 蒸馏回 VLA
   → π'_VLA (99%)   ⟲ 可重复（Self-Improving Loop）
```

- **Stage 1（工程基础）**：❄ 冻结 VLA backbone（结构性防遗忘）+ 🔌 外挂残差 MLP（10M，非 LoRA）+ ✂ 残差动作 clip 在 $[-\xi,\xi]$ + 🌡 Cal-QL warmup critic。**5 GB 显存可跑**，60% → 99% 约 250k 步。
- **Stage 2 ★（灵魂）**：时间步 $0 \to \alpha T$ 让 VLA 单独走（probing 探险），$\alpha T \to T$ 让 (VLA+残差) 接管救场。**α 消融：α=0 → 95% / α=0.6 → 99.5% ★ / α=1 → 94%**。反直觉点：不 probing 直接采会毁了 VLA（数据只覆盖最优区域，部署飘到边缘就崩）。
- **Stage 3（看似普通 SFT）**：$\mathcal{L}_{\text{SFT}} = -\mathbb{E}_{(s,a^*)\sim D^{\text{PLD}}}[\log\pi_{\text{VLA}}(a^*|s)]$。**防遗忘根本机制在 Stage 2 而非 Stage 3**——Stage 2 把数据分布调到 ≈ VLA 现有分布 → SFT loss 小 → 梯度小 → 不灾难性遗忘。这是与 MIFO（事后冻结参数）的本质区别：PLD 是事前避免。

## 双 Buffer 机制（继承 RLPD 四件套）

PLD 的双 Buffer **完全沿用 RLPD（Ball et al., ICML 2023, arXiv:2302.02948）**，创新在 Stage 2/3 而非 buffer 本身。

- **架构**：Offline Buffer $B_{\text{off}}$（50 条示教，永久固定，提供"成功长什么样"的**锚定**）+ Online Buffer $B_{\text{on}}$（FIFO 滚动，提供"现在在哪探索"的**新鲜**）。各取 128 → Batch(256) → Cal-QL update（UTD=20）。
- **四件套**：对称采样 50/50 + LayerNorm critic + Ensemble（10 Q 取 Min-Q 保守估计）+ 高 UTD（20）。改动：Offline 源改为 VLA 自跑、critic 改 Cal-QL、actor 改 Residual MLP。
- **三个易被低估的关键**：① **50/50 是 sweet spot**（80/20、20/80 都退化）；② **LayerNorm 不可省**——否则 Cal-QL 在 OOD action 上 Q 值发散到天文数字，很多复现失败都因漏掉它；③ **250× 样本利用**＝高 UTD×Ensemble Min-Q，把"86 天真机数据"压成可行工程。
- **Residual + 双 Buffer 是一对**：残差 MLP 是独立 Gaussian 策略 → 可 off-policy 复用历史数据；若改用 LoRA → 必须 on-policy → 数据用完即丢 → 双 buffer 失去意义。

**三大结构性劣势**：① **Flat Structure**（所有 transition 平等存储，无时序/任务标签/失败模式聚类）；② **被动累积不感知漂移**（FIFO 不区分"过时失败"与"当前失败"）；③ **Per-Task 不共享**（90 个 LIBERO 任务 → 90 个独立 buffer + 90 个 residual MLP，0 skill share）。

## ICLR 2026 审稿意见 + 9 条件打分

4 位审稿人**独立**指出 4 大不足，与 9 条件框架**两个独立 lens 殊途同归**：

| 审稿人 | 质疑要点 | 对应 9 条件 |
|---|---|---|
| yzFS | 任务增长需多少 specialist？skill 能否跨任务复用？ | #6 + #7 |
| F1ZL | 多任务下残差策略是否每任务独立训练？ | #7 |
| BWyM | 评估仅短 horizon，未测长 horizon / 时序依赖 | #2 + #5 |
| wXQ9 | 仅二值 reward 限制效率；训练 closed-loop 但蒸馏后 open-loop chunking | dense reward + 闭/开环 |

**9 条件覆盖度 = 1/9**：仅满足必要条件 #1（rollout 高效复用），#2–#9 全 ✗。根因——PLD 双 buffer 的设计目标是"训练时如何高效复用 rollout"，而非"部署后如何应对环境变化与持续学习"。

---

# RECAP / π\*0.6 深度解读

RECAP / π\*0.6（Physical Intelligence, Sergey Levine 团队, 2025.11, arXiv:2511.14759，**未开源**）是第二条主线——**信号驱动**（vs PLD 数据驱动）。

> **一句话**：RECAP = 用 value function 把 episode-level reward 升级到 step-level advantage，再用 advantage 作为 conditioning + CFG 推理，**绕开 PPO** 实现 self-improving。

## 三阶段与"每轮 from pretrain 重训"

| 阶段 | 名字 | 频率 | 输出 |
|---|---|---|---|
| A | PRETRAIN | 一次性 | $V_{\text{pre}}, \pi_{\text{pre}}$（数万小时多机器人数据） |
| B | SPECIALIZE | 每新任务一次 | $\pi^0_\ell$（demo 微调起点） |
| C | ITERATE | 每任务 K 轮 | $\pi^*_\ell$（采数据→重训 V→重训 π→循环） |

**反直觉设计**：阶段 C 每轮 V 和 π 都**从 $V_{\text{pre}}/\pi_{\text{pre}}$ 重训**（非增量更新）——增量会忘掉通用预训练知识，每轮 from pretrain = 把预训练当永久 anchor。

## 数据池、reward 与 step-level label

单池 $\mathcal{D}_\ell$ **只增不减**，三来源：$\mathcal{D}_{\text{demo}}$ + $\mathcal{D}_{\text{auto}}$（含失败）+ $\mathcal{D}_{\text{correction}}$（HG-DAgger 人接管）。每条 episode 仅标 1 bit success/failure，reward：

$$r_t = \begin{cases} 0 & t=T \text{ 且 success} \\ -C_{\text{fail}} & t=T \text{ 且 failure} \\ -1 & \text{otherwise} \end{cases}$$

Value function 学到的物理含义 = 负的"剩余步数到成功"，归一化到 $[-1,0]$。**Step-level advantage indicator $I_t$ 规则**：demo / correction **强制 $I_t=$ True**（绕过 V 评估，视为理想动作）；autonomous 成功段 $I_t=\mathbb{1}[A>\epsilon_\ell]$；autonomous 失败段保留（学"这种动作不好"）。

## π 训练（CFG-style 双支）

$$\min_\theta \mathbb{E}\left[-\log\pi_\theta(a_t\mid o_t,\ell) - \alpha\log\pi_\theta(a_t\mid I_t,o_t,\ell)\right]$$

第一项 = unconditional BC，第二项 = conditional BC。工程要点：① 二值化阈值 $\epsilon_\ell$ = demo value 分布 **30 percentile**（per-task 自校准）；② advantage 通过 **prefix 文本注入**（"Advantage: positive/negative"），**模型架构完全不动**；③ **Random drop $I_t$（50%）** 等价随机 conditional/unconditional 双支，无需算两遍 forward；④ 推理时换 positive/negative prompt forward 两次 = CFG，$\hat\pi \propto \pi\cdot(\pi_c/\pi)^\beta$（$\beta=1$ 单次足够）；⑤ **多轮迭代的微观机制**：旧 episode 数据不变，但新 $V^k$ 重算 advantage → 同一 step 的 $I_t$ 可能 flip，这是"持续改进"的来源。

## 数据流对比 PLD

| 维度 | PLD | RECAP |
|---|---|---|
| 筛选 | Stage 2 Probing 主动选 200–300 条高质量 | **不筛选**，全收（数千 episode/task，brute-force） |
| 利用 | 双 buffer 50/50 + Cal-QL UTD=20（250× 复用） | 单池累积 + 4 label 规则 + 每轮 from $\pi_{\text{pre}}$ 重训 |
| 存储 | offline 50 永久 + online FIFO 1M | **单池只增不减**（无压缩/归档/遗忘，推测单 task 100 GB+） |

PLD 用更少数据（200–300 + 50）；RECAP 用更多（折衣 600 / 装盒 1200，37.5% 含 expert intervention），换来长 horizon 连续运行 + 免 probing + 工业直接适用。

---

# 第三部分 · 综合评价与 HEARS-Buffer

# 综合评价 + 8 条 Buffer 设计原则

## 三个独立 lens 殊途同归

| 评价 lens | PLD | RECAP |
|---|---|---|
| 🎓 学术审稿 | ICLR 2026，4 审稿独立指出 4 大不足 | 未走审稿，社区认可高，尚未被独立审视 |
| 🏭 工业顾问 | 5 GB 消费级友好；但"1h autonomous"是 state machine 拼接 | 13h espresso 最强；但整 episode 算 advantage，**"局部正确但最终失败"的段被一刀切** |
| 📊 9 条件覆盖 | **1/9** | **1–2/9** |

**四个共同空白**（证明非单视角 bias）：① 失败 rollout 段级利用；② 跨任务 skill share；③ Buffer 主动管理；④ 推理时 Retrieval。

## 8 条 Buffer 设计原则（需求侧 9 条件 ↔ 能力侧 8 原则）

| # | 原则 | 对应条件 | 核心机制 | 依赖库 |
|---|---|---|---|---|
| ① | 智能筛选 | #1,#4 | embedding 到 centroid 距离 + TD-error + new-cluster 三级判定 | VLA backbone |
| ② | 段级切分 | #2,#5 | subtask prediction $\hat\ell$ 变化 + advantage 突变/intervention | VLA backbone |
| ③ | **失败样本利用 ★** | 业界建议 | 段级 quality 标 + 失败段 KNN 找 recovery target 朝相似成功段 SFT | FAISS |
| ④ | 存储空间上界 | #8 | size cap + per-cluster quota + archived ref-only | LeRobot v3.0 |
| ⑤ | 数据压缩 | #8 | 3 层 Frame/Segment/Cluster | ffmpeg, Parquet |
| ⑥ | **主动遗忘 ★** | #3,#4 | 周期 capability check + archive + 复发 trigger | 新设计 |
| ⑦ | 跨任务复用 | #6,#7 | embedding + HDBSCAN 增量聚类 + skill_id 倒排索引 | hdbscan, FAISS |
| ⑧ | 运行时检索 | #9 | FAISS HNSW + KNN top-k | FAISS |

**两个 take-away**：8 条中 **6 个可基于成熟开源库**；**仅 ③⑥ 是全新设计点**（段级混训 + 主动遗忘），研究差异化集中于此。三概念精确区分——**新颖度**（到 centroid 距离）决定是否写入；**质量标注**（段内平均 advantage）决定 SFT/RL 权重；**库内对比**（KNN match）决定 bad 段 recovery target。

---

# HEARS-Buffer 设计方案

**研究定位**：HEARS（Hierarchical Episodic Adaptive Retrieval Skill-library Buffer）**不是**又一个 self-improving 算法或 VLA 模型，而是该训练范式的**第 4 个组件——数据层基础设施**，可 plug-in 进 PLD 也可进 RECAP（**互补而非竞争**）。

## 三层架构 × 五支柱

```
顶层 · 操作 API 层    write / sample / retrieve / evict / snapshot
                      └ 同时服务 Action Expert 训练 + 推理
中层 · 索引层（5 支柱，核心研究贡献）   A② B⑦ C★③ D④⑤⑥ E⑧
底层 · 存储层（LeRobot v3.0，不重新发明）  Parquet + MP4 + Parquet meta
                      └ 3 层压缩 → 10 task 长期部署 6–10 GB（端测可行）
```

| 支柱 | 名字 | 原则 | 核心机制 | 解决条件 |
|---|---|---|---|---|
| A | Episodic + Segment-aware | ② | LeRobot v3.0 + segment metadata | #2,#5 |
| B | Cross-task Skill Library | ⑦ | HDBSCAN 在 VLA visual embedding 聚类（espresso/cappuccino 共享"磨豆""压粉"） | #6,#7 |
| **C★** | **Mixed SFT+RL Segment Annotation** | **③** | **失败 rollout 段级混训** | #1 升级 |
| D | Capability-aware Lifecycle | ④⑤⑥ | cluster SR 评估 + 主动归档/遗忘 + 复发触发 | #3,#4,#8 |
| E | Retrieval-augmented Decision | ⑧ | FAISS HNSW + KNN（v0.3+） | #9 |

底层可靠性：`episodes/` 是 source of truth，segments/clusters/faiss 索引层可从原始数据完全重建。

## Pillar C ★：段级 SFT+RL 混训（核心创新）

**动机（业界顾问关键建议）**：长程任务 sparse reward 信号粗。一条 espresso 失败 episode（500 步）可能磨豆/压粉都对、只是装手柄歪了，整条被当负样本 → 学偏。**bad rollout 用来纠偏，good rollout 用来强化与探索**。现有所有范式都没显式处理"局部正确但最终失败"的段（PLD 直接丢失败 / RECAP 整 episode advantage 被全局失败拉低 / 14 篇 replay 论文都是 step-level 无 segment 概念）。

**C1 段切分（4 信号融合 cut score）**：
$$\text{cut\_score}(t) = w_1|A(t){-}A(t{-}1)| + w_2|V(t){-}V(t{-}1)| + w_3\mathbb{1}[\hat\ell(t)\neq\hat\ell(t{-}1)] + w_4\mathbb{1}[\text{intervention}]$$
推荐 $w_1{=}0.3, w_2{=}0.3, w_3{=}0.2, w_4{=}0.5$（**intervention boundary 权重最高**——expert 接管/松手是 ground-truth 级段边界）。滑窗 5 步、平均 > 0.6 切分。

**C2 段级 quality 标注（3 类）**：

| Quality | 判据 | SFT_w | RL_w | 用法 |
|---|---|:--:|:--:|---|
| good | 段内平均 $A>\epsilon_\ell$ + V 上升 | 0 | 1.0 | RL 强化 + entropy 探索 |
| bad | 段内平均 $A<0$ + V 下降 + episode 失败 | 1.0 | 0 | SFT 朝 recovery target |
| unclear | 介于两者 | 0.3 | 0.3 | 弱双修 |

**核心特例规则**：autonomous + 失败 episode + 段内 $A>0$ → quality = **good**。这就是"局部正确但最终失败"的段——RECAP 整 episode advantage 被全局失败拉低，**HEARS 段级 advantage 能识别局部正确**。

**C3 Recovery target 匹配（KNN）**：对每个 bad 段取起始 state 的 visual embedding → 在 quality=good 段库 FAISS HNSW KNN top-5（约束：距离 < θ + 同 cluster 优先 + 同 task 优先）→ 训练时随机抽 1 作 SFT target（防过拟合）。

## 段级标注 → Action Expert 训练接口

落到 Action Expert（flow matching VLA，如 π0）的统一 loss：

$$\mathcal{L}_\sigma = \text{SFT}_w\cdot\mathcal{L}_{\text{BC}}(\sigma,\text{recovery\_target}) + \text{RL}_w\cdot\mathcal{L}_{\text{flow-adv}}(\sigma,I_{\text{advantage}})$$

- $\mathcal{L}_{\text{BC}} = -\mathbb{E}_{(o,a)\in\sigma}[\log\pi(a_{\text{target}}|o)]$（BC 朝 KNN recovery target）
- $\mathcal{L}_{\text{flow-adv}}$ = flow matching MSE + advantage prefix conditioning（借用 RECAP lower bound）
- **batch 按 segment 而非 episode 取样**，Random drop $I_t$（CFG）仍适用。

**创新三合一** = RECAP advantage conditioning（推理 trick 反用于训练）+ 业界顾问段级混训建议 + KNN recovery 朝相似成功段 SFT。Buffer 段级标注 ↔ Action Expert 训练**端到端可微**。

## 工程要点

- **底层 LeRobot v3.0**（vs RLDS：Parquet+MP4 支持 append、视频压缩、HF 生态原生，OpenVLA/π0 已迁移）。
- **3 层压缩**：Frame（MP4 H.264，10×）→ Segment（unclear 丢/bad 留 keyframes/good 全留，~60% 保留）→ Cluster（归档老 cluster 保留 centroid+5 representatives，90% 压缩）。**100 GB → 6–10 GB**。
- **D 支柱复发触发**：archive 只删数据保留 centroid，新段距 archived centroid 近时自动 revive（类比人类"提取性遗忘"，Recall failure not storage failure）。
- **入库流程 ~3s/episode**（远低于 rollout 30–60s）：value forward 算 advantage → changepoint 切段 → quality 标 → embedding → HDBSCAN 增量聚类 → bad 段查 recovery → 写 LeRobot → 异步 capability check。

---

# 验证路径与 Roadmap

## 为什么强制先复现 PLD + RECAP

HEARS 是数据层基础设施、本身非 self-improving 算法，**必须有 baseline 算法的 rollout 数据才能验证价值**：① 对照实验 baseline；② 提供 rollout 数据集（C 支柱段切分在其 rollout 上做 PoC）；③ 参考实现来源（RECAP advantage conditioning + CFG 是训练接口的直接借鉴）。

## 四 Phase Roadmap（6 个月）

| Phase | 内容 | 时长 | 资源 | 风险 | 验收 |
|---|---|---|---|---|---|
| **0a** PLD 复现 | ICLR 开源代码 + LIBERO-Long 10 task | 4 周 | 1× RTX 4090 | 低 | SR ≥ 95%；产出 200–300 rollout/task |
| **0b** RECAP 复现 | 从 arXiv 重建 Algorithm 1 + Eq.1/3/4，用 π0 替 π0.6 | 8 周 | 4–8× A100 | **高（PI 不开源）** | advantage cond + CFG 跑通，vs π0 +10% SR |
| **1** HEARS C 支柱 PoC | 段切分 + Recovery KNN + 段级 loss | 4 周 | 1–2× A100 | 中 | 段切分 F1 > 0.7，quality 准确率 > 75% |
| **2** HEARS 完整集成 | 5 支柱接进 PLD/RECAP，4 setup ablation + 论文 | 8 周 | 4–8× A100 | 中 | HEARS+PLD vs PLD baseline SR 提升 ≥ 3% |

**关键里程碑**：4w PLD baseline / 12w RECAP baseline / 16w C 支柱 PoC / **24w 论文 draft**。**Phase 0b 备选**：Plan B 只复现 value function（4 周，用作段切分判据）；Plan C 等 PI 开源（2026 H2）。**目标会议**：CoRL 2026（9 月 deadline）/ ICLR 2027。

## 风险登记（最重要 5 个）

| # | 风险 | 概率 | 严重 | 缓解 |
|---|---|:--:|:--:|---|
| **R2** | RECAP 复现工程量超估 | 高 | 高 | Plan B 只复现 V；Plan C 等开源 |
| **R7** | GPU 资源不足（0b/2） | 高 | 中 | 申请集群；LoRA 替代 full FT |
| R3 | HDBSCAN 聚不出有意义 cluster | 中 | 高 | 退回 K-Means + subtask label 监督 |
| R4 | 段切分 4 信号权重难调 | 中 | 中 | Grid search + 100 gold standard 校准 |
| R6 | 段级 SFT/RL 混训不稳定 | 低 | 高 | 借鉴 RECAP random drop；逐渐加 RL weight |

**两个红色风险**（高概率+高严重）= R2（RECAP 复现）与 R7（GPU 资源），是项目最大瓶颈。

## 诊断部分总结（5 句话）

1. **SFT/RL 单独都救不了 VLA**（纯 RL 1000 rollout 后识别红色 95%→50%、任务 60%→99% 但通用能力丢失），必须混合训练。
2. **已有两条路线**：PLD（数据驱动，残差 RL 生成对齐数据再蒸馏）+ RECAP（信号驱动，value function 生成训练信号直接 FT），都很强但都有结构性短板。
3. **3 个 lens 综合评价**（学术审稿 + 工业顾问 + 9 条件框架）共同识别 4 个空白。
4. **HEARS-Buffer 5 支柱**（A Episodic / B Skill / **C Mixed SFT+RL★** / D Lifecycle / E Retrieval）对应 8 设计原则，C 支柱配 Action Expert 训练接口。
5. **6 个月验证路径**：先复现 PLD+RECAP baseline，再 HEARS PoC，目标 CoRL 2026。

---

# 第四部分 · 训练动力学机制

> 本章为独立机制解读，为前三部分的工程选择提供"为什么"的理论支撑：为什么 SFT 与 RL 必须混合、为什么 RL 能修复 SFT 遗忘、为什么必须显式管熵。综合本地调研 30+ 篇与 2025–2026 最新研究。

# RL/SFT 训练动力学机制解读

## 核心对立：SFT「记忆/压缩」 vs RL「探索/扩展」

源头是 [SFT Memorizes, RL Generalizes (arXiv:2501.17161, ICML 2025)]：SFT 倾向记住训练分布、OOD 即崩；结果奖励训练的 RL 能跨规则/视觉变体泛化。但这句口号已被 2025 下半年研究反复修正，需区分三层：

| 层面 | SFT 在做什么 | RL 在做什么 |
|---|---|---|
| 目标 | 最大化专家轨迹似然 | 最大化期望奖励 `E[R]` |
| 数据来源 | off-policy（固定专家数据） | on-policy（自身采样 rollout） |
| 分布效果 | **扩张/同质化**（覆盖更多 mode） | **压缩/锐化**（集中到高奖励 mode） |
| 熵 | 不直接约束，常使熵升高/漂移 | 系统性降低熵（探索预算被消耗） |
| 风险 | OOD 遗忘、格式锁死 | 熵坍缩、mode collapse |

**物理直觉**：SFT 把概率质量铺到老师走过的所有路径上（拟合更宽分布）；RL 把质量从错误路径抽走、堆到对的路径上（再分配，不创造新路径）。二者方向相反，这是它们互补又互相覆盖的根因。**关键澄清**：「SFT 记忆、RL 泛化」是现象级总结而非机制真理——SFT 的问题不在"记忆"而在"OOD 早峰后遗忘"，RL 很大程度是"修复"而非凭空泛化（见下）。

## 分布层面：sharpening vs expansion，及"RL 是否扩展边界"之争

- **RL Squeezes / SFT Expands** [arXiv:2509.21128]：RL 把推理功能集中到少数关键 step，SFT 均摊到许多 step。参数层面对偶——MIFO [arXiv:2510.04454]：**SFT 更新冗余幅度大、RL 更新节俭幅度小**，故后续 SFT 会覆盖先前 RL，需参数冻结保护。
- **争议（须向导师呈现两面）**：观点 A [arXiv:2504.13837, NeurIPS 2025] —— pass@1 RL 胜 base，但 **pass@k（大 k）base 反超**，RL 可解集是 base 子集，**只提采样效率不创造新模式**；观点 B [ProRL, arXiv:2505.24864] —— 长程 RL + KL 控制 + 参考重置能解 base 全失败的问题，**扩展边界**。**调和**：决定 RL 是"锐化"还是"扩展"的开关，是**探索预算（熵）是否被维持**。

## 梯度与 token 层面

- **SFT 隐含坏 reward** [DFT, arXiv:2508.05629, ICLR 2026]：标准 SFT 梯度 ≡ 隐式奖励 **1/p(y|x)** 的策略梯度——越不确定的 token 被赋越大奖励（逆概率加权、无界方差），这是泛化差的根因。修正一行：`loss = −p.detach()·log p`，使 SFT 接近 RL 泛化。
- **负梯度 = 锐化动力** [arXiv:2505.18830]：对错误路径施加负梯度产生 squeezing effect，把质量从错误 mode 抽走——SFT 只有正梯度（只抬对的、不压错的），做不到。
- **统一视角** [UPGE/HPT, arXiv:2509.04419]：SFT、PPO、GRPO 是同一策略梯度估计器的特例，**SFT 是"优势恒 +1、无负样本"的退化 RL**。

## 熵动力学

[The Entropy Mechanism, arXiv:2505.22617]：无干预 RL 中策略熵早期急剧下降、探索枯竭。经验定律 **R = −a·e^H + b**——**性能用熵"换"**，天花板在熵耗尽前可预测。**熵就是 RL 的探索预算**，花光就跳不出当前 mode（这是"常规 RL 只锐化"的微观解释；ProRL 能扩展正因避免熵过早耗尽）。熵还可作 SFT/RL 判别信号：[SRFT, arXiv:2506.19767] 用熵区分"SFT 全局漂移 vs RL 细粒度优化"动态调权。

## 灾难性遗忘与 OOD：RL 如何"修复" SFT（核心 takeaway）

[RL Fine-Tuning Heals OOD Forgetting in SFT, arXiv:2509.12235]：真相不是"SFT 记忆/RL 泛化"，而是 **OOD 性能在 SFT 早期达峰，随后随 ID 优化而遗忘；RL 在恢复被遗忘的能力**。谱分析：遗忘 = 权重矩阵**奇异向量旋转**（方向丢了），奇异值基本不变（幅度没丢）。限制：RL 只能从有限范围的 SFT checkpoint 修复——**SFT 过度了救不回来**，呼应 [Quagmires, arXiv:2510.01624]"高 SFT 分数会误导你"。**结论：SFT 别练过头，在 OOD 峰附近就切 RL。**

## 何时用 SFT / RL / 混合

| 场景 | 该做什么 |
|---|---|
| 连格式都不对、0 奖励率高 | 先 SFT 稳定格式（SFT 是 RL 前提） |
| 已有格式、要泛化 | 切 RL |
| 想长期边界扩展 | 长程 RL + 维持熵（KL 控制 + 参考重置） |
| 最小改动提升 SFT 泛化 | DFT 一行替换 |

**三类自动切换信号**：熵触发（SRFT）｜梯度范数+散度触发（SASR, arXiv:2505.13026）｜rollout 正确率触发（HPT）。**趋势**从"两阶段顺序"→"单 step 双梯度统一 loss"（CHORD, arXiv:2508.11408，已并入 ms-swift）。

## 对 VLA 真机后训练的启示

1. **「RL 修复 SFT 遗忘」更关键**——VLA 的 SFT 同样 OOD 早峰后遗忘，换光照/物体就崩；[Continual VLA via RFT, arXiv:2602.10503]、[VLA-OPD, arXiv:2603.26666] 均报告 on-policy RL 比 SFT 抗遗忘。
2. **熵坍缩在动作空间更危险**——连续动作熵坍缩 = mode collapse 到单一抓取姿态；VLA-OPD 用有界 mode-seeking 目标避开熵爆炸/坍缩。
3. **rollout 极贵 → 必须 off-policy + replay**——这是 PLD 走 off-policy、本报告主张 HEARS replay 的根本动因。
4. **梯度不对称 → 参数保护 + token 级重加权**（DFT + RLSD 方向/幅度解耦可迁移到动作 token）。
5. **不能只看 loss/成功率均值**——必须看 OOD 成功率与多样性指标。

> **一句话**：VLA 后训练应是「SFT 求覆盖（求准）→ 在 OOD 峰附近切 RL 求锐化与修复（求泛化），全程显式管熵、用 off-policy replay 缓解 rollout 成本」。完整文献与机制细节见仓库 `docs/01_训练动力学机制解读.md`。

---

# 附：关键文献

**VLA 后训练范式**：PLD（Xiao et al., ICLR 2026, arXiv:2511.00091）｜RECAP/π\*0.6（Physical Intelligence, 2025.11, arXiv:2511.14759）｜RLPD（Ball et al., ICML 2023, arXiv:2302.02948）｜Cal-QL（Nakamoto et al., NeurIPS 2023, arXiv:2303.05479）｜VLA-OPD（arXiv:2603.26666）。

**训练动力学**完整文献见第 9 章末尾「参考文献」。

**基础设施**：LeRobot v3.0 ｜ HDBSCAN ｜ FAISS。
