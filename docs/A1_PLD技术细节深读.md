# PLD（Probe-Learn-Distill）技术细节深度报告

**论文**：*Self-Improving Vision-Language-Action Models with Data Generation via Residual RL*（Xiao et al., NVIDIA / CMU / UC Berkeley / UT Austin, arXiv:2511.00091, ICLR 2026 Poster）

> 本文是主报告第 3 章的全细节支撑版，逐节核对论文原文（方法 §3 + Eq.2 + Algorithm 1 + Appendix B.1 + Table 1/2/3/5/6 + Figure 11/13/14）。标"（解读）"者为精读笔记推断。

PLD 的一句话定位：**冻结 VLA 主干，用 off-policy 残差 RL 训练一个轻量专家去"接管"VLA 的失败状态，再用一种"基策略探针（probing）+ 残差救场"的混合 rollout 生成一批"贴着 VLA 部署分布、且含纠错行为"的数据，最后用标准 SFT 把这批数据蒸馏回 VLA 本体。** 全文真正的新颖点不是残差策略、也不是双 buffer（这些来自 ResiP / RLPD），而是**用残差 RL 做"部署对齐的数据策展（data curation）"，训完即丢专家**。

---

## 1. 问题设定与形式化目标

**被训对象的形式。** PLD 把 VLA 当作 base policy $\pi_b$（默认 $\pi_0$ flow-matching head；也验证 OpenVLA 离散 token head）。策略消费 $(o_t, g)$ 输出 $a_t = D_\phi(h_\theta(o_t, g))$，$h_\theta$ 为 VLM 主干、$D_\phi$ 为动作头。动作 7-DoF（6-DoF delta 末端位姿 + 1-DoF 连续夹爪）。真机用 action chunking（GPU 装配 chunk size = 26、execution horizon = 15）。

**控制过程。** goal-conditioned MDP，稀疏二元奖励 $r(s,a,g)=\mathbf{1}[d(\phi(s),g)\le\varepsilon]$。off-policy 目标为标准 GCRL 折扣回报 $J(\pi)=\mathbb{E}\big[\sum_t \gamma^t r(s_t,a_t,g)\big]$。

**为什么不能直接对 VLA 做 RL。** (i) sparse reward 下语言条件操作使 RL 不稳定且样本低效；(ii) 直接 RL fine-tune 大 VLA 资源极重——OpenVLA-OFT 在 LIBERO、batch 8 需 ~62.5 GB 单卡显存。故 PLD 选择 decoupled：冻结 $\pi_b$，只训轻量残差。

---

## 2. 残差策略（Residual Policy）的精确形式

执行动作 $\bar a = a_b + a_\delta$，$a_b\sim\pi_b$，$a_\delta\in[-\xi,\xi]$。残差是任务专属高斯策略 $\pi_\delta(\cdot|s,a_b)$（条件于 $s$ 与 VLA 想做的 $a_b$）。

**参数化。** 3 层 MLP 高斯策略（hidden 256、latent 256、Tanh、LayerNorm），视觉用冻结 ResNetV1-10。单 task **5 GB VRAM peak、RTX 4090 可跑**（~10M 参数为解读）。

**为什么 clip 到 $[-\xi,\xi]$。** $\xi$ 太大 → 早期组合策略大幅偏离 $\pi_b$、不稳定、早期崩（Figure 13 action scale 1.0 早期塌陷）；$\xi$ 太小 → 探索不足、渐近差。结论：LIBERO $\xi=0.5$、SimplerEnv $\xi=0.1$。

**为什么残差而非 LoRA / 直接 fine-tune。** 残差是结构独立高斯策略，可用任意 off-policy 算法训 → 可维护 replay buffer、老 rollout 反复用。LoRA 嵌在 VLA 内部：反向传播穿过整个 VLA（显存照炸）、对 flow/diffusion 头做最大化 Q 的 RL 极难。冻结主干同时根除灾难性遗忘——残差用完即弃。

---

## 3. Stage 1（Probe + Learn）：Cal-QL critic + 残差 actor

**critic 目标（Cal-QL，Appendix B.1）：**
$$\min_\theta\ \alpha\,\mathbb{E}_{s\sim\mathcal{D},a\sim\pi}\big[\max(Q_\theta(s,a),\,V^\mu(s))\big]\;-\;\tfrac12\,\mathbb{E}_{s,a\sim\mathcal{D}}\big[(Q_\theta(s,a)-\mathcal{B}^\pi\bar Q(s,a))^2\big].$$
第一项 calibrated CQL 保守项（用 $V^\mu$ 做下界校准，避免 CQL 过度低估）；第二项 TD-error。

**TD 更新（Eq.2）用组合策略 bootstrap：** $Q^{\bar\pi}(s_t,\bar a_t)\leftarrow r+\gamma\,\mathbb{E}[Q^{\bar\pi}_{\text{target}}(s_{t+1},\bar a_{t+1})]$，$\bar a=a_b+a_\delta$。

**actor 更新** 通过最大化 SAC target（带 entropy，target entropy $-\text{act\_dim}/2$，自动调温），**不加 BC 正则**——使专家不被数据质量/base 性能拖住。

**warmup critic（§3.1）：** 先只用 $\pi_b$ 数据初始化 Cal-QL critic、actor 不更新（warmup 100 episodes）。意义：防遗忘 + Cal-QL 预训练 critic 渐近更优、对 $\alpha$ 鲁棒（Figure 14：Cal-QL > IQL > CQL，CQL 严重遗忘）。

**UTD / ensemble：** critic:actor = 2:1；Q ensemble 2 + Clipped Double Q + LayerNorm（RLPD trick 压制高 UTD 下 OOD Q 爆炸）；Polyak 0.005。性能对 update frequency（1→500）不敏感（Figure 15）。

**冻结 VLA：** 梯度只过 $\pi_\delta$ 与 $Q_\theta$，$\nabla_{\theta_{\text{VLA}}}=0$。

---

## 4. Stage 2（Probing）：混合 rollout 的精确机制

**控制律：** $\pi(s_t)=a_{\text{base}}$ 若 $t<\alpha T$（VLA 单独走，探针）；$=a_{\text{base}}+a_\delta$ 若 $t\ge\alpha T$（残差救场）。收集轨迹 = 前段起点 $s_0\sim p_0^{\pi_b}$（VLA 自身访问分布）+ 后段专家从次优区恢复。**探针步只用于状态初始化、不进 replay buffer。**

**为什么防遗忘（support 语言）：** 全程用组合体采（α=0）→ 数据是高度最优、单峰的专家轨迹，集中在最优窄带，VLA 部署会访问的 OOD/失败状态 underrepresented → SFT 学不到实际失败的解。Probing 把数据 support 拉回 VLA 访问分布，SFT 只需小幅更新。原文 §3.2：纯专家"narrow distribution of unimodal expert behavior may leave OOD and failure states underrepresented... risks overfitting and harming robustness and generalization."

**α 消融（Figure 11）：** 随 α 增大成功轨迹平均长度上升，fine-tuning 性能 **α=0.6 plateau，>0.6 下降**。

**Table 3（数据来源对比，LIBERO-90 $\pi_0$ SFT，按 coverage 的 Overall SR）：**

| coverage | PLD Data | Base Rollout(α=0) | Human |
|---:|---:|---:|---:|
| 0.1 | **0.314** | 0.103 | 0.272 |
| 0.3 | **0.470** | 0.068 | 0.419 |
| 0.6 | **0.637** | 0.328 | 0.611 |
| 0.8 | **0.745** | 0.344 | 0.694 |
| 1.0 | **0.871** | 0.488 | 0.815 |

未见任务 10% coverage 时 PLD 已 24.4% zero-shot，base rollout 几乎不泛化。

---

## 5. Stage 3（Distill）：蒸馏 SFT 与"防遗忘在 Stage 2"

**SFT 目标**（按动作头实例化）：token head $-\mathbb{E}[\log p_\theta(u_k|u_{<k},x)]$；diffusion head $\mathbb{E}[\|\epsilon-\epsilon_\theta(\cdot)\|^2]$；flow head $L_2$ flow-matching。概念形式 $\mathcal{L}_{\text{SFT}}=-\mathbb{E}_{(s,a^*)\sim\mathcal{D}^{\text{PLD}}}[\log\pi_{\text{VLA}}(a^*|s)]$。实现 LoRA rank 32、8×L40。

**为什么防遗忘在 Stage 2：** SFT loss 梯度幅度 ∝ 数据分布与当前策略分布的偏离。遗忘根因是"SFT 数据离 $\pi_b$ 太远 → loss 大 → 梯度大 → 覆盖通用能力"。PLD 数据起点在 $\pi_b$ 访问分布上 → $a^*$ 与 $\pi_b(\cdot|s)$ 差异小 → loss 初值小、梯度小、更新温和。KL 语言（引 "RL's Razor"）：PLD 数据 biased toward base ⟹ SFT 诱导 KL 小 ⟹ forgets less。与 MIFO 事后冻结参数本质不同：PLD 从数据侧根除遗忘。

**蒸馏后无残差：** 行为永久写进 VLA 权重，部署 zero overhead，distilled generalist 超越 average specialist。

---

## 6. 双 Buffer + Cal-QL 的样本高效

对称回放（承袭 RLPD）：$\mathcal{B}_{\text{offline}}$（$\pi_b$ 成功 rollout，永久锚定）+ $\mathcal{B}_{\text{online}}$（组合体 transition）。每 minibatch 各取一半。offline 一半防 Q 因 online 早期全失败而把所有动作判烂；online 一半提供新鲜信号。

**250× 复用：** buffer 250k、训练 ~250k actor steps，一条真机数据进 buffer 后参与 ~250k 次梯度更新。真机：Franka 200 条 teleop SFT 后 2 小时达 100%。

---

## 7. 实验与消融硬数字

**LIBERO（Table 1）：** $\pi_0$ 93.4→**97.2**（Goal +7.9）；OpenVLA 91.8→**99.2**（Goal +15.7）。
**SimplerEnv（Table 2）：** Octo Avg 71.8→**96.6**；Pick Carrot 43.3→**93.9**（+50.6）。
**真机 Franka：** peg insertion 30/30；cube pick-up 随机化 PLD 30/30 vs RLPD 16/30 vs Human 10/30。失败模式（Figure 8）：RLPD/Human 训出的策略把方块推进左上角卡死（没访问过角落状态），PLD 探针到这些状态生成 recovery 救回。
**真机 unseen（Table 6）：** Pick Blue Cube(cluttered) PLD 28/30 vs Human 12/30。
**YAM 双臂 GPU 装配：** 4 stage 串联，连续 1 小时无人介入循环。

**ablation：** α plateau@0.6（Fig.11）；action scale LIBERO 0.5/SimplerEnv 0.1（Fig.13）；Cal-QL>IQL>CQL（Fig.14）；reward bias 大 bias(-1.0) 显著伤性能 → 用 0.0（Fig.12）；JSRL 部分 task 不收敛、PLD 全收敛（Fig.17）。

**超参（Table 5）：** batch 256、buffer 250k、γ 0.99、lr 3e-4、AdamW、warmup 100、critic:actor 2、target entropy $-\text{act\_dim}/2$、$\xi$ 0.5、Q ensemble 2、Polyak 0.005、ResNetV1-10、hidden 256、LayerNorm。

---

## 8. 审稿局限及真实技术根因

4 审稿（F1ZL 8 / wXQ9 4→support / yzFS 6 / BWyM 6），Accept(Poster)。

1. **只用 sparse binary reward**（未解决）：Cal-QL 学 0/1，难训任务无梯度密度。
2. **competence threshold——base 必须够强**：残差只在 $a_b$ 邻域修正 + probing 依赖 base 走到有意义状态。base>80%→99%，base<50%→救不回（Figure 10）。
3. **closed-loop 残差 vs open-loop chunking 蒸馏**：训练逐步闭环、部署 chunk 开环，闭环纠错被打包进 chunk，高动态任务失效。
4. **per-task 残差无 skill 共享**：90 task = 90 MLP，训练成本线性增长。
5. **flat FIFO replay 无能力感知**：不按 TD-error/语义价值优先保留。
6. **episode-level 信用分配**：reward 仅末尾 0/1，放大对双 buffer + LayerNorm 依赖。
7. **"Self-Improving"只跑一轮**：多轮收益递减未实证。

rebuttal 说服点：1h autonomous 真机 demo、competence threshold 定量边界、recovery-data ablation（去掉 recovery 数据策略卡 failure mode）、5 GB VRAM/task。

---

## 与 RPD 的一句话区分

同起点（VLA SFT ~60%）但目标相反：**PLD 改 VLA 本体**（off-policy Cal-QL + 双 buffer + probing + 蒸馏，真机可行，部署单一 VLA）；**RPD**（Jülg et al. IROS 2026）训新小学生（on-policy PPO + MSE 蒸馏 $\mathcal{L}=\mathcal{L}_{\text{PPO}}-\mathcal{L}_{\text{MSE}}$，仿真 only，百万样本，部署 100M student，VLA 训完即扔）。PLD 选 off-policy 是因真机 rollout 太贵不能丢——不是 off-policy 更好，而是真机没得选。
