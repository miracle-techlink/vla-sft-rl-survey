# RECAP / π\*0.6 技术机制精读报告

**论文**：Physical Intelligence, *π\*0.6: a VLA That Learns From Experience*, arXiv:2511.14759v2（2025-11-19，**未开源**）
**方法名**：RECAP = RL with Experience and Corrections via Advantage-conditioned Policies

> 本文是主报告第 4 章的全细节支撑版。凡精读笔记与原文不一致处以原文为准并标注（如 dropout 30% 非 50%、后训练 advantage 用 N=50 lookahead）。

定位：把 episode 级 0/1 成功标签，经分布式 value function 升级成 step 级 advantage，再把 advantage 当作**条件输入（conditioning）而非损失项（loss）**注入策略，用纯加权监督学习实现 self-improving，**完全绕开 PPO/REINFORCE**。

---

## 1. 框架：PRETRAIN → SPECIALIZE → ITERATE，及"每轮从 anchor 重训"

Algorithm 1（10 行）：
```
1: 用 Eq.1 在 D_demo 上训 V_pre          ┐ PRETRAIN（一次性，π_pre 即 π0.6）
2: 用 Eq.3 + V_pre 训 π_pre              ┘
3: D_ℓ ← D_demo(ℓ)                       ┐ SPECIALIZE
4: 从 V_pre 训 V⁰_ℓ（Eq.1, D_ℓ）          │
5: 从 π_pre 训 π⁰_ℓ（Eq.3, V⁰_ℓ）          ┘
6: for k=1…K:                            ┐ ITERATE
7:   用 π^{k-1} 采数据并入 D_ℓ             │
8:   从 V_pre 训 V^k（Eq.1, 整个 D_ℓ）      │
9:   从 π_pre 训 π^k（Eq.3, V^k）           │
10: end for                              ┘
```

**反直觉核心：第 8/9 行每轮都从 $V_{\text{pre}}/\pi_{\text{pre}}$ 重训，而非从上一轮增量更新。** 原文 V-D："finetuned from the pre-trained checkpoint, rather than the policy and value from the last iteration ... useful for avoiding drift over multiple iterations." 机制：从 $\pi^{k-1}$ 增量更新会逐轮累积分布偏移（连续微调下的灾难性遗忘）；把 $\pi_{\text{pre}}$ 当永久 anchor、每轮回原点，唯一在变的是数据池越来越好，改进由数据质量驱动而非参数漂移 → 多轮单调可控。

---

## 2. 奖励与 value function

**Reward（Eq.5）：** $r_t=0$（$t=T$ 且 success）/ $-C_{\text{fail}}$（$t=T$ 且 failure，$C_{\text{fail}}$ 大常数，论文未给精确值）/ $-1$（otherwise）。无折扣下成功轨迹 MC return $R_t=\sum_{t'\ge t}r_{t'}=-(\text{剩余步数})$ → **value 学的是"到成功的负剩余步数"**，归一化 $[-1,0]$。失败轨迹末步 $-C_{\text{fail}}$ 拉低 return → value 显式学会识别失败状态。

**分布式 value（Eq.1）：** 小 VLM backbone（~670M），输出 $V$ 在 **201 bin** 上 softmax，对离散化 MC return 做交叉熵：
$$\min_\phi\ \mathbb{E}_{\tau}\Big[\textstyle\sum_{o_t} H\big(R^B_t(\tau),\,p_\phi(V|o_t,\ell)\big)\Big],\quad V(o_t)=\textstyle\sum_b p_\phi(V{=}b|o_t)\,v(b).$$

**为什么分布式 + CE 而非 MSE：** (i) return 多峰，MSE 压成条件均值丢模态；(ii) CE 把回归变分类、目标固定 one-hot、梯度有界、对失败极端负 return 鲁棒，MSE 对长尾大残差易震荡。**为什么 MC 而非 TD：** TD bootstrap 在大模型+异质 off-policy+稀疏奖励下易发散（需 Cal-QL 一整套）；MC 无自举发散，代价是高方差且反映行为策略 return（非 $V^*$），但 value 只用来算 advantage 做条件、不进 actor 反向传播，偏差可容忍——只需可靠的相对好坏信号。

---

## 3. Advantage 与二值 indicator

**后训练 advantage（Appendix F，N=50 lookahead）：** $A^\pi(o_t,a_t)=\sum_{t'=t}^{t+49} r_{t'}+V(o_{t+50})-V(o_t)$。预训练用纯 MC（$N=T$）。

**indicator：** $I_t=\mathbb{1}[A>\epsilon_\ell]$。阈值 per-task 校准：预训练约 30% 为正、微调约 40%、折衣（demo 质量高）约 10%。**为什么阈值而非推理期 $\beta$：** 高 CFG weight "drive action distribution to corners of support → aggressive behavior"，故把"该多优化"从推理期挪到训练期 $\epsilon_\ell$。

**三类数据不对称：** demo 强制 $I_t$=True（绕过 value）；human correction 强制 True（假设专家纠正总好）；autonomous 真按 $\mathbb{1}[A>\epsilon]$（失败段多 False，教"别这么做"）。失败 episode 不丢——既训 value 又教策略避错。

---

## 4. 策略训练：CFG 双支 + prefix 注入 + random drop

**目标（Eq.3）：**
$$\min_\theta\ \mathbb{E}_{\mathcal{D}}\Big[-\log\pi_\theta(a_t|o_t,\ell)-\alpha\log\pi_\theta(a_t|I_t,o_t,\ell)\Big].$$
unconditional 支学行为平均；conditional 支学已知好坏标签下的策略。**本质 = 对全部数据（成功+失败、好+坏）的加权/条件监督学习**，无 PG、无 importance ratio、无 KL。

**prefix 注入：** $I_t$=True 输入 "Advantage: positive"、否则 "negative"，位置在 $\ell$ 之后、动作之前 → 只影响动作 log-likelihood、不改架构。

**random drop $I_t$（30%，非笔记的 50%）：** 30% 步训 unconditional、70% 训 conditional → 一个 trick 同训两模型省一遍 forward，**等价替代权重 $\alpha$**（"randomly omit $I_t$ instead of tuning $\alpha$"）。这是 CFG 训练侧标准做法。

---

## 5. CFG 推理与 flow-matching 下界

**闭式（Eq.2）：** $\hat\pi\propto\pi_{\text{ref}}(a|o,\ell)\big(\pi_{\text{ref}}(a|I,o,\ell)/\pi_{\text{ref}}(a|o,\ell)\big)^\beta$。$\beta=1$ 退化为直接用 conditional 支（**单次 forward**，默认部署）；$\beta>1$ 锐化 improvement 方向（2× forward）。

**flow score 外推（Appendix E）：** $\nabla_a\log\pi_\theta(a|o,\ell)+\beta(\nabla_a\log\pi_\theta(a|I,o,\ell)-\nabla_a\log\pi_\theta(a|o,\ell))$——与图像扩散 CFG 形式完全一致，"类别条件"换成 "Advantage: positive"。

**下界（Eq.4/9）：** flow 无解析 $\log\pi$，用扩散↔flow 等价把 flow-matching MSE 当 log-likelihood 下界：$\log\pi_\theta(a|I,o,\ell)\ge\mathbb{E}[\log p_\theta(a^{\text{disc}}|\cdot)-\alpha_\eta\|\omega-a-f_\theta(a^{\eta,\omega},I,o,\ell)\|^2]$（离散 token 自回归似然 + 连续动作 flow MSE）。

---

## 6. 为什么绕开 PPO：AWR 视角与下界

regularized RL 目标 $\mathcal{J}=\mathbb{E}[\sum\gamma^t r_t]-\beta\mathbb{E}[D(\pi\|\pi_{\text{ref}})]$ 的经典解是 AWR $\tilde\pi\propto\pi_{\text{ref}}\exp(A/\beta)$。RECAP 不用它（AWR "discard or significantly downweight a significant portion of the data → filtered imitation"），而用 **less well-known result**：$\tilde\pi\propto\pi_{\text{ref}}(a|o)\,p(I|A)$，只要 $p(I|A)$ 是 advantage 单调递增函数即保证 $\mathcal{J}(\tilde\pi)\ge\mathcal{J}(\pi_{\text{ref}})$（单调改进），取阈值 delta 分布 + Bayes 重写得 Eq.2/Eq.3。整个推导只涉及监督式 log-likelihood，advantage 不进梯度。

| 维度 | PPO | RECAP |
|---|---|---|
| 数据 | on-policy | off-policy/全离线，单池只增 |
| advantage 进哪 | loss/梯度 | conditioning 通道（prefix） |
| importance ratio | 需 + clip | 无 |
| KL/信任域 | 需 | 无显式（隐含在 anchor 重训 + 阈值） |
| flow 兼容 | 难 | 用下界天然兼容 |

实验（Fig.11）：PPO（信任域收到 $\eta=0.01$ 才稳但性能差）、AWR（策略慢、throughput 低）均显著差于 RECAP。

---

## 7. 迭代微观机制：$I_t$ flip

旧 episode $(o_t,a_t,r_t)$ 永不变，但每轮从 $V_{\text{pre}}$ 重训新 $V^k$（用更大 $\mathcal{D}_\ell$）→ $A^{V^k}\neq A^{V^{k-1}}$ → $I_t^{(k)}$ 可能 flip。随策略变强、行为策略平均（$V$）抬升，上轮"高于平均"的动作这轮可能"低于新平均" → **自举式标准提升**，advantage 永远衡量"相对当前水平的进步"，阈值按"约 40% 为正"自动上移及格线——这是跨多轮持续涨不饱和的引擎。

---

## 8. 实验

**硬件：** 静态双臂 2×6-DoF + 平行夹爪，50 Hz，3 相机。三任务族：Cafe/Box/Laundry。

**标志能力：** espresso 连续 **13 小时**；折陌生衣物 2 小时+；装纸箱 600 秒全流程。
**核心改善：** 最难任务 throughput **>2×**、failure **~2× reduction**，多数任务 90%+。折衣 collar-up 严格判据 2 轮（每轮 600 条）达 97%。

**数据/人力（Appendix F）：** T 恤每轮 4 机器人 300 条、2 轮、**0% intervention**；diverse laundry 450 eval + 287 correction；失败模式去除 ~1000 auto +(280+378) correction；box 每轮 600 demo + 360 correction（intervention 最高）；cafe 单轮 429 correction + 414 auto。

**ablation：** advantage conditioning 链式增益（$\pi_{0.5}$→$\pi_{0.6}$无indicator→RL-pretrained→offline-RL+SFT→full RECAP）；vs AWR/PPO 远超；多轮（Fig.9/10）T 恤首轮破 90%、长程 box 2 轮 throughput 2×；默认 $\beta=1$ 已达报告性能。

---

## 9. 与精读笔记的差异更正

1. **dropout = 30%**（笔记 50%）。
2. **后训练 advantage 用 N=50 lookahead**，非纯 $R_t-V$；纯 MC 仅预训练。
3. **微调阈值约 40%、折衣约 10%、预训练约 30%**。
4. $C_{\text{fail}}$ 精确值、最终 checkpoint 均未开源，引用须标"论文未给"。

---

**一句话总结：** 把 RL 的 advantage 从"loss 通道"挪到"conditioning 通道"——分布式 MC value 把 0/1 升级成 step advantage，二值化成 "Advantage: positive/negative" prefix 注入 flow VLA，靠 CFG 式 random drop 联合训双支，把 self-improving 实现为"从永久 anchor 反复重训的 advantage 加权监督学习"，绕开 PPO 的 importance ratio/KL/on-policy，同时天然兼容长程（episode reward）与 flow 架构（log-likelihood 下界）。
