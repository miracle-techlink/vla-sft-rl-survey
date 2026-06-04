# 支撑机制：Cal-QL / RLPD / AWR-AWAC / CFG

> 本文是主报告第 2 章的全细节支撑版。PLD 依赖 Cal-QL/RLPD（离线-到-在线 RL），RECAP 依赖 AWR/AWAC + CFG（advantage 加权/条件）。每个机制给出精确目标函数与设计动机。CFG 部分论文未涵盖，按通用文献给出，标"（解读）"。

来源：`storage_replay/S04_CalQL_2303.05479.pdf`、`S03_RLPD_2302.02948.pdf`、`pld_improvement/P07_AWAC_2006.09359.pdf`、`P06_IQL_2110.06169.pdf`。

---

## 0. 共同前置：离线 RL 的 OOD 高估病理

Bellman 备份 $\mathcal{B}^\pi\bar Q(s,a)=r+\gamma\mathbb{E}_{a'\sim\pi(s')}[\bar Q(s',a')]$。策略改进 $\pi(s)=\arg\max_a Q_\theta(s,a)$ 选出的 $a'$ 落在数据外（OOD）时，$\bar Q(s',a')$ 是函数逼近器在未见区域的外推、无数据约束。$\arg\max$ **系统性挑中被高估的 OOD 动作** → 高估经备份回传放大 → Q 单调爆炸发散。离线比在线更致命：无法采新数据纠偏，错误外推永不修正。三条应对：显式保守（CQL→Cal-QL）、隐式正则+架构抑制（RLPD）、隐式策略约束（AWR/AWAC）。

---

## 1. CQL → Cal-QL

**CQL 保守目标（Eq.3.1）：**
$$\min_\theta\ \underbrace{\alpha\big(\mathbb{E}_{s\sim\mathcal{D},a\sim\pi}[Q_\theta]-\mathbb{E}_{s,a\sim\mathcal{D}}[Q_\theta]\big)}_{\text{push down OOD; push up data}}+\tfrac12\mathbb{E}_{\mathcal{D}}[(Q_\theta-\mathcal{B}^\pi\bar Q)^2].$$
第一项压低当前策略（可能 OOD）动作 Q、抬高数据动作 Q → $Q_\theta$ 成真值下界，$\arg\max$ 不选 OOD。

**Cal-QL 动机：** CQL 常把 Q 压得过低（远低于真值）。在线微调时坏动作的真实回报仍高于被严重低估的离线 Q → 坏动作"看似最优" → 策略 unlearn 已学好的离线初始化 → offline-to-online 早期性能塌陷（dip）。

**校准定义（Def.4.1）：** $\mathbb{E}_{a\sim\pi}[Q_\theta^\pi(s,a)]\ge\mathbb{E}_{a\sim\mu}[Q^\mu(s,a)]=V^\mu(s)$。Q 既下界真值（保守）又上界参考策略真值 $V^\mu$（校准），$\mu$ 取行为策略（$V^\mu$ 由 return-to-go 回归、无 bootstrap）。

**Calibrated objective（Eq.5.1）：**
$$\mathbb{E}_{s\sim\mathcal{D},a\sim\pi}\big[\max(Q_\theta(s,a),V^\mu(s))\big]-\mathbb{E}_{s,a\sim\mathcal{D}}[Q_\theta(s,a)].$$
$\max(Q_\theta,V^\mu)$ 把 push-down 截断在 $V^\mu$ 以上 → Q 被保证 $\gtrsim V^\mu$，坏动作（真实回报 $\le V^\mu$）不再显得最优 → **消除 dip、不掉性能、加速**，同时仍下界真值（保守保留）。Theorem 6.1：regret = miscalibration + overestimation 两项受控，$V^\mu$ 越接近 $V^*$ 界越紧。

---

## 2. RLPD：四要素

不做离线预训练、不加显式保守项，naive SAC + replay 靠四件套既稳又高效。

**① 对称采样 50/50：** 每 minibatch 一半 offline 一半 online，每步恒有 offline 转移把 Q 钉在数据区域 → 隐式正则，无需显式约束。50/50 在稀疏奖励方差最低（80/20 退化）。

**② Critic LayerNorm：** $\|Q_{\theta,w}(s,a)\|=\|w^\top\text{relu}(\psi_\theta)\|\le\|w\|\|\psi_\theta\|\le\|w\|$，Q 被权重范数上界（即便 OOD）→ **封顶外推爆炸**，不用显式保守项也稳。只压数值外推、不约束探索 → 防发散又保探索。去掉它 Q 发散性能崩。

**③ Ensemble + Min-Q：** $y=r+\gamma\min_{i\in\mathcal{Z}}Q_{\theta_i'}(s',\tilde a')$，取 min = 悲观分位，抵消"$\max$+函数逼近"的正向高估偏置。随机集成蒸馏（$E$ 个抽 $Z\in\{1,2\}$）兼顾抑高估与多样性。

**④ 高 UTD：** 每交互一步做 $G$ 次梯度更新（10–20），数据反复 bootstrap → 样本效率大增。但高 UTD 对同批数据大量梯度步易过拟合/放大高估 → **必须依赖①②③才不发散**。四要素协同：UTD 提供效率，前三者提供稳定性。Cal-QL §7.2 追高 UTD 时直接复用 RLPD 的 LayerNorm+ensemble。

---

## 3. AWR / AWAC（RECAP 理论近亲）

从带 KL 信赖域的策略改进 $\max_\pi\mathbb{E}_{a\sim\pi}[A^{\pi_k}]$ s.t. $D_{\mathrm{KL}}(\pi\|\pi_\beta)\le\epsilon$ 出发，KKT 解 $\pi^\star\propto\pi_\beta\exp(A/\lambda)$ 投影回参数类得 **AWR 目标（Eq.13）：**
$$\theta_{k+1}=\arg\max_\theta\ \mathbb{E}_{(s,a)\sim\mathcal{D}}\big[\log\pi_\theta(a|s)\cdot\exp(\tfrac1\lambda A^{\pi_k}(s,a))\big].$$

性质：(i) 解以 $\pi_\beta$ 为基底、回归只在数据样本上做 → 结果策略天然拉向数据支撑（隐式 KL 约束）、**从不查询 OOD Q**；(ii) advantage 用 off-policy bootstrap Q 估计、可复用任意数据；(iii) 权重直接作用于采样 $(s,a)$、**无需重要性采样、无需显式拟合 $\pi_\beta$**（避免在线微调时行为模型漂移失配 → 既不过度保守又能持续改进）。

**与 RECAP 异同：** 共享内核（advantage 引导策略偏向高优势动作）。区别——AWR 把 advantage 压成回归权重 $\exp(A/\lambda)$（软筛选，下采大量数据）；RECAP 把 advantage 作生成模型条件 + 推理时 CFG，可在推理期连续调控"想要多高优势"（调 $\beta$）而无需重训。

---

## 4. Classifier-Free Guidance（解读）

要从 $p(x|c)$ 采样（RECAP 中 $x$=动作、$c$=目标 advantage），CFG 用同一网络联合训 $p_\theta(x|c)$ 与 $p_\theta(x)$（训练时随机丢 $c$），推理时外推：
$$\hat p(x|c)\propto p(x)\big(p(x|c)/p(x)\big)^\beta,\quad \tilde\epsilon=\epsilon_\theta(x,\varnothing)+\beta(\epsilon_\theta(x,c)-\epsilon_\theta(x,\varnothing)).$$
$\beta>1$ 放大条件方向，把采样推向更强满足 $c$ 的样本，同时以 $p(x)$ 为底盘保证留在数据流形。$c$=高 advantage 时 = 推理时可连续调节的策略改进旋钮（$\beta$ 大→更 exploitative，$\beta\to0$→退回行为先验更安全）。与 LayerNorm/隐式 KL 精神一致：不显式查 OOD 而偏向高价值动作。random drop 保证 $p(x)$ 与 $p(x|c)$ 由同一网络一致表示，二者之差才是有意义的条件方向。

---

## 5. 机制总览

| 机制 | 核心目标 | 解决 | 一句话 |
|---|---|---|---|
| OOD 高估 | $\max_a Q_\theta$ 选 OOD 外推并放大 | — | 离线无法纠偏，错误自强化 |
| CQL | push down $\mathbb{E}_\pi Q$ − push up $\mathbb{E}_{\mathcal D}Q$ | OOD 高估 | Q 成下界，不选 OOD |
| Cal-QL | $\max(Q_\theta,V^\mu)$ 截断 | CQL 过度保守→unlearning | 抬下界 $\ge V^\mu$，不掉性能且加速 |
| 对称采样 | batch 50/50 | 隐式正则 | 每步 offline 转移钉住数据区 |
| LayerNorm | $\|Q\|\le\|w\|$ | 外推爆炸 | 封顶外推不约束探索 |
| min-Q | $y=r+\gamma\min Q$ | maximization bias | 悲观分位抵消正偏 |
| 高 UTD | 每步 $G$ 次更新 | 样本效率 | 多次备份，须靠前三者防发散 |
| AWR/AWAC | $\max\mathbb{E}[\log\pi\cdot\exp(A/\lambda)]$ | 过度保守/OOD 查询 | 隐式 KL 约束、off-policy、免 IS |
| CFG（解读） | $\hat p\propto p(x)(p(x|c)/p(x))^\beta$ | 推理时可控优势引导 | 以行为先验为底放大优势方向 |
