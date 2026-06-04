# VLA 真机后训练：PLD 与 RECAP 的机制级解读 + HEARS-Buffer 方向

> 从**算法机制**角度解读 VLA（Vision-Language-Action）真机后训练的两条自改进路线——PLD（off-policy 残差 RL 做数据生成）与 RECAP/π\*0.6（迭代 offline RL，advantage 作 conditioning）——讲清目标函数、算法、推导与失败模式，并由二者共同的技术开放问题导出数据层方向 HEARS-Buffer。
>
> 刘越（同济大学计算机学院 空间智能课题组） · 2026-06

---

## 📄 核心文档

| 文档 | 说明 |
|---|---|
| **[技术报告_VLA后训练_PLD与RECAP机制解读.md](./技术报告_VLA后训练_PLD与RECAP机制解读.md)** | **主报告**（7 章：问题设定 → 支撑机制 → PLD → RECAP → 机制对比 → HEARS 方向 → 训练动力学）+ PDF |
| [docs/A1_PLD技术细节深读.md](./docs/A1_PLD技术细节深读.md) | PLD 全细节（残差/Cal-QL/probing/蒸馏/消融，逐节核对论文） |
| [docs/A2_RECAP技术细节深读.md](./docs/A2_RECAP技术细节深读.md) | RECAP 全细节（分布式 value/advantage/CFG/下界/迭代，含对原文数字的更正） |
| [docs/A3_支撑机制_CalQL_RLPD_AWR_CFG.md](./docs/A3_支撑机制_CalQL_RLPD_AWR_CFG.md) | Cal-QL/RLPD/AWR-AWAC/CFG 精确目标函数与动机 |
| [docs/01_训练动力学机制解读.md](./docs/01_训练动力学机制解读.md) | RL/SFT 训练动力学（分布/梯度/熵/遗忘 + 30+ 文献） |
| [docs/02_论文索引.md](./docs/02_论文索引.md) | 调研论文索引 |

> PDF 与 .md 同名同目录。`.md` 在 GitHub 直接渲染（含公式），`.pdf` 用 pandoc + Noto CJK 编译，便于直接发导师。

---

## 🎯 两条主线一句话

- **PLD**：冻结 VLA 主干，用 off-policy 残差 RL（Cal-QL + RLPD 双 buffer）训一个轻量残差专家接管失败状态；用"基策略探针（前 αT 步 VLA 单独走）+ 残差救场（后段接管）"生成贴着部署分布、含纠错行为的数据，再普通 SFT 蒸馏回 VLA。**RL 产物是数据，专家训完即弃；防遗忘机制在 probing（数据侧），不在蒸馏。**
- **RECAP / π\*0.6**：用分布式 value（201-bin 交叉熵 + MC return）把 episode 级 0/1 标签升级成 step 级 advantage，二值化成 "Advantage: positive/negative" 文本 prefix 注入 flow VLA，靠 CFG 式 random drop 联合训 conditional/unconditional 双支。**advantage 走 conditioning 通道而非 loss，绕开 PPO；防遗忘靠每轮从 anchor 重训（流程侧）。**

## 🔬 机制对比的关键

| | PLD | RECAP |
|---|---|---|
| 把 RL 变可行的方式 | 生成对齐数据 | 把弱标签升级成 advantage 信号 |
| credit assignment | 精确 Q（必须 Cal-QL 抗 OOD 高估） | 相对 value（MC 足矣，躲开 TD 发散） |
| 防遗忘 | 数据侧（probing → 小 KL） | 流程侧（每轮回 anchor 重训） |
| 取舍 | 样本效率优先（250× 复用，5GB 可跑） | 算力换简单（单池只增，集群 brute-force） |

**共同技术空白** → HEARS-Buffer 切入点：① 失败 rollout 段级利用（PLD 丢失败 / RECAP episode 级 advantage 把"局部正确全局失败"段拉低）② 跨任务 skill 复用 ③ buffer 能力感知主动管理 ④ 长 horizon 闭环-开环鸿沟。

---

## 📁 仓库结构

```
vla-sft-rl-survey/
├── README.md
├── 技术报告_VLA后训练_PLD与RECAP机制解读.md(+pdf)   # 主报告
└── docs/
    ├── A1_PLD技术细节深读.md(+pdf)
    ├── A2_RECAP技术细节深读.md(+pdf)
    ├── A3_支撑机制_CalQL_RLPD_AWR_CFG.md(+pdf)
    ├── 01_训练动力学机制解读.md(+pdf)
    └── 02_论文索引.md(+pdf)
```

> 私有仓库。`未开源`方法（RECAP）的精确常数论文未给；arXiv 26xx.xxxxx 为整理时预印本，引用前请核对。
