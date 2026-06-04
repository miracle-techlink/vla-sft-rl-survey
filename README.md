# VLA 真机 SFT+RL 混合训练：调研 + HEARS-Buffer 设计 + 训练动力学

> 面向 VLA（Vision-Language-Action）真机后训练的经验**采样 / 存储 / 利用**新范式调研。诊断 PLD / RECAP 两条 self-improving 路线，提出数据层方案 **HEARS-Buffer**，并附 RL/SFT 训练动力学机制解读。
>
> 研究者：刘越（同济大学计算机学院 空间智能课题组） · 2026-06

---

## 📄 核心文档

| 文档 | 说明 |
|---|---|
| **[导师汇报_VLA_SFT-RL调研与HEARS-Buffer.md](./导师汇报_VLA_SFT-RL调研与HEARS-Buffer.md)** | **主报告**（精简结构化，9 章，含训练动力学摘要章）— 给导师汇报用 |
| [docs/01_训练动力学机制解读.md](./docs/01_训练动力学机制解读.md) | RL/SFT 训练动力学机制**完整版**（分布 / 梯度 / 熵 / 遗忘四层 + 30+ 篇文献） |
| [docs/02_论文索引.md](./docs/02_论文索引.md) | 调研论文索引（VLA 后训练 + SFT+RL 混合 + Replay Buffer） |
| `*.pdf` | 主报告 PDF 导出（pandoc + Noto CJK） |

---

## 🎯 一页速览

**问题**：真机 RL 一条 rollout 30–60 秒 + 人工监管，经验极贵。"怎么采、怎么存、怎么用"是 self-improving VLA 的最大工程瓶颈。

**诊断**：
- VLA 直接 RL 触发灾难性遗忘（通用能力 95%→50%，4 因素叠加）→ 必须 SFT+RL 混合。
- 四范式（PLD / RPD / VLA-OPD / RECAP）都在"绕开直接 RL VLA"。
- 用「9 条件框架」量化 Buffer 必要性：**PLD 1/9，RECAP 1–2/9**。
- 两条主线：**PLD**（数据驱动，残差 RL 生成对齐数据再蒸馏）+ **RECAP/π\*0.6**（信号驱动，value function 生成训练信号直接 FT）。

**方案 — HEARS-Buffer**（self-improving VLA 的"第 4 个组件：数据层基础设施"，plug-in PLD/RECAP）：

| 支柱 | 功能 |
|---|---|
| A Episodic + Segment-aware | 段级 episodic 存储 |
| B Cross-task Skill Library | HDBSCAN 聚类跨任务技能复用 |
| **C★ Mixed SFT+RL Segment Annotation** | **核心创新：失败 rollout 段级混训** |
| D Capability-aware Lifecycle | 主动归档 / 遗忘 / 复发触发 |
| E Retrieval-augmented Decision | FAISS 运行时检索 |

**核心创新**：段级 advantage 能识别"局部正确但最终失败"的段（RECAP 整 episode advantage 做不到）；bad 段 KNN 找 recovery target 朝相似成功段 SFT，good 段 RL 强化——三合一新设计。

**验证**：6 个月 roadmap，先复现 PLD（4w）+ RECAP（8w）baseline，再 HEARS C 支柱 PoC（4w）+ 完整集成（8w）。目标 CoRL 2026 / ICLR 2027。

---

## 📚 关键文献

PLD (ICLR 2026, arXiv:2511.00091) · RECAP/π\*0.6 (PI, arXiv:2511.14759) · RLPD (ICML 2023, arXiv:2302.02948) · Cal-QL (NeurIPS 2023, arXiv:2303.05479)

训练动力学完整文献见 [docs/01](./docs/01_训练动力学机制解读.md)。

---

## 📁 仓库结构

```
vla-sft-rl-survey/
├── README.md
├── 导师汇报_VLA_SFT-RL调研与HEARS-Buffer.md   # 主报告
├── docs/
│   ├── 01_训练动力学机制解读.md
│   └── 02_论文索引.md
└── figures/                                    # 图素材（可选）
```

> 私有仓库，研究调研笔记。引用 arXiv 编号前请按实际发表信息核对。
