# UCLA ECE C147A / 247A: 期末科研大作业全套成果归档 (Final Project Hub)

> **论文题目**: *Bilateral Spatial Coverage Matters More Than Electrode Density in Continuous sEMG Typing Decoding*  
> **核心交付**: [**`final_paper.pdf` (8页 NeurIPS 终稿)**](./final_paper.pdf) | [**`paper/main.tex` (LaTeX源码)**](./paper/main.tex) | [**`SUBMISSION_CHECKLIST.md` (交付核验 8/8)**](./SUBMISSION_CHECKLIST.md)  
> **学术审计**: [**`FINAL_AUDIT.md` (科研审计总报告)**](./FINAL_AUDIT.md) | [**`MASK_ORDER_AUDIT.md` (纯净重训记录)**](./MASK_ORDER_AUDIT.md)  
> **算法代码**: [`emg2qwerty/` (核心算法库与测试套件)](./emg2qwerty/) | [`scripts/` (自动化核验与评测脚本)](./scripts/) | [`patchkit/` (研究补丁归档)](./patchkit/)

---

## 一、核心科研发现概览 (Executive Summary)

本研究基于 Meta Reality Labs `emg2qwerty` 基准数据集（受试者 `#89335547`，全量 18 会话），深入探讨了连续表面肌电（sEMG）按键解码任务中的神经网络时序建模、电极空间布局与跨会话泛化特性。

```
                                 [双腕 32 通道基准: A1-Clean]
                                    Val CER: 17.96 ± 0.37%
                                              |
                   +--------------------------+--------------------------+
                   | 电极减半 (-50%)                                     | 单侧堆叠 (16ch 单侧)
                   v                                                     v
       [双腕对称稀疏: F3-Clean]                              [单侧密集: F1 / F2 Clean]
         Val CER: 20.13 ± 0.64%                                左腕 F1: 52.89 ± 3.62% (对侧盲区 89.3%)
      (配对代价仅 +2.17 ± 0.65 pp)                             右腕 F2: 47.08 ± 2.42% (对侧盲区 78.9%)
```

### 1. 核心定量结论
1. **电极减半代价极小（Unconfounded Decimation Penalty）**：
   在名义模型容量（5.4M）与无旋转增强协议严格对齐的受控对比下，将电极从 32 通道减半至双腕 16 通道（8+8），验证 CER 仅由 **$17.96\%$** 上升至 **$20.13\%$**。逐种子配对差值 $\Delta_s = \text{CER}_{\text{F3}, s} - \text{CER}_{\text{Full32}, s}$ 稳定在 **$+2.17 \pm 0.65\text{ pp}$**（相对误差仅微增 12.1%）。
2. **双腕空间覆盖支配单腕电极密度**：
   在相同 16 个有效电极预算下，双腕 8+8 布局相比单右腕（F2-Clean）领先 **$26.95\text{ pp}$**，相比单左腕（F1-Clean）领先 **$32.76\text{ pp}$**。细粒度动态规划分解表明，单腕传感器在对侧击键上存在严重的信息获取局限性（遗漏率达 $80.3\% \sim 86.4\%$）。
3. **架构归纳偏置与工程权衡**：
   在基线协议下对比，官方 TDS-CNN 基线（$19.20 \pm 0.32\%$）与提出的 CNN-BiGRU-CTC（$19.34 \pm 0.19\%$）验证 CER 处于同等水平。BiGRU 虽取得更低的 CTC 损失与替换率，但训练用时增加 24% 且推断无法切片并行，未展现整体 CER 优势。

---

## 二、完整实验结果汇总矩阵 (Validation & Frozen Test)

| 实验编号 | 物理电极构型 | 激活通道 | 名义参数量 | 验证 Loss | 验证 CER (%) | 测试 CER (%) | 字符替换率 | 字符遗漏率 | 训练时长 |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **A0** | TDS-CNN 官方基线 | 32 (2$\times$16) | 5.3M | 0.745 | **19.10%** ($19.20 \pm 0.32$) | 21.33% | 13.36% | 4.21% | 14.6 min |
| **A1** | CNN-BiGRU-CTC 主模型 | 32 (2$\times$16) | 5.4M | **0.638** | 19.49% ($19.34 \pm 0.19$) | **21.05%** | **12.80%** | 5.23% | 18.2 min |
| **C1** | 单左腕切片 (16L) | 16 (1$\times$16) | 3.1M | 1.761 | 60.08% | 60.97% | 9.15% | 50.75% | 15.1 min |
| **C2** | 单右腕切片 (16R) | 16 (1$\times$16) | 3.1M | 1.541 | 45.86% | 47.96% | 19.81% | 24.94% | 15.2 min |
| **C3** | 双腕等距稀疏 (8L+8R) | 16 (2$\times$8) | 5.3M | 0.749 | 24.17% | 25.29% | 16.55% | 4.94% | 17.2 min |
| **C4** | 双腕极度稀疏 (4L+4R) | 8 (2$\times$4) | 5.2M | 0.878 | 27.49% | 33.67% | 19.81% | 5.38% | 16.7 min |
| **A1-Clean** | 双腕全通道无旋转基准 | 32 (2$\times$16) | 5.4M | 0.601 | $\mathbf{17.96 \pm 0.37\%}$ | -- | 11.56% | 4.58% | 18.0 min |
| **F3-Clean** | 双腕对称固定容量 (8+8) | 16 (2$\times$8) | 5.4M | 0.646 | $\mathbf{20.13 \pm 0.64\%}$ | -- | 13.41% | 4.62% | 18.1 min |
| **F1-Clean** | 单左腕固定容量 (16L) | 16 (1$\times$16) | 5.4M | 1.729 | $52.89 \pm 3.62\%$ | -- | 13.81% | 38.18% | 18.0 min |
| **F2-Clean** | 单右腕固定容量 (16R) | 16 (1$\times$16) | 5.4M | 1.556 | $47.08 \pm 2.42\%$ | -- | 16.33% | 29.74% | 18.1 min |

> *注：Clean 系列严格保留测试集隔离（标记为 `--`），三随机种子指标采用 Bessel 校正无偏样本标准差（$ddof=1$）。*

---

## 三、逐种子配对差值核算表 ($\Delta_s = \text{CER}_{\text{F3-Clean}, s} - \text{CER}_{\text{A1-Clean}, s}$)

| 随机种子 ($s$) | A1-Clean (32ch) | F3-Clean (8+8ch) | 配对差值 ($\Delta_s$) | 相对增幅 |
| :---: | :---: | :---: | :---: | :---: |
| **Seed 42** | 18.08% (18.0771%) | 19.49% (19.4949%) | **+1.42 pp** (+1.4178 pp) | +7.84% |
| **Seed 123** | 18.25% (18.2543%) | 20.78% (20.7798%) | **+2.53 pp** (+2.5255 pp) | +13.83% |
| **Seed 2026** | 17.55% (17.5454%) | 20.12% (20.1152%) | **+2.57 pp** (+2.5698 pp) | +14.65% |
| **均值与样本标准差** | **$17.96 \pm 0.37\%$** | **$20.13 \pm 0.64\%$** | **$+2.17 \pm 0.65\text{ pp}$** | **$+12.11\%$** |

---

## 四、目录结构与交付文件索引

```
Final_Project/
├── final_paper.pdf              # 最终 8 页定稿论文 PDF (NeurIPS 2024 排版, 实体物理文件)
├── SUBMISSION_CHECKLIST.md      # 提交终验 8/8 全项 PASS 报告
├── FINAL_AUDIT.md               # 阶段一至阶段五全套科研真实性审计报告
├── MASK_ORDER_AUDIT.md          # 通道污染排查、数学反例证明与重训归档
├── ECE147_Final_Project_Handout.pdf # 课程官方 Handout #18 (问题陈述与提交规范)
├── paper/                       # 论文工程目录
│   ├── main.pdf                 # NeurIPS 2024 正式排版 PDF (8 页, 0 Overfull 警告)
│   ├── main.tex                 # 论文 LaTeX 源码
│   ├── neurips_2024.sty         # NeurIPS 官方宏包
│   └── references.bib           # 参考文献 BibTeX 数据库
├── emg2qwerty/                  # 核心算法实现库
│   ├── emg2qwerty/              # 模块代码 (CNN-BiGRU, MaskChannels, CTC 守恒)
│   ├── tests/                   # 包含 channel_invariance_test 等不可回归自动化单测
│   └── results/                 # 全量实验原始 Checkpoint、预测与评测 JSON 结果
├── scripts/                     # 验证与测试工具箱
│   ├── audit_stage1.py          # 真实性动态规划重算工具
│   ├── evaluate_frozen_test.py  # 严格隔离测试集评测工具
│   └── gpu_smoke_test.py        # GPU 排除通道不变性冒烟测试
└── patchkit/                    # 研究覆盖包归档
    ├── ece147_final_project_research_overlay.zip
    └── README_RESEARCH_PATCHKIT.md
```

---

## 五、复现与验证命令 (Quick Commands)

```bash
# 1. 运行排除通道绝对不变性回归测试 (6 tests)
pytest emg2qwerty/tests/channel_invariance_test.py -v

# 2. 运行完整自动化测试套件 (38 tests)
pytest emg2qwerty/tests/channel_invariance_test.py emg2qwerty/tests/charset_test.py emg2qwerty/tests/data_test.py

# 3. 运行 GPU 端到端冒烟测试
python scripts/gpu_smoke_test.py

# 4. 重新编译论文 PDF
cd paper && tectonic main.tex
```
