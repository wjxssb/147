# UCLA ECE C147A / C247A Final Project: 最终提交核验报告 (Submission Checklist)

**项目名称**: Continuous Keystroke Prediction from Surface Electromyography (sEMG) Signals  
**论文题目**: *Bilateral Spatial Coverage Matters More Than Electrode Density in Continuous sEMG Typing Decoding*  
**受试者编号**: `#89335547` (全量 18 会话: 16 训练, 1 验证, 1 严格隔离测试)  
**计算平台**: Debian GNU/Linux 13 trixie, NVIDIA Tesla V100-SXM2-16GB, Driver 580.196.07, CUDA 13.0  
**核验状态**: **8 / 8 全部通过 (100% PASS)**  

---

## 一、八项最终交付核验清单 (Checklist 8/8)

- [x] **1. 论文中 A0/A1 与 Clean 实验使用各自正确的对照协议**
  - **A0 vs A1 (架构比较)**：保持在原始带旋转数据增强协议下对比（A0 Val CER: $19.20 \pm 0.32\%$, A1 Val CER: $19.34 \pm 0.19\%$），两者 CER 处于同等水平，BiGRU 训练用时增加 24%，绝不宣称 BiGRU 具备决定性优势。
  - **A1-Clean vs F3-Clean (电极数量比较)**：在统一无旋转纯净协议下对比（32ch: $17.96 \pm 0.37\%$ vs 16ch: $20.13 \pm 0.64\%$），配对差值 $\Delta_s = +2.17 \pm 0.65\text{ pp}$。
  - **F1-Clean vs F2-Clean vs F3-Clean (空间分布比较)**：在相同 16 电极预算与 5.4M 参数量下对比，双腕 8+8 以 $26.95 \sim 32.76\text{ pp}$ 优势压倒单腕集中分布。
  - **严格杜绝交叉混用**：绝对不拿旧版 A0 (19.20%) 与 A1-Clean (17.96%) 跨协议宣称 BiGRU 优于 TDS-CNN。

- [x] **2. Full32-Clean 与 F3-Clean 的逐种子配置、Checkpoint 和预测记录一致**
  - Seeds 42, 123, 2026 的 Hydra 运行时配置（`hydra_configs/config.yaml`）、模型权重 Checkpoint、验证集预测记录（`val_predictions.json`）、动态规划错误分解（`error_analysis.json`）及硬件开销日志（`hardware_metrics.json`）均完备保存在 `emg2qwerty/results/` 下。

- [x] **3. 所有 CER、Loss、Mean ± Sample Std 均从原始结果自动生成**
  - 均由自动化 Python 脚本读取原始 JSON 计算，标准差一律采用无偏样本标准差（Bessel 校正，$ddof=1$），并同时明示了无四舍五入的浮点精确值。
  - 补充计算了逐种子配对差值 $\Delta_s = \text{CER}_{\text{F3}, s} - \text{CER}_{\text{Full32}, s}$：$+1.42\text{ pp}$ (Seed 42), $+2.53\text{ pp}$ (Seed 123), $+2.57\text{ pp}$ (Seed 2026)，配对均值与样本标准差为 **$+2.17 \pm 0.65\text{ pp}$**。

- [x] **4. Legacy F3 电极污染结果明确标记为无效对照，不混入正式均值**
  - 旧版受 `RandomBandRotation` 循环移位污染的 F3 实验在 `MASK_ORDER_AUDIT.md` 与 `FINAL_AUDIT.md` 中被醒目标记为 `[INVALID FOR FIXED-ELECTRODE-BUDGET COMPARISON]`，论文 Table 1 与正文彻底替换为纯净重训的 F3-Clean。

- [x] **5. Clean 模型未报告未经评测的 Test CER**
  - 严格遵守测试集隔离纪律，Clean 系列（A1-Clean, F1-Clean, F2-Clean, F3-Clean）在 Table 1 中的 Test Loss 和 Test CER 严格填为 `--`，绝无偷跑或编造测试指标。

- [x] **6. 课程要求的 Contribution Statement、AI Usage Statement 和页数已核验**
  - 严格遵照 UCLA ECE C147A 官方 Handout 规则：**正文严格不超过 7 页**，贡献声明、AI 使用声明与参考文献不计入页数限制。
  - 生成的 `paper/main.pdf` 精确为 **8 页**：第 1 至 7 页为完整正文（在第 7 页底端完结），第 8 页专门承载贡献声明、AI 使用声明与参考文献。

- [x] **7. main.pdf 实际打开后，图表、公式、引用和文字均可正常阅读**
  - 使用 Tectonic 重新编译，编译输出中 **Overfull \hbox 警告为 0**。
  - 表格经过 `\resizebox{\textwidth}{!}{...}` 精细适配，两栏公式均采用折行排版，无任何右侧文本溢出。

- [x] **8. 最终提交文件、README 和复现命令与当前代码版本对应**
  - 代码库与复现命令在本地 Python 3.12 及 Debian V100 GPU 环境下均经过端到端运行检验，自动化单测通过率 100%（38/38 Passed）。

---

## 二、逐种子配对差值深入分析 (Paired Seed Analysis)

为了更精确地衡量“电极减半（32 通道 $\to$ 双腕 8+8 通道）”在单用户验证集上的实际精度代价，我们计算了逐种子配对差值：
$$ \Delta_s = \text{CER}_{\text{F3-Clean}, s} - \text{CER}_{\text{A1-Clean}, s} $$

| 随机种子 ($s$) | A1-Clean (32ch) Val CER | F3-Clean (8+8ch) Val CER | 配对差值 ($\Delta_s$) | 相对变化率 |
| :---: | :---: | :---: | :---: | :---: |
| **Seed 42** | 18.08% (18.0771%) | 19.49% (19.4949%) | **+1.42 pp** (+1.4178 pp) | +7.84% |
| **Seed 123** | 18.25% (18.2543%) | 20.78% (20.7798%) | **+2.53 pp** (+2.5255 pp) | +13.83% |
| **Seed 2026** | 17.55% (17.5454%) | 20.12% (20.1152%) | **+2.57 pp** (+2.5698 pp) | +14.65% |
| **配对统计 (Mean $\pm$ Std)** | **$17.96 \pm 0.37\%$** | **$20.13 \pm 0.64\%$** | **$+2.17 \pm 0.65\text{ pp}$** | **$+12.11\%$** |

> **科学含义**：逐种子配对消除了初始化幸运/不幸种子对全局误差水平的系统性平移，表明双腕电极减半对当前单用户造成的稳定精度损失在 **$1.42 \sim 2.57$ 个百分点** 之间，平均仅付出 **$+2.17\text{ pp}$** 的代价即可降低 50% 的传感器硬件开销。

---

## 三、一键复现命令索引 (Reproduction Commands)

### 1. 自动化回归测试与硬件冒烟测试
```bash
cd /home/frank/下载/课程/ECE147/Final_Project/emg2qwerty

# 运行排除通道绝对不变性回归测试 (6 tests)
pytest emg2qwerty/tests/channel_invariance_test.py -v

# 运行核心单元测试套件 (38 tests)
pytest emg2qwerty/tests/channel_invariance_test.py emg2qwerty/tests/charset_test.py emg2qwerty/tests/data_test.py

# 运行 GPU 端到端前向/反向冒烟测试
python scripts/gpu_smoke_test.py
```

### 2. 独立重新计算错误指标与键盘分区统计
```bash
cd /home/frank/下载/课程/ECE147/Final_Project/emg2qwerty
python scripts/audit_stage1.py
```

### 3. 论文 PDF 编译与页数核查
```bash
cd /home/frank/下载/课程/ECE147/Final_Project/paper
tectonic main.tex
pdfinfo main.pdf | grep "Pages:"  # 必须精确输出 Pages: 8
```
