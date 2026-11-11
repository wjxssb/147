# ECE C147A / C247A Final Project — F3 电极污染风险专项审计与修复报告 (MASK_ORDER_AUDIT.md)

**审计主题**: 针对固定容量双腕掩码控制组（F3 Dual-8ch FixedCap）数据增强时序与电极泄漏风险的专项排查与确定性修复  
**执行环境**: 本地 Linux 工作区 & Debian 12 Tesla V100-SXM2-16GB 计算集群  
**审计基准**: 严格科研真实性标准（Zero-Fabrication, Strict Excluded-Channel Invariance）  
**报告日期**: 2026-11-10  

---

## 一、P0 风险排查结论：确认原实现存在电极污染严重缺陷

### 1. 缺陷定位与机理分析
在排查 Hydra 解析配置（`results/F3_Dual_8ch_FixedCap_seed42/hydra_configs/config.yaml`）与官方数据增强流水线（[`config/transforms/log_spectrogram.yaml`](file:///home/frank/下载/课程/ECE147/Final_Project/emg2qwerty/config/transforms/log_spectrogram.yaml)）时，确认了以下原始执行顺序：

```yaml
# 原有（缺陷）训练数据增强流水线
transforms:
  train:
    - ${to_tensor}          # 1. 加载双腕完整 16 通道原始数据 (T, 2, 16)
    - ${band_rotation}      # 2. RandomBandRotation: offsets = [-1, 0, 1] 随机循环移位！
    - ${temporal_jitter}    # 3. 左右腕时序对齐抖动
    - ${channel_mask}       # 4. MaskChannels: keep_electrodes = [0, 2, 4, 6, 8, 10, 12, 14]
    - ${logspec}            # 5. Log-Spectrogram 计算
    - ${specaug}            # 6. 频谱掩码增强
```

### 2. 数学机理与泄漏证明
F3 的核心研究目的，是评估在**仅有 8 个偶数物理电极**（$E_0, E_2, E_4, E_6, E_8, E_{10}, E_{12}, E_{14}$）的硬件约束下，双腕空间分布的解码表现。其余 8 个奇数电极（$E_1, E_3, E_5, E_7, E_9, E_{11}, E_{13}, E_{15}$）属于**必须严格排除的物理通道**。

然而，当 `RandomBandRotation` 在 `MaskChannels` 之前执行时：
- 当旋转偏移量 $\text{offset} = +1$ 时（即 `tensor.roll(1, dims=-1)`）：
  $$\text{物理通道 } E_{15} \to \text{索引 } 0, \quad E_1 \to \text{索引 } 2, \quad E_3 \to \text{索引 } 4, \quad \dots, \quad E_{13} \to \text{索引 } 14$$
  随后 `MaskChannels` 保留偶数索引 $[0, 2, 4, \dots, 14]$。**此时保留下来的全部是有且仅有的奇数（本应排除的）物理电极！**
- 当旋转偏移量 $\text{offset} = -1$ 时：
  $$\text{物理通道 } E_1 \to \text{索引 } 0, \quad E_3 \to \text{索引 } 2, \quad \dots, \quad E_{15} \to \text{索引 } 14$$
  保留下来的同样是 $100\%$ 的奇数（本应排除的）物理电极！
- 仅当 $\text{offset} = 0$ 时，保留的才是原本的偶数物理电极。

### 3. 合成通道标识数据验证（Synthetic Identity Verification）
通过构建合成张量 $X_{t, b, c} = c$（将通道索引作为物理信号标识），执行原始数据流水线验证：
```python
Original channel values: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
When offset = +1:
  Active positions: [0, 2, 4, 6, 8, 10, 12, 14]
  Physical channels present: [15, 1, 3, 5, 7, 9, 11, 13]
  Contaminated channels: 8 out of 8 active slots (100% 奇数被排除通道泄入模型！)
```

### 4. 危害性评估
1. **训练与验证严重分布失配**：在验证集流水线中，未启用 `RandomBandRotation`，模型在验证时仅接触纯偶数电极；然而在训练过程中，有 $\frac{2}{3}$ 的批次模型实际接触并利用了奇数物理电极的肌电特征；
2. **消融有效性失效**：模型在训练阶段隐式利用了全部 16 个物理电极的信息，不能代表“硬件仅有 8 个电极”的真实物理约束；
3. **旧版结果判定**：原 F3 实验结果（Val CER 20.67%）被判定为 **`[INVALID FOR FIXED-ELECTRODE-BUDGET COMPARISON]`**。旧版 Checkpoint 与日志将严格作为审计历史保留，不得作为论文核心证据。

---

## 二、P1 确定性修复规范与不可回归测试体系

### 1. 修复原则与配置变更 (Config Diff)

为彻底阻断任何电极污染并消除额外混杂变量，实施两项关键修复：
1. **通道掩码绝对前置**：`channel_mask` 必须紧随 `to_tensor` 执行，在任何可能重新排列或混合通道的变换之前，将已排除物理电极直接置零；
2. **受控实验关闭旋转增强**：F1、F2、F3 统一关闭 `RandomBandRotation`（设置 `offsets: [0]`），避免不同稀疏几何构型与环形旋转产生交互干扰；
3. **全流程确定性一致**：训练集、验证集、测试集严格使用完全相同的物理通道掩码。

#### 修复前后的 YAML 差异
```diff
--- config/transforms/log_spectrogram_legacy.yaml
+++ config/transforms/log_spectrogram.yaml
@@ -32,10 +32,10 @@
 transforms:
   train:
     - ${to_tensor}
+    - ${channel_mask}       # 修复：掩码绝对前置，最先清零排除通道
     - ${band_rotation}
     - ${temporal_jitter}
-    - ${channel_mask}       # 原缺陷：旋转后掩码造成奇数电极泄露
     - ${logspec}
     - ${specaug}
 
   val:
     - ${to_tensor}
     - ${channel_mask}
     - ${logspec}
```

并在 `run_experiments.py` 中为固定容量对照组补充确定性覆盖项：
```python
"band_rotation.transform.offsets=[0]"  # 显式关闭环形旋转增强，确保物理通道绝对锁定
```

### 2. 频谱与特征前端的被排除通道行为审计
1. **`LogSpectrogram` 行为**：
   输入全 0 信号时，STFT 频域全为 0，经功率计算 $|0|^2=0$ 与对数转换 $\log(\max(0, \epsilon))$ 后，生成全常数矩阵 $\log(\epsilon)$；
2. **`SpectrogramNorm` 独立性**：
   官方实现为 `nn.BatchNorm2d(channels=32)`，统计量沿 $(N, \text{freq}, T)$ 维度独立计算。我们通过单测验证：改变被排除通道的输入信号，活跃通道的归一化输出最大绝对误差为 `0.00e+00`，证明通道间统计量绝对解耦；
3. **`MultiBandRotationInvariantMLP`**：
   在 `band_rotation` 关闭（offsets=[0]）且输入被掩码置零的前提下，排除通道输入为常数底噪，仅提供固定的偏置截距项，不再传递任何动态肌电调制信号。

### 3. 新增不可回归自动化测试套件 (`emg2qwerty/tests/channel_invariance_test.py`)
我们在代码库中建立了包含 6 个测试用例的严密自动化回归测试体系：
- `test_synthetic_channel_identity_leakage_demonstration`: 自动重现并断言旧版先 roll 后 mask 的 100% 通道泄漏缺陷；
- `test_clean_pipeline_channel_identity`: 验证修复后只有选定通道保留原始信号，排除通道严格为 0；
- `test_spectrogram_norm_channel_independence`: 严格证明 BatchNorm2d 在通道间的零串扰；
- `test_excluded_channel_invariance_end_to_end`: **端到端排除通道不变性测试**。向被排除通道注入幅度高达 $\pm 1000.0$ 的极端随机噪声，断言模型前向输出 Logits 的最大绝对误差必须为 `0.0`：
  $$\max_{t, c} \big| \text{Logits}_{\text{clean}}(t, c) - \text{Logits}_{\text{corrupted}}(t, c) \big| \equiv 0.000000$$

#### 本地与 GPU 集群单测执行证据
```bash
# 本地 Python 3.12 测试通过
pytest emg2qwerty/tests/channel_invariance_test.py
====== 6 passed in 3.36s ======

# Debian Tesla V100 GPU 测试通过
pytest emg2qwerty/tests/channel_invariance_test.py emg2qwerty/tests/charset_test.py emg2qwerty/tests/data_test.py
====== 38 passed in 2.81s ======

# 专用 GPU Smoke Test (scripts/gpu_smoke_test.py)
Running GPU Smoke Test on: cuda (Tesla V100-SXM2-16GB)
GPU Logits Diff on corrupted excluded channels: 0.0
✓ GPU Excluded-Channel Invariance Test PASSED!
✓ GPU Forward + Backward + Optimizer Step PASSED!
```

---

## 三、P2 干净协议重新训练执行结果 (Seeds 42, 123, 2026)

遵循“旧版结果保留归档、新版结果严格重训”的学术规范，在 Debian Tesla V100 GPU 上完成了统一干净协议（关闭旋转增强、掩码绝对前置）下的全量独立重训（Task-1798）：

### 表 1：纯净重训（Clean）与历史缺陷版（Legacy）验证集结果对照表 (Bessel 校正 $ddof=1$)

| 模型配置 | 实验版本 | Seed 42 | Seed 123 | Seed 2026 | Mean ± Sample Std ($ddof=1$) | 总体标准差 ($ddof=0$) | 对侧键区 CER |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **A1 双腕全通道 (32ch, 5.4M)** | **Clean (纯净无旋转基准)** | **18.08%** | **18.25%** | **17.55%** | **17.96 ± 0.37%** | 17.96 ± 0.30% | 23.87% (双侧均衡) |
| **F3 双腕 8+8 (16ch, 5.4M)** | **Clean (纯净重训)** | **19.49%** | **20.78%** | **20.12%** | **20.13 ± 0.64%** | 20.13 ± 0.52% | 26.40% (双侧均衡) |
| | Legacy (缺陷历史) | 20.67% | 22.18% | 21.40% | 21.41 ± 0.75% | 21.41 ± 0.62% | 28.10% (包含奇数泄漏) |
| **F1 单左腕 (16L, 5.4M)** | **Clean (纯净重训)** | **57.07%** | **50.64%** | **50.97%** | **52.89 ± 3.62%** | 52.89 ± 2.95% | **89.29% (右手盲区)** |
| | Legacy (缺陷历史) | 61.32% | 59.08% | 58.22% | 59.54 ± 1.60% | 59.54 ± 1.31% | 97.83% (右手盲区) |
| **F2 单右腕 (16R, 5.4M)** | **Clean (纯净重训)** | **49.82%** | **45.26%** | **46.15%** | **47.08 ± 2.42%** | 47.08 ± 1.98% | **78.88% (左手盲区)** |
| | Legacy (缺陷历史) | 48.18% | 46.99% | 46.72% | 47.30 ± 0.78% | 47.30 ± 0.64% | 76.92% (左手盲区) |

### 核心科研结论确立
1. **彻底排除了电极污染后，双腕优于单腕的核心结论更加坚不可摧**：
   - 纯净协议下，F3 Clean（双腕 8+8）验证 CER 达到 **$20.13 \pm 0.64\%$**；
   - 相比 F1 Clean（单左腕 $52.89\%$），双腕空间覆盖降低了 **$32.76$ 个百分点**；相比 F2 Clean（单右腕 $47.08\%$），降低了 **$26.95$ 个百分点**；
   - 即使彻底阻断任何通道旋转泄漏，单侧腕带对对侧击键依然存在高达 **$78.9\% \sim 89.3\%$** 的条件错误率（近千次字符被直接遗漏/删除）。
2. **严格协议对齐下的无混杂电极减半代价（Unconfounded Decimation Penalty）**：
   - 在完全统一的无旋转协议下，双腕 32 通道基准（A1 Clean）均值达到 **$17.96 \pm 0.37\%$**；
   - 将电极减半至双腕 16 通道（F3 Clean，8+8），CER 仅由 $17.96\%$ 微增至 $20.13\%$，**真实代价仅为 $+2.17$ 个百分点**（CER 相对变化仅 1.12 倍）；
   - 这决定性地证明：双腕对称稀疏布局能以极小代价保留绝大部分信息，而单侧布局则会导致系统性信息崩溃。

---

## 四、P3 统计口径与因果推断边界的最终收紧

在最新版论文与审计报告中，对以下统计与推断表达完成了全面纠偏：
1. **杜绝跨口径泛化差值计算**：
   - **禁止操作**：严禁将单一种子（如 Seed 42）的 Test CER 直接减去三种子的 Validation Mean 并声称为“平均泛化差距”；
   - **规范表述**：泛化差距若基于 Seed 42 单次评估计算，必须明确标注为“单次评测下的点估计差距（Point Estimate Gap）”；若基于多种子，只能报告同种子的验证与测试成对统计。
2. **训练随机种子的统计学界定**：
   - 三种随机种子（42, 123, 2026）仅反映在**单受试者、固定数据划分下神经网络权重随机初始化与数据批次排列的数值稳定性**；
   - **严禁将其解释为跨受试者（Inter-subject）或跨会话（Cross-session）的统计显著性**。论文中所有 Mean ± Std 均明确标记为单受试者随机初始化的稳定性描述。
3. **生理机制与工程下限断言剔除**：
   - 剔除所有“证明了对侧不存在神经传导”的表述，更正为“反映了单腕传感器在信息获取上的物理局限性（Observed Information Limitation），导致 CTC 对齐塌陷为删除”；
   - 剔除“证明了商业化工业级硬件绝对下限”与未经直接传感器采集的“$\pm 5^\circ$ 佩戴滑动”、“汗液阻抗失真”断言，严格界定为“在该受试者、环形布局及评估协议下的实证下限（Empirical Lower Bound）”。
