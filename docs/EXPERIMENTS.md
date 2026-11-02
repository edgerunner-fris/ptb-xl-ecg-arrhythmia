# 实验日志

记录每个阶段做了什么、得到什么、怎么解释，以及失败和返工。所有数值均来自 `results/*.json`；
完整汇总见 [`RESULTS_TABLE.md`](RESULTS_TABLE.md)（由 `src/m6_master_table.py` 生成）。

协议固定如下，后续不再更改：

- 数据 PTB-XL v1.0.3 `records100`，100 Hz、10 s、12 导联。
- 切分官方 `strat_fold`：train 1–8、val 9、test 10；患者交集 0（`results/m0_data_report.json`）。
- 双头多标签：5 个诊断超类 + 12 个节律类；缺失 statement 的记录在对应头 masked。
- 主指标 test macro-AUC（statement-masked），次指标 macro-F1@0.5；早停/选模只看 val。
- 单 seed 42；仅 M3 的 1%/10%/100% 追加 seed 43 复核。

---

## M0 数据校验

`python src/m0_inspect.py`，308.1 s。

- 21,799 条记录、18,869 名患者；`.hea`/`.dat` 缺失 0，读取错误 0。
- 全部记录 12 导联、`fs=100`、长度 1000；导联顺序全库唯一（I, II, III, aVR, aVL, aVF, V1–V6）。
- 切分：train 17,418 条 / 15,023 人，val 2,183 / 1,942，test 2,198 / 1,904；
  `train∩val = train∩test = val∩test = 0`，跨 fold 患者 0。
- 训练集正例：CD 3,907、HYP 2,119、MI 4,379、NORM 7,596、STTC 4,186；
  节律 SR 13,404、AFIB 1,211、……、TRIGU 16、PSVT 19、SVTAC 21。

结论：数据可直接用官方切分，不需要重新分层；长尾类在 test 上只有个位数正例，macro-AUC 必然噪声大，
后续同时报 `stable_min10` / `stable_min20`。

## M1 1D ResNet 基线

`python src/m1_train.py --tag m1_resnet`，25 epoch 预算，早停 patience 8，best epoch 9，162.8 s。

- 结构：width 64，blocks [2,2,2,2]，3,857,553 参数；两个头共享 backbone。
- val：super 0.9260 / rhythm 0.9562；test（masked）：super **0.9239** / rhythm **0.9320**，
  macro-F1 0.7391 / 0.4066。
- 稳定性口径：super min10/min20 = 0.9239；rhythm min20 = 0.9178（12 类全在 min10 口径内时 0.9136）。

节律头 F1 0.4066 低，是 0.5 阈值与长尾分布共同造成的；AUC 为主指标，不用 F1 选模。

## M2 patch Transformer

`python src/m2_train.py --tag m2_transformer`，best epoch 14，218.1 s。

- Conv1d tokenizer：patch 25 / stride 12 → 82 token（含 CLS），d=128、4 层、4 头，844,945 参数。
- test（masked）：super 0.9133 / rhythm 0.8662，F1 0.7148 / 0.3594。

加宽版（d=192、6 层、6 头，2,155,601 参数）best epoch 10：super 0.9130 / rhythm 0.8824。

读法：

- 超类头 ResNet 领先 0.0106（0.9239 vs 0.9133），加宽也没有补上（0.9130）——在当前数据量下，
  卷积的局部/平移归纳偏置对超类更有效。
- 节律头加宽有 +0.0162（0.8662 → 0.8824），说明差距主要来自容量而不是 tokenizer。
- **失败**：宽模型训练完成并保存了 best checkpoint，但最终评估阶段 OpenBLAS 报
  “Memory allocation still failed after 10 retries”（系统内存耗尽），进程退出。
  用 `src/m2_eval_ckpt.py` 以 `num_workers=0`、batch 128 重新评估，结果可信，但
  `elapsed_sec` 记为 null（原始墙钟不可恢复）。

## M3 自监督与下游适配

### 预训练

- 掩码：8 段 × 50 个原始采样点，名义 40%；实际掩码率 0.3349（MSM）/ 0.3339（joint）。
- 早期版本在 token 级掩码，25/12 的重叠导致被掩码 token 的信息从相邻 token 泄漏；改为
  **采样点级 span 掩码 + 仅对 masked 采样点算 MSE** 后重跑，之前的结果全部作废。
- MSM-only：30 epoch，332.5 s，最终 MSE 0.7749。
- 联合（MSM + NT-Xent，τ=0.1，λ=0.2）：30 epoch，962.4 s，MSE 0.8109、CL loss 0.0511。
  联合模型 MSE 更高是预期内的（容量要同时满足两个目标），不能据此说 MSM 变差。

### 标签比例曲线（test，masked）

| 比例 | 训练记录 | joint super | joint rhythm | scratch super | scratch rhythm |
|---|---|---|---|---|---|
| 1% | 165 | 0.7943 | 0.6745 | 0.7596 | 0.5872 |
| 5% | 867 | 0.8587 | 0.7807 | 0.8536 | 0.7514 |
| 10% | 1,748 | 0.8764 | 0.7640 | 0.8712 | 0.7292 |
| 25% | 4,351 | 0.8804 | 0.8086 | 0.8761 | 0.7870 |
| 100% | 17,418 | 0.9130 | 0.8836 | 0.9133 | 0.8662 |

MSM-only 全量微调：10% 0.8709 / 0.7732，100% 0.9149 / 0.9005。

配对差（同一患者子集）seed 42：

| 比例 | joint−scratch super | joint−scratch rhythm |
|---|---|---|
| 1% | +0.0347 | +0.0873 |
| 5% | +0.0050 | +0.0293 |
| 10% | +0.0052 | +0.0347 |
| 25% | +0.0043 | +0.0216 |
| 100% | −0.0002 | +0.0174 |

seed 43 复核（`results/m3_seed_check.json`）：

| 比例 | joint−scratch super | joint−scratch rhythm | joint−MSM super | joint−MSM rhythm |
|---|---|---|---|---|
| 1% | −0.0039 | +0.1081 | — | — |
| 10% | +0.0045 | +0.0146 | −0.0163 | +0.0002 |
| 100% | +0.0044 | +0.0369 | −0.0037 | −0.0081 |

（seed 42 的 joint−MSM：10% +0.0054 / −0.0092，100% −0.0019 / −0.0169。）

### 适配模式（test，masked）

| run | super | rhythm |
|---|---|---|
| joint linear 10% | 0.8619 | 0.7396 |
| joint linear 100% | 0.8780 | 0.8663 |
| MSM linear 10% | 0.8308 | 0.7039 |
| MSM linear 100% | 0.8657 | 0.8385 |
| scratch linear 10% | 0.8226 | 0.7516 |
| joint adapter 10% | 0.8499 | 0.7871 |
| MSM adapter 10% | 0.8464 | 0.7647 |
| scratch adapter 10% | 0.8356 | 0.7393 |
| joint adapter 100% | 0.9014 | 0.8743 |

### 结论（哪些站得住）

1. **节律头上的 SSL 增益在两个 seed 上都为正**：1% +0.0873/+0.1081，10% +0.0347/+0.0146，
   100% +0.0174/+0.0369。方向稳定，幅度在 10%/100% 明显缩水。
2. **超类头 1% 的 +0.0347 没有复现**（seed 43 为 −0.0039），该结论已撤回；
   超类头在全量数据上基本无增益（−0.0002 / +0.0044）。
3. **全量微调时 MSM-only 略好于 joint**：100% 下 joint−MSM 超类 −0.0019/−0.0037、
   节律 −0.0169/−0.0081，两个 seed 同号。对比项不是"越多越好"，它在冻结特征时才有用
   （linear probe 100%：joint 0.8780/0.8663 vs MSM 0.8657/0.8385）。
4. adapter 在 10% 时比 scratch adapter 好（0.8499/0.7871 vs 0.8356/0.7393），说明冻结编码器
   上的预训练特征确实可迁移；100% 全量微调仍最高（0.9130/0.8836）。

### 注意

- 不同比例的标签子集是**按患者独立抽样**，不是嵌套的；5%→10% 节律头的回落（0.7807→0.7640）
  是抽样差异，不是训练效应。
- scratch linear 100%、MSM adapter 100%、scratch adapter 100% 没有跑，不能从表中推断。

## M4 温度标定

`python src/m4_calibration.py --tag m1_resnet`，val 拟合、test 评估（masked）。

| head | T | ECE pooled 前 | 后 | NLL 前 | 后 | Brier 前 | 后 |
|---|---|---|---|---|---|---|---|
| super | 1.1113 | 0.0164 | 0.0140 | 0.2771 | 0.2757 | 0.0837 | 0.0836 |
| rhythm | 0.8426 | 0.0074 | 0.0043 | 0.0603 | 0.0598 | 0.0150 | 0.0147 |

- AUC 不变（最大 |ΔAUC| 1.1e-16，浮点噪声），符合温度缩放是单调变换的性质。
- 改善幅度很小，因为模型本身已经很校准（pooled ECE 0.0164 / 0.0074）。
- 节律头 T = 0.8426 < 1，说明模型**欠自信**（概率被压得过于接近 0/1）。
- 逐类并非都改善：HYP ECE 0.0416→0.0495 变差；SARRH 0.0402→0.0166、SR 0.0747→0.0546 明显改善。
  单一温度不可能对所有类都最优，这是全局标定的固有局限。

## M4 conformal 预测

`python src/m4_conformal.py --tag m1_resnet --alpha 0.1`，逐类 split conformal，校准集 = val，
评估集 = test，名义覆盖 0.9。

边际版本（α=0.1）：

| head | 平均类覆盖 | 最低类覆盖 | 联合覆盖 | 正类召回（类均值） | 正集合数 | 空集/记录 |
|---|---|---|---|---|---|---|
| super | 0.9032 | 0.8957 | 0.6645 | 0.7430 | 1.3244 | 0.0000 |
| rhythm | 0.8913 | 0.8823 | 0.5620 | 0.1808 | 0.8279 | 1.1735 |

- 超类头可用：覆盖达标，空集 0，正类召回 0.7430（CD 0.6351、HYP 0.6107、MI 0.7582、
  NORM 0.9107、STTC 0.8004）。
- 节律头的 0.8913 是**靠空集换来的**：平均每条记录 1.1735 个空集，41.3% 的记录至少有一个空集；
  正类召回只有 0.1808，AFLT/BIGU/PSVT/SARRH/SVARR/SVTAC/TRIGU 的正类召回为 0。
  也就是说"覆盖 0.9"主要来自把稀有类的集合做成空集（自动判负）。
- α 曲线（超类）：α=0.01/0.05/0.10/0.50 时覆盖 0.9883/0.9539/0.9032/0.4991，
  联合覆盖 0.9504/0.8193/0.6645/0.1515。边际覆盖随 α 单调，联合覆盖始终低得多——
  多标签下"每类 90%"不等于"整条记录 90%"。

Mondrian（正负分开取分位数）：

| head | 合并覆盖 | 正类召回均值 | 最低类（正类） | 正集合数 | both 集合/记录 | 退化类 |
|---|---|---|---|---|---|---|
| super | 0.9055 | 0.8938 | 0.8779 | 1.9458 | 0.5737 | 无 |
| rhythm | 0.8915 | 0.9280 | 0.6786 | 7.2707 | 5.6511 | 5 个 |

- 超类头：正类召回从 0.7430 提到 0.8938，代价是正集合数 1.3244→1.9458、both 0.5737/记录。
  仍可作为决策参考。
- 节律头：正类召回 0.9280 是**由退化和膨胀共同造成的**。AFLT/BIGU/PSVT/SVTAC/TRIGU
  校准集正例分别为 7/8/3/3/2，小于 α=0.1 下非退化所需的最小值 9，正类分位数取到 1.0，
  正负阈值倒置，产生 5.6511 个 both 集合/记录、正集合数 7.2707/12——不可用。
- 即使在非退化类中，PACE 的正类召回仍只有 0.6786：校准集正例平均置信度 0.9233，
  test 0.7578（−0.1655），分布偏移直接把覆盖拉低。该类边际覆盖 0.8937，也低于名义 0.9。

**交换性假设被破坏**：模型在 val 上早停，又在同一 val 上估计 conformal 分位数，
val 分数不是 test 分数的无偏样本。上面的覆盖是经验值，不是有限样本保证。

## M5 可解释性

### gradient×input（ResNet）

`python src/m5_saliency.py --tag m1_resnet`，取置信度最高的 3 个正确 + 2 个错误样本。

- top-10% 归因质量均值 0.5712（均匀基线 0.10），即注意力确实集中。
- 逐导联 top-10% 质量都在 0.39–0.78 之间，没有出现"只盯某一个导联"的模式；
  正确样本里 V4–V6 略高，但样本只有 5 个，不能当结论。

### attention rollout（Transformer）

`python src/m5_attention.py --tag m2_transformer`，同样 3+2 个样本。

- top-10% 质量均值 0.1952，仅略高于均匀的 0.10；同批样本 gradient×input 为 0.5423。
- rollout 是 **class-agnostic** 的：解释的是 CLS 表示，不是某个类的 logit；且注意力没有导联轴，
  热图只能落在时间轴上，不能回答"哪个导联"。
- 结论：rollout 在这个模型上只能给出弱定位，不能作为诊断解释；要类特定解释应回到
  gradient×input 或做类条件注意力。

图：`figures/m5_saliency_{correct,wrong}.png`、`figures/m5_attention_{correct,wrong}.png`。

## M5 知识蒸馏

`python src/m5_distill.py --teacher m1_resnet --tag m5_kd_student`，best epoch 17，342.2 s。

- 教师 ResNet 3,857,553 参数；学生 patch Transformer 844,945 参数（同 M2 结构）。
- 损失：硬标签 masked BCE + 0.5 × 教师软标签 masked BCE。
- test（masked）：super 0.9152 / rhythm **0.9160**，F1 0.7216 / 0.3673。

对比同架构从头训练（0.9133 / 0.8662）：超类 +0.0019，节律 **+0.0499**。仍低于教师
（0.9239 / 0.9320），但用 21.9% 的参数量把节律差距从 0.0658 缩到 0.0160。
这比 M3 的自监督更有效——教师提供的软标签直接包含类间关系，而 SSL 只有信号重构/对比目标。

## M5 动态 int8 量化

`python src/m5_quantize.py --tag m5_kd_student`，CPU、单线程、warmup 10 次、50 次取中位数。

- 10 个 `nn.Linear` 被量化；Conv1d tokenizer 与注意力计算仍是 fp32。
- state_dict：3.40 MB → 1.83 MB（1.86×）。
- 延迟：batch 1 13.07 → 18.21 ms（**变慢**，x0.72）；batch 32 349.02 → 328.57 ms（x1.06）。
- 精度：super 0.9152 → 0.9153，rhythm 0.9161 → 0.9157；F1 0.7216→0.7201、0.3673→0.3712。

读法：动态量化在这里换来的是体积，不是速度。batch 1 时反量化/量化开销占主导；
只有 batch 32 才有微弱收益。若目标是延迟，应做静态量化（需要校准数据）或直接换更小的模型。

**失败与修复**：首次运行在 `TransformerEncoderLayer.forward` 报
`AttributeError: 'function' object has no attribute 'device'`。原因是动态量化把 `nn.Linear`
换成 `torch.ao.nn.quantized.dynamic.Linear`，其 `.weight` 是绑定方法而不是张量，
而 fast path 的守卫会读 `self.linear1.weight.device`。修法是给 fp32 和 int8 **都**关闭
MHA fast path（`torch.backends.mha.set_fastpath_enabled(False)`），保证两者走同一代码路径。
代价：这两组延迟与"默认开启 fast path"的运行不可直接比较，docstring 已注明。

## 返工与撤回清单

| 问题 | 处理 |
|---|---|
| token 级 MSM 因 patch 重叠泄漏 | 改为采样点级 span 掩码，只对 masked 采样点算 MSE，重跑全部 SSL |
| adapter 与编码器设备不一致 | adapter `.to(next(model.parameters()).device)` |
| linear probe 保存的是最后状态而非最优状态 | 克隆最优权重后再保存 |
| attention rollout 与 fast path 结果不一致 | rollout 时关闭 fast path |
| 宽 Transformer 最终评估 OOM | `m2_eval_ckpt.py` 低内存重评，elapsed 记 null |
| 量化模型 fast path 崩溃 | 两侧关闭 fast path，见上 |
| M4 图例被裁掉 | 先 `tight_layout(rect=...)` 再放图例 |
| 曲线把只有 2 个点的 MSM-only 画成连续线 | 少于 3 个点时只画 marker |
| 超类 1% 的 SSL 增益（+0.0347） | seed 43 未复现（−0.0039），已撤回 |
| "对比学习在全量微调下反而更差"的幅度 | seed 43 确认方向但幅度缩小，措辞已降级 |
| `predict.py` 导联重排按列选取而不是按行 | 改为按行重排；用 test 记录的缓存输入核对，预处理逐位一致（max \|Δ\|=0） |

## 没做的事

- 图片/扫描件输入的数字化（导联分割、网格标定、波形描迹）。`predict.py` 只接受 WFDB/npy/CSV 数字信号。
- 多 seed 全量方差；目前只有 M3 三个点做了 seed 43。
- 外部数据集验证、500 Hz 版本。
- 删除/插入式 faithfulness 实验；attribution 只有 5 个样本的定性图。
- 静态量化与 GPU 推理基准。
