# 12 导联 ECG 多标签心律失常检测（PTB-XL v1.0.3）

从零实现、可复现的心电多标签分类项目：1D 卷积 tokenizer + self-attention（patch Transformer）、
掩码信号建模 + 对比自监督、患者级严格切分、多标签长尾评测、温度标定 + conformal 预测、
注意力/gradient 可解释性，最后做知识蒸馏与动态 int8 量化。

- 数据：PTB-XL v1.0.3 `records100`（100 Hz，10 s = 1000 点，12 导联），21,799 条记录 / 18,869 名患者。
- 切分：官方 `strat_fold`，train=1–8 / val=9 / test=10（17,418 / 2,183 / 2,198 条记录）。
  患者交集已验证为 0（见 `results/m0_data_report.json`）。
- 双头：5 个诊断超类（CD/HYP/MI/NORM/STTC）+ 12 个节律类，均为多标签。
- **所有数字来自 `results/*.json` 的真实运行**，汇总见 [`docs/RESULTS_TABLE.md`](docs/RESULTS_TABLE.md)
  （由 [`src/m6_master_table.py`](src/m6_master_table.py) 生成，不手抄）。

## 怎么启动

项目已经跑完，`results/`、`figures/`、`models/` 里是全部产物。三种用法：

1. **只看结果**：读 [`docs/RESULTS_TABLE.md`](docs/RESULTS_TABLE.md)，图在 `figures/`，实验日志见
   [`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md)。
2. **单独复跑某一阶段**：例如 `python src/m4_conformal.py --tag m1_resnet`，命令见下方
   [分阶段命令](#分阶段命令)。脚本会读 `data/processed/` 的缓存、写 `results/`，不会重训上游模型。
3. **从原始数据全流程复现**：按 M0 → M6 的顺序执行（见下）。
4. **验收**：`python src/m6_acceptance.py`（不训练、不联网，12 项检查），说明与手工检查清单见
   [`docs/ACCEPTANCE.md`](docs/ACCEPTANCE.md)。

环境（本机实测）：

```
Python 3.13.12
torch 2.9.1+cu128   numpy 2.5.4   scipy 1.18.1   scikit-learn 1.9.1
pandas 3.1.0        matplotlib 3.11.2   wfdb 4.3.1
GPU: RTX 4060 Laptop 8 GB    RAM: 15.64 GB    Windows 11
```

```powershell
cd ptb-xl-ecg-arrhythmia  # or the path where you cloned this repository
python -m venv .venv
.\.venv\Scripts\pip install -r requirements.txt
```

数值稳定：所有训练脚本都在 `OPENBLAS_NUM_THREADS=1`、`OMP_NUM_THREADS=1` 下运行；
DataLoader `workers=2`（M1/M2 原始配置为 4）。

## 单条记录推理（仅数字信号）

[`src/predict.py`](src/predict.py) 读取一条 12 导联数字心电图，走与训练完全相同的预处理
（导联重排 → 100 Hz → ±5 mV 截断 → 0.5–40 Hz 零相位带通 → 训练集 z-score），
输出 5 个超类与 12 个节律的概率，以及一句正常/异常判定。

```powershell
# WFDB（.hea/.dat），例如 PTB-XL 的 records100；路径带不带 .hea 都行
python src/predict.py --wfdb data/raw/ptb-xl/1.0.3/records100/00000/00009_lr

# NumPy：(12, N) 或 (N, 12)，单位 mV；500 Hz 会自动重采样到 100 Hz
python src/predict.py --npy my_ecg.npy --fs 500

# CSV：12 个导联列（可有 time/index 列），单位 mV
python src/predict.py --csv my_ecg.csv --fs 100

# CSV 是 µV 时加 --scale 1e-3
python src/predict.py --csv my_ecg_uv.csv --fs 100 --scale 1e-3
```

输入规格：

| 格式 | 形状 / 列 | 单位 | 采样率 | 导联名 |
|---|---|---|---|---|
| WFDB | 头文件声明的 12 路信号 | 按 header 自动换算成 mV（mV / µV / V） | 读 header | 按 `sig_name` 重排（标准名 I,II,III,AVR/AVL/AVF,V1–V6），12 个必须齐全 |
| NPY | `(12, N)` 或 `(N, 12)` | mV（µV 用 `--scale 1e-3`） | `--fs`，默认 100 | 无导联名，按 I,II,III,aVR,aVL,aVF,V1–V6 解释并打印警告 |
| CSV | 12 个导联列，表头可有可无 | mV（µV 用 `--scale 1e-3`） | `--fs`，默认 100 | 有表头时按名字匹配（大小写不敏感，AVR/aVR 都行）；无表头用 `--no-header`（必须正好 12 列），按固定顺序 |

所有格式共用同一条时间轴处理：先重采样到 100 Hz，再居中裁剪或零填充到 1000 点（10 s）。
导联缺失、重复、含 NaN 都会直接报错；某导联完全平坦会给出警告。旧格式的 ADC 计数不能直接输入，
需要先换算成 mV。如果输入时长不在 5–15 s 或幅值大于 20（很可能是 µV/ADC 计数），脚本会打印
warning；**出现 warning 时不要相信 verdict**，先修正 `--fs` / `--scale`。实测把 500 Hz 文件按
100 Hz 读，判定会从 normal 变成 abnormal（MI）；µV 文件不加 `--scale 1e-3` 也会误判。

用 test 记录 9 验证过（`python src/m6_predict_check.py`，结果在
[`results/predict_selfcheck.json`](results/predict_selfcheck.json)）：WFDB（mV）、
WFDB（header 写 µV）、NPY、带 `Unnamed: 0` + `Time (s)` 列的 CSV、12 列无表头 CSV
五种输入输出一致（`--raw`：NORM 0.9594、CD 0.0336）。预处理与训练缓存逐位一致
（max |Δ| = 0）；与保存的 test 预测最大概率差 2.05e-05（CPU batch 1 对比 GPU batch 256）。

默认应用 M4 的温度缩放（super T=1.1113、rhythm T=0.8426）；`--raw` 关闭，用于复现
`results/m1_resnet_pred.npz`。判定规则：NORM ≥ 0.5 且 CD/HYP/MI/STTC 都 < 0.5 → 正常；
任一异常超类 ≥ 0.5 → 异常；其余 → 不确定。阈值可用 `--threshold` 调整，但没有经过临床验证。

**图片不能直接输入。** 模型输入是 `(12, 1000)` 的 mV 数值，不是像素。照片/扫描件要先做
导联分割、网格标定和波形描迹，才能变成这个张量；而且多数 12 导联打印件是 3×4 布局、
每导联只有 2.5 s，模型却要求每导联 10 s，缺失的 7.5 s 无法从图里恢复。本项目没有实现
数字化，也没有成对的“图像-信号”数据来验证它，所以不对图片输入做任何承诺。如果医院能导出
WFDB / XML / SCP-ECG 或数值文件，用上面的命令；只有 PDF/照片时，需要另做一个数字化项目
并单独验证。

注意：这是 PTB-XL 上的研究模型，不是医疗器械；换设备、换导联位置、换滤波设置都会影响结果，
不能用它排除疾病。

## 数据与标签

| 项目 | 取值 |
|---|---|
| 来源 | PhysioNet `physionet-open` S3 镜像的 PTB-XL v1.0.3，仅取 `records100/` + 两个元数据 CSV |
| 样本 | 21,799 条 × 12 导联 × 1000 点，`fs=100`；读取错误 0，长度异常 0 |
| 导联顺序 | 全库唯一：I, II, III, aVR, aVL, aVF, V1–V6 |
| 主标签 | `diagnostic_superclass` → CD / HYP / MI / NORM / STTC |
| 次标签 | `rhythm` → SR / AFIB / AFLT / STACH / SBRAD / SARRH / SVTAC / PSVT / BIGU / TRIGU / SVARR / PACE |
| 缺失标签 | 无对应 statement 的记录**不参与该头的 loss 与指标**（masked），不当负样本 |

训练集超类正例数：CD 3,907、HYP 2,119、MI 4,379、NORM 7,596、STTC 4,186；
节律类中 SR 13,404、AFIB 1,211，而 TRIGU 仅 16、PSVT 19、SVTAC 21（长尾）。

预处理（`src/m1_preprocess.py`）：±5 mV 截断 → 0.5–40 Hz 4 阶 Butterworth 零相位带通
（`sosfiltfilt`）→ **仅在训练集上**拟合的逐导联 z-score。统计量存于 `results/m1_norm_stats.json`。

## 方法

| 阶段 | 内容 | 关键文件 |
|---|---|---|
| M0 | 数据完整性、标签分布、官方切分与患者交集校验 | `src/m0_inspect.py` |
| M1 | 1D ResNet 基线（dual head，3,857,553 参数） | `src/m1_train.py` |
| M2 | patch Transformer：Conv1d tokenizer（patch 25 / stride 12 → 82 token，d=128，4 层 4 头，844,945 参数） | `src/m2_train.py` |
| M3 | 自监督（sample 级 span 掩码 MSM + NT-Xent 对比），三种适配：full / Houlsby adapter / linear probe，标签比例 1/5/10/25/100% | `src/m3_pretrain.py`, `src/m3_finetune.py` |
| M4 | 逐头温度标定（ECE/NLL/Brier），逐类 split conformal + Mondrian 变体 | `src/m4_calibration.py`, `src/m4_conformal.py` |
| M5 | attention rollout、gradient×input、ResNet→Transformer 知识蒸馏、动态 int8 量化 | `src/m5_*.py` |
| M6 | 全量数字汇总、文档 | `src/m6_master_table.py` |

自监督掩码在**原始采样点**上做（8 段 × 50 点，名义 40%，实际 0.335），避免 token 重叠导致的泄漏；
对比学习用同一记录的两种增强视图（幅度、±100 ms 循环移位、高斯噪声、逐导联 dropout），τ=0.1、λ=0.2。

## 主要结果（test，statement-masked）

| 模型 | 参数 | super macro-AUC | super macro-F1 | rhythm macro-AUC | rhythm macro-F1 |
|---|---|---|---|---|---|
| ResNet1d（M1） | 3,857,553 | **0.9239** | **0.7391** | **0.9320** | **0.4066** |
| PatchTransformer（M2） | 844,945 | 0.9133 | 0.7148 | 0.8662 | 0.3594 |
| PatchTransformer-wide（M2） | 2,155,601 | 0.9130 | 0.7091 | 0.8824 | 0.3455 |
| KD 蒸馏学生（M5） | 844,945 | 0.9152 | 0.7216 | 0.9160 | 0.3673 |

完整的分层指标（含 min10/min20 稳定性口径）、M3 曲线、M4 标定/覆盖、M5 蒸馏与量化见
[`docs/RESULTS_TABLE.md`](docs/RESULTS_TABLE.md)。

### 这些数字说明什么、不说明什么

- **蒸馏把同架构 Transformer 的节律 AUC 从 0.8662 提到 0.9160**（+0.0499），参数不变；
  但仍低于 3.86M 参数的 ResNet 教师（0.9320）。
- **SSL 的收益集中在节律头，且标签越少收益越大**：1% 时节律 AUC 提升 seed42 +0.0873 /
  seed43 +0.1081；超类头在 1% 的正增益（+0.0347）**没有复现**（seed43 −0.0039），已撤回该结论。
- **全量微调下 MSM-only 略优于 MSM+对比**（100%：joint−msm 超类 −0.0019/−0.0037，
  节律 −0.0169/−0.0081，两个 seed 同号）。对比项在冻结特征（linear probe）时才有增益。
- **温度标定几乎不改变结果**：ECE pooled 超类 0.0164→0.0140、节律 0.0074→0.0043，
  Brier/NLL 变化在 1e-3 量级，AUC 不变（最大 |ΔAUC| 1.1e-16，浮点噪声）。
- **节律头的边际 conformal 覆盖 0.8913 是"假达标"**：靠 1.1735 个空集/记录换来，
  正类召回仅 0.1808。Mondrian 把正类召回提到 0.9280，但 5 个类校准集正例 < 9（有限样本退化），
  集合大小涨到 7.2707/12，5.6511 个 both 集合/记录——**不能直接当决策规则**。超类头可用
  （Mondrian 正类召回 0.8938，1.9458 个正预测/记录）。
- **动态 int8 只影响 `nn.Linear`**（10 个模块），模型体积 3.40→1.83 MB，但 1 线程 CPU 上
  batch 1 延迟反而从 13.07 ms 升到 18.21 ms（x0.72），batch 32 仅 x1.06。
  对该规模模型，动态量化换来的是体积，不是延迟。
- **attention rollout 的定位能力有限**：5 个样本 top-10% 质量均值 0.1952（均匀基线 0.10），
  且是 class-agnostic、不带导联轴；同批样本的 gradient×input 为 0.5423。这是示例性展示，
  不是 faithfulness 基准。

## 分阶段命令

所有命令在项目根目录执行；已存在的 `results/<tag>.json` 会被脚本复用或跳过（各脚本行为见 docstring）。

```powershell
# 0) 数据（可选：数据已在 data/raw；脚本内为绝对路径，换机器需改 OUT）
python src/download_ptbxl.py --workers 32 --verify-md5

# 1) M0：建 raw 缓存 + 完整性/标签/切分报告（308.1 s）
python src/m0_inspect.py

# 2) 预处理：滤波 + 训练集 z-score
python src/m1_preprocess.py

# 3) M1 基线 + masked 指标
python src/m1_train.py --tag m1_resnet
python src/eval_masked.py m1_resnet
python src/m1_post.py m1_resnet

# 4) M2 Transformer（窄 + 宽）
python src/m2_train.py --tag m2_transformer
python src/eval_masked.py m2_transformer
python src/m2_train.py --tag m2_transformer_wide --d-model 192 --depth 6 --nhead 6
#    宽模型首次在最终评估阶段因系统内存耗尽中断（训练已完成、checkpoint 已保存）：
python src/m2_eval_ckpt.py --tag m2_transformer_wide --d-model 192 --depth 6 --nhead 6
python src/eval_masked.py m2_transformer_wide

# 5) M3 自监督 + 下游
python src/m3_pretrain.py --mode msm
python src/m3_pretrain.py --mode joint
python src/m3_run_all.py          # 21 个 full/adapter/linear 运行 + masked 指标 + 曲线
python src/m3_run_extra.py        # MSM-only full、scratch adapter
python src/m3_seed_check.py       # seed 43 复核 1%/10%/100%

# 6) M4 标定 + conformal
python src/m4_calibration.py --tag m1_resnet
python src/m4_conformal.py --tag m1_resnet --alpha 0.1

# 7) M5 可解释性、蒸馏、量化
python src/m5_saliency.py --tag m1_resnet
python src/m5_attention.py --tag m2_transformer
python src/m5_distill.py --teacher m1_resnet --tag m5_kd_student
python src/eval_masked.py m5_kd_student
python src/m5_quantize.py --tag m5_kd_student

# 8) M6 汇总 + 数字审计
python src/m6_master_table.py
python src/m6_check_numbers.py   # 文档里每个小数都必须能在 results/ 中找到

# 9) 验收（不训练、不联网；退出码 0 = 12 项全过）
python src/m6_acceptance.py

# 10) 推理输入自检（五种输入格式输出一致性）
python src/m6_predict_check.py
```

单次训练耗时（本机）：M1 162.8 s，M2 218.1 s，MSM 332.5 s，联合 SSL 962.4 s，蒸馏 342.2 s。

## 目录

```
configs/m0.yaml          冻结的数据/标签/切分/评测决策
splits/main_v1.json      主切分（ecg_id 级别）
splits/m3_frac_*.json    M3 标签比例子集（按患者抽取，seed 42/43）
data/raw/                下载的 PTB-XL（与脚本分离）
data/processed/*.npy     原始缓存与滤波后缓存
src/                     全部脚本（见上表）
models/                  各 run 的 checkpoint（.pt）
results/                 每次运行的 config + metrics JSON、预测 npz、日志
figures/                 曲线、混淆/覆盖图、attention/saliency 叠加图
docs/                    RESULTS_TABLE.md（生成）、EXPERIMENTS.md、INTERVIEW_QA.md、RESUME.md
```

## 评测口径

- 主指标 macro-AUC，次指标 macro-F1@0.5；逐类 AUC/F1、正例率全部保存。
- **statement-masked**：某类没有对应 statement 的记录不计入该类（既不算正也不算负）。
  因此每个头的评测样本数不同：超类 2,158 条，节律 2,098 条。
- 另报 `stable_min10` / `stable_min20`：只保留正例数 ≥10 / ≥20 的类，缓解长尾类 AUC 噪声。
- 早停与模型选择只看 val（`mean(super AUC, rhythm AUC)`），test 只在最后评一次。
- 所有结果单 seed 42；只有 M3 的 1%/10%/100% 做了 seed 43 复核，其差值见
  `results/m3_seed_check.json`。

## 已知局限

1. **单 seed**：除 M3 三个点外没有多 seed 方差，跨 run 的 0.001–0.005 差异不能当作显著。
2. **子集非嵌套**：M3 不同标签比例是独立的患者抽样，曲线不保证单调（5%→10% 的回落是抽样，不是训练效应）。
3. **conformal 交换性被破坏**：模型在 val 上早停，又在同一 val 上估计分位数；PACE 校准集正例平均置信度
   0.9233、test 0.7578（−0.1655），该类覆盖 0.8937 低于名义值。
4. **无外部验证**：只在一个中心的数据上验证，跨中心/跨设备泛化未测。
5. **100 Hz**：未使用 500 Hz 版本，高频形态信息被低通截断。
6. **attribution 只做了 5 个样本**的定性叠加，没有删除/插入式的 faithfulness 实验。
7. **量化只作用于 Linear**，Conv1d tokenizer 仍是 fp32；延迟结论仅限 1 线程 CPU。
8. **推理只接受数字信号**：`predict.py` 不做图片数字化；单条记录的 0.5 阈值未做临床校准。

## 文档

- [`docs/RESULTS_TABLE.md`](docs/RESULTS_TABLE.md)：所有实测数字（自动生成）
- [`docs/ACCEPTANCE.md`](docs/ACCEPTANCE.md)：启动步骤、自动验收 12 项、手工检查清单、常见失败
- [`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md)：实验日志，含失败与返工
- [`docs/INTERVIEW_QA.md`](docs/INTERVIEW_QA.md)：15 个面试问答题
- [`docs/RESUME.md`](docs/RESUME.md)：200 字 / 300 字简历文本
