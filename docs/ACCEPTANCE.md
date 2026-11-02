# 启动与验收

面向第一次拿到这个目录的人。两步：**跑起来看结果**，**用检查项确认结果没被动过**。

## 1. 启动

### 只看（不需要 GPU，不需要数据）

```powershell
cd ptb-xl-ecg-arrhythmia  # or the path where you cloned this repository
# 结果总表（自动生成，含所有实测数字）
notepad docs\RESULTS_TABLE.md
# 图
explorer figures
```

### 复跑已有结果的某一阶段

脚本只读 `data/processed/`、写 `results/`，不会重训上游模型。

```powershell
$env:OPENBLAS_NUM_THREADS='1'; $env:OMP_NUM_THREADS='1'
.\.venv\Scripts\python.exe src\m4_conformal.py --tag m1_resnet --alpha 0.1
.\.venv\Scripts\python.exe src\eval_masked.py m1_resnet
.\.venv\Scripts\python.exe src\m6_master_table.py
```

### 从原始数据全流程

先确认 `data/raw/ptb-xl/1.0.3/` 已有 21,799 对 `.hea/.dat`；没有则运行
`python src/download_ptbxl.py --workers 32 --verify-md5`（脚本内是绝对路径，换机器要改）。
之后按 README 的"分阶段命令"M0 → M6 顺序执行。单机全流程约 2–3 小时（不含下载）。

## 2. 自动验收（一条命令）

```powershell
$env:OPENBLAS_NUM_THREADS='1'; $env:OMP_NUM_THREADS='1'
.\.venv\Scripts\python.exe src\m6_acceptance.py
```

约 30 秒，不训练、不联网。退出码 0 表示 12 项全部通过。每项在验证什么：

| # | 检查 | 通过意味着 |
|---|---|---|
| 1 | 环境与缓存 | 版本与结果产出一致；`ptbxl_100hz_filt.npy`、`meta.parquet`、切分文件存在 |
| 2 | 原始文件数 | `records100` 下有 21,799 `.hea` / 21,799 `.dat`，没有缺记录 |
| 3 | 信号缓存 | 形状 `(21799, 12, 1000)`、float32，与 100 Hz / 10 s / 12 导联一致 |
| 4 | 切分 | train/val/test = 17,418/2,183/2,198，`ecg_id` 无重复、两两交集为 0 |
| 5 | 患者级隔离 | 患者数 15,023/1,942/1,904，两两交集为 0，且与 M0 报告一致 |
| 6 | 标签 | 从 `meta.parquet` 重算的训练集正例数与 M0 报告逐类相等 |
| 7 | 产物清单 | 29 个 M3 运行 + 29 个 masked、对应 checkpoint、14 张图、核心 JSON 都在 |
| 8 | 指标复算 | 从 4 个模型的 `*_pred.npz` 重算 masked test AUC/F1，与 JSON 差值 0 |
| 9 | 标定 | 温度为正；AUC 变化 ≤1e-12；pooled ECE 下降 |
| 10 | conformal | 用存下的 q 重算逐类覆盖与 Mondrian 覆盖，与 JSON 差值 0 |
| 11 | 汇总一致性 | `m6_master_table.json` 与各 run 的 masked JSON 一致（防止汇总过期） |
| 12 | 文档数字 | README/docs 里 551 处小数都能在 `results/` 中找到 |

任何一项 FAIL 都会打印具体位置和原因，并以退出码 1 结束。

## 3. 手工验收（约 10 分钟）

自动检查保证"自洽"，不保证"方法对"。建议人工看这几处：

1. **切分**：打开 `results/m0_data_report.json`，确认
   `patient_intersection` 三个值都是 0、`patients_with_multiple_folds` 为 0。
2. **主结果**：`docs/RESULTS_TABLE.md` 第一张表。注意表头是 statement-masked；
   不要拿它和"全量当负样本"的数字比。
3. **低标签结论**：对比 M3 曲线表与 seed check 表。超类 1% 的正增益在 seed 43 消失，
   文档里已撤回；如果看到"低标签下超类也提升"的说法，说明文档被改了。
4. **conformal 的坏消息**：节律头边际覆盖 0.8913，但空集 1.1735/记录、正类召回 0.1808；
   Mondrian 正类召回 0.9280，但 5 个类退化、集合大小 7.2707/12。这两组数必须同时出现。
5. **量化**：`figures/m5_quantize.png` 与 `results/m5_quantize.json`。体积减半，
   但 batch 1 延迟 13.07→18.21 ms；如果报告只说"加速"，是错的。
6. **图**：`figures/m4_reliability.png`、`m4_coverage_vs_size.png`、`m3_label_fraction.png`
   与 JSON 中的数字对得上。
7. **复现一个数**：任选一个 tag，运行 `python src/eval_masked.py <tag>`，
   应与原 `_masked.json` 完全一致（脚本会覆盖写）。

## 4. 常见失败

| 现象 | 含义 / 处理 |
|---|---|
| 检查 2 失败 | 原始数据不全：重跑 `download_ptbxl.py`，或该目录没挂载 |
| 检查 3 失败 | `ptbxl_100hz_filt.npy` 缺失/损坏：先 `m0_inspect.py` 再 `m1_preprocess.py` |
| 检查 7 失败 | 删过 `results/` 或 `figures/` 中的文件；按 README 命令补跑对应阶段 |
| 检查 8/10 失败 | 预测 npz 与 JSON 不是同一次运行；不要手工改 JSON，重跑该阶段 |
| 检查 11 失败 | 改了某个 run 的 JSON 但没重跑 `m6_master_table.py` |
| 检查 12 失败 | 文档里的数字写错或引用了不存在的值；按报错行号修文档或重跑对应实验 |

## 5. 这个项目不提供什么

- **图片/扫描件输入**。`src/predict.py` 只接受 12 导联数字信号（WFDB / npy / CSV）；
  照片需要先数字化，而多数打印件每导联只有 2.5 s（模型要 10 s），缺失段无法恢复。
  这一步未实现、未验证，因此不做任何精度承诺。
- conformal 集合的在线输出；`results/m4_conformal.json` 是离线分析。
- 多 seed 全量方差、外部数据集验证、500 Hz 实验。这些在 `docs/EXPERIMENTS.md` 末尾列为"没做的事"。
