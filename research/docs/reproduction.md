# STAGE 复现说明

本文档说明如何在当前工程结构下复现 STAGE 的实验结果。文档围绕三个运行模式组织：

~~~text
verify     检查 reference artifact 是否齐全。
reproduce  基于 reference artifact 重新计算 source metrics。
rebuild    基于 workspace 重新生成实验产物。
~~~

需要注意：`source metrics` 和 `paper tables` 已经拆开。实验任务只生成中间统计结果，只有显式运行 `tables` 任务时才生成论文表格。

---

## 1. 目录与 artifact 空间

当前工程使用三个 artifact 空间。

### 1.1 reference

`reference` 保存论文实验中已经保留的产物，用于快速检查和复现统计结果。

主要位置：

~~~text
data/reference/graphs
data/reference/graphs_shifted
data/reference/splits

outputs/reference/checkpoints/stage
outputs/reference/checkpoints/ablation
outputs/reference/checkpoints/baselines
outputs/reference/checkpoints/explainers

outputs/reference/explanations
outputs/reference/llm_reports
outputs/reference/metrics
~~~

`reference` 是只读基准空间。`reproduce` 模式会读取这些文件，但不应该修改它们。

### 1.2 reproduce

`reproduce` 是基于 `reference` 重新生成的结果空间。

主要位置：

~~~text
outputs/reproduce/metrics/source_metrics
outputs/reproduce/metrics/paper_tables
outputs/reproduce/metrics/benchmarks
outputs/reproduce/metrics/sensitivity
~~~

`reproduce` 不重新构图，不重新训练模型。它用于验证：在已有论文产物的基础上，评估与统计流程能否重新得到对应结果。

### 1.3 workspace

`workspace` 是重新构建实验产物的工作空间。

主要位置：

~~~text
outputs/workspace/graphs
outputs/workspace/graphs_shifted
outputs/workspace/splits
outputs/workspace/checkpoints
outputs/workspace/metrics
outputs/workspace/llm_reports
~~~

`rebuild` 模式只读写 `workspace`。如果 workspace 中缺少必要文件，命令应该直接报错，而不是回退到 reference。

---

## 2. verify 模式

`verify` 用于检查环境和 reference artifact 是否齐全。

~~~bash
PYTHONPATH=src python tools/run.py verify
~~~

它会检查：

~~~text
Python 模块是否能导入
reference graph chunks
shifted graph folders
five-fold split files
STAGE checkpoints
ablation checkpoints
baseline checkpoints
PGExplainer checkpoint
explanation artifacts
LLM reports
source metric CSV files
~~~

`verify` 不生成实验结果，也不修改任何 artifact。

---

## 3. reproduce 模式

`reproduce` 使用 `reference` 产物重新计算结果，并写入 `outputs/reproduce/`。

### 3.1 运行检测实验

~~~bash
PYTHONPATH=src python tools/run.py reproduce detection
~~~

该命令会基于 reference graph、split 和 checkpoint 重新计算检测相关 source metrics。

也可以只运行局部模型或某一折：

~~~bash
PYTHONPATH=src python tools/run.py reproduce detection --model stage --fold 2
PYTHONPATH=src python tools/run.py reproduce detection --model FlowTransformer --fold 2
PYTHONPATH=src python tools/run.py reproduce detection --model gcn_only --fold 2
~~~

局部 detection 只生成 source metrics，不生成最终论文表格。

### 3.2 运行鲁棒性实验

~~~bash
PYTHONPATH=src python tools/run.py reproduce robustness
~~~

该命令使用 reference shifted graphs 和已有 checkpoint 重新统计鲁棒性结果。

### 3.3 运行解释器 benchmark

~~~bash
PYTHONPATH=src python tools/run.py reproduce explanation
~~~

该命令会重新运行解释器 benchmark，用于比较 endogenous evidence、GNNExplainer、PGExplainer、SHAP 和 LIME 等解释方法。

这是长任务。运行前需要确保 PGExplainer 预训练权重存在：

~~~text
outputs/reference/checkpoints/explainers/pgexplainer_weights.pt
~~~

### 3.4 运行 K-sensitivity 分析

~~~bash
PYTHONPATH=src python tools/run.py reproduce sensitivity
~~~

该命令重新运行 K-sensitivity 分析，输出 sensitivity 相关 CSV。该任务通常耗时较长。

### 3.5 运行 LLM 统计

~~~bash
PYTHONPATH=src python tools/run.py reproduce llm
~~~

该命令读取 reference LLM JSON 结果并重新统计 LLM 攻击链分析指标。它不重新调用 LLM。

### 3.6 reproduce all

~~~bash
PYTHONPATH=src python tools/run.py reproduce all
~~~

`reproduce all` 会依次运行：

~~~text
detection
robustness
explanation
sensitivity
llm
~~~

它不会自动运行 `tables`。这是为了避免局部结果或未完成结果误生成最终论文表格。

---

## 4. rebuild 模式

`rebuild` 使用 `outputs/workspace/` 重新生成实验产物。它不回退到 reference。

### 4.1 重新构图

推荐入口：

~~~bash
PYTHONPATH=src python tools/run.py rebuild graphs
~~~

该命令会调用图构建流程，并将结果写入 workspace 图目录。

如果只想查看构图计划，可以运行：

~~~bash
PYTHONPATH=src python tools/build_graphs.py
~~~

该命令默认是 dry-run，只打印输入输出路径，不真正构图。

如果直接调用图构建脚本并真正执行，需要显式加 `--run`：

~~~bash
PYTHONPATH=src python tools/build_graphs.py --run
~~~

### 4.2 重新生成五折划分

~~~bash
PYTHONPATH=src python tools/run.py rebuild splits
~~~

该命令读取 workspace graphs，并生成：

~~~text
outputs/workspace/splits/fold_0.json
outputs/workspace/splits/fold_1.json
outputs/workspace/splits/fold_2.json
outputs/workspace/splits/fold_3.json
outputs/workspace/splits/fold_4.json
outputs/workspace/splits/split_metadata.json
~~~

五折划分使用分层 K 折，默认折数和随机种子在脚本中固定。

### 4.3 重新训练模型

完整训练：

~~~bash
PYTHONPATH=src python tools/run.py rebuild train
~~~

训练单个模型：

~~~bash
PYTHONPATH=src python tools/run.py rebuild train --model stage --fold 0
PYTHONPATH=src python tools/run.py rebuild train --model FlowTransformer --fold 0
PYTHONPATH=src python tools/run.py rebuild train --model DIDS-MFL --fold 0
PYTHONPATH=src python tools/run.py rebuild train --model GraphIDS --fold 0
PYTHONPATH=src python tools/run.py rebuild train --model gcn_only --fold 0
~~~

高级 baseline 的跨数据集命令为：

~~~bash
PYTHONPATH=src python tools/run.py rebuild train --model DIDS-MFL --fold all --dataset DAPT2020
PYTHONPATH=src python tools/run.py rebuild detection --model DIDS-MFL --fold all --dataset DAPT2020
PYTHONPATH=src python tools/run.py rebuild train --model GraphIDS --fold all --dataset CIC-IDS-2018+Sandworm
PYTHONPATH=src python tools/run.py rebuild detection --model GraphIDS --fold all --dataset CIC-IDS-2018+Sandworm
~~~

DAPT2020 与 CIC-IDS-2018+Sandworm 的新五折文件写入 `outputs/workspace/splits/<dataset>/`，固定使用 `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`。ID-APT 始终要求已有 workspace 五折文件，不会静默重建。高级 baseline checkpoint 只写入 `outputs/workspace/checkpoints/baselines/`。

训练结果写入：

~~~text
outputs/workspace/checkpoints/
~~~

训练过程会显示进度条。若输出重定向到日志文件，进度条会自动关闭，避免日志被刷新字符污染。

### 4.4 重新评估检测结果

~~~bash
PYTHONPATH=src python tools/run.py rebuild detection
~~~

局部评估示例：

~~~bash
PYTHONPATH=src python tools/run.py rebuild detection --model stage --fold 2
PYTHONPATH=src python tools/run.py rebuild detection --model FlowTransformer --fold 2
PYTHONPATH=src python tools/run.py rebuild detection --model gcn_only --fold 2
~~~

这些命令只生成 workspace source metrics，不生成 paper tables。

### 4.5 重新评估鲁棒性

~~~bash
PYTHONPATH=src python tools/run.py rebuild robustness
~~~

该命令需要 workspace shifted graphs 和 workspace checkpoints。

### 4.6 重新运行解释器 benchmark

~~~bash
PYTHONPATH=src python tools/run.py rebuild explanation
~~~

该命令需要 workspace STAGE checkpoint 和 workspace PGExplainer 权重：

~~~text
outputs/workspace/checkpoints/explainers/pgexplainer_weights.pt
~~~

### 4.7 重新运行 K-sensitivity

~~~bash
PYTHONPATH=src python tools/run.py rebuild sensitivity
~~~

这是长任务。它同样需要 workspace PGExplainer 权重。

### 4.8 LLM 相关任务

准备 exam papers：

~~~bash
PYTHONPATH=src python tools/run.py rebuild llm-prepare
~~~

调用 LLM 生成回答：

~~~bash
PYTHONPATH=src python tools/run.py rebuild llm-answer
~~~

统计 LLM 结果：

~~~bash
PYTHONPATH=src python tools/run.py rebuild llm-grade
~~~

其中 `llm-grade` 通常较轻，主要读取已有 JSON 并生成 source metrics。`llm-answer` 可能依赖本地或远程 LLM 服务，运行前应确认模型服务可用。

### 4.9 rebuild all

~~~bash
PYTHONPATH=src python tools/run.py rebuild all
~~~

`rebuild all` 会依次运行 workspace 重构流程中的主要任务，但不会自动运行 `tables`。如需生成论文表格，需要单独执行 `rebuild tables`。

---

## 5. source metrics 与 paper tables

当前工程明确区分两类输出。

### 5.1 source metrics

source metrics 是实验任务直接生成的中间统计结果。

典型目录：

~~~text
outputs/reproduce/metrics/source_metrics
outputs/workspace/metrics/source_metrics
~~~

例如：

~~~text
stage_detection.csv
baseline_detection.csv
ablation_summary.csv
robustness_summary.csv
table10_llm.csv
~~~

### 5.2 paper tables

paper tables 是从 source metrics 整理得到的论文表格。

典型目录：

~~~text
outputs/reproduce/metrics/paper_tables
outputs/workspace/metrics/paper_tables
~~~

只有显式运行以下命令时才会生成：

~~~bash
PYTHONPATH=src python tools/run.py reproduce tables
PYTHONPATH=src python tools/run.py rebuild tables
~~~

如果 source metrics 不完整，`tables` 任务会明确报错。这是预期行为。

---

## 6. detection 模型选择

`detection` 支持以下 `--model` 取值：

~~~text
all
stage

baselines
TS-IDS
FlowTransformer
E-T-GraphSAGE

ablations
gcn_only
lstm_only
inter_serial
fusion_add
fusion_concat
fusion_cross_attn
gcn_trans_rvg
gat_lstm_rvg
~~~

示例：

~~~bash
PYTHONPATH=src python tools/run.py rebuild detection --model stage --fold 2
PYTHONPATH=src python tools/run.py rebuild detection --model FlowTransformer --fold 2
PYTHONPATH=src python tools/run.py rebuild detection --model gcn_only --fold 2
~~~

`--fold` 支持：

~~~text
all
0
1
2
3
4
~~~

单模型、单折评估只用于生成局部 source metrics，不代表最终论文表格。

---

## 7. PGExplainer 预训练权重

PGExplainer 是唯一需要预训练权重的 post-hoc explainer。

reference 模式需要：

~~~text
outputs/reference/checkpoints/explainers/pgexplainer_weights.pt
~~~

workspace 模式需要：

~~~text
outputs/workspace/checkpoints/explainers/pgexplainer_weights.pt
~~~

如果该文件缺失，`reproduce explanation`、`reproduce sensitivity`、`rebuild explanation` 或 `rebuild sensitivity` 会提前报错。

本地开发时该文件可以是软链接；artifact 交付时应确保权重文件本体或有效目标被一并提供，避免软链接在其他机器上失效。

---

## 8. 长任务运行建议

对于长任务，建议使用 `screen` 或类似工具运行。

示例：

~~~bash
screen -S rep_explanation
CUDA_VISIBLE_DEVICES=2 PYTHONPATH=src python tools/run.py reproduce explanation
~~~

退出 screen 但保持任务运行：

~~~text
Ctrl + A，然后按 D
~~~

重新进入：

~~~bash
screen -r rep_explanation
~~~

如果希望只写日志而不在终端刷屏：

~~~bash
CUDA_VISIBLE_DEVICES=2 PYTHONPATH=src python tools/run.py reproduce explanation > outputs/logs/reproduce/explanation.log 2>&1
~~~

---

## 9. 推荐复现流程

### 9.1 快速检查

~~~bash
PYTHONPATH=src python tools/run.py verify
~~~

### 9.2 基于 reference 重新统计结果

先运行较轻任务：

~~~bash
PYTHONPATH=src python tools/run.py reproduce detection
PYTHONPATH=src python tools/run.py reproduce robustness
PYTHONPATH=src python tools/run.py reproduce llm
~~~

再安排长任务：

~~~bash
PYTHONPATH=src python tools/run.py reproduce explanation
PYTHONPATH=src python tools/run.py reproduce sensitivity
~~~

source metrics 齐全后生成表格：

~~~bash
PYTHONPATH=src python tools/run.py reproduce tables
~~~

### 9.3 workspace 调试流程

如果只想检查 rebuild 模式能否跑起来，可以使用局部命令：

~~~bash
PYTHONPATH=src python tools/run.py rebuild detection --model stage --fold 2
PYTHONPATH=src python tools/run.py rebuild detection --model FlowTransformer --fold 2
PYTHONPATH=src python tools/run.py rebuild detection --model gcn_only --fold 2
~~~

训练入口可以用 timeout 做启动测试：

~~~bash
timeout 300 bash -c 'CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src python tools/run.py rebuild train --model stage --fold 0'
~~~

如果进入训练进度条且没有路径、导入或模型初始化错误，就说明训练入口已能正常启动。

---

## 10. 常见问题

### 10.1 为什么 detection 不自动生成表格？

因为 detection 支持单模型、单折局部评估。局部结果并不等于最终论文表格。为了避免误生成不完整表格，表格生成被拆到 `tables` 任务中。

### 10.2 为什么 build_graphs.py 默认不构图？

构图可能耗时很长并写入大量文件。默认 dry-run 可以先确认输入输出路径。真正构图需要使用 `--run`，更推荐通过 `tools/run.py rebuild graphs` 进入 workspace 构图流程。

### 10.3 为什么 verify 通过但 explanation 仍然可能很慢？

`verify` 只检查 artifact 是否存在，不运行解释器 benchmark。`reproduce explanation` 会重新计算解释器结果，因此可能耗时较长。

## 11. 数据处理消融实验复现

数据处理消融实验用于验证抗捷径预处理的必要性。该实验不属于默认 `verify`、`reproduce` 或 `rebuild` 流程，需要单独运行。所有输出均位于：

~~~text
outputs/shortcut_ablation/
~~~

完整流程如下：

~~~bash
PYTHONPATH=src python -m shortcut_ablation.cli synthesize --max-files 10000 --reset
PYTHONPATH=src python -m shortcut_ablation.cli build-graphs
PYTHONPATH=src python -m shortcut_ablation.cli build-shifted-graphs
PYTHONPATH=src python -m shortcut_ablation.cli make-splits
PYTHONPATH=src python -m shortcut_ablation.cli train --model all --fold all
PYTHONPATH=src python -m shortcut_ablation.cli robustness --model all --fold all
~~~

新增 DIDS-MFL / GraphIDS 的表 6 身份捷径实验由隔离入口执行。`w/o` 使用
`outputs/shortcut_ablation/graphs{,_shifted}` 及独立五折（原折号 1--5
映射为这里的 0--4），将已有身份编码器与高级 baseline 联合训练；
`w/` 使用去标识化 ID-APT 图及其原有五折 checkpoint。所有新权重与指标
写在 `outputs/workspace/checkpoints/advanced_robustness/` 和
`outputs/workspace/metrics/advanced_robustness/`。

~~~bash
STAGE_PATH_PROFILE=workspace PYTHONPATH=src python tools/advanced_robustness.py \
  --model DIDS-MFL --fold 0 --condition w/o --train --evaluate
STAGE_PATH_PROFILE=workspace PYTHONPATH=src python tools/advanced_robustness.py \
  --model GraphIDS --fold 0 --condition w/o --train --evaluate
STAGE_PATH_PROFILE=workspace PYTHONPATH=src python tools/advanced_robustness.py \
  --model DIDS-MFL --fold 0 --condition w/ --evaluate
~~~

`w/o` 的外层测试折绝不参与 checkpoint 选择或 GraphIDS 阈值校准；
内部从训练折按类别分层取 10% 用于验证。GraphIDS 原生只检测
Benign/Attack，不能生成 12 个家族分类 F1；家族级表 7 仅输出
DIDS-MFL 的逐家族分类结果。原 ID-APT checkpoint 的选模与阈值
使用既有五折验证集，因此其数值沿用原协议，不能解释成独立的
嵌套测试估计。

旧项目的 shifted-domain 生成脚本只处理 0、2、3、4 折；缺失的
fold 1 可按同一扰动规则在 workspace 单独生成（种子 43），
不会写到 `data/reference/`：

~~~bash
STAGE_PATH_PROFILE=workspace PYTHONPATH=src python tools/generate_missing_shifted_fold.py
STAGE_PATH_PROFILE=workspace PYTHONPATH=src python tools/check_advanced_robustness_inputs.py
python tools/aggregate_advanced_results.py --task robustness
python tools/aggregate_advanced_results.py --task detection
~~~

汇总工具严格要求相应的五折或三个数据集结果齐全；缺文件时仅列出
缺项，不写 final CSV。GraphIDS 的表 6 行应标明 binary task，
不能与 13 类 ID-APT Macro-F1 混作同一指标。第三数据集还需
提供原 CIC-IDS-2018+Sandworm 合并、特征对齐及构图协议或原图块。

训练阶段默认使用与主实验一致的训练设置。调试时可以临时指定较小的 `--max-files`、单个 `--model` 或单个 `--fold`，但正式结果应使用完整数据规模、五折划分和主实验一致的训练配置。

该实验的 shifted domain 与主鲁棒性实验保持同一思想：模拟身份和流量表层特征变化。由于 shortcut-ablation 数据显式暴露 IP identity，shifted graph 生成阶段额外加入拓扑保持的 IP identity reassignment。该操作只改变模型可见的 IP 标识，不改变 `edge_index`、`edge_attr` 或标签。
