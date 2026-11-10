# STAGE 模型与实验任务说明

本文档说明 STAGE 项目中的模型、训练入口和实验任务。完整复现流程请见 `docs/reproduction.md`，artifact 清单请见 `docs/artifact_manifest.md`，数据说明请见 `docs/dataset.md`。

本文档回答以下问题：

~~~text
1. 项目中有哪些模型？
2. 主模型、baseline 和 ablation 的关系是什么？
3. 如何训练不同模型？
4. 如何评估 detection、robustness、explanation、sensitivity 和 LLM 任务？
5. PGExplainer 为什么需要预训练权重？
6. source metrics 和 paper tables 的关系是什么？
~~~

---

## 1. 模型分组

STAGE 项目中的模型分为三类：

~~~text
STAGE 主模型：
  论文中的最终模型，用于主要检测结果、鲁棒性实验和解释器实验。

Baselines：
  与 STAGE 对比的外部检测模型。

Ablations：
  用于分析 STAGE 内部结构贡献的消融模型。
~~~

三类模型共享同一套图数据和五折划分，但 checkpoint 路径、训练入口和评估结果文件不同。

---

## 2. STAGE 主模型

STAGE 是论文中的最终模型，用于学习 APT 攻击链中的时空行为模式。

在当前项目结构中，STAGE 主模型对应：

~~~text
--model stage
~~~

训练时，`stage` 会映射到内部训练模式：

~~~text
stage_final
~~~

主要 checkpoint 位置为：

~~~text
outputs/reference/checkpoints/stage/
outputs/workspace/checkpoints/stage/
~~~

其中：

~~~text
outputs/reference/checkpoints/stage/
  保存论文实验使用的 reference checkpoint。

outputs/workspace/checkpoints/stage/
  保存 rebuild 模式重新训练得到的 checkpoint。
~~~

STAGE 主模型主要用于：

~~~text
detection:
  生成 STAGE 检测结果。

robustness:
  评估 shifted-domain 下的鲁棒性。

explanation:
  作为解释器 benchmark 的被解释模型。

sensitivity:
  分析核心证据数量变化对解释结果的影响。

LLM attack-chain analysis:
  为 LLM 任务提供检测结果和核心证据。
~~~

---

## 3. Baseline 模型

Baseline 模型用于和 STAGE 主模型进行检测性能对比。

当前支持的 baseline 包括：

~~~text
TS-IDS
FlowTransformer
E-T-GraphSAGE
DIDS-MFL
GraphIDS
~~~

在 detection 和训练命令中，可以直接指定单个 baseline：

~~~bash
python tools/run.py rebuild detection --model FlowTransformer --fold 2
python tools/run.py rebuild train --model FlowTransformer --fold 0
~~~

也可以指定全部 baseline：

~~~bash
python tools/run.py rebuild detection --model baselines --fold 2
python tools/run.py rebuild train --model baselines --fold 0
~~~

Baseline checkpoint 位置为：

~~~text
outputs/reference/checkpoints/baselines/
outputs/workspace/checkpoints/baselines/
~~~

Baseline 训练入口由以下模块负责：

~~~text
tools/train_baselines.py
src/stage/training/baseline_trainer.py
~~~

`baseline_trainer.py` 中已经加入训练进度条。终端运行时会显示每个 epoch 的 batch 进度和当前 loss；如果输出被重定向到日志文件，进度条会自动关闭，避免日志被刷屏。

### 3.1 DIDS-MFL 适配

DIDS-MFL 官方代码以 `TemporalData` 事件流为输入。STAGE 的一个图样本已经是按时间排序的流窗口，因此适配层以 GRU 保存窗口内时间记忆，以动态图扩散聚合 `edge_index`，并保留 implicit/explicit 双分支解耦、正交/去相关约束和多尺度融合。ID-APT 使用原有图级攻击家族标签；外部数据若只提供逐流二元标签，则“窗口包含任一恶意流”定义为恶意图。

### 3.2 GraphIDS 适配

GraphIDS 官方实现把网络流表示为边；STAGE 已把流转换为节点，并用共享源/目的地址连接流节点。适配层因此在该流图上执行 mean-aggregation E-GraphSAGE，再以长度 512 的无位置编码 Transformer 自编码器重构嵌入。训练仅使用训练折中的 benign 图；阈值只在验证折选择。输出任务固定为 benign/attack 二分类，并在 CSV 中记录 `Task=binary`。

这两种适配均复用图样本级五折索引，不修改 `data/reference` 或 `outputs/reference`。实现入口为：

~~~text
src/stage/baselines/dids_mfl/
src/stage/baselines/graphids/
src/stage/training/advanced_baseline_trainer.py
src/stage/data/baseline_protocol.py
~~~

---

## 4. Ablation 模型

Ablation 模型用于分析 STAGE 不同组件的贡献。

当前支持的 ablation 模式包括：

~~~text
gcn_only
lstm_only
inter_serial
fusion_add
fusion_concat
fusion_cross_attn
gcn_trans_rvg
gat_lstm_rvg
~~~

可以运行全部 ablation：

~~~bash
python tools/run.py rebuild detection --model ablations --fold 2
python tools/run.py rebuild train --model ablations --fold 0
~~~

也可以只运行某一个 ablation：

~~~bash
python tools/run.py rebuild detection --model gcn_only --fold 2
python tools/run.py rebuild train --model gcn_only --fold 0
~~~

Ablation checkpoint 位置为：

~~~text
outputs/reference/checkpoints/ablation/
outputs/workspace/checkpoints/ablation/
~~~

Ablation 与 STAGE 主模型共享训练入口：

~~~text
tools/train.py
src/stage/training/trainer.py
~~~

`trainer.py` 中已经加入训练进度条，用于显示训练和验证阶段的 batch 进度。

---

## 5. 训练入口

统一训练入口为：

~~~bash
python tools/run.py rebuild train
~~~

默认情况下，该命令会训练 STAGE、baseline 和 ablation 相关模型。由于完整训练耗时很长，调试时建议使用 `--model` 和 `--fold` 进行局部测试。

### 5.1 训练 STAGE 主模型

~~~bash
python tools/run.py rebuild train --model stage --fold 0
~~~

### 5.2 训练单个 baseline

~~~bash
python tools/run.py rebuild train --model FlowTransformer --fold 0
~~~

### 5.3 训练单个 ablation

~~~bash
python tools/run.py rebuild train --model gcn_only --fold 0
~~~

### 5.4 训练全部模型

~~~bash
python tools/run.py rebuild train --model all --fold all
~~~

完整训练会写入：

~~~text
outputs/workspace/checkpoints/
~~~

训练不会覆盖 reference checkpoint。

---

## 6. Detection 评估

Detection 评估用于重新计算 STAGE、baseline 和 ablation 的检测性能。

统一入口为：

~~~bash
python tools/run.py reproduce detection
python tools/run.py rebuild detection
~~~

二者区别是：

~~~text
reproduce detection:
  读取 reference graph、split 和 checkpoint。
  输出到 outputs/reproduce/metrics/source_metrics/。

rebuild detection:
  读取 workspace graph、split 和 checkpoint。
  输出到 outputs/workspace/metrics/source_metrics/。
~~~

Detection 支持局部评估。`--model` 可以选择：

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
python tools/run.py rebuild detection --model stage --fold 2
python tools/run.py rebuild detection --model FlowTransformer --fold 2
python tools/run.py rebuild detection --model gcn_only --fold 2
~~~

Detection 任务只生成 source metrics，不自动生成最终论文表格。

常见 source metric 文件包括：

~~~text
stage_detection.csv
baseline_detection.csv
ablation_summary.csv
~~~

---

## 7. Robustness 评估

Robustness 任务用于评估模型在 shifted-domain 图数据上的鲁棒性。

命令为：

~~~bash
python tools/run.py reproduce robustness
python tools/run.py rebuild robustness
~~~

输入数据包括：

~~~text
data/reference/graphs_shifted/
outputs/workspace/graphs_shifted/
~~~

如果 workspace 中没有 shifted-domain 图数据，`rebuild robustness` 应直接失败，而不应 fallback 到 reference。

Robustness 任务输出 source metrics，常见文件包括：

~~~text
robustness_summary.csv
~~~

如果需要生成论文表格，需要在 source metrics 齐全后显式运行：

~~~bash
python tools/run.py reproduce tables
python tools/run.py rebuild tables
~~~

---

## 8. Explanation benchmark

Explanation benchmark 用于比较 STAGE 内生解释和若干 post-hoc explainer 的效果与效率。

命令为：

~~~bash
python tools/run.py reproduce explanation
python tools/run.py rebuild explanation
~~~

当前解释器包括：

~~~text
Ours / Endogenous explainer
GNNExplainer
PGExplainer
SHAP
LIME
~~~

该任务会对样本运行解释器，对比解释质量和计算开销。它不是轻量命令，运行时间可能较长。

额外的计算开销分析使用相同的 fold-2 验证集、STAGE checkpoint、解释器实现和
正确分类且具有节点真值的样本筛选条件：

~~~bash
STAGE_PATH_PROFILE=workspace PYTHONPATH=src python tools/analyze_explanation_overhead.py --samples 20
~~~

结果保存在 `outputs/workspace/metrics/benchmarks/computational_overhead.csv`，
包含每样本的预测加解释耗时、模型前向总次数、相对于共同分类前向的额外次数、
进程内 PyTorch 峰值 GPU allocated memory、相对方法调用前的增量峰值、图节点和边数。GPU 计时前后同步；
解释器初始化、样本筛选和质量评估不计入。STAGE 使用分类前向返回的内部权重
直接计算节点分数，因此总前向次数为 1、额外次数为 0。其他方法先运行同样的
分类前向，再调用原有解释器；PGExplainer 的离线训练成本不包含在表中。
无 GPU 时显存列为空。旧 `explanation_benchmark.csv` 的 `time` 列没有 GPU
同步，且 STAGE 解释器在筛选前向后又运行一次前向，不能与新实验的时间列混用。
为避免解释器缓存影响峰值显存的横向比较，正式报告应分别在独立进程中运行每种方法。

输出位置包括：

~~~text
outputs/reproduce/metrics/benchmarks/
outputs/reproduce/metrics/source_metrics/
outputs/workspace/metrics/benchmarks/
outputs/workspace/metrics/source_metrics/
~~~

Explanation benchmark 只生成 source metrics。需要最终表格时，再显式运行 `tables` 任务。

---

## 9. PGExplainer 预训练权重

PGExplainer 是当前解释器中唯一需要预训练权重的 post-hoc explainer。

Reference 模式需要：

~~~text
outputs/reference/checkpoints/explainers/pgexplainer_weights.pt
~~~

Workspace 模式需要：

~~~text
outputs/workspace/checkpoints/explainers/pgexplainer_weights.pt
~~~

`verify` 会检查 reference PGExplainer checkpoint 是否存在：

~~~bash
python tools/run.py verify
~~~

如果缺少 PGExplainer 权重，`reproduce explanation` 和 `reproduce sensitivity` 会提前失败，而不是运行到中途才报 `PGExplainer is not yet fully trained`。

本地开发时，`pgexplainer_weights.pt` 可以是软链接；artifact 交付时，应提供真实文件或确保软链接目标有效。

---

## 10. Sensitivity 分析

Sensitivity 任务用于分析不同核心证据数量设置对解释结果的影响。

命令为：

~~~bash
python tools/run.py reproduce sensitivity
python tools/run.py rebuild sensitivity
~~~

该任务通常耗时较长。输出位置为：

~~~text
outputs/reproduce/metrics/sensitivity/
outputs/workspace/metrics/sensitivity/
~~~

常见输出文件包括：

~~~text
sensitivity_analysis_fold2.csv
~~~

Sensitivity 同样依赖 PGExplainer 预训练权重。如果缺少权重，任务会提前报错。

---

## 11. LLM 攻击链分析任务

LLM 任务用于基于核心证据生成和评估攻击链分析结果。

相关 rebuild 任务包括：

~~~bash
python tools/run.py rebuild llm-prepare
python tools/run.py rebuild llm-answer
python tools/run.py rebuild llm-grade
~~~

含义如下：

~~~text
llm-prepare:
  准备 exam papers。

llm-answer:
  调用指定 LLM 生成回答。

llm-grade:
  基于已有 LLM 输出重新统计评分结果。
~~~

`reproduce llm` 不重新调用 LLM，而是读取 reference 中已有的 LLM JSON，重新统计 source metrics：

~~~bash
python tools/run.py reproduce llm
~~~

常见输入位置：

~~~text
outputs/reference/llm_reports/
outputs/workspace/llm_reports/
~~~

常见输出文件：

~~~text
table10_llm.csv
~~~

---

## 12. Source metrics 与 paper tables

本项目区分两类结果文件：

~~~text
source metrics:
  实验任务直接生成的原始统计结果。

paper tables:
  从 source metrics 整理得到的论文表格。
~~~

实验任务只生成 source metrics，不自动生成 paper tables。

生成 paper tables 需要显式执行：

~~~bash
python tools/run.py reproduce tables
python tools/run.py rebuild tables
~~~

`reproduce all` 和 `rebuild all` 不自动包含 `tables`，这是为了避免局部实验或单折实验误生成最终论文表格。

---

## 13. 常用输出位置

### 13.1 Reference checkpoint

~~~text
outputs/reference/checkpoints/stage/
outputs/reference/checkpoints/baselines/
outputs/reference/checkpoints/ablation/
outputs/reference/checkpoints/explainers/
~~~

### 13.2 Workspace checkpoint

~~~text
outputs/workspace/checkpoints/stage/
outputs/workspace/checkpoints/baselines/
outputs/workspace/checkpoints/ablation/
outputs/workspace/checkpoints/explainers/
~~~

### 13.3 Source metrics

~~~text
outputs/reproduce/metrics/source_metrics/
outputs/workspace/metrics/source_metrics/
~~~

### 13.4 Paper tables

~~~text
outputs/reproduce/metrics/paper_tables/
outputs/workspace/metrics/paper_tables/
~~~

---

## 14. Smoke test 建议

如果只是确认 rebuild 模式能跑起来，不需要完整训练所有模型。

可以先测试 detection：

~~~bash
python tools/run.py rebuild detection --model stage --fold 2
python tools/run.py rebuild detection --model FlowTransformer --fold 2
python tools/run.py rebuild detection --model gcn_only --fold 2
~~~

再测试训练入口：

~~~bash
timeout 300 bash -c 'CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src python tools/run.py rebuild train --model stage --fold 0'
timeout 300 bash -c 'CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src python tools/run.py rebuild train --model FlowTransformer --fold 0'
~~~

看到 graph chunks 加载成功、进入 fold 训练并显示训练进度条，即可认为训练入口调试通过。

---

## 15. 注意事项

1. `reference` 应视为只读基准空间。
2. `workspace` 用于重新构建实验产物，可以随时清理。
3. 局部 detection 只生成 source metrics，不生成最终论文表格。
4. 生成论文表格必须显式运行 `tables` 任务。
5. PGExplainer 权重是 explanation 和 sensitivity 的必要 artifact。
6. 完整训练、explanation benchmark 和 sensitivity 分析都可能耗时较长，建议使用 screen 或其他会话管理工具运行。

## 16. 数据处理消融模型

数据处理消融实验在 `src/shortcut_ablation/` 中提供四个隔离的 shortcut-visible 模型版本：

~~~text
STAGE
TS-IDS
FlowTransformer
E-T-GraphSAGE
~~~

这些模型不修改主实验中的 `src/stage/` 实现。它们共享同一个 identity encoder，将以下字段编码为模型可见特征：

~~~text
src_ip_idx_global
dst_ip_idx_global
sport_idx_raw
dport_idx_raw
proto_idx
flags_idx
~~~

其中，原始端口使用 0--65535 的完整取值范围，IP 使用全局稳定编码。对于 baseline 模型，identity encoder 输出会与原始连续流量特征 `x` 拼接后输入模型；对于 STAGE，shortcut-visible identity features 被接入空间编码器输入端。

该设计的目的不是提出新模型，而是构造一个受控消融条件：在训练流程、模型集合和评价指标保持一致的前提下，观察显式暴露 IP 和端口捷径是否会导致模型在 shifted domain 下退化。
