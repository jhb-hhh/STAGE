# STAGE 数据说明

本文档说明 STAGE 项目中的数据产物、图数据目录、五折划分和图构建入口。它只关注数据层，不重复模型训练、解释器 benchmark 或 LLM 统计的完整流程；这些内容分别见 `docs/model_details.md` 和 `docs/reproduction.md`。

当前项目将数据空间分成两类：

```text
data/reference/
  已保留的 reference 数据产物，用于 verify 和 reproduce。

outputs/workspace/
  rebuild 模式重新生成的数据产物。
```

`reference` 应视为只读基准空间。`workspace` 是用户重新构建实验产物的运行时空间。

---

## 1. 数据目录概览

当前推荐的数据目录结构如下：

```text
data/
├── reference/
│   ├── graphs/
│   ├── graphs_shifted/
│   └── splits/
└── processed/
    ├── synthesized_pcaps/
    ├── labels/
    └── graphs_generated/
```

其中：

```text
data/reference/graphs/
  reference 主图数据。
  reproduce detection、explanation、sensitivity 等任务会读取这里。

data/reference/graphs_shifted/
  reference shifted-domain 图数据。
  reproduce robustness 会读取这里。

data/reference/splits/
  reference 五折划分。
  reproduce detection 和训练/评估相关任务依赖这里。

data/processed/synthesized_pcaps/
  重新构图时的输入 PCAP 目录。

data/processed/labels/labels.csv
  重新构图时的标签文件。

data/processed/graphs_generated/
  直接运行 `tools/build_graphs.py --run` 时的默认输出目录。
```

`data/reference/` 是快速复现所需的核心数据空间；`data/processed/` 用于从中间数据重新构图。

---

## 2. Reference graph chunks

主图数据位于：

```text
data/reference/graphs/
```

该目录至少应包含：

```text
graphs_chunk_*.pt
label_mapping.json
```

当前 reference 应包含 48 个 graph chunk。可以使用以下命令检查：

```bash
python tools/run.py verify
```

`verify` 会统计 graph chunk 数量，并检查 reference 数据目录是否存在。

图数据以 chunk 形式保存，避免一次性加载超大单文件。训练、检测评估、解释器 benchmark 和 LLM 取样任务都会通过数据集类逐 chunk 加载这些 graph snapshots。

---

## 3. Shifted-domain graph chunks

鲁棒性实验使用 shifted-domain 图数据：

```text
data/reference/graphs_shifted/
```

该目录用于模拟端口变化、特征扰动或分布偏移等测试场景。`reproduce robustness` 会读取该目录：

```bash
python tools/run.py reproduce robustness
```

`rebuild robustness` 则读取 workspace 版本：

```text
outputs/workspace/graphs_shifted/
```

如果 workspace 中缺少 shifted graphs，`rebuild robustness` 应直接报错，而不应 fallback 到 reference。

---

## 4. 五折划分

Reference 五折划分位于：

```text
data/reference/splits/
```

该目录包含：

```text
fold_0.json
fold_1.json
fold_2.json
fold_3.json
fold_4.json
```

每个文件包含：

```json
{
  "train": [...],
  "val": [...]
}
```

这些索引对应 `data/reference/graphs/` 中按顺序展开后的图样本。

---

## 5. 重新生成五折划分

在 rebuild 模式下，重新生成五折划分使用：

```bash
python tools/run.py rebuild splits
```

该命令会读取：

```text
outputs/workspace/graphs/
```

并写入：

```text
outputs/workspace/splits/
```

输出包括：

```text
fold_0.json
fold_1.json
fold_2.json
fold_3.json
fold_4.json
split_metadata.json
```

`tools/make_splits.py` 使用五折分层划分：

```text
n_splits = 5
shuffle = True
random_state = 42
```

这与原始五折划分逻辑保持一致。我们曾验证过：在相同 graph 顺序下，workspace 新生成的 split 与 reference split 完全一致。

注意：`rebuild splits` 只应写入 `outputs/workspace/splits/`，不应覆盖 `data/reference/splits/`。

---

## 6. 图构建入口

图构建相关入口有两个层次。

### 6.1 查看构图计划

直接运行：

```bash
python tools/build_graphs.py
```

默认是 dry-run，只打印构图计划，不真正构图。输出会显示：

```text
path_profile
input_pcap_dir
label_file
output_graph_dir
```

这样设计是为了避免用户误触发耗时构图任务。

### 6.2 辅助脚本真实构图

如果直接调用辅助脚本并希望真实构图，需要显式加 `--run`：

```bash
python tools/build_graphs.py --run
```

在 default profile 下，该命令默认写入：

```text
data/processed/graphs_generated/
```

### 6.3 推荐的 workspace 构图入口

普通 rebuild 流程推荐使用统一入口：

```bash
python tools/run.py rebuild graphs
```

该命令会在 workspace profile 下执行真实构图，输出到：

```text
outputs/workspace/graphs/
```

这是 `rebuild` 模式后续训练、检测评估和解释器实验的输入。

---

## 7. Reference 与 workspace 的关系

`reference` 和 `workspace` 的关系如下：

```text
reference:
  已保留的论文产物。
  用于 verify 和 reproduce。
  不应被 rebuild 覆盖。

workspace:
  用户重新生成实验产物的空间。
  用于 rebuild。
  可以删除、重建、替换。
```

测试 rebuild 模式时，可以临时将 reference 图数据软链接到 workspace，例如：

```bash
ln -sfn "$PWD/data/reference/graphs" outputs/workspace/graphs
```

这只适合调试。测试结束后可以删除：

```bash
rm -rf outputs/workspace
```

删除 workspace 目录只会删除软链接本身，不会删除软链接指向的 reference 数据。

---

## 8. 与运行模式的关系

数据层与三种运行模式的关系如下：

```text
verify:
  检查 data/reference/ 和 outputs/reference/ 是否齐全。

reproduce:
  读取 data/reference/graphs、data/reference/splits 和其他 reference artifact。
  不重新构图，不重新生成 splits。

rebuild:
  读取或生成 outputs/workspace/graphs 和 outputs/workspace/splits。
  不 fallback 到 reference。
```

因此，如果 workspace 缺少 graph chunks，下面命令应直接失败：

```bash
python tools/run.py rebuild detection --model stage --fold 2
```

并提示先运行：

```bash
python tools/run.py rebuild graphs
```

或为调试准备 workspace graph 链接。

---

## 9. 数据相关常见检查

检查 reference 数据是否齐全：

```bash
python tools/run.py verify
```

查看构图计划：

```bash
python tools/build_graphs.py
```

重新生成 workspace splits：

```bash
python tools/run.py rebuild splits
```

查看 workspace split 文件：

```bash
ls -lh outputs/workspace/splits/
```

清理 workspace 测试产物：

```bash
rm -rf outputs/workspace
```

---

## 10. Git 管理原则

以下内容不应提交到 Git：

```text
outputs/workspace/
outputs/reproduce/
data/processed/graphs_generated/
```

这些目录属于运行时输出或重新生成的数据产物。

以下内容通常通过 artifact 包或软链接提供，而不是直接提交到 Git：

```text
data/reference/graphs/
data/reference/graphs_shifted/
outputs/reference/checkpoints/
outputs/reference/llm_reports/
```

如果 reference artifact 使用软链接，交付时必须确保软链接目标有效，或者将软链接替换为真实文件。

## 11. 高级 baseline 外部数据

DIDS-MFL 和 GraphIDS 通过 `--dataset` 支持以下图目录：

~~~text
ID-APT
  workspace profile 的 outputs/workspace/graphs

DAPT2020
  data/external/dapt2020/graphs_temporal

CIC-IDS-2018+Sandworm
  data/external/cicids2018_sandworm/graphs
~~~

外部图至少需要 `x`、`edge_index`、`edge_attr` 和 `y`。当 `y` 是逐流/逐边二元标签时，只要窗口包含一个恶意流，适配层就把该图标为恶意；序列化图本身不会被修改。外部五折只写到 `outputs/workspace/splits/`。当前仓库未随源码提供 CIC-IDS-2018+Sandworm 图块时，命令会给出明确的缺失路径，而不会回退到 ID-APT。

## 12. 数据处理消融数据集

除主实验使用的去标识化 ID-APT 数据外，项目还提供一个隔离的数据处理消融数据集，用于评估抗捷径预处理的贡献。该数据集不会覆盖主实验数据，所有产物均位于：

~~~text
outputs/shortcut_ablation/
~~~

该消融数据集与主数据集的关键区别如下：

~~~text
1. PCAP 合成阶段不执行动态 IP 重映射；
2. 保留原始 victim IP 和 malicious IP；
3. 图数据中额外保存全局 IP 编码；
4. 图数据中额外保存 0--65535 范围内的原始端口；
5. shifted domain 只扰动模型可见的身份和表层流量特征，不改变图拓扑和标签。
~~~

生成后的 original shortcut graphs 位于：

~~~text
outputs/shortcut_ablation/graphs/
~~~

shifted shortcut graphs 位于：

~~~text
outputs/shortcut_ablation/graphs_shifted/
~~~

每个 shortcut graph 在原有字段基础上额外包含：

~~~text
src_ip_idx_global
dst_ip_idx_global
sport_idx_raw
dport_idx_raw
~~~

其中 `src_ip_idx_global` 和 `dst_ip_idx_global` 使用全局稳定 IP 编码，而不是每个图窗口内部重新编码。这样可以显式暴露 IP identity，使该实验能够检测模型是否依赖身份相关捷径。

shifted shortcut graphs 保持以下字段不变：

~~~text
edge_index
edge_attr
y
~~~

因此，shifted-domain 下降主要反映模型对 IP、端口和数值表层特征的依赖，而不是图拓扑或标签变化。
