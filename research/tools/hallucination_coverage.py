#!/usr/bin/env python
"""
实体包含率实验：验证 STAGE 证据是否有效减少 LLM 幻觉。

对比:
  - Baseline: 全图统计摘要 → LLM
  - Ours: Top-K 证据节点 → LLM
  - Random (可选): 随机节点 → LLM

指标: 实体包含率 = (报告中提到的节点 ∩ 证据节点) / 报告中提到的节点总数
"""

import os
import sys
import json
import random
import re
import torch
import numpy as np
from tqdm import tqdm
from collections import Counter
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, 'src')

from stage.data.dataset import IDAPTDataset
from stage.models.stage import StageModel

# ============================================================
# 1. 配置
# ============================================================
NUM_SAMPLES = 50  # 抽取样本数
TOP_K = 20  # 证据数量
OUTPUT_DIR = "outputs/hallucination_coverage"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# 2. 加载数据和模型
# ============================================================
print("Loading dataset and model...")
dataset = IDAPTDataset(root_dir='data/reference/graphs')
test_indices = list(range(int(len(dataset) * 0.8), len(dataset)))
sample_indices = random.sample(test_indices, min(NUM_SAMPLES, len(test_indices)))
print(f"✅ Sampled {len(sample_indices)} test samples.")

model = StageModel(num_classes=13)
checkpoint = torch.load('outputs/reference/checkpoints/stage/best_model_fold_2.pth', map_location='cpu')
if 'state_dict' in checkpoint:
    state_dict = checkpoint['state_dict']
else:
    state_dict = checkpoint
model.load_state_dict(state_dict)
model.eval()


# ============================================================
# 3. 工具函数
# ============================================================
def extract_topk_evidence(data, k=20):
    """提取 STAGE 选出的 Top-K 证据节点"""
    with torch.no_grad():
        data.batch = torch.zeros(data.num_nodes, dtype=torch.long)
        _, expl = model(data, return_explanation=True)
        importance = expl['final_attn'].squeeze(-1)
        topk_indices = importance.argsort(descending=True)[:k].tolist()
        return topk_indices


def extract_random_nodes(data, k=20):
    """随机抽取 k 个节点（用于 Random Baseline）"""
    all_nodes = list(range(data.num_nodes))
    return random.sample(all_nodes, min(k, len(all_nodes)))


def build_baseline_prompt(data):
    """组 A：全图统计摘要"""
    stats = {
        'num_nodes': data.num_nodes,
        'num_edges': data.edge_index.shape[1],
        'avg_degree': data.edge_index.shape[1] / max(1, data.num_nodes),
    }
    return f"""
You are an APT attack chain analyst. Based on the following statistical summary of a network traffic graph, reconstruct the attack chain step by step.

**Graph Summary**:
- Total nodes: {stats['num_nodes']}
- Total edges: {stats['num_edges']}
- Average degree: {stats['avg_degree']:.2f}

**Instructions**:
1. Identify possible attack phases: Reconnaissance, Initial Access, Delivery, C2, Exfiltration.
2. Mention specific node IDs (e.g., node_5) in your report.
3. If evidence is insufficient, state "Insufficient evidence" rather than guessing.
"""


def build_evidence_prompt(evidence_nodes):
    """组 B：Top-K 证据节点"""
    nodes_str = ', '.join([f'node_{n}' for n in evidence_nodes[:20]])
    return f"""
You are an APT attack chain analyst. The following nodes have been identified as critical evidence by a detection model.

**Evidence Nodes**: {nodes_str}

**Instructions**:
1. Based only on these evidence nodes, reconstruct the attack chain.
2. Identify phases: Reconnaissance, Initial Access, Delivery, C2, Exfiltration.
3. Mention specific node IDs in your report.
4. If evidence is insufficient, state "Insufficient evidence" rather than guessing.
"""


def build_random_prompt(random_nodes):
    """组 C：随机节点（可选）"""
    nodes_str = ', '.join([f'node_{n}' for n in random_nodes[:20]])
    return f"""
You are an APT attack chain analyst. The following nodes have been randomly selected from the network graph.

**Random Nodes**: {nodes_str}

**Instructions**:
1. Based on these nodes, try to reconstruct the attack chain.
2. Identify phases if possible.
3. Mention specific node IDs.
4. If evidence is insufficient, state "Insufficient evidence" rather than guessing.
"""


def extract_mentioned_nodes(report_text):
    """从 LLM 报告中提取提到的节点 ID"""
    return set(map(int, re.findall(r'node[_\s]*(\d+)', report_text, re.IGNORECASE)))


def compute_coverage(mentioned_nodes, evidence_nodes):
    """计算实体包含率"""
    if not mentioned_nodes:
        return 0.0  # 没有提到任何节点 -> 无法评估
    evidence_set = set(evidence_nodes)
    covered = mentioned_nodes.intersection(evidence_set)
    return len(covered) / len(mentioned_nodes)


def mock_llm_call(prompt):
    """
    Mock LLM 调用（用于测试脚本流程）
    实际使用时替换为真实的 LLM 接口
    """
    # 模拟：从 prompt 中提取一些节点
    nodes = re.findall(r'node_(\d+)', prompt)
    if not nodes:
        return "Insufficient evidence to reconstruct attack chain."
    # 随机返回其中一部分
    selected = random.sample(nodes, min(len(nodes), random.randint(3, 10)))
    selected_str = ', '.join([f'node_{n}' for n in selected])
    phases = ['initial access', 'c2', 'exfiltration']
    selected_phases = random.sample(phases, min(len(phases), random.randint(1, 3)))
    return f"The attack chain includes nodes {selected_str}, with phases: {', '.join(selected_phases)}."


def real_llm_call(prompt):
    """
    TODO: 替换为真实的 LLM 接口
    根据你的项目，可能需要:
      - 调用 tools/llm.py 中的函数
      - 调用 src/stage/llm/ 中的模块
      - 或使用 OpenAI API / 本地 Qwen
    """
    # ===== 如果你已有 LLM 接口，取消下面的注释 =====
    # from tools.llm import generate_report
    # return generate_report(prompt)

    # ===== 临时使用 Mock =====
    return mock_llm_call(prompt)


# ============================================================
# 4. 主实验循环
# ============================================================
print("\nRunning experiment...")
results = []

for idx in tqdm(sample_indices, desc="Processing samples"):
    data = dataset[idx]

    # 提取证据
    stage_nodes = extract_topk_evidence(data, TOP_K)
    random_nodes = extract_random_nodes(data, TOP_K)

    # 构造 Prompt
    prompt_baseline = build_baseline_prompt(data)
    prompt_evidence = build_evidence_prompt(stage_nodes)
    prompt_random = build_random_prompt(random_nodes)

    # 调用 LLM（实际使用时替换 real_llm_call）
    report_baseline = real_llm_call(prompt_baseline)
    report_evidence = real_llm_call(prompt_evidence)
    report_random = real_llm_call(prompt_random)

    # 提取提到的节点
    mentioned_baseline = extract_mentioned_nodes(report_baseline)
    mentioned_evidence = extract_mentioned_nodes(report_evidence)
    mentioned_random = extract_mentioned_nodes(report_random)

    # 计算包含率
    coverage_baseline = compute_coverage(mentioned_baseline, stage_nodes)
    coverage_evidence = compute_coverage(mentioned_evidence, stage_nodes)
    coverage_random = compute_coverage(mentioned_random, stage_nodes)

    # 注意：这里 stage_nodes 是证据集合，三个组的“包含率”都基于同一组证据计算
    # 这样 Baseline 和 Random 看的是“在没有证据的情况下，LLM 的报告能碰巧提到多少证据节点”

    results.append({
        'sample_id': idx,
        'coverage_baseline': coverage_baseline,
        'coverage_evidence': coverage_evidence,
        'coverage_random': coverage_random,
        'num_nodes': data.num_nodes,
        'num_edges': data.edge_index.shape[1],
        'mentioned_baseline': len(mentioned_baseline),
        'mentioned_evidence': len(mentioned_evidence),
        'mentioned_random': len(mentioned_random),
    })

# ============================================================
# 5. 结果统计
# ============================================================
df = pd.DataFrame(results)

print("\n" + "=" * 60)
print("实体包含率实验 (Entity Coverage Rate)")
print("=" * 60)
print(f"组 A (Baseline, 全图统计): {df['coverage_baseline'].mean():.2%}")
print(f"组 B (Ours, Top-K 证据):   {df['coverage_evidence'].mean():.2%}")
print(f"组 C (Random, 随机节点):   {df['coverage_random'].mean():.2%}")
print("-" * 60)
print(f"✅ Ours vs Baseline: +{df['coverage_evidence'].mean() - df['coverage_baseline'].mean():.2%}")
print(f"✅ Ours vs Random:   +{df['coverage_evidence'].mean() - df['coverage_random'].mean():.2%}")
print("=" * 60)

# 保存结果
df.to_csv(os.path.join(OUTPUT_DIR, 'coverage_results.csv'), index=False)
print(f"✅ Results saved to {OUTPUT_DIR}/coverage_results.csv")

# ============================================================
# 6. 画图（可选）
# ============================================================
plt.figure(figsize=(8, 6))
means = [df['coverage_baseline'].mean(), df['coverage_evidence'].mean(), df['coverage_random'].mean()]
labels = ['Baseline\n(Full Stats)', 'Ours\n(Top-K Evidence)', 'Random\nBaseline']
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
bars = plt.bar(labels, means, color=colors, edgecolor='black', linewidth=1.5)
plt.ylabel('Entity Coverage Rate', fontsize=12)
plt.title('LLM Hallucination Reduction: Entity Coverage Comparison', fontsize=14)
plt.ylim(0, 1.0)
for bar, val in zip(bars, means):
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
             f'{val:.2%}', ha='center', va='bottom', fontsize=12, fontweight='bold')
plt.grid(axis='y', linestyle='--', alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'coverage_comparison.pdf'), bbox_inches='tight')
plt.savefig(os.path.join(OUTPUT_DIR, 'coverage_comparison.png'), bbox_inches='tight')
print(f"✅ Figure saved to {OUTPUT_DIR}/coverage_comparison.pdf")