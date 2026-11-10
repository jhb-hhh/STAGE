import json
import random
import re
import torch
import sys
import numpy as np
from collections import Counter
from tqdm import tqdm

sys.path.insert(0, 'src')

from stage.data.dataset import IDAPTDataset
from stage.models.stage import StageModel
from stage.llm.llm_client import LLMClient  # 请根据你实际的 LLM 调用方式修改导入

# ========== 配置 ==========
NUM_SAMPLES = 50
OUTPUT_DIR = "outputs/hallucination_experiment"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ========== 加载数据 ==========
dataset = IDAPTDataset(root_dir='data/reference/graphs')
# 假设测试集索引为 80% 之后（具体看你的 splits）
test_indices = list(range(int(len(dataset) * 0.8), len(dataset)))
sample_indices = random.sample(test_indices, min(NUM_SAMPLES, len(test_indices)))

print(f"Sampled {len(sample_indices)} test samples.")

# ========== 加载模型（用于提取证据） ==========
model = StageModel(num_classes=13)
checkpoint = torch.load('outputs/reference/checkpoints/stage/best_model_fold_2.pth', map_location='cpu')
if 'state_dict' in checkpoint:
    state_dict = checkpoint['state_dict']
else:
    state_dict = checkpoint
model.load_state_dict(state_dict)
model.eval()


# ========== 工具函数：从图中提取 Top-K 证据 ==========
def extract_topk_evidence(data, k=20):
    """运行模型，提取 Top-K 重要性节点"""
    with torch.no_grad():
        data.batch = torch.zeros(data.num_nodes, dtype=torch.long)
        _, expl = model(data, return_explanation=True)
        importance = expl['final_attn'].squeeze(-1)  # (num_nodes,)
        topk_indices = importance.argsort(descending=True)[:k].tolist()
        return topk_indices, importance[topk_indices].tolist()


# ========== 工具函数：构建两种 Prompt ==========
def build_baseline_prompt(data):
    """组 A：全量图统计摘要（不含证据）"""
    stats = {
        'num_nodes': data.num_nodes,
        'num_edges': data.edge_index.size(1),
        'avg_degree': data.edge_index.size(1) / data.num_nodes,
        'protocols': Counter(data.proto_idx.tolist()).most_common(3),
        'top_degree_nodes': sorted(range(data.num_nodes),
                                   key=lambda i: (data.edge_index[0] == i).sum() + (data.edge_index[1] == i).sum(),
                                   reverse=True)[:5],
        'label': data.y[0].item() if data.y.numel() > 0 else -1,
    }
    prompt = f"""
You are an APT attack chain analyst. Given the following statistical summary of a network traffic graph, reconstruct the attack chain step by step.

**Graph Summary**:
- Total nodes: {stats['num_nodes']}
- Total edges: {stats['num_edges']}
- Average degree: {stats['avg_degree']:.2f}
- Top protocols (proto_idx): {stats['protocols']}
- Top 5 nodes by degree: {stats['top_degree_nodes']}
- Label (if provided): {stats['label']}

**Instructions**:
1. Identify possible attack phases (Reconnaissance, Initial Access, Delivery, C2, Exfiltration).
2. For each phase, mention the relevant node IDs and edges if inferred.
3. If evidence is insufficient, state "Insufficient evidence" rather than guessing.
"""
    return prompt


def build_evidence_prompt(data, topk_nodes, topk_weights):
    """组 B：只给 Top-K 证据节点"""
    # 提取子图边（仅连接 Top-K 节点之间的边）
    node_set = set(topk_nodes)
    edge_indices = []
    for i, (src, dst) in enumerate(data.edge_index.t().tolist()):
        if src in node_set and dst in node_set:
            edge_indices.append((src, dst))

    prompt = f"""
You are an APT attack chain analyst. The following nodes have been identified as the most critical evidence by a detection model.

**Top-K Evidence Nodes** (node_id: importance_score):
{', '.join([f"node_{n}: {w:.4f}" for n, w in zip(topk_nodes, topk_weights)])}

**Edges between evidence nodes**:
{edge_indices[:20]}  # 最多展示 20 条边

**Instructions**:
1. Based only on these evidence nodes and their connections, reconstruct the attack chain.
2. Identify phases (Reconnaissance, Initial Access, Delivery, C2, Exfiltration) and map nodes to phases.
3. If a phase cannot be confidently determined, state "Insufficient evidence" rather than guessing.
"""
    return prompt


# ========== 工具函数：解析报告中的实体 ==========
def extract_entities(report_text):
    """从 LLM 报告中提取提到的节点 ID 和攻击阶段"""
    node_ids = set(re.findall(r'node[_\s]*(\d+)', report_text, re.IGNORECASE))
    phases = []
    for p in ['reconnaissance', 'initial access', 'delivery', 'c2', 'exfiltration', 'lateral movement']:
        if p in report_text.lower():
            phases.append(p)
    return node_ids, phases


# ========== 工具函数：与真实数据对比 ==========
def compute_fact_accuracy(node_ids, phases, data):
    """计算事实准确率 = (正确提及数) / (总提及数 + 1)"""
    # 真实节点：所有有恶意标签的节点（或全部节点，取决于基准）
    real_node_ids = set(range(data.num_nodes))  # 简化：所有节点都存在
    # 更精确的做法：取 data.y 中为恶意的节点
    malicious_nodes = set((data.y > 0).nonzero().squeeze(-1).tolist()) if data.y.numel() > 0 else set()

    correct_nodes = node_ids.intersection(real_node_ids)
    node_acc = len(correct_nodes) / len(node_ids) if node_ids else 1.0

    # 攻击阶段准确性：比较报告中的阶段与真实标签（如果有）
    # 简化版本：只报告提及的节点是否真实存在
    return node_acc


# ========== 主实验循环 ==========
baseline_scores = []
evidence_scores = []
baseline_reports = []
evidence_reports = []

for idx in tqdm(sample_indices, desc="Processing samples"):
    data = dataset[idx]

    # 1. 提取 Top-K 证据
    topk_nodes, topk_weights = extract_topk_evidence(data)

    # 2. 构造两组 Prompt
    prompt_baseline = build_baseline_prompt(data)
    prompt_evidence = build_evidence_prompt(data, topk_nodes, topk_weights)

    # 3. 调用 LLM（这里用 mock，你需要替换成真实的 LLM 调用）
    # 假设你有一个 llm_client.generate(prompt) 方法
    # report_baseline = llm_client.generate(prompt_baseline)
    # report_evidence = llm_client.generate(prompt_evidence)

    # Mock 示例（请替换）
    report_baseline = "The graph shows node_3 and node_5 communicating over HTTP. Insufficient evidence for attack chain."
    report_evidence = "node_12 and node_45 show initial access via phishing, node_89 indicates C2 communication."

    baseline_reports.append(report_baseline)
    evidence_reports.append(report_evidence)

    # 4. 解析并计算准确率
    nodes_b, phases_b = extract_entities(report_baseline)
    nodes_e, phases_e = extract_entities(report_evidence)

    acc_b = compute_fact_accuracy(nodes_b, phases_b, data)
    acc_e = compute_fact_accuracy(nodes_e, phases_e, data)

    baseline_scores.append(acc_b)
    evidence_scores.append(acc_e)

# ========== 结果统计 ==========
print("\n" + "=" * 60)
print("Hallucination Reduction Experiment")
print("=" * 60)
print(f"Baseline (Full Graph Stats) Fact Accuracy: {np.mean(baseline_scores):.2%}")
print(f"Ours (Top-K Evidence) Fact Accuracy: {np.mean(evidence_scores):.2%}")
print(f"Improvement: {np.mean(evidence_scores) - np.mean(baseline_scores):.2%}")
print("=" * 60)

# 保存报告
with open(os.path.join(OUTPUT_DIR, "baseline_reports.json"), "w") as f:
    json.dump(baseline_reports, f, indent=2)
with open(os.path.join(OUTPUT_DIR, "evidence_reports.json"), "w") as f:
    json.dump(evidence_reports, f, indent=2)

print(f"Reports saved to {OUTPUT_DIR}")