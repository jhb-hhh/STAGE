import torch
import sys
import numpy as np
sys.path.insert(0, 'src')

from stage.models.stage import StageModel
from stage.data.dataset import IDAPTDataset
from torch_geometric.loader import DataLoader

# ========== 加载原始 ID-APT 数据集 ==========
dataset = IDAPTDataset(root_dir='data/reference/graphs')
loader = DataLoader(dataset, batch_size=1, shuffle=False)

print(f"Total graph chunks in ID-APT: {len(dataset)}")

# ========== 加载预训练模型 ==========
model = StageModel(num_classes=13)
checkpoint = torch.load('outputs/reference/checkpoints/stage/best_model_fold_2.pth', map_location='cpu')
if 'state_dict' in checkpoint:
    state_dict = checkpoint['state_dict']
else:
    state_dict = checkpoint
model.load_state_dict(state_dict)
model.eval()

print("\n" + "="*60)
print("Analyzing GCN Encoder Input Layer Weights on ID-APT")
print("(This reflects the model's reliance on original 56-dim features)")
print("="*60)

# 获取 GCN 第一层的线性权重
gcn1_weight = model.spatial_encoder.gcn1.lin.weight  # shape: [hidden_dim, 56]
mean_abs_weights = gcn1_weight.abs().mean(dim=0).cpu().detach().numpy()

# 定义 56 维输入特征的名称映射
feature_names = []
# 0~7: 基础统计特征 (data.x)
base_names = ['duration', 'pkt_count', 'byte_count', 'rate',
              'pkt_len_mean', 'pkt_len_std', 'iat_mean', 'iat_std']
feature_names.extend(base_names)
# 8~15: 协议嵌入
for i in range(8):
    feature_names.append(f'proto_emb_{i}')
# 16~31: 源端口嵌入
for i in range(16):
    feature_names.append(f'sport_emb_{i}')
# 32~47: 目的端口嵌入
for i in range(16):
    feature_names.append(f'dport_emb_{i}')
# 48~55: TCP标志嵌入
for i in range(8):
    feature_names.append(f'flags_emb_{i}')

# 按权重绝对值排序，取 Top-10 和 Bottom-10
sorted_idx = np.argsort(mean_abs_weights)[::-1]
top10_idx = sorted_idx[:10]
bottom10_idx = sorted_idx[-10:]

print("\nTop-10 Influential Input Features:")
for idx in top10_idx:
    print(f"  Index {idx:2d} | {feature_names[idx]:20s} | weight = {mean_abs_weights[idx]:.6f}")

print("\nBottom-10 Least Influential Input Features:")
for idx in bottom10_idx:
    print(f"  Index {idx:2d} | {feature_names[idx]:20s} | weight = {mean_abs_weights[idx]:.6f}")

# 统计各类特征的平均贡献
categories = {
    'Base stats (0~7)': mean_abs_weights[0:8].mean(),
    'Protocol emb (8~15)': mean_abs_weights[8:16].mean(),
    'Sport emb (16~31)': mean_abs_weights[16:32].mean(),
    'Dport emb (32~47)': mean_abs_weights[32:48].mean(),
    'Flags emb (48~55)': mean_abs_weights[48:56].mean()
}
print("\nAverage Contribution by Feature Category:")
for cat, val in categories.items():
    print(f"  {cat}: {val:.6f}")

print("="*60)