import torch
import sys
import numpy as np

checkpoint_path = 'outputs/shortcut_ablation/checkpoints/STAGE/STAGE_fold_1.pth'
print(f"Loading checkpoint: {checkpoint_path}")

checkpoint = torch.load(checkpoint_path, map_location='cpu')
if 'state_dict' in checkpoint:
    state_dict = checkpoint['state_dict']
else:
    state_dict = checkpoint

# 提取 GCN 第一层权重
gcn1_weight_key = 'spatial_encoder.gcn1.lin.weight'
if gcn1_weight_key in state_dict:
    gcn1_weight = state_dict[gcn1_weight_key]
    mean_abs_weights = gcn1_weight.abs().mean(dim=0).cpu().detach().numpy()
    input_dim = gcn1_weight.shape[1]
    print(f"Input dimension: {input_dim}")

# 88 维特征名称
feature_names = []
# 0~7: 基础统计特征
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

# 56~87: IP 身份嵌入
for i in range(16):
    feature_names.append(f'src_ip_emb_{i}')
for i in range(16):
    feature_names.append(f'dst_ip_emb_{i}')

# 按索引顺序打印所有特征
print("\n" + "="*70)
print(f"All {input_dim} Input Features and Their Weights (Shortcut Model)")
print("="*70)
print(f"{'Index':>6} | {'Feature Name':<24} | {'Weight':>12}")
print("-"*70)

# 按索引顺序显示
for i in range(input_dim):
    if i < len(feature_names):
        name = feature_names[i]
    else:
        name = f'unknown_{i}'
    print(f"{i:>6} | {name:<24} | {mean_abs_weights[i]:>12.6f}")

# 按权重排序显示 Top-20 和 Bottom-20
sorted_idx = np.argsort(mean_abs_weights)[::-1]
top20_idx = sorted_idx[:20]
bottom20_idx = sorted_idx[-20:]

print("\n" + "="*70)
print("Top-20 Highest Weights")
print("="*70)
print(f"{'Rank':>6} | {'Index':>6} | {'Feature Name':<24} | {'Weight':>12}")
print("-"*70)
for rank, idx in enumerate(top20_idx, 1):
    if idx < len(feature_names):
        name = feature_names[idx]
    else:
        name = f'unknown_{idx}'
    print(f"{rank:>6} | {idx:>6} | {name:<24} | {mean_abs_weights[idx]:>12.6f}")

print("\n" + "="*70)
print("Bottom-20 Lowest Weights")
print("="*70)
print(f"{'Rank':>6} | {'Index':>6} | {'Feature Name':<24} | {'Weight':>12}")
print("-"*70)
for rank, idx in enumerate(bottom20_idx, 1):
    if idx < len(feature_names):
        name = feature_names[idx]
    else:
        name = f'unknown_{idx}'
    print(f"{rank:>6} | {idx:>6} | {name:<24} | {mean_abs_weights[idx]:>12.6f}")

# 类别统计
print("\n" + "="*70)
print("Category Average Weights")
print("="*70)
categories = {
    'Base stats (0~7)': mean_abs_weights[0:8].mean(),
    'Protocol emb (8~15)': mean_abs_weights[8:16].mean(),
    'Sport emb (16~31)': mean_abs_weights[16:32].mean(),
    'Dport emb (32~47)': mean_abs_weights[32:48].mean(),
    'Flags emb (48~55)': mean_abs_weights[48:56].mean(),
    'Src IP emb (56~71)': mean_abs_weights[56:72].mean(),
    'Dst IP emb (72~87)': mean_abs_weights[72:88].mean(),
}
for cat, val in categories.items():
    print(f"  {cat}: {val:.6f}")
print("="*70)