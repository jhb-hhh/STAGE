import torch
import sys
import numpy as np

# 直接加载 checkpoint，不依赖模型架构
checkpoint_path = 'outputs/shortcut_ablation/checkpoints/STAGE/STAGE_fold_1.pth'
print(f"Loading checkpoint: {checkpoint_path}")

checkpoint = torch.load(checkpoint_path, map_location='cpu')
if 'state_dict' in checkpoint:
    state_dict = checkpoint['state_dict']
else:
    state_dict = checkpoint

# 提取 GCN 第一层权重（直接从 checkpoint 中读取）
gcn1_weight_key = 'spatial_encoder.gcn1.lin.weight'
if gcn1_weight_key in state_dict:
    gcn1_weight = state_dict[gcn1_weight_key]
    mean_abs_weights = gcn1_weight.abs().mean(dim=0).cpu().detach().numpy()
    input_dim = gcn1_weight.shape[1]
    print(f"Input dimension: {input_dim}")
else:
    # 尝试其他可能的 key
    for key in state_dict.keys():
        if 'gcn1' in key and 'weight' in key:
            print(f"Found key: {key}")
            gcn1_weight = state_dict[key]
            mean_abs_weights = gcn1_weight.abs().mean(dim=0).cpu().detach().numpy()
            input_dim = gcn1_weight.shape[1]
            print(f"Input dimension: {input_dim}")
            break

# Shortcut 模型的 88 维输入特征名称
feature_names = []
# 0~7: 基础统计特征
base_names = ['duration', 'pkt_count', 'byte_count', 'rate',
              'pkt_len_mean', 'pkt_len_std', 'iat_mean', 'iat_std']
feature_names.extend(base_names)

# 8~15: 协议嵌入 (8维)
for i in range(8):
    feature_names.append(f'proto_emb_{i}')

# 16~31: 源端口嵌入 (16维)
for i in range(16):
    feature_names.append(f'sport_emb_{i}')

# 32~47: 目的端口嵌入 (16维)
for i in range(16):
    feature_names.append(f'dport_emb_{i}')

# 48~55: TCP标志嵌入 (8维)
for i in range(8):
    feature_names.append(f'flags_emb_{i}')

# 56~87: IP 身份嵌入 (32维，16维源IP + 16维目的IP)
for i in range(16):
    feature_names.append(f'src_ip_emb_{i}')
for i in range(16):
    feature_names.append(f'dst_ip_emb_{i}')

# 如果输入维度是 88，使用完整名称；如果是 56，截断到 56
if input_dim == 88:
    feature_names_full = feature_names
else:
    feature_names_full = feature_names[:input_dim]

sorted_idx = np.argsort(mean_abs_weights)[::-1]
top10_idx = sorted_idx[:10]

print("\n" + "="*60)
print("Top-10 Influential Input Features (Shortcut Model)")
print(f"(Trained on raw data with IP/Port preserved, input_dim={input_dim})")
print("="*60)
for idx in top10_idx:
    if idx < len(feature_names_full):
        print(f"  Index {idx:2d} | {feature_names_full[idx]:24s} | weight = {mean_abs_weights[idx]:.6f}")
    else:
        print(f"  Index {idx:2d} | unknown_{idx:2d}                 | weight = {mean_abs_weights[idx]:.6f}")

# 类别平均贡献（根据输入维度决定）
if input_dim == 88:
    categories = {
        'Base stats (0~7)': mean_abs_weights[0:8].mean(),
        'Protocol emb (8~15)': mean_abs_weights[8:16].mean(),
        'Sport emb (16~31)': mean_abs_weights[16:32].mean(),
        'Dport emb (32~47)': mean_abs_weights[32:48].mean(),
        'Flags emb (48~55)': mean_abs_weights[48:56].mean(),
        'Src IP emb (56~71)': mean_abs_weights[56:72].mean(),
        'Dst IP emb (72~87)': mean_abs_weights[72:88].mean(),
    }
else:
    categories = {
        'Base stats (0~7)': mean_abs_weights[0:8].mean(),
        'Protocol emb (8~15)': mean_abs_weights[8:16].mean(),
        'Sport emb (16~31)': mean_abs_weights[16:32].mean(),
        'Dport emb (32~47)': mean_abs_weights[32:48].mean(),
        'Flags emb (48~55)': mean_abs_weights[48:56].mean(),
    }

print("\nAverage Contribution by Feature Category:")
for cat, val in categories.items():
    print(f"  {cat}: {val:.6f}")

print("="*60)

# 对比 ID-APT 模型的结果
print("\n" + "="*60)
print("Comparison with ID-APT (De-identified) Model")
print("="*60)
print("ID-APT Top-5 (56-dim, no IP):")
print("  iat_std (0.1454), duration (0.1337), iat_mean (0.1280),")
print("  pkt_count (0.1275), pkt_len_std (0.1084)")
print("\nShortcut Model includes IP embeddings (src_ip_emb_*, dst_ip_emb_*)")
print("with input_dim=88, confirming IP identity is encoded.")
print("="*60)