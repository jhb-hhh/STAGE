import torch
import sys
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
import torch_geometric.utils
from scipy.stats import mode

sys.path.insert(0, 'src')

from stage.models.stage import StageModel
from stage.data.dataset import IDAPTDataset
from torch_geometric.loader import DataLoader

# ========== 加载数据 ==========
dataset = IDAPTDataset(root_dir='data/external/dapt2020/graphs_temporal')
loader = DataLoader(dataset, batch_size=1, shuffle=False)

print(f"Total graph chunks loaded: {len(dataset)}")

# ========== 加载预训练模型（冻结） ==========
model = StageModel(num_classes=13)
checkpoint = torch.load('outputs/reference/checkpoints/stage/best_model_fold_2.pth', map_location='cpu')
if 'state_dict' in checkpoint:
    state_dict = checkpoint['state_dict']
else:
    state_dict = checkpoint
model.load_state_dict(state_dict)
model.eval()

for param in model.parameters():
    param.requires_grad = False

# ========== 提取特征和二元标签 ==========
features = []
binary_labels = []

for data in loader:
    # 修复节点特征维度（DAPT只有3维，补零到8维）
    if data.x.size(1) < 8:
        padding = torch.zeros(data.x.size(0), 8 - data.x.size(1))
        data.x = torch.cat([data.x, padding], dim=1)

    with torch.no_grad():
        feat_spatial = model.spatial_encoder(data)
        x_dense, mask = torch_geometric.utils.to_dense_batch(feat_spatial, data.batch)
        x_dense = x_dense + model.pos_encoder[:, :x_dense.size(1), :].to(x_dense.device)
        feat_temporal_dense, _ = model.temporal_encoder(x_dense)
        feat_temporal = feat_temporal_dense[mask]
        feat_fused, _, _ = model.gate_module(feat_spatial, feat_temporal)
        importance_mask = model.node_importance_net(feat_fused)
        feat_weighted = feat_fused * importance_mask
        out_max = torch_geometric.nn.global_max_pool(feat_weighted, data.batch)
        out_mean = torch_geometric.nn.global_mean_pool(feat_weighted, data.batch)
        embedding = torch.cat([out_max, out_mean], dim=1).cpu().numpy()
        features.append(embedding[0])

    # 图级标签：取 y 的众数
    y_np = data.y.numpy()
    if y_np.ndim > 1:
        y_np = y_np.argmax(axis=1)
    if y_np.size == 1:
        majority_label = int(y_np.item())
    else:
        mode_res = mode(y_np, keepdims=False)
        if np.isscalar(mode_res.mode):
            majority_label = int(mode_res.mode)
        else:
            majority_label = int(mode_res.mode[0])
    binary_labels.append(majority_label)

X = np.array(features)
y = np.array(binary_labels)

print(f"Extracted {X.shape[0]} samples, feature dim: {X.shape[1]}")
print(f"Class distribution: Benign={sum(y == 0)}, Malicious={sum(y == 1)}")

# ============================================================
# 1. 标准 75/25 划分（原实验）
# ============================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

clf = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
clf.fit(X_train_scaled, y_train)
y_pred = clf.predict(X_test_scaled)

print("\n" + "=" * 60)
print("Binary Classification on DAPT 2020 (Benign vs Malicious)")
print("Using Temporal Window-split Data (305 chunks) - 75/25 Split")
print("=" * 60)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Benign', 'Malicious']))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("=" * 60)

# ============================================================
# 2. 5折交叉验证（新添加）
# ============================================================
print("\n" + "=" * 60)
print("5-Fold Cross-Validation on DAPT 2020 (305 chunks)")
print("=" * 60)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_accuracies = []
cv_reports = []
all_y_true = []
all_y_pred = []

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    X_train_fold, X_val_fold = X[train_idx], X[val_idx]
    y_train_fold, y_val_fold = y[train_idx], y[val_idx]

    scaler_fold = StandardScaler()
    X_train_fold_scaled = scaler_fold.fit_transform(X_train_fold)
    X_val_fold_scaled = scaler_fold.transform(X_val_fold)

    clf_fold = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
    clf_fold.fit(X_train_fold_scaled, y_train_fold)
    y_pred_fold = clf_fold.predict(X_val_fold_scaled)

    acc_fold = accuracy_score(y_val_fold, y_pred_fold)
    cv_accuracies.append(acc_fold)
    all_y_true.extend(y_val_fold)
    all_y_pred.extend(y_pred_fold)

    print(f"Fold {fold + 1}: Accuracy = {acc_fold:.4f}")

print(f"\nMean CV Accuracy: {np.mean(cv_accuracies):.4f} ± {np.std(cv_accuracies):.4f}")
print("\nOverall Classification Report (across all folds):")
print(classification_report(all_y_true, all_y_pred, target_names=['Benign', 'Malicious']))
print("Confusion Matrix (across all folds):")
print(confusion_matrix(all_y_true, all_y_pred))
print("=" * 60)