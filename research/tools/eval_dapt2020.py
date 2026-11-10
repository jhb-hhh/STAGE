import torch
import sys
sys.path.insert(0, 'src')

from stage.models.stage import StageModel
from stage.data.dataset import IDAPTDataset
from torch_geometric.loader import DataLoader

# 加载 DAPT 2020 数据集
dataset = IDAPTDataset(root_dir='data/external/dapt2020/graphs')
loader = DataLoader(dataset, batch_size=1, shuffle=False)

# 加载模型
model = StageModel(num_classes=13)
checkpoint = torch.load('outputs/reference/checkpoints/stage/best_model_fold_2.pth', map_location='cpu')

if 'state_dict' in checkpoint:
    state_dict = checkpoint['state_dict']
else:
    state_dict = checkpoint

model.load_state_dict(state_dict)
model.eval()

print("✅ Model loaded. Running inference on DAPT 2020...")

for i, data in enumerate(loader):
    if i >= 5:
        break

    # 修复节点特征维度：确保 data.x 为8维
    if data.x.size(1) < 8:
        padding = torch.zeros(data.x.size(0), 8 - data.x.size(1))
        data.x = torch.cat([data.x, padding], dim=1)

    with torch.no_grad():
        logits, expl = model(data, return_explanation=True)
        print(f"Sample {i}: logits shape {logits.shape}, gate_spa shape {expl['gate_spa'].shape}")

print("✅ Inference complete.")