"""Detection evaluation for the final STAGE model."""

from pathlib import Path

import torch
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from stage.data.dataset import IDAPTDataset
from stage.evaluation.common import load_label_map, load_split_indices, resolve_device
from stage.evaluation.metrics import calculate_metrics_dict
from stage.models.stage import StageModel
from stage.training.checkpoint import load_model_checkpoint
from stage.utils.config import get_path


def evaluate_stage_fold(fold_id=2, checkpoint_path=None, dataset=None, batch_size=16, device=None):
    """Evaluate the final STAGE model on one validation fold."""
    device = resolve_device(device)

    label_map = load_label_map()
    num_classes = len(label_map)

    if dataset is None:
        dataset = IDAPTDataset()

    indices = load_split_indices(fold_id)
    val_dataset = dataset[indices["val"]]

    loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    model = StageModel(num_classes=num_classes).to(device)

    if checkpoint_path is None:
        checkpoint_path = get_path("outputs", "stage_checkpoint_dir") / f"best_model_fold_{fold_id}.pth"
    else:
        checkpoint_path = Path(checkpoint_path)

    load_result = load_model_checkpoint(model, checkpoint_path, map_location=device, strict=False)

    if load_result.missing_keys or load_result.unexpected_keys:
        print("Warning: checkpoint loaded with key differences.")
        print("missing keys:", load_result.missing_keys)
        print("unexpected keys:", load_result.unexpected_keys)

    model.eval()
    y_true, y_pred = [], []

    with torch.no_grad():
        for batch in tqdm(loader, desc=f"Evaluating STAGE fold {fold_id}"):
            batch = batch.to(device)
            batch.x = torch.nan_to_num(batch.x.float(), nan=0.0)

            logits = model(batch)
            preds = logits.argmax(dim=1)

            y_true.extend(batch.y.cpu().numpy().tolist())
            y_pred.extend(preds.cpu().numpy().tolist())

    return calculate_metrics_dict(y_true, y_pred)


if __name__ == "__main__":
    metrics = evaluate_stage_fold(fold_id=2, batch_size=16)
    print(metrics)
