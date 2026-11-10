"""Robustness evaluation under shifted-domain graph perturbations."""

import glob
import json
from pathlib import Path

import pandas as pd
import torch
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from stage.evaluation.common import load_label_map, load_split_indices, resolve_device
from stage.evaluation.metrics import calculate_metrics_dict
from stage.models.stage import StageModel
from stage.training.checkpoint import load_model_checkpoint
from stage.utils.config import get_path

DEVICE = resolve_device()


def perturb_graph(data):
    """Apply the shifted-domain perturbation used by the robustness experiment."""
    port_offset = 23333

    if hasattr(data, "sport_idx"):
        mask = data.sport_idx > 1024
        new_sport = data.sport_idx.clone()
        new_sport[mask] = (new_sport[mask] + port_offset) % 65535
        data.sport_idx = new_sport

    if hasattr(data, "dport_idx"):
        mask = data.dport_idx > 1024
        new_dport = data.dport_idx.clone()
        new_dport[mask] = (new_dport[mask] + port_offset) % 65535
        data.dport_idx = new_dport

    if hasattr(data, "x"):
        noise = 0.9 + (0.2 * torch.rand_like(data.x))
        data.x = data.x * noise

    mapping_rules = [
        ("192.168.", "10.0."),
        ("10.", "172.16."),
        ("172.", "192.168."),
    ]

    if hasattr(data, "src_ip_list"):
        new_ips = []

        for ip in data.src_ip_list:
            if isinstance(ip, str):
                temp_ip = ip

                for old, new in mapping_rules:
                    if old in temp_ip:
                        temp_ip = temp_ip.replace(old, new)
                        break

                new_ips.append(temp_ip)
            else:
                new_ips.append(ip)

        data.src_ip_list = new_ips

    return data


def load_original_val_graphs(fold_id):
    """Load original validation graphs for one fold."""
    indices = load_split_indices(fold_id)
    val_indices = set(indices["val"])

    graph_dir = get_path("data", "graph_dir")
    chunk_files = sorted(glob.glob(str(Path(graph_dir) / "graphs_chunk_*.pt")))

    val_graphs = []
    global_idx = 0

    for chunk_path in tqdm(chunk_files, desc="Scanning original graphs"):
        data_list = torch.load(chunk_path, map_location="cpu", weights_only=False)

        for graph in data_list:
            if global_idx in val_indices:
                val_graphs.append(graph)

            global_idx += 1

    return val_graphs


def load_shifted_val_graphs(fold_id):
    """Load shifted-domain validation graphs for one fold."""
    shifted_root = get_path("data", "shifted_graph_dir")
    shifted_dir = shifted_root / f"fold_{fold_id}"

    if not shifted_dir.exists():
        raise FileNotFoundError(f"Shifted graph directory not found: {shifted_dir}")

    chunk_files = sorted(glob.glob(str(shifted_dir / "*.pt")))

    shifted_graphs = []

    for chunk_path in tqdm(chunk_files, desc="Loading shifted graphs"):
        shifted_graphs.extend(torch.load(chunk_path, map_location="cpu", weights_only=False))

    return shifted_graphs


def load_stage_model(fold_id=2, device=DEVICE):
    """Load the final STAGE model for robustness evaluation."""
    label_map = load_label_map()

    model = StageModel(num_classes=len(label_map)).to(device)
    ckpt_path = get_path("outputs", "stage_checkpoint_dir") / f"best_model_fold_{fold_id}.pth"

    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    result = load_model_checkpoint(model, ckpt_path, map_location=device, strict=False)

    if result.missing_keys or result.unexpected_keys:
        print("[WARN] checkpoint key mismatch")
        print("missing:", result.missing_keys)
        print("unexpected:", result.unexpected_keys)

    model.eval()
    return model


def evaluate_graph_list(model, graphs, batch_size=64, device=DEVICE, desc="Evaluating"):
    """Evaluate a model on an in-memory graph list."""
    loader = DataLoader(graphs, batch_size=batch_size, shuffle=False, num_workers=0)

    y_true, y_pred = [], []

    with torch.no_grad():
        for batch in tqdm(loader, desc=desc):
            batch = batch.to(device)
            batch.x = torch.nan_to_num(batch.x.float(), nan=0.0)

            logits = model(batch)
            preds = logits.argmax(dim=1)

            y_true.extend(batch.y.cpu().numpy().tolist())
            y_pred.extend(preds.cpu().numpy().tolist())

    return calculate_metrics_dict(y_true, y_pred)


def evaluate_stage_robustness(
    fold_id=2,
    batch_size=64,
    save=True,
    original_graphs=None,
    shifted_graphs=None,
):
    """Evaluate STAGE on original and shifted-domain validation graphs."""
    model = load_stage_model(fold_id=fold_id)

    if original_graphs is None:
        original_graphs = load_original_val_graphs(fold_id)
    if shifted_graphs is None:
        shifted_graphs = load_shifted_val_graphs(fold_id)

    original = evaluate_graph_list(
        model,
        original_graphs,
        batch_size=batch_size,
        desc="Original",
    )
    shifted = evaluate_graph_list(
        model,
        shifted_graphs,
        batch_size=batch_size,
        desc="Shifted",
    )

    f1_drop = (original["F1_Macro"] - shifted["F1_Macro"]) / original["F1_Macro"] * 100.0

    rows = [
        {
            "Model": "STAGE",
            "Test Set": "Original",
            "F1-Score": original["F1_Macro"],
            "Recall": original["Recall"],
            "Precision": original["Precision"],
            "Accuracy": original["Accuracy"],
            "F1 Drop": "-",
        },
        {
            "Model": "STAGE",
            "Test Set": "Shifted",
            "F1-Score": shifted["F1_Macro"],
            "Recall": shifted["Recall"],
            "Precision": shifted["Precision"],
            "Accuracy": shifted["Accuracy"],
            "F1 Drop": f"{f1_drop:.2f}%",
        },
    ]

    df = pd.DataFrame(rows)

    if save:
        out_dir = get_path("outputs", "metrics_dir") / "robustness"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"stage_robustness_fold_{fold_id}.csv"
        df.to_csv(out_path, index=False)
        print(f"saved: {out_path}")

    return df


def load_baseline_model(name, fold_id=2, device=DEVICE):
    from stage.baselines.registry import build_baseline
    from stage.evaluation.baselines import get_baseline_checkpoint_path, load_baseline_checkpoint

    label_map = load_label_map()
    model = build_baseline(name, num_classes=len(label_map), in_channels=8).to(device)

    ckpt_path = get_baseline_checkpoint_path(name, fold_id)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    result = load_baseline_checkpoint(model, ckpt_path, device)

    if result.missing_keys or result.unexpected_keys:
        print(f"[WARN] baseline checkpoint key mismatch for {name}")
        print("missing:", result.missing_keys)
        print("unexpected:", result.unexpected_keys)

    model.eval()
    return model


def evaluate_baseline_robustness(
    name,
    fold_id=2,
    batch_size=64,
    save=False,
    original_graphs=None,
    shifted_graphs=None,
):
    model = load_baseline_model(name, fold_id=fold_id)

    if original_graphs is None:
        original_graphs = load_original_val_graphs(fold_id)
    if shifted_graphs is None:
        shifted_graphs = load_shifted_val_graphs(fold_id)

    original = evaluate_graph_list(
        model,
        original_graphs,
        batch_size=batch_size,
        desc=f"{name} Original",
    )
    shifted = evaluate_graph_list(
        model,
        shifted_graphs,
        batch_size=batch_size,
        desc=f"{name} Shifted",
    )

    f1_drop = (original["F1_Macro"] - shifted["F1_Macro"]) / original["F1_Macro"] * 100.0

    rows = [
        {
            "Model": name,
            "Fold": fold_id,
            "Test Set": "Original",
            "F1": original["F1_Macro"],
            "Rec": original["Recall"],
            "Prec": original["Precision"],
            "Acc": original["Accuracy"],
            "Drop%": "-",
        },
        {
            "Model": name,
            "Fold": fold_id,
            "Test Set": "Shifted",
            "F1": shifted["F1_Macro"],
            "Rec": shifted["Recall"],
            "Prec": shifted["Precision"],
            "Acc": shifted["Accuracy"],
            "Drop%": f"{f1_drop:.2f}%",
        },
    ]

    df = pd.DataFrame(rows)

    if save:
        out_dir = get_path("outputs", "metrics_dir") / "robustness"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{name}_robustness_fold_{fold_id}.csv"
        df.to_csv(out_path, index=False)
        print(f"saved: {out_path}")

    return df


def evaluate_baseline_robustness_all(
    names=None,
    fold_id=2,
    batch_size=64,
    save=True,
    original_graphs=None,
    shifted_graphs=None,
):
    from stage.baselines.registry import CANONICAL_BASELINE_NAMES

    if names is None:
        names = CANONICAL_BASELINE_NAMES

    if original_graphs is None:
        original_graphs = load_original_val_graphs(fold_id)
    if shifted_graphs is None:
        shifted_graphs = load_shifted_val_graphs(fold_id)

    dfs = []
    for name in names:
        dfs.append(
            evaluate_baseline_robustness(
                name=name,
                fold_id=fold_id,
                batch_size=batch_size,
                save=False,
                original_graphs=original_graphs,
                shifted_graphs=shifted_graphs,
            )
        )

    output_df = pd.concat(dfs, ignore_index=True)

    if save:
        out_dir = get_path("outputs", "metrics_dir") / "robustness"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"robustness_baselines_fold_{fold_id}.csv"
        output_df.to_csv(out_path, index=False)
        print(f"saved: {out_path}")

    return output_df
