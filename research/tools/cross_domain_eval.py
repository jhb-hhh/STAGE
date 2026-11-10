"""Evaluate ID-APT-trained checkpoints on target graphs.

Target graphs must use the ID-APT graph schema: eight flow-node features and
two edge features. Target labels are used only for reporting metrics.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch_geometric.loader import DataLoader
from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from stage.baselines.registry import build_baseline  # noqa: E402
from stage.data.baseline_protocol import GraphTargetDataset  # noqa: E402
from stage.data.dataset import IDAPTDataset  # noqa: E402
from stage.evaluation.metrics import calculate_metrics_dict, summarize_fold_metrics  # noqa: E402


PAPER_MODELS = {"TS-IDS", "FlowTransformer", "E-T-GraphSAGE", "STAGE"}
LEGACY_ROOT = Path("data/legacy")
SOURCE_GRAPH_DIR = LEGACY_ROOT / "output_v2" / "graph_data"
SOURCE_SPLIT_DIR = LEGACY_ROOT / "output_v2" / "checkpoints" / "kfold_indices"


def checkpoint_path(model: str, fold: int) -> Path:
    if model in {"TS-IDS", "FlowTransformer", "E-T-GraphSAGE"}:
        return LEGACY_ROOT / "src_v2_core" / "baselines_compare" / "outputs" / f"{model}_fold_{fold}.pth"
    if model == "STAGE":
        return (
            LEGACY_ROOT
            / "output_v3"
            / "ablation_exp"
            / "kfold_experiment_stage_final"
            / f"best_model_fold_{fold}.pth"
        )
    return ROOT / "outputs" / "workspace" / "checkpoints" / "baselines" / f"{model}_fold_{fold}.pth"


def _attach_unknown_target_identities(batch):
    """Supply shortcut-model identity fields without inventing target mappings.

    Cross-domain IPs are outside the source vocabulary, so both endpoint IDs use
    the source model's unknown/fallback index 0.  The converted target artifacts
    retain the protocol/flag and capped port indices, which are reused here.
    """
    batch.sport_idx_raw = batch.sport_idx.long()
    batch.dport_idx_raw = batch.dport_idx.long()
    batch.src_ip_idx_global = torch.zeros_like(batch.proto_idx, dtype=torch.long)
    batch.dst_ip_idx_global = torch.zeros_like(batch.proto_idx, dtype=torch.long)
    return batch


def load_model(model_name: str, fold: int, device):
    path = checkpoint_path(model_name, fold)
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if model_name in PAPER_MODELS:
        if model_name == "STAGE":
            sys.path.insert(0, str(LEGACY_ROOT))
            from src_v3_core.ablation_exp.model_stage_final import StageFinalModel

            model = StageFinalModel(num_classes=13).to(device)
        else:
            model = build_baseline(
                model_name, num_classes=13, in_channels=8, edge_dim=2
            ).to(device)
        model.load_state_dict(checkpoint, strict=True)
        model.eval()
        return model, checkpoint

    if int(checkpoint["in_channels"]) != 8 or int(checkpoint["edge_dim"]) != 2:
        raise ValueError(f"{path} is not an ID-APT-schema checkpoint")

    model = build_baseline(
        model_name,
        num_classes=int(checkpoint["num_classes"]),
        in_channels=8,
        edge_dim=2,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint


def collect_scores(
    model_name,
    model,
    dataset,
    device,
    batch_size,
    indices=None,
    source=False,
    use_penultimate=False,
):
    adapted = GraphTargetDataset(dataset, task="binary", indices=indices)
    effective_batch = 1 if model_name in {"FlowTransformer", "E-T-GraphSAGE"} else batch_size
    if model_name == "GraphIDS":
        effective_batch = min(effective_batch, 4)
    loader = DataLoader(adapted, batch_size=effective_batch, shuffle=False, num_workers=0)
    true, scores, legacy, features = [], [], [], []
    captured = []
    hook = None
    if use_penultimate:
        if model_name == "GraphIDS":
            raise ValueError("GraphIDS does not expose a classification penultimate layer")
        final_linear = [module for module in model.modules() if isinstance(module, torch.nn.Linear)][-1]
        hook = final_linear.register_forward_pre_hook(
            lambda _module, inputs: captured.append(inputs[0].detach())
        )
    with torch.no_grad():
        for batch in tqdm(loader, desc=f"{model_name} {'source-val' if source else 'target'}"):
            batch = batch.to(device)
            batch.x = torch.nan_to_num(batch.x.float(), nan=0.0)
            if model_name == "GraphIDS":
                score = model.anomaly_score(batch).view(-1)
                legacy_pred = torch.zeros_like(score, dtype=torch.long)
                binary_features = score[:, None]
            else:
                # A binary attack score is better defined as total posterior
                # mass outside the benign class than as a 13-way argmax.  The
                # latter lets twelve attack logits collectively dominate an
                # unseen-domain benign sample.
                logits = model(batch)
                score = 1.0 - torch.softmax(logits, dim=-1)[:, 0]
                legacy_pred = (logits.argmax(dim=-1) != 0).long()
                binary_features = captured.pop(0) if use_penultimate else logits
            true.extend(batch.graph_y.view(-1).cpu().tolist())
            scores.extend(score.detach().cpu().tolist())
            legacy.extend(legacy_pred.detach().cpu().tolist())
            features.append(binary_features.detach().cpu().numpy())
    if hook is not None:
        hook.remove()
    return (
        np.asarray(true, dtype=np.int64),
        np.asarray(scores, dtype=np.float64),
        np.asarray(legacy, dtype=np.int64),
        np.concatenate(features, axis=0),
    )


def select_source_threshold(labels, scores, steps=1001):
    """Maximize binary macro-F1 on ID-APT validation only.

    Target labels never enter threshold selection.  Ties prefer the value
    closest to 0.5 for deterministic, conservative calibration.
    """
    lo, hi = float(scores.min()), float(scores.max())
    candidates = np.linspace(lo, hi, steps) if hi > lo else np.asarray([lo])
    ranked = []
    for threshold in candidates:
        value = f1_score(labels, scores >= threshold, average="macro", zero_division=0)
        ranked.append((value, -abs(float(threshold) - 0.5), float(threshold)))
    return max(ranked)[2]


def evaluate_fold(
    model_name: str,
    fold: int,
    dataset,
    device,
    batch_size: int,
    decision: str,
    source_dataset=None,
    source_thresholds=None,
):
    model, checkpoint = load_model(model_name, fold, device)
    threshold = None
    if decision == "source-calibrated":
        if source_thresholds is not None:
            threshold = float(source_thresholds[fold])
        else:
            if source_dataset is None:
                raise ValueError("source_dataset is required for source calibration")
            with open(SOURCE_SPLIT_DIR / f"fold_{fold}.json", "r") as handle:
                val_indices = json.load(handle)["val"]
            source_true, source_scores, _, _ = collect_scores(
                model_name, model, source_dataset, device, batch_size, indices=val_indices, source=True
            )
            if model_name == "GraphIDS" and "threshold" in checkpoint:
                # GraphIDS already stores its source-validation anomaly threshold.
                threshold = float(checkpoint["threshold"])
            else:
                threshold = select_source_threshold(source_true, source_scores)

    binary_adapter = None
    if decision in {"source-logistic", "source-embedding-logistic"}:
        if source_dataset is None:
            raise ValueError("source_dataset is required for the source binary adapter")
        with open(SOURCE_SPLIT_DIR / f"fold_{fold}.json", "r") as handle:
            val_indices = json.load(handle)["val"]
        source_true, _, _, source_features = collect_scores(
            model_name,
            model,
            source_dataset,
            device,
            batch_size,
            indices=val_indices,
            source=True,
            use_penultimate=decision == "source-embedding-logistic",
        )
        binary_adapter = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                class_weight="balanced", max_iter=1000, random_state=42
            ),
        )
        binary_adapter.fit(source_features, source_true)

    true, scores, legacy, target_features = collect_scores(
        model_name,
        model,
        dataset,
        device,
        batch_size,
        use_penultimate=decision == "source-embedding-logistic",
    )
    if decision == "legacy-argmax":
        if model_name == "GraphIDS":
            output = scores > float(checkpoint["threshold"])
        else:
            output = legacy
    elif decision == "binary-posterior":
        output = scores >= 0.5
    elif decision in {"source-logistic", "source-embedding-logistic"}:
        scores = binary_adapter.predict_proba(target_features)[:, 1]
        output = scores >= 0.5
    else:
        output = scores >= threshold
    metrics = calculate_metrics_dict(true, output.astype(np.int64))
    if np.unique(true).size == 2:
        metrics["AUROC"] = float(roc_auc_score(true, scores))
    if len(true) % 2 == 0 and np.all(true[0::2] == 0) and np.all(true[1::2] == 1):
        paired_delta = scores[1::2] - scores[0::2]
        metrics["PairedScoreDelta"] = float(paired_delta.mean())
        metrics["PairwiseWinRate"] = float((paired_delta > 0).mean())
    predictions = pd.DataFrame(
        {
            "Fold": fold,
            "True": true,
            "Score": scores,
            "Prediction": output.astype(np.int64),
        }
    )
    return metrics, threshold, predictions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        choices=("DIDS-MFL", "GraphIDS", "TS-IDS", "FlowTransformer", "E-T-GraphSAGE", "STAGE"),
        required=True,
    )
    parser.add_argument(
        "--threshold-file",
        type=Path,
        help="Reuse Fold/Threshold values previously calibrated on ID-APT.",
    )
    parser.add_argument("--target", required=True)
    parser.add_argument("--graph-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "workspace" / "metrics" / "cross_domain")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--fold",
        type=int,
        choices=(-1, 0, 1, 2, 3, 4),
        default=-1,
        help="Evaluate all folds (-1) or one fold for a fast feasibility check.",
    )
    parser.add_argument(
        "--decision",
        choices=(
            "legacy-argmax",
            "binary-posterior",
            "source-calibrated",
            "source-logistic",
            "source-embedding-logistic",
        ),
        default="legacy-argmax",
        help="Calibrated mode selects each fold threshold on ID-APT validation only.",
    )
    args = parser.parse_args()

    dataset = IDAPTDataset(root_dir=args.graph_dir)
    sample = dataset[0]
    observed = (int(sample.x.size(-1)), int(sample.edge_attr.size(-1)))
    if observed != (8, 2):
        raise ValueError(f"Target graph schema {observed}, expected (8, 2)")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    source_thresholds = None
    if args.threshold_file:
        threshold_frame = pd.read_csv(args.threshold_file)
        source_thresholds = {
            int(row.Fold): float(row.Threshold)
            for row in threshold_frame.itertuples(index=False)
        }
    source_dataset = None
    if (
        args.decision
        in {"source-calibrated", "source-logistic", "source-embedding-logistic"}
        and source_thresholds is None
    ):
        source_dataset = IDAPTDataset(root_dir=SOURCE_GRAPH_DIR)
    rows = []
    prediction_frames = []
    fold_ids = range(5) if args.fold == -1 else [args.fold]
    for fold in fold_ids:
        metrics, threshold, predictions = evaluate_fold(
            args.model,
            fold,
            dataset,
            device,
            args.batch_size,
            args.decision,
            source_dataset,
            source_thresholds,
        )
        rows.append({"Fold": fold, "Decision": args.decision, "Threshold": threshold, **metrics})
        prediction_frames.append(predictions)

    summary = summarize_fold_metrics(
        [
            {k: v for k, v in r.items() if k not in {"Fold", "Decision", "Threshold"}}
            for r in rows
        ]
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    decision_suffix = args.decision.replace("-", "_")
    stem = f"idapt_to_{args.target.lower().replace('+', '_').replace('-', '_')}_{args.model.lower().replace('-', '_')}_{decision_suffix}"
    if args.fold != -1:
        stem += f"_fold_{args.fold}"
    pd.DataFrame(rows).to_csv(args.output_dir / f"{stem}_folds.csv", index=False)
    pd.concat(prediction_frames, ignore_index=True).to_csv(
        args.output_dir / f"{stem}_predictions.csv", index=False
    )
    pd.DataFrame([{"Model": args.model, "Source": "ID-APT", "Target": args.target, **summary}]).to_csv(
        args.output_dir / f"{stem}_summary.csv", index=False
    )
    print(pd.DataFrame([summary]).to_string(index=False))


if __name__ == "__main__":
    main()
