"""Endogenous evidence extraction for STAGE."""

import heapq
import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from stage.data.dataset import IDAPTDataset
from stage.models.stage import StageModel
from stage.training.checkpoint import load_model_checkpoint
from stage.utils.config import get_path

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("STAGE_EndogenousExplainer")


class EndogenousExplainer:
    """Extract node-level evidence from internal STAGE signals."""

    def __init__(self, model_path=None, load_dataset=True):
        self.device = torch.device(DEVICE)
        self.output_dir = get_path("outputs", "explanation_dir")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        label_map_path = get_path("data", "label_map")
        if not label_map_path.exists():
            raise FileNotFoundError(f"Label map not found: {label_map_path}")

        with open(label_map_path, "r", encoding="utf-8") as file:
            self.label_map = json.load(file)
            self.idx_to_label = {value: key for key, value in self.label_map.items()}

        if model_path is None:
            model_path = self._auto_find_model_path()

        self.model = self._load_model(model_path)
        self.dataset = IDAPTDataset() if load_dataset else None

    def _auto_find_model_path(self):
        """Return the first available reference STAGE checkpoint path."""
        checkpoint_dir = get_path("outputs", "stage_checkpoint_dir")

        for fold_id in [2, 0, 1, 3, 4]:
            path = checkpoint_dir / f"best_model_fold_{fold_id}.pth"
            if path.exists():
                return path

        raise FileNotFoundError(f"STAGE checkpoint not found in {checkpoint_dir}.")

    def _load_model(self, model_path):
        """Load the STAGE model used for endogenous explanation."""
        logger.info(f"Loading STAGE model: {model_path}")

        model = StageModel(num_classes=len(self.label_map))

        try:
            load_result = load_model_checkpoint(model, model_path, map_location=self.device, strict=False)

            if load_result.missing_keys:
                logger.warning(f"Missing checkpoint keys: {load_result.missing_keys}")
            if load_result.unexpected_keys:
                logger.warning(f"Unexpected checkpoint keys: {load_result.unexpected_keys}")

            logger.info("STAGE model loaded successfully.")
        except Exception as exc:
            logger.error(f"Model loading failed: {exc}")
            raise

        model.to(self.device)
        model.eval()

        return model

    def explain(self, data):
        """Compute SAGA node-importance scores from internal STAGE signals."""
        data = data.to(self.device)
        self.model.eval()

        try:
            with torch.no_grad():
                _, internals = self.model(data, return_explanation=True)

                p_uni = internals["final_attn"].squeeze().detach().cpu().numpy()
                if p_uni.ndim == 0:
                    p_uni = np.array([p_uni])

                w_spa_tensor = internals["gate_spa"].detach().cpu().numpy()
                w_tem_tensor = internals["gate_tem"].detach().cpu().numpy()

                if w_spa_tensor.ndim > 1:
                    g_spa = w_spa_tensor.mean(axis=1)
                    g_tem = w_tem_tensor.mean(axis=1)
                else:
                    g_spa = w_spa_tensor.mean()
                    g_tem = w_tem_tensor.mean()

                if isinstance(g_spa, float) or getattr(g_spa, "size", 1) == 1:
                    g_spa = np.ones_like(p_uni) * g_spa
                    g_tem = np.ones_like(p_uni) * g_tem

                saga_scores = (g_spa + g_tem) * np.log1p(p_uni)
                bias_ratio = g_spa.mean() / (g_spa.mean() + g_tem.mean() + 1e-9)

            top_k = min(5, len(saga_scores))
            top_k_indices = saga_scores.argsort()[-top_k:][::-1]

            return top_k_indices, saga_scores, len(saga_scores), bias_ratio

        except Exception as exc:
            logger.error(f"Explanation failed: {exc}")
            return [], np.zeros(data.num_nodes), 0, 0.5

    @staticmethod
    def _as_list(value):
        """Normalize PyG metadata fields that may be wrapped by a batch dimension."""
        if isinstance(value, list) and len(value) == 1 and isinstance(value[0], (list, tuple)):
            return list(value[0])

        if isinstance(value, tuple):
            return list(value)

        return value

    def generate_report(self, data, top_k_indices, scores, confidence=None, bias_ratio=None):
        """Generate a JSON evidence report for selected nodes."""
        raw_filename = data.filename[0] if isinstance(getattr(data, "filename", None), list) else getattr(data, "filename", "unknown")
        label_idx = data.y.item()
        true_label = self.idx_to_label.get(label_idx, "Unknown")

        report = {
            "meta": {
                "filename": raw_filename,
                "confidence": confidence,
                "bias_ratio": float(bias_ratio) if bias_ratio is not None else None,
            },
            "timeline_evidence": [],
        }

        if hasattr(data, "src_ip_list"):
            src_ips = self._as_list(data.src_ip_list)
            dst_ips = self._as_list(data.dst_ip_list)
            start_times = self._as_list(getattr(data, "start_time_list", [0.0] * len(src_ips)))
            durations = self._as_list(getattr(data, "raw_duration_list", [0.0] * len(src_ips)))
            protos = self._as_list(getattr(data, "raw_proto_list", [0] * len(src_ips)))
            src_ports = self._as_list(getattr(data, "raw_src_port_list", [0] * len(src_ips)))
            dst_ports = self._as_list(getattr(data, "raw_dst_port_list", [0] * len(src_ips)))
            pkt_counts = self._as_list(getattr(data, "raw_pkt_count_list", [0] * len(src_ips)))
            byte_counts = self._as_list(getattr(data, "raw_byte_count_list", [0] * len(src_ips)))
            flags = self._as_list(getattr(data, "raw_tcp_flags_list", [""] * len(src_ips)))
            edge_index = data.edge_index.cpu().numpy()

            for rank, idx in enumerate(top_k_indices):
                idx = int(idx)

                if idx >= len(src_ips):
                    continue

                neighbors = edge_index[1][edge_index[0] == idx]
                valid_neighbors = [
                    node_id
                    for node_id in np.unique(neighbors)
                    if node_id < len(src_ips) and node_id != idx
                ]
                context = [
                    f"Connected to {src_ips[node_id]} (Node {node_id})"
                    for node_id in valid_neighbors[:3]
                ] or ["Isolated Node"]

                report["timeline_evidence"].append(
                    {
                        "rank": rank + 1,
                        "timestamp": f"{float(start_times[idx]):.2f}",
                        "duration": f"{float(durations[idx]):.4f}s",
                        "source": f"{src_ips[idx]}:{src_ports[idx]}",
                        "destination": f"{dst_ips[idx]}:{dst_ports[idx]}",
                        "protocol": str(protos[idx]),
                        "flags": str(flags[idx]),
                        "packet_count": int(pkt_counts[idx]),
                        "byte_count": int(byte_counts[idx]),
                        "importance_score": float(scores[idx]),
                        "topological_context": context,
                    }
                )

        save_name = f"[{true_label.replace('/', '_')}]_report_{raw_filename}.json"
        save_path = self.output_dir / save_name

        with open(save_path, "w", encoding="utf-8") as file:
            json.dump(report, file, indent=4, ensure_ascii=False)

        logger.info(f"Report saved: {save_name}")

        return report

    def visualize(self, data, scores, valid_len, bias_ratio):
        """Save a bar chart of normalized SAGA node scores."""
        plt.figure(figsize=(10, 4))

        color = "tab:red" if bias_ratio > 0.5 else "tab:blue"
        label_str = self.idx_to_label.get(data.y.item(), "Unknown")
        display_scores = scores[:valid_len]

        if len(display_scores) > 0 and display_scores.max() > 0:
            display_scores = display_scores / display_scores.max()

        plt.bar(np.arange(valid_len), display_scores, color=color, alpha=0.8)
        plt.title(f"SAGA Attribution: {label_str} (Spatial Bias: {bias_ratio:.2f})")
        plt.xlabel("Flow Index")
        plt.ylabel("Normalized SAGA Score")

        filename = data.filename[0] if isinstance(getattr(data, "filename", None), list) else getattr(data, "filename", "unknown")
        plt.savefig(self.output_dir / f"vis_{filename}.png")
        plt.close()


def main():
    """Generate reports and visualizations for high-confidence malicious samples."""
    explainer = EndogenousExplainer()
    loader = DataLoader(explainer.dataset, batch_size=1, shuffle=False)

    top_candidates = []
    top_k = 5
    pbar = tqdm(total=len(explainer.dataset), desc="Scanning")

    for index, batch in enumerate(loader):
        pbar.update(1)

        if batch.y.item() == 0:
            continue

        with torch.no_grad():
            batch_gpu = batch.to(explainer.device)
            batch_gpu.x = torch.nan_to_num(batch_gpu.x.float(), nan=0.0)
            logits = explainer.model(batch_gpu)
            probs = torch.softmax(logits, dim=1)
            conf, pred = probs.max(dim=1)

        if pred.item() == batch.y.item():
            current_conf = conf.item()
            batch_cpu = batch.clone().cpu()
            record = (current_conf, index, batch_cpu)

            if len(top_candidates) < top_k:
                heapq.heappush(top_candidates, record)
            elif current_conf > top_candidates[0][0]:
                heapq.heapreplace(top_candidates, record)

    pbar.close()
    top_candidates.sort(key=lambda item: item[0], reverse=True)

    logger.info(f"Generating reports for {len(top_candidates)} selected samples.")

    for confidence, _, batch_data in top_candidates:
        top_k_indices, scores, valid_len, bias = explainer.explain(batch_data)
        explainer.generate_report(batch_data, top_k_indices, scores, confidence, bias)
        explainer.visualize(batch_data, scores, valid_len, bias)

    logger.info("Report generation completed.")


NIDSExplainer = EndogenousExplainer


if __name__ == "__main__":
    main()
