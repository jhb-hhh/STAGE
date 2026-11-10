"""K-sensitivity analysis for STAGE explanation results."""

import json
import logging
import os
import random
import time
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Subset
from torch_geometric.data import Batch
from torch_geometric.explain import Explainer, GNNExplainer, PGExplainer
from torch_geometric.loader import DataLoader
from torch_geometric.nn import MessagePassing
from tqdm import tqdm

from stage.data.dataset import IDAPTDataset
from stage.explanation.endogenous import EndogenousExplainer
from stage.models.stage import StageModel
from stage.training.checkpoint import load_model_checkpoint
from stage.utils.config import get_path

warnings.filterwarnings("ignore")

try:
    import shap
except ImportError:
    shap = None

try:
    from lime import lime_tabular
except ImportError:
    lime_tabular = None

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LABEL_FILE_PATH = str(get_path("data", "label_file"))
LABEL_MAP_PATH = str(get_path("data", "label_map"))
RESULTS_DIR = str(get_path("outputs", "metrics_dir") / "sensitivity")
os.makedirs(RESULTS_DIR, exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("STAGE_ExplanationSensitivity")

if shap is not None:
    logging.getLogger("shap").setLevel(logging.WARNING)


class SenseConfig:
    """Default configuration for K-sensitivity analysis."""

    NUM_SAMPLES = 200
    K_LIST = [1, 3, 5, 8, 10, 15, 20, 30, 50]
    ROBUST_ALPHA1 = 0.5
    ROBUST_ALPHA2 = 0.2
    PERTURBATION_SAMPLES = 50
    MODEL_PATH = str(get_path("outputs", "stage_checkpoint_dir") / "stage_reference_checkpoint.pth")
    PGEXPLAINER_WEIGHT_PATH = str(get_path("outputs", "explainer_checkpoint_dir") / "pgexplainer_weights.pt")


def set_seed(seed=42):
    """Set random seeds for reproducible sensitivity analysis."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def clear_pyg_state(model):
    """Reset PyG explanation state and remove residual masks."""
    for module in model.modules():
        if isinstance(module, MessagePassing):
            module.explain = False
            module._edge_mask = None
            module._loop_mask = None
            module._apply_sigmoid = True


def target_to_int(target):
    """Convert a target tensor or scalar to a Python integer."""
    if hasattr(target, "item"):
        return int(target.item())

    return int(target)


class HybridModelWrapper(nn.Module):
    """Wrap STAGE so PyG explainers can receive tensor-level inputs."""

    def __init__(self, model):
        super().__init__()
        self.model = model
        self.context = None

    def set_context(self, data):
        """Store non-feature graph fields required by the wrapped model."""
        self.context = {
            "proto_idx": data.proto_idx,
            "sport_idx": data.sport_idx,
            "dport_idx": data.dport_idx,
            "flags_idx": data.flags_idx,
            "edge_attr": data.edge_attr,
            "batch": data.batch if hasattr(data, "batch") else torch.zeros(
                data.x.size(0),
                dtype=torch.long,
                device=data.x.device,
            ),
        }

    def forward(self, x, edge_index, edge_attr=None, batch=None, **kwargs):
        """Reconstruct a PyG batch and run the wrapped STAGE model."""
        if self.context is None:
            raise RuntimeError("Context is not set.")

        current_batch = batch if batch is not None else self.context["batch"]
        current_edge_attr = edge_attr if edge_attr is not None else self.context["edge_attr"]

        if current_edge_attr is not None and edge_index.size(1) != current_edge_attr.size(0):
            current_edge_attr = torch.zeros(
                (edge_index.size(1), current_edge_attr.size(1)),
                device=x.device,
            )

        reconstructed_data = Batch(
            x=x,
            edge_index=edge_index,
            edge_attr=current_edge_attr,
            batch=current_batch,
            proto_idx=self.context["proto_idx"],
            sport_idx=self.context["sport_idx"],
            dport_idx=self.context["dport_idx"],
            flags_idx=self.context["flags_idx"],
        )

        return self.model(reconstructed_data)


class GroundTruthManager:
    """Create node-level malicious-flow masks from label metadata."""

    def __init__(self, label_file_path):
        self.file_index = {}

        if not os.path.exists(label_file_path):
            return

        try:
            self.df = pd.read_csv(label_file_path)
            self.file_index = self.df.set_index("filename").to_dict("index")
        except Exception as exc:
            logger.error(f"Failed to read label CSV: {exc}")

    @staticmethod
    def _normalize_filename(filename):
        if isinstance(filename, list):
            filename = filename[0]

        return os.path.basename(filename) if filename else ""

    @staticmethod
    def _parse_malicious_ips(value):
        value = str(value)

        if not value or value.lower() in {"nan", "n/a", "none", ""}:
            return set()

        value = value.replace("'", "").replace('"', "").replace("[", "").replace("]", "")

        return {ip.strip() for ip in value.split(",") if ip.strip()}

    def get_node_ground_truth(self, data):
        """Return a binary mask over nodes that involve known malicious IPs."""
        filename = self._normalize_filename(getattr(data, "filename", None))
        record = self.file_index.get(filename)

        if record is None:
            return torch.zeros(data.num_nodes, device=DEVICE)

        malicious_ips = self._parse_malicious_ips(record.get("malicious_ips", ""))
        gt_mask = torch.zeros(data.num_nodes, device=DEVICE)

        if not malicious_ips:
            return gt_mask

        if hasattr(data, "src_ip_list") and hasattr(data, "dst_ip_list"):
            src_list = data.src_ip_list
            dst_list = data.dst_ip_list

            if src_list and isinstance(src_list[0], list):
                src_list = src_list[0]
            if dst_list and isinstance(dst_list[0], list):
                dst_list = dst_list[0]

            for index in range(min(len(src_list), data.num_nodes)):
                if src_list[index] in malicious_ips or dst_list[index] in malicious_ips:
                    gt_mask[index] = 1.0

        return gt_mask


class StageExplainerAdapter:
    """Adapter for STAGE endogenous explanation scores."""

    def __init__(self, model):
        self.explainer_instance = EndogenousExplainer(model_path=None, load_dataset=False)
        self.explainer_instance.model = model

    def explain(self, data, target=None):
        """Return node-level STAGE endogenous explanation scores."""
        _, saga_scores_cpu, _, _ = self.explainer_instance.explain(data)
        scores_tensor = torch.from_numpy(saga_scores_cpu).float().to(DEVICE)

        if len(scores_tensor) < data.num_nodes:
            padding = torch.zeros(data.num_nodes - len(scores_tensor), device=DEVICE)
            scores_tensor = torch.cat([scores_tensor, padding])

        return scores_tensor


class GNNExplainerWrapper:
    """PyG GNNExplainer adapter."""

    def __init__(self, model):
        self.wrapper = HybridModelWrapper(model)
        self.explainer = Explainer(
            model=self.wrapper,
            algorithm=GNNExplainer(epochs=30),
            explanation_type="model",
            node_mask_type="attributes",
            edge_mask_type=None,
            model_config={
                "mode": "multiclass_classification",
                "task_level": "graph",
                "return_type": "raw",
            },
        )

    def explain(self, data, target=None):
        """Return node-importance scores from GNNExplainer."""
        self.wrapper.set_context(data)
        old_cudnn = torch.backends.cudnn.enabled
        torch.backends.cudnn.enabled = False
        x_isolated = data.x.detach().clone().requires_grad_(True)

        try:
            with torch.enable_grad():
                explanation = self.explainer(
                    x=x_isolated,
                    edge_index=data.edge_index,
                    target=target,
                )
        finally:
            torch.backends.cudnn.enabled = old_cudnn

        node_imp = torch.zeros(data.num_nodes, device=DEVICE)

        if explanation.node_mask is not None:
            node_imp += explanation.node_mask.mean(dim=1)

        return node_imp / (node_imp.max() + 1e-9)


class PGExplainerWrapper:
    """PyG PGExplainer adapter."""

    def __init__(self, model, weight_path):
        self.wrapper = HybridModelWrapper(model)
        algorithm = PGExplainer(epochs=10, lr=0.001)
        self.explainer = Explainer(
            model=self.wrapper,
            algorithm=algorithm,
            explanation_type="phenomenon",
            node_mask_type=None,
            edge_mask_type="object",
            model_config={
                "mode": "multiclass_classification",
                "task_level": "graph",
                "return_type": "raw",
            },
        )

        if os.path.exists(weight_path):
            state_dict = torch.load(weight_path, map_location=DEVICE)

            if "model_state_dict" in state_dict:
                state_dict = state_dict["model_state_dict"]

            state_dict = {key.replace("module.", ""): value for key, value in state_dict.items()}
            self.explainer.algorithm.load_state_dict(state_dict)
            self.explainer.algorithm.to(DEVICE)
            self.explainer.algorithm.mlp.eval()
            self.explainer.algorithm._curr_epoch = 100
        else:
            logger.warning(f"PGExplainer weights not found at {weight_path}")

    def explain(self, data, target=None):
        """Return node-importance scores aggregated from PGExplainer edge masks."""
        self.wrapper.set_context(data)
        old_cudnn = torch.backends.cudnn.enabled
        torch.backends.cudnn.enabled = False

        try:
            with torch.enable_grad():
                explanation = self.explainer(
                    x=data.x,
                    edge_index=data.edge_index,
                    target=target,
                    edge_attr=data.edge_attr,
                )
        finally:
            torch.backends.cudnn.enabled = old_cudnn

        node_imp = torch.zeros(data.num_nodes, device=DEVICE)

        if explanation.edge_mask is not None:
            row, col = data.edge_index
            node_imp.scatter_add_(0, row, explanation.edge_mask)
            node_imp.scatter_add_(0, col, explanation.edge_mask)

        return node_imp / (node_imp.max() + 1e-9)


class ShapExplainerWrapper:
    """SHAP adapter for node-mask attribution."""

    def __init__(self, model):
        self.model = model

    def explain(self, data, target=None):
        """Return node-importance scores from SHAP KernelExplainer."""
        if shap is None:
            return torch.zeros(data.num_nodes, device=DEVICE)

        target_index = target_to_int(target)
        num_nodes = data.num_nodes

        def predict_fn(masks):
            probabilities = []

            for index in range(masks.shape[0]):
                mask = torch.from_numpy(masks[index]).to(DEVICE).float()
                data_masked = data.clone()
                data_masked.x = data.x * mask.unsqueeze(1)

                with torch.no_grad():
                    probs = torch.softmax(self.model(data_masked), dim=1)
                    probabilities.append(probs[0, target_index].item())

            return np.array(probabilities)

        background = np.zeros((1, num_nodes))
        explainer = shap.KernelExplainer(predict_fn, background)
        shap_values = explainer.shap_values(
            np.ones((1, num_nodes)),
            nsamples=SenseConfig.PERTURBATION_SAMPLES,
            silent=True,
        )

        if isinstance(shap_values, list):
            node_imp = np.abs(shap_values[target_index][0])
        else:
            node_imp = np.abs(shap_values[0])

        node_imp = torch.from_numpy(node_imp.flatten()).to(DEVICE)

        if node_imp.max() > 1e-9:
            node_imp = node_imp / node_imp.max()

        return node_imp


class LimeExplainerWrapper:
    """LIME adapter for node-mask attribution."""

    def __init__(self, model):
        self.model = model

    def explain(self, data, target=None):
        """Return node-importance scores from LIME."""
        if lime_tabular is None:
            return torch.zeros(data.num_nodes, device=DEVICE)

        target_index = target_to_int(target)
        num_nodes = data.num_nodes

        def predict_fn(features_array):
            probabilities = []

            for index in range(features_array.shape[0]):
                mask = torch.from_numpy(features_array[index]).to(DEVICE).float()
                data_masked = data.clone()
                data_masked.x = data.x * mask.unsqueeze(1)

                with torch.no_grad():
                    probabilities.append(torch.softmax(self.model(data_masked), dim=1).cpu().numpy()[0])

            return np.array(probabilities)

        explainer = lime_tabular.LimeTabularExplainer(
            training_data=np.ones((1, num_nodes)),
            mode="classification",
            feature_names=[str(index) for index in range(num_nodes)],
            discretize_continuous=False,
            verbose=False,
        )
        explanation = explainer.explain_instance(
            data_row=np.ones(num_nodes),
            predict_fn=predict_fn,
            labels=(target_index,),
            num_features=num_nodes,
            num_samples=SenseConfig.PERTURBATION_SAMPLES,
        )

        node_imp = torch.zeros(num_nodes, device=DEVICE)

        for feature_idx, weight in explanation.as_map()[target_index]:
            if weight > 0:
                node_imp[feature_idx] = weight

        if node_imp.max() > 1e-9:
            node_imp = node_imp / node_imp.max()

        return node_imp


def calculate_metrics_for_k(node_mask, gt_mask, model, data, k, target_class):
    """Compute explanation metrics for one K value."""
    scores = node_mask.detach().cpu().numpy()
    labels = gt_mask.detach().cpu().numpy()
    real_k = min(k, len(scores))

    if real_k == 0:
        return {}

    top_k_indices = np.argsort(scores)[-real_k:]
    true_indices = np.where(labels == 1)[0]

    if len(true_indices) > 0:
        intersection = len(set(top_k_indices) & set(true_indices))
        union = len(set(top_k_indices) | set(true_indices))
        iou = intersection / union
        precision = intersection / real_k
    else:
        iou = 0.0
        precision = 0.0

    with torch.no_grad():
        original_prob = torch.softmax(model(data), dim=1)[0, target_class].item()

    def get_masked_prob(indices_to_zero):
        data_temp = data.clone()
        data_temp.x = data.x.clone()
        data_temp.x[indices_to_zero] = 0.0

        with torch.no_grad():
            prob = torch.softmax(model(data_temp), dim=1)[0, target_class].item()

        return prob

    fid_plus = original_prob - get_masked_prob(top_k_indices)
    all_indices = np.arange(data.num_nodes)
    remove_others = np.setdiff1d(all_indices, top_k_indices)
    fid_minus = original_prob - get_masked_prob(remove_others)

    if len(top_k_indices) > 0:
        num_drop = int(len(top_k_indices) * SenseConfig.ROBUST_ALPHA1)
        drop_idx = np.random.choice(top_k_indices, num_drop, replace=False)
        rob_fid_plus = original_prob - get_masked_prob(drop_idx)
    else:
        rob_fid_plus = fid_plus

    background_indices = np.setdiff1d(all_indices, top_k_indices)

    if len(background_indices) > 0:
        num_keep = int(len(background_indices) * SenseConfig.ROBUST_ALPHA2)
        background_keep = np.random.choice(background_indices, num_keep, replace=False)
        keep_total = np.concatenate([top_k_indices, background_keep])
        mask_rob_minus = np.setdiff1d(all_indices, keep_total)
        rob_fid_minus = original_prob - get_masked_prob(mask_rob_minus)
    else:
        rob_fid_minus = fid_minus

    return {
        "precision": precision,
        "iou": iou,
        "fid+": fid_plus,
        "fid-": fid_minus,
        "rob_fid+": rob_fid_plus,
        "rob_fid-": rob_fid_minus,
    }


def calculate_consistency_overlap(mask_orig, mask_pert, k):
    """Compute Top-K overlap between original and perturbed explanations."""
    scores1 = mask_orig.detach().cpu().numpy()
    scores2 = mask_pert.detach().cpu().numpy()
    real_k = min(k, len(scores1))

    if real_k == 0:
        return 0.0

    topk1 = set(np.argsort(scores1)[-real_k:])
    topk2 = set(np.argsort(scores2)[-real_k:])
    union = len(topk1 | topk2)

    return len(topk1 & topk2) / union if union > 0 else 0.0


def load_stage_model(label_map):
    """Load the STAGE checkpoint used by sensitivity analysis."""
    model = StageModel(num_classes=len(label_map)).to(DEVICE)

    if not os.path.exists(SenseConfig.MODEL_PATH):
        raise FileNotFoundError(f"Model not found at {SenseConfig.MODEL_PATH}")

    load_result = load_model_checkpoint(
        model,
        SenseConfig.MODEL_PATH,
        map_location=DEVICE,
        strict=False,
    )

    if load_result.missing_keys or load_result.unexpected_keys:
        logger.warning(f"Missing checkpoint keys: {load_result.missing_keys}")
        logger.warning(f"Unexpected checkpoint keys: {load_result.unexpected_keys}")

    model.eval()

    return model


def make_validation_loader(dataset):
    """Create the validation loader used for sensitivity analysis."""
    split_path = get_path("data", "split_dir") / "fold_2.json"

    if split_path.exists():
        with open(split_path, "r", encoding="utf-8") as file:
            val_indices = json.load(file)["val"]

        val_subset = Subset(dataset, val_indices)
        logger.info("Evaluation is restricted to fold 2 validation samples.")

        return DataLoader(val_subset, batch_size=1, shuffle=True)

    logger.warning("Fold 2 validation indices were not found; sampling from the full dataset.")

    return DataLoader(dataset, batch_size=1, shuffle=True)


def build_explainers(model):
    """Create the explanation methods included in the sensitivity analysis."""
    explainers = {
        "STAGE": StageExplainerAdapter(model),
        "GNNExplainer": GNNExplainerWrapper(model),
        "PGExplainer": PGExplainerWrapper(model, SenseConfig.PGEXPLAINER_WEIGHT_PATH),
    }

    if shap is not None:
        explainers["SHAP"] = ShapExplainerWrapper(model)
    if lime_tabular is not None:
        explainers["LIME"] = LimeExplainerWrapper(model)

    return explainers


def make_perturbed_graph(data):
    """Create a lightly perturbed copy of a graph for consistency analysis."""
    data_perturbed = data.clone()
    num_edges = data.edge_index.size(1)

    if num_edges > 10:
        permutation = torch.randperm(num_edges)
        keep = int(num_edges * 0.95)
        data_perturbed.edge_index = data.edge_index[:, permutation[:keep]]
        data_perturbed.edge_attr = data.edge_attr[permutation[:keep]]

    return data_perturbed


def main():
    """Run K-sensitivity analysis."""
    set_seed(42)

    if not os.path.exists(LABEL_MAP_PATH):
        raise FileNotFoundError(f"Label map not found: {LABEL_MAP_PATH}")

    with open(LABEL_MAP_PATH, "r", encoding="utf-8") as file:
        label_map = json.load(file)

    model = load_stage_model(label_map)
    dataset = IDAPTDataset()
    loader = make_validation_loader(dataset)
    ground_truth = GroundTruthManager(LABEL_FILE_PATH)
    explainers = build_explainers(model)

    all_results = []
    count = 0
    pbar = tqdm(total=SenseConfig.NUM_SAMPLES, desc="Evaluating K-Sensitivity")

    for data in loader:
        if count >= SenseConfig.NUM_SAMPLES:
            break

        data = data.to(DEVICE)
        data.x = torch.nan_to_num(data.x.float(), nan=0.0)

        clear_pyg_state(model)

        with torch.no_grad():
            pred = model(data).argmax(dim=1).item()

        if data.y.item() == 0 or pred != data.y.item():
            continue

        gt_mask = ground_truth.get_node_ground_truth(data)

        if gt_mask.sum() == 0:
            continue

        count += 1
        pbar.update(1)

        data_perturbed = make_perturbed_graph(data)

        for name, explainer in explainers.items():
            try:
                clear_pyg_state(model)
                start_time = time.time()
                mask_orig = explainer.explain(data, target=data.y)
                time_cost = time.time() - start_time

                clear_pyg_state(model)
                mask_perturbed = explainer.explain(data_perturbed, target=data.y)

                for k_value in SenseConfig.K_LIST:
                    metrics = calculate_metrics_for_k(mask_orig, gt_mask, model, data, k_value, pred)

                    if not metrics:
                        continue

                    consistency = calculate_consistency_overlap(mask_orig, mask_perturbed, k_value)

                    record = {
                        "Method": name,
                        "k": k_value,
                        "Sample_ID": count,
                        "time": time_cost,
                        "consistency": consistency,
                    }
                    record.update(metrics)
                    all_results.append(record)

            except Exception as exc:
                logger.error(f"Error in {name}: {exc}")

    pbar.close()

    if not all_results:
        print("No results generated.")
        return

    df = pd.DataFrame(all_results)
    save_path = os.path.join(RESULTS_DIR, "sensitivity_analysis_fold2.csv")
    df.to_csv(save_path, index=False)

    print(f"Data saved to: {save_path}")
    print("\n[Preview: Benchmark at K=5]")

    df_k5 = df[df["k"] == 5]

    if not df_k5.empty:
        summary = df_k5.groupby("Method")[["precision", "iou", "fid+", "consistency"]].mean()
        print(summary)


if __name__ == "__main__":
    main()
