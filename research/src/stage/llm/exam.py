"""LLM attribution exam preparation and execution."""

import ast
import glob
import heapq
import json
import logging
import os
import random
import re
import time
from pathlib import Path

import numpy as np
import torch
from openai import OpenAI
from torch_geometric.loader import DataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from stage.data.dataset import IDAPTDataset
from stage.llm.prompts import build_exam_prompt
from stage.models.stage import StageModel
from stage.training.checkpoint import load_model_checkpoint
from stage.utils.config import get_path

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger("STAGE_LLM_Exam")


class ExamDataPreparer:
    """Prepare compressed Top-K evidence reports for LLM attribution exams."""

    def __init__(self, checkpoint_path=None, load_dataset=True):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.exam_dir = get_path("outputs", "llm_report_dir") / "exam_papers"
        self.exam_dir.mkdir(parents=True, exist_ok=True)

        with open(get_path("data", "label_map"), "r", encoding="utf-8") as file:
            self.label_map = json.load(file)
            self.idx_to_label = {value: key for key, value in self.label_map.items()}
            self.num_classes = len(self.label_map)

        self.dataset = IDAPTDataset() if load_dataset else None
        self.model = self._load_model(checkpoint_path)

    def _load_model(self, checkpoint_path=None):
        """Load the STAGE checkpoint used for evidence extraction."""
        if checkpoint_path is None:
            checkpoint_path = get_path("outputs", "stage_checkpoint_dir") / "best_model_fold_2.pth"

        model = StageModel(num_classes=self.num_classes)
        load_model_checkpoint(model, checkpoint_path, map_location=self.device, strict=False)
        model.to(self.device).eval()

        return model

    def explain(self, data):
        """Return Top-5 endogenous evidence nodes and their scores."""
        data = data.to(self.device)

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
            top_k = min(5, len(saga_scores))
            top_k_indices = saga_scores.argsort()[-top_k:][::-1]

            return top_k_indices, saga_scores

    def run_sampling(self, top_k=100):
        """Select high-confidence malicious samples and generate exam papers."""
        if self.dataset is None:
            raise RuntimeError("Dataset is not loaded. Initialize with load_dataset=True.")

        loader = DataLoader(self.dataset, batch_size=1, shuffle=False)
        top_candidates = []

        logger.info(f"Scanning dataset for top-{top_k} high-confidence correct malicious samples...")
        pbar = tqdm(total=len(self.dataset), desc="Scanning")

        for index, batch in enumerate(loader):
            pbar.update(1)
            y_true = batch.y.item()

            if y_true == 0:
                continue

            with torch.no_grad():
                batch_gpu = batch.to(self.device)
                batch_gpu.x = torch.nan_to_num(batch_gpu.x.float(), nan=0.0)

                start_model = time.perf_counter()
                logits = self.model(batch_gpu)
                t_model = time.perf_counter() - start_model

                probs = torch.softmax(logits, dim=1)
                conf, pred = probs.max(dim=1)

            if pred.item() == y_true:
                current_conf = conf.item()
                batch_cpu = batch.clone().cpu()
                original_nodes = batch_gpu.num_nodes
                record = (current_conf, index, batch_cpu, t_model, original_nodes)

                if len(top_candidates) < top_k:
                    heapq.heappush(top_candidates, record)
                elif current_conf > top_candidates[0][0]:
                    heapq.heapreplace(top_candidates, record)

        pbar.close()
        top_candidates.sort(key=lambda item: item[0], reverse=True)

        logger.info(f"Extracting SAGA evidence for {len(top_candidates)} selected samples...")

        for _, _, batch_data, t_model, original_nodes in top_candidates:
            start_saga = time.perf_counter()
            top_k_indices, saga_scores = self.explain(batch_data)
            t_saga = time.perf_counter() - start_saga
            self._generate_report(batch_data, top_k_indices, saga_scores, t_model, t_saga, original_nodes)

        logger.info(f"Exam papers generated at: {self.exam_dir}")

    @staticmethod
    def _as_list(value):
        """Normalize PyG metadata fields that may be wrapped by a batch dimension."""
        if isinstance(value, list) and len(value) == 1 and isinstance(value[0], (list, tuple)):
            return list(value[0])

        if isinstance(value, tuple):
            return list(value)

        return value

    def _generate_report(self, data, top_k_indices, scores, t_model, t_saga, original_nodes):
        """Write one compressed evidence report for the attribution exam."""
        raw_filename = data.filename[0] if isinstance(data.filename, list) else data.filename
        label_idx = data.y.item()
        true_label = self.idx_to_label.get(label_idx, "Unknown")

        report = {
            "meta": {
                "filename": raw_filename,
                "true_label_id": label_idx,
                "metrics": {
                    "original_nodes": original_nodes,
                    "t_model_sec": t_model,
                    "t_saga_sec": t_saga,
                },
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
        save_path = self.exam_dir / save_name

        with open(save_path, "w", encoding="utf-8") as file:
            json.dump(report, file, indent=4, ensure_ascii=False)

        return save_path


class LLMExamEngine:
    """Run LLM attribution exams over prepared exam papers."""

    def __init__(
        self,
        model_name="qwen2.5-7b",
        deploy_type="local",
        api_key=None,
        base_url=None,
        model_dir=None,
        load_model=True,
    ):
        self.model_name = model_name
        self.deploy_type = deploy_type

        self.exam_dir = get_path("outputs", "llm_report_dir") / "exam_papers"
        self.result_dir = get_path("outputs", "llm_report_dir") / f"exam_results_{model_name}"
        self.result_dir.mkdir(parents=True, exist_ok=True)

        self.system_prompt = build_exam_prompt()
        self.client = None
        self.tokenizer = None
        self.model = None

        if self.deploy_type == "local":
            self.model_dir = model_dir or os.environ.get("STAGE_LOCAL_LLM_DIR")
            if load_model:
                self._load_local_model()
        elif self.deploy_type == "api":
            self.api_key = api_key or self._get_api_key_from_env(model_name)
            self.base_url = base_url or self._get_base_url_from_env(model_name)
            if load_model:
                self._init_api_client()
        else:
            raise ValueError(f"Unsupported deploy_type: {deploy_type}")

    @staticmethod
    def _get_api_key_from_env(model_name):
        """Return the API key environment variable for a model name."""
        if "deepseek" in model_name.lower():
            return os.environ.get("DEEPSEEK_API_KEY")

        return os.environ.get("OPENAI_API_KEY")

    @staticmethod
    def _get_base_url_from_env(model_name):
        """Return the API base URL environment variable for a model name."""
        if "deepseek" in model_name.lower():
            return os.environ.get("DEEPSEEK_BASE_URL")

        return os.environ.get("OPENAI_BASE_URL")

    def _load_local_model(self):
        """Load a local causal language model."""
        if not self.model_dir:
            raise ValueError("No local model directory. Set STAGE_LOCAL_LLM_DIR or pass model_dir.")

        model_dir = Path(self.model_dir)
        if not model_dir.exists():
            raise FileNotFoundError(f"Local model directory not found: {model_dir}")

        logger.info(f"Loading local model {self.model_name} from {model_dir}")

        self.tokenizer = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            str(model_dir),
            device_map="cuda:0",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        ).eval()

    def _init_api_client(self):
        """Initialize the OpenAI-compatible API client."""
        if not self.api_key:
            raise ValueError(
                f"No API key found for {self.model_name}. Set OPENAI_API_KEY or DEEPSEEK_API_KEY."
            )

        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)

    def _generate_local(self, messages):
        """Generate a response with a local model."""
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Local model is not loaded.")

        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

        start_time = time.perf_counter()

        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=1024,
                temperature=0.1,
                do_sample=False,
            )

        t_llm = time.perf_counter() - start_time
        generated_ids = [
            output_ids[len(input_ids):]
            for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
        ]
        response_str = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

        return response_str, t_llm

    def _generate_api(self, messages):
        """Generate a response with an OpenAI-compatible API."""
        if self.client is None:
            raise RuntimeError("API client is not initialized.")

        start_time = time.perf_counter()
        is_json_format = "gpt" in self.model_name.lower() or "deepseek" in self.model_name.lower()

        request_args = {
            "model": self.model_name,
            "messages": messages,
            "temperature": 0.1,
        }

        if is_json_format:
            request_args["response_format"] = {"type": "json_object"}

        response = self.client.chat.completions.create(**request_args)
        t_llm = time.perf_counter() - start_time

        return response.choices[0].message.content, t_llm

    @staticmethod
    def parse_exam_response(response_str):
        """Parse an attribution-exam response as JSON."""
        clean_content = response_str.strip()
        clean_content = clean_content.replace("```json", "").replace("```", "").strip()

        start_idx = clean_content.find("{")
        end_idx = clean_content.rfind("}")

        if start_idx != -1 and end_idx != -1:
            clean_content = clean_content[start_idx:end_idx + 1]

        try:
            return json.loads(clean_content)
        except json.JSONDecodeError:
            match = re.search(
                r'\[\s*(?:["\'][a-zA-Z_0-9]+["\']\s*,\s*)*["\'][a-zA-Z_0-9]+["\']\s*\]',
                response_str,
            )

            if match:
                try:
                    extracted_tactics = ast.literal_eval(match.group(0).upper())
                    return {
                        "reasoning": "Format Error: Extracted via regex fallback.",
                        "observed_tactics": extracted_tactics,
                    }
                except Exception:
                    pass

            raise

    def take_exam(self, max_samples=100):
        """Run the selected LLM over prepared exam papers."""
        files = glob.glob(str(self.exam_dir / "*.json"))

        if max_samples and max_samples < len(files):
            files.sort()
            random.seed(42)
            files = random.sample(files, max_samples)

        logger.info(f"Starting {self.model_name} exam over {len(files)} papers...")

        for file_path in files:
            file_path = Path(file_path)
            filename = file_path.name

            with open(file_path, "r", encoding="utf-8") as file:
                evidence = json.load(file)

            user_prompt = f"```json\n{json.dumps(evidence, indent=2)}\n```"
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_prompt},
            ]

            try:
                if self.deploy_type == "local":
                    response_str, t_llm = self._generate_local(messages)
                else:
                    response_str, t_llm = self._generate_api(messages)

                try:
                    response_json = self.parse_exam_response(response_str)
                except Exception:
                    logger.error(f"{filename}: failed to parse response.")
                    continue

                response_json["t_llm_sec"] = t_llm

                save_path = self.result_dir / f"result_{filename}"
                with open(save_path, "w", encoding="utf-8") as file:
                    json.dump(response_json, file, indent=4, ensure_ascii=False)

                logger.info(f"[{self.model_name}] finished {filename} ({t_llm:.2f}s)")

            except Exception as exc:
                logger.error(f"{filename}: inference failed: {exc}")
