"""LLM analyst for STAGE evidence reports."""

import glob
import json
import logging
import os
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from stage.llm.prompts import build_cyberguardian_prompt
from stage.utils.config import get_path

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("STAGE_LLM_Analyst")


class CyberGuardianAgent:
    """Generate structured analyst responses from STAGE evidence reports."""

    def __init__(
        self,
        model_dir=None,
        input_report_dir=None,
        output_response_dir=None,
        load_model=True,
        device_map="cuda:0",
    ):
        self.model_dir = model_dir or os.environ.get("STAGE_LOCAL_LLM_DIR")
        self.input_report_dir = Path(input_report_dir) if input_report_dir else get_path("outputs", "explanation_dir")
        self.output_response_dir = (
            Path(output_response_dir)
            if output_response_dir
            else get_path("outputs", "llm_report_dir") / "analyst_responses"
        )

        self.device_map = device_map
        self.system_prompt = build_cyberguardian_prompt()
        self.tokenizer = None
        self.model = None

        if load_model:
            self.load_model()

    def load_model(self):
        """Load the local causal language model."""
        if not self.model_dir:
            raise ValueError("No local LLM path provided. Set STAGE_LOCAL_LLM_DIR or pass model_dir explicitly.")

        model_dir = Path(self.model_dir)
        if not model_dir.exists():
            raise FileNotFoundError(f"Local LLM path not found: {model_dir}")

        logger.info(f"Loading local LLM from: {model_dir}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            str(model_dir),
            trust_remote_code=True,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            str(model_dir),
            device_map=self.device_map,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        self.model.eval()

        logger.info("Local LLM loaded successfully.")

    def build_messages(self, evidence_json, filename="unknown.json"):
        """Build chat messages from an evidence report."""
        evidence_list = evidence_json.get("timeline_evidence", [])
        if not evidence_list:
            return None

        input_for_llm = {
            "meta": evidence_json.get("meta", {"filename": filename}),
            "timeline_evidence": evidence_list,
        }

        user_prompt = (
            "Here is the forensic report data:\n"
            "```json\n"
            f"{json.dumps(input_for_llm, indent=2)}\n"
            "```"
        )

        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    @staticmethod
    def parse_json_response(response_content):
        """Parse a JSON response, preserving raw output on parse failure."""
        clean_content = response_content.strip()

        if "```json" in clean_content:
            clean_content = clean_content.split("```json", 1)[1].split("```", 1)[0].strip()
        elif "```" in clean_content:
            clean_content = clean_content.replace("```", "").strip()

        try:
            return json.loads(clean_content)
        except Exception:
            logger.warning("JSON parsing failed; saving raw output.")
            return {"raw_output": clean_content}

    def generate_response(self, report_path, save=True):
        """Generate one analyst response from an evidence report."""
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("LLM is not loaded. Initialize with load_model=True or call load_model().")

        report_path = Path(report_path)
        filename = report_path.name

        if not report_path.exists():
            logger.warning(f"Report not found: {report_path}")
            return None

        try:
            with open(report_path, "r", encoding="utf-8") as file:
                evidence_json = json.load(file)
        except Exception as exc:
            logger.warning(f"Failed to read report {report_path}: {exc}")
            return None

        messages = self.build_messages(evidence_json, filename=filename)
        if messages is None:
            logger.warning(f"Report {filename} has no timeline_evidence; skipped.")
            return None

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

        try:
            generated_ids = self.model.generate(
                **model_inputs,
                max_new_tokens=2048,
                temperature=0.1,
                top_p=0.9,
                do_sample=True,
            )
            generated_ids = [
                output_ids[len(input_ids):]
                for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]
            response_content = self.tokenizer.batch_decode(
                generated_ids,
                skip_special_tokens=True,
            )[0]

            response_json = self.parse_json_response(response_content)

            if save:
                self.output_response_dir.mkdir(parents=True, exist_ok=True)
                save_path = self.output_response_dir / f"response_{filename}"
                with open(save_path, "w", encoding="utf-8") as file:
                    json.dump(response_json, file, indent=4, ensure_ascii=False)
                logger.info(f"Analyst response saved: {save_path}")

            return response_json

        except Exception as exc:
            logger.error(f"LLM inference failed for {filename}: {exc}")
            return None

    def run_batch(self, limit=None):
        """Generate responses for a batch of evidence reports."""
        files = sorted(glob.glob(str(self.input_report_dir / "*.json")))

        if limit is not None:
            files = files[:limit]

        logger.info(f"Found {len(files)} evidence reports.")

        results = []
        for file_path in files:
            result = self.generate_response(file_path, save=True)
            results.append(result)

        return results
