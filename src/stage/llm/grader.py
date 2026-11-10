"""LLM attribution grading utilities."""

import glob
import json
from pathlib import Path

import pandas as pd

from stage.utils.config import get_path

TRUE_CAPABILITIES = {
    0: {"BENIGN"},
    1: {"LOADER", "STEALER", "C2_RAT"},
    2: {"LOADER", "STEALER", "C2_RAT"},
    3: {"LOADER", "STEALER", "C2_RAT"},
    4: {"LOADER", "C2_RAT"},
    5: {"LOADER", "STEALER", "C2_RAT"},
    6: {"LOADER", "C2_RAT"},
    7: {"STEALER", "C2_RAT"},
    8: {"LOADER", "C2_RAT"},
    9: {"LOADER", "C2_RAT"},
    10: {"LOADER", "STEALER", "C2_RAT"},
    11: {"LOADER", "STEALER", "C2_RAT"},
    12: {"LOADER", "C2_RAT"},
}


def normalize_tactics(pred_tactics):
    """Normalize predicted tactics to uppercase labels."""
    if isinstance(pred_tactics, str):
        pred_tactics = [pred_tactics]

    normalized = set()

    for item in pred_tactics:
        if not isinstance(item, str):
            continue

        item = item.strip().upper()
        if item:
            normalized.add(item)

    return normalized


def grade_single_model(model_name, llm_root=None):
    """Compute attribution metrics for one LLM result directory."""
    if llm_root is None:
        llm_root = get_path("outputs", "llm_report_dir")
    else:
        llm_root = Path(llm_root)

    result_dir = llm_root / f"exam_results_{model_name}"
    exam_dir = llm_root / "exam_papers"

    files = sorted(glob.glob(str(result_dir / "result_*.json")))

    if not files:
        return None

    total_samples = 0
    valid_detection_hits = 0
    total_scs = 0.0
    total_ip = 0.0
    total_f1 = 0.0
    total_latency_list = []
    compression_rates = []

    for file_path in files:
        file_path = Path(file_path)

        try:
            with open(file_path, "r", encoding="utf-8") as file:
                data = json.load(file)
        except Exception:
            continue

        original_filename = file_path.name.replace("result_", "")
        exam_paper_path = exam_dir / original_filename

        if not exam_paper_path.exists():
            continue

        try:
            with open(exam_paper_path, "r", encoding="utf-8") as file:
                exam_data = json.load(file)
        except Exception:
            continue

        pred_set = normalize_tactics(data.get("observed_tactics", []))

        true_id = int(exam_data["meta"]["true_label_id"])
        original_nodes = int(exam_data["meta"]["metrics"]["original_nodes"])
        t_model = float(exam_data["meta"]["metrics"]["t_model_sec"])
        t_saga = float(exam_data["meta"]["metrics"]["t_saga_sec"])
        t_llm = float(data.get("t_llm_sec", 0.0))

        true_set = TRUE_CAPABILITIES.get(true_id, set())
        intersection = pred_set.intersection(true_set)

        if len(intersection) > 0:
            valid_detection_hits += 1

        scs = len(intersection) / len(true_set) if len(true_set) > 0 else 0.0
        ip = len(intersection) / len(pred_set) if len(pred_set) > 0 else 0.0
        f1 = 2 * (ip * scs) / (ip + scs) if (ip + scs) > 0 else 0.0

        total_scs += scs
        total_ip += ip
        total_f1 += f1

        total_latency_list.append(t_model + t_saga + t_llm)
        compression_rates.append((original_nodes - 5) / original_nodes * 100)

        total_samples += 1

    if total_samples == 0:
        return None

    return {
        "Model": model_name,
        "Count": total_samples,
        "BHR": valid_detection_hits / total_samples * 100,
        "SCS": total_scs / total_samples * 100,
        "IP": total_ip / total_samples * 100,
        "F1": total_f1 / total_samples * 100,
        "Latency": sum(total_latency_list) / len(total_latency_list),
        "Compression": sum(compression_rates) / len(compression_rates),
    }


def run_evaluation(models=None, save=True, output_dir=None, llm_root=None):
    """Compute LLM attribution metrics for one or more model outputs."""
    if models is None:
        models = ["gpt-4o-mini", "deepseek-chat", "llama-3-8b", "qwen2.5-7b"]

    rows = []

    for model in models:
        metrics = grade_single_model(model, llm_root=llm_root)
        if metrics is not None:
            rows.append(metrics)

    df = pd.DataFrame(rows)

    if not df.empty:
        for column in ["BHR", "SCS", "IP", "F1", "Latency", "Compression"]:
            df[column] = df[column].astype(float).map(lambda value: f"{value:.2f}")

    if save:
        if output_dir is None:
            out_dir = get_path("outputs", "metrics_dir") / "paper_tables"
        else:
            out_dir = Path(output_dir)

        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "table10_llm.csv"
        df.to_csv(out_path, index=False, encoding="utf-8")
        print(f"saved: {out_path}")

    return df


if __name__ == "__main__":
    result_df = run_evaluation(save=True)
    print(result_df.to_string(index=False))
