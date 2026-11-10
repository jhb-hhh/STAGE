#!/usr/bin/env python3
"""Unified runner for the STAGE compact artifact."""

import argparse
import json
import shutil
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

REFERENCE_CHECKPOINT = ROOT / "reference_artifacts/checkpoints/stage/stage_reference_checkpoint.pth"
TRAINED_CHECKPOINT = ROOT / "outputs/workspace/checkpoints/training/stage_trained_checkpoint.pth"
TRAINING_OUTPUT_ROOT = ROOT / "outputs/workspace/checkpoints/training"


def check_exists(path: Path, name: str, required: bool = True):
    status = "OK" if path.exists() else "MISSING"
    print(f"[{status}] {name}: {path}")
    if required and not path.exists():
        return False
    return True


def resolve_checkpoint(name_or_path: str):
    if name_or_path == "reference":
        return REFERENCE_CHECKPOINT, "reference"
    if name_or_path == "trained":
        return TRAINED_CHECKPOINT, "trained"
    return Path(name_or_path), "custom"


def verify():
    print("== STAGE artifact verification ==")

    required_paths = [
        (ROOT / "configs", "configs/"),
        (ROOT / "src/stage", "src/stage/"),
        (ROOT / "tools", "tools/"),
        (ROOT / "sample_data/graphs", "sample_data/graphs/"),
        (ROOT / "sample_data/splits", "sample_data/splits/"),
        (ROOT / "sample_data/label_mapping.json", "sample_data/label_mapping.json"),
        (ROOT / "sample_data/graphs/graphs_chunk_000.pt", "sample graph chunk"),
        (REFERENCE_CHECKPOINT, "STAGE reference checkpoint"),
        (ROOT / "reference_artifacts/metrics/source_metrics", "reference source metrics"),
    ]

    ok = True
    for path, name in required_paths:
        ok = check_exists(path, name) and ok

    print()
    print("== Import checks ==")

    imports = [
        ("stage", "stage"),
        ("stage.models.stage", "StageModel"),
        ("stage.data.dataset", "IDAPTDataset"),
        ("stage.training.trainer", "train_kfold"),
        ("stage.evaluation.detection", "evaluate_stage_fold"),
        ("stage.explanation.endogenous", "EndogenousExplainer"),
        ("stage.analysis.paper_tables", "paper_tables"),
        ("stage.llm.grader", "LLM grader"),
    ]

    for module_name, label in imports:
        try:
            __import__(module_name)
            print(f"[OK] import {label}")
        except Exception as exc:
            print(f"[MISSING] import {label} failed: {exc}")
            ok = False

    print()
    if ok:
        print("Artifact verification passed.")
        return 0

    print("Artifact verification failed.")
    return 1


def train(epochs: int = 1, batch_size: int = 4, fold_id: int = 2):
    print("== STAGE training on the compact artifact subset ==")
    print("This validates the training pipeline on a small sanitized dataset.")
    print("It is not intended to reproduce full-scale paper metrics.")

    from stage.training.trainer import train_kfold

    train_kfold(
        mode_name="stage",
        fold=fold_id,
        batch_size=batch_size,
        epochs=epochs,
        output_root=TRAINING_OUTPUT_ROOT,
    )

    produced = (
        TRAINING_OUTPUT_ROOT
        / "kfold_experiment_stage_final"
        / f"best_model_fold_{fold_id}.pth"
    )

    if not produced.exists():
        print(f"[MISSING] expected trained checkpoint: {produced}")
        return 1

    TRAINED_CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(produced, TRAINED_CHECKPOINT)

    try:
        produced.unlink()
    except OSError:
        pass

    print()
    print("Training finished.")
    print(f"Trained checkpoint: {TRAINED_CHECKPOINT}")
    return 0


def evaluate(checkpoint: str = "reference", fold_id: int = 2, batch_size: int = 8, device: str = "cpu"):
    print("== STAGE evaluation on the compact artifact subset ==")
    print("Checkpoint mode:", checkpoint)

    from stage.evaluation.detection import evaluate_stage_fold

    checkpoint_path, checkpoint_label = resolve_checkpoint(checkpoint)

    if not checkpoint_path.exists():
        print(f"[MISSING] checkpoint: {checkpoint_path}")
        if checkpoint_label == "trained":
            print("Run first: PYTHONPATH=src python tools/run.py train-stage")
        return 1

    metrics = evaluate_stage_fold(
        fold_id=fold_id,
        checkpoint_path=checkpoint_path,
        batch_size=batch_size,
        device=device,
    )

    out_dir = ROOT / "outputs/reproduce/evaluation"
    out_dir.mkdir(parents=True, exist_ok=True)

    out_file = out_dir / f"stage_metrics_{checkpoint_label}.json"
    with open(out_file, "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2, ensure_ascii=False)

    print()
    print("Evaluation metrics:")
    for key, value in metrics.items():
        print(f"{key}: {value}")

    print()
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Saved to: {out_file}")
    return 0


def eval_robustness(checkpoint: str = "reference", fold_id: int = 2, batch_size: int = 8, device: str = "cpu"):
    print("== STAGE compact robustness evaluation ==")
    print("This evaluates STAGE on original and perturbed compact validation graphs.")
    print("It validates the robustness evaluation pipeline, not full-scale paper metrics.")

    import copy
    import csv
    import torch

    from stage.data.dataset import IDAPTDataset
    from stage.evaluation.common import load_label_map, load_split_indices
    from stage.evaluation.robustness import perturb_graph, evaluate_graph_list
    from stage.models.stage import StageModel
    from stage.training.checkpoint import load_model_checkpoint

    checkpoint_path, checkpoint_label = resolve_checkpoint(checkpoint)

    if not checkpoint_path.exists():
        print(f"[MISSING] checkpoint: {checkpoint_path}")
        if checkpoint_label == "trained":
            print("Run first: PYTHONPATH=src python tools/run.py train-stage")
        return 1

    label_map = load_label_map()
    model = StageModel(num_classes=len(label_map)).to(device)
    load_model_checkpoint(model, checkpoint_path, map_location=device, strict=False)
    model.eval()

    dataset = IDAPTDataset()
    indices = load_split_indices(fold_id)
    val_indices = indices["val"]

    original_graphs = [dataset[i].clone() for i in val_indices]
    shifted_graphs = [perturb_graph(copy.deepcopy(graph)) for graph in original_graphs]

    original = evaluate_graph_list(
        model,
        original_graphs,
        batch_size=batch_size,
        device=device,
        desc="Original compact graphs",
    )
    shifted = evaluate_graph_list(
        model,
        shifted_graphs,
        batch_size=batch_size,
        device=device,
        desc="Shifted compact graphs",
    )

    if original["F1_Macro"] == 0:
        f1_drop = None
    else:
        f1_drop = (original["F1_Macro"] - shifted["F1_Macro"]) / original["F1_Macro"] * 100.0

    result = {
        "checkpoint": str(checkpoint_path.relative_to(ROOT)) if checkpoint_path.is_relative_to(ROOT) else str(checkpoint_path),
        "fold": fold_id,
        "num_graphs": len(original_graphs),
        "note": "Compact robustness evaluation generated by applying perturb_graph() to the compact validation subset.",
        "original": original,
        "shifted": shifted,
        "f1_drop_percent": f1_drop,
    }

    out_dir = ROOT / "outputs/reproduce/robustness"
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / f"stage_robustness_{checkpoint_label}.json"
    with open(json_path, "w", encoding="utf-8") as file:
        json.dump(result, file, indent=2, ensure_ascii=False)

    csv_path = out_dir / f"stage_robustness_{checkpoint_label}.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["Checkpoint", "TestSet", "Accuracy", "Precision", "Recall", "F1_Macro", "FPR", "MCC", "F1_Drop_Percent"],
        )
        writer.writeheader()
        writer.writerow({
            "Checkpoint": checkpoint_label,
            "TestSet": "Original",
            "Accuracy": original["Accuracy"],
            "Precision": original["Precision"],
            "Recall": original["Recall"],
            "F1_Macro": original["F1_Macro"],
            "FPR": original["FPR"],
            "MCC": original["MCC"],
            "F1_Drop_Percent": "",
        })
        writer.writerow({
            "Checkpoint": checkpoint_label,
            "TestSet": "Shifted",
            "Accuracy": shifted["Accuracy"],
            "Precision": shifted["Precision"],
            "Recall": shifted["Recall"],
            "F1_Macro": shifted["F1_Macro"],
            "FPR": shifted["FPR"],
            "MCC": shifted["MCC"],
            "F1_Drop_Percent": "" if f1_drop is None else f"{f1_drop:.2f}",
        })

    print()
    print("Original metrics:")
    for key, value in original.items():
        print(f"  {key}: {value}")

    print()
    print("Shifted metrics:")
    for key, value in shifted.items():
        print(f"  {key}: {value}")

    print()
    print("F1 drop percent:", "undefined" if f1_drop is None else f"{f1_drop:.2f}%")
    print(f"Saved JSON to: {json_path}")
    print(f"Saved CSV to: {csv_path}")

    from table_utils import print_csv_table
    print_csv_table(csv_path, title="Robustness evaluation table")

    return 0


def explain(checkpoint: str = "reference", limit: int = 5, top_k: int = 5, device: str = "cpu"):
    print("== STAGE endogenous explanation on the compact artifact subset ==")
    print("This is a lightweight explanation run, not the full explanation benchmark.")

    import torch
    from stage.explanation.endogenous import EndogenousExplainer

    checkpoint_path, checkpoint_label = resolve_checkpoint(checkpoint)

    if not checkpoint_path.exists():
        print(f"[MISSING] checkpoint: {checkpoint_path}")
        if checkpoint_label == "trained":
            print("Run first: PYTHONPATH=src python tools/run.py train-stage")
        return 1

    graph_path = ROOT / "sample_data/graphs/graphs_chunk_000.pt"
    if not graph_path.exists():
        print(f"[MISSING] graph chunk: {graph_path}")
        return 1

    graphs = torch.load(graph_path, map_location="cpu", weights_only=False)[:limit]

    explainer = EndogenousExplainer(model_path=checkpoint_path, load_dataset=False)
    explainer.device = torch.device(device)
    explainer.model.to(explainer.device)

    records = []

    for graph_index, data in enumerate(graphs):
        data_for_pred = data.clone().to(explainer.device)
        data_for_pred.x = torch.nan_to_num(data_for_pred.x.float(), nan=0.0)

        with torch.no_grad():
            logits = explainer.model(data_for_pred)
            probs = torch.softmax(logits, dim=1)
            confidence, pred = probs.max(dim=1)

        top_indices, scores, valid_len, bias_ratio = explainer.explain(data.clone())
        top_indices = top_indices[:top_k]

        top_nodes = []
        for rank, node_idx in enumerate(top_indices, start=1):
            node_idx = int(node_idx)
            score = float(scores[node_idx]) if node_idx < len(scores) else 0.0
            top_nodes.append(
                {
                    "rank": rank,
                    "node_index": node_idx,
                    "importance_score": score,
                }
            )

        record = {
            "graph_index": graph_index,
            "true_label": int(data.y.view(-1)[0].item()),
            "predicted_label": int(pred.view(-1)[0].item()),
            "confidence": float(confidence.view(-1)[0].item()),
            "valid_length": int(valid_len),
            "bias_ratio": float(bias_ratio),
            "top_k_nodes": top_nodes,
        }
        records.append(record)

        print(
            f"graph={graph_index} true={record['true_label']} "
            f"pred={record['predicted_label']} "
            f"conf={record['confidence']:.4f} "
            f"top_nodes={[item['node_index'] for item in top_nodes]}"
        )

    out_dir = ROOT / "outputs/reproduce/explanations"
    out_dir.mkdir(parents=True, exist_ok=True)

    out_file = out_dir / f"endogenous_explanation_{checkpoint_label}.json"
    payload = {
        "checkpoint": str(checkpoint_path.relative_to(ROOT)) if checkpoint_path.is_relative_to(ROOT) else str(checkpoint_path),
        "graph_file": str(graph_path.relative_to(ROOT)),
        "limit": len(graphs),
        "top_k": top_k,
        "note": "Lightweight endogenous explanation on sanitized compact graph data.",
        "records": records,
    }

    with open(out_file, "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)

    print()
    print(f"Saved explanation output to: {out_file}")
    return 0


def tables(include_llm: bool = False):
    print("== STAGE paper-table reproduction ==")
    print("This regenerates paper-ready tables from reference source metrics.")

    source_dir = ROOT / "reference_artifacts/metrics/source_metrics"
    output_dir = ROOT / "outputs/reproduce/paper_tables"

    required = [
        "baseline_detection.csv",
        "ablation_summary.csv",
        "robustness_baselines.csv",
        "robustness_stage.csv",
        "explanation_benchmark.csv",
    ]

    ok = True
    for filename in required:
        metric_path = source_dir / filename
        if metric_path.exists():
            print(f"[OK] source metric: {metric_path}")
        else:
            print(f"[MISSING] source metric: {metric_path}")
            ok = False

    if not ok:
        return 1

    from stage.analysis.paper_tables import (
        set_table_dirs,
        make_table5_detection,
        make_table6_robustness,
        make_table8_ablation,
        make_table9_explainers,
    )

    set_table_dirs(source_dir=source_dir, output_dir=output_dir)

    makers = [
        make_table5_detection,
        make_table6_robustness,
        make_table8_ablation,
        make_table9_explainers,
    ]

    generated = []
    for maker in makers:
        output_path, _ = maker()
        generated.append(output_path)

    if include_llm:
        from stage.analysis.paper_tables import make_table10_llm
        output_path, _ = make_table10_llm()
        generated.append(output_path)
    else:
        print("[INFO] LLM table is skipped by default in quick mode.")

    print()
    print("Generated tables:")
    for path in generated:
        print(f"- {path}")

    from table_utils import print_csv_table
    for path in generated:
        print_csv_table(path, title=f"Generated table: {Path(path).name}")

    return 0


def llm():
    print("== STAGE offline LLM input preparation ==")
    print("This command does not execute LLM generation.")
    print("It prepares an offline LLM input sample from endogenous explanation output.")

    explanation_path = ROOT / "outputs/reproduce/explanations/endogenous_explanation_reference.json"

    if not explanation_path.exists():
        print(f"[MISSING] explanation output: {explanation_path}")
        print("Run first: PYTHONPATH=src python tools/run.py explain-stage")
        return 1

    with open(explanation_path, "r", encoding="utf-8") as file:
        explanation = json.load(file)

    records = explanation.get("records", [])

    llm_cases = []
    for record in records:
        evidence_lines = []
        for node in record.get("top_k_nodes", []):
            evidence_lines.append(
                {
                    "rank": node.get("rank"),
                    "node_index": node.get("node_index"),
                    "importance_score": node.get("importance_score"),
                }
            )

        llm_cases.append(
            {
                "case_id": f"compact_case_{record.get('graph_index')}",
                "true_label": record.get("true_label"),
                "predicted_label": record.get("predicted_label"),
                "confidence": record.get("confidence"),
                "bias_ratio": record.get("bias_ratio"),
                "evidence_nodes": evidence_lines,
                "task_instruction": (
                    "Given the selected high-importance graph nodes and model prediction, "
                    "summarize the likely attack-chain evidence and explain why these nodes "
                    "may support the predicted APT behavior. This compact artifact only "
                    "prepares the LLM input; LLM generation is not executed."
                ),
            }
        )

    out_dir = ROOT / "outputs/reproduce/llm"
    out_dir.mkdir(parents=True, exist_ok=True)

    out_file = out_dir / "llm_input_sample.json"
    payload = {
        "source_explanation": str(explanation_path.relative_to(ROOT)),
        "note": (
            "Offline LLM input sample generated from compact explanation output. "
            "The artifact does not execute LLM generation, does not require "
            "API keys, and does not provide local LLM weights."
        ),
        "cases": llm_cases,
    }

    with open(out_file, "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)

    readme_file = out_dir / "README.md"
    readme_file.write_text(
        "# Offline LLM Input\n\n"
        "This directory contains an offline LLM input sample generated from the compact explanation output.\n\n"
        "The compact artifact does not call online LLM APIs and does not provide local LLM weights.\n\n"
        "The purpose of this step is to show how extracted evidence is converted into an LLM-consumable input format.\n\n"
        "Paper-scale LLM results are represented by the reference source metrics and paper tables.\n",
        encoding="utf-8",
    )

    print(f"Saved LLM input sample to: {out_file}")
    print(f"Saved LLM note to: {readme_file}")
    return 0


def run_reference():
    print("== STAGE reference-checkpoint workflow ==")
    print("This workflow uses artifact-provided reference checkpoints on sample_data.")
    print("It validates executable evaluation, robustness, explanation, and LLM-input preparation paths.")
    print("It does not train models and does not regenerate paper tables.")
    print("For paper-table regeneration, run: PYTHONPATH=src python tools/run.py make-tables")

    steps = [
        ("verify", lambda: verify()),
        ("eval-stage", lambda: evaluate(checkpoint="reference")),
        ("eval-baselines", None),
        ("eval-ablations", None),
        ("eval-robustness", lambda: eval_robustness(checkpoint="reference")),
        ("explain-stage", lambda: explain(checkpoint="reference")),
        ("prepare-llm-input", lambda: llm()),
    ]

    from baseline_runner import eval_baselines
    from ablation_runner import eval_ablations

    concrete_steps = []
    for name, fn in steps:
        if name == "eval-baselines":
            fn = lambda: eval_baselines(root=ROOT, checkpoint_type="reference")
        elif name == "eval-ablations":
            fn = lambda: eval_ablations(root=ROOT, checkpoint_type="reference")
        concrete_steps.append((name, fn))

    for name, fn in concrete_steps:
        print()
        print("=" * 80)
        print(f"[run-reference] {name}")
        print("=" * 80)
        code = fn()
        if code != 0:
            print(f"[FAILED] {name}")
            return code

    print()
    print("Reference-checkpoint workflow completed.")
    return 0


def run_training():
    print("== STAGE sample-data training workflow ==")
    print("This workflow trains selected models on sample_data.")
    print("It validates training and re-evaluation pipelines, not full-paper metrics.")

    from baseline_runner import train_baselines, eval_baselines
    from ablation_runner import train_ablations, eval_ablations

    steps = [
        ("train-stage", lambda: train()),
        ("eval-stage --checkpoint trained", lambda: evaluate(checkpoint="trained")),
        (
            "train-baselines --models TS-IDS",
            lambda: train_baselines(root=ROOT, models="TS-IDS", epochs=1, batch_size=4),
        ),
        (
            "eval-baselines --checkpoint trained --models TS-IDS",
            lambda: eval_baselines(root=ROOT, models="TS-IDS", checkpoint_type="trained"),
        ),
        (
            "train-ablations --models stage_final,gcn_only",
            lambda: train_ablations(root=ROOT, models="stage_final,gcn_only", epochs=1, batch_size=4),
        ),
        (
            "eval-ablations --checkpoint trained --models stage_final,gcn_only",
            lambda: eval_ablations(root=ROOT, models="stage_final,gcn_only", checkpoint_type="trained"),
        ),
        ("eval-robustness --checkpoint trained", lambda: eval_robustness(checkpoint="trained")),
        ("explain-stage --checkpoint trained", lambda: explain(checkpoint="trained")),
    ]

    for name, fn in steps:
        print()
        print("=" * 80)
        print(f"[run-training] {name}")
        print("=" * 80)
        code = fn()
        if code != 0:
            print(f"[FAILED] {name}")
            return code

    print()
    print("Sample-data training workflow completed.")
    return 0


def build_parser():
    parser = argparse.ArgumentParser(description="STAGE compact artifact runner")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("verify")
    subparsers.add_parser("run-reference")
    subparsers.add_parser("run-training")

    train_parser = subparsers.add_parser("train-stage")
    train_parser.add_argument("--epochs", type=int, default=1)
    train_parser.add_argument("--batch-size", type=int, default=4)
    train_parser.add_argument("--fold", type=int, default=2)

    evaluate_parser = subparsers.add_parser("eval-stage")
    evaluate_parser.add_argument("--checkpoint", default="reference")
    evaluate_parser.add_argument("--fold", type=int, default=2)
    evaluate_parser.add_argument("--batch-size", type=int, default=8)
    evaluate_parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])

    train_baseline_parser = subparsers.add_parser("train-baselines")
    train_baseline_parser.add_argument("--models", default="all")
    train_baseline_parser.add_argument("--epochs", type=int, default=1)
    train_baseline_parser.add_argument("--batch-size", type=int, default=4)
    train_baseline_parser.add_argument("--fold", type=int, default=2)

    eval_baseline_parser = subparsers.add_parser("eval-baselines")
    eval_baseline_parser.add_argument("--models", default="all")
    eval_baseline_parser.add_argument("--checkpoint", default="reference")
    eval_baseline_parser.add_argument("--fold", type=int, default=2)
    eval_baseline_parser.add_argument("--batch-size", type=int, default=8)
    eval_baseline_parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])

    train_ablation_parser = subparsers.add_parser("train-ablations")
    train_ablation_parser.add_argument("--models", default="all")
    train_ablation_parser.add_argument("--epochs", type=int, default=1)
    train_ablation_parser.add_argument("--batch-size", type=int, default=4)
    train_ablation_parser.add_argument("--fold", type=int, default=2)

    eval_ablation_parser = subparsers.add_parser("eval-ablations")
    eval_ablation_parser.add_argument("--models", default="all")
    eval_ablation_parser.add_argument("--checkpoint", default="reference")
    eval_ablation_parser.add_argument("--fold", type=int, default=2)
    eval_ablation_parser.add_argument("--batch-size", type=int, default=8)
    eval_ablation_parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])

    robustness_parser = subparsers.add_parser("eval-robustness")
    robustness_parser.add_argument("--checkpoint", default="reference")
    robustness_parser.add_argument("--fold", type=int, default=2)
    robustness_parser.add_argument("--batch-size", type=int, default=8)
    robustness_parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])

    explain_parser = subparsers.add_parser("explain-stage")
    explain_parser.add_argument("--checkpoint", default="reference")
    explain_parser.add_argument("--limit", type=int, default=5)
    explain_parser.add_argument("--top-k", type=int, default=5)
    explain_parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])

    tables_parser = subparsers.add_parser("make-tables")
    tables_parser.add_argument("--include-llm", action="store_true")

    subparsers.add_parser("prepare-llm-input")

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "run-reference":
        raise SystemExit(run_reference())

    if args.command == "run-training":
        raise SystemExit(run_training())

    if args.command == "verify":
        raise SystemExit(verify())

    if args.command == "train-stage":
        raise SystemExit(
            train(
                epochs=args.epochs,
                batch_size=args.batch_size,
                fold_id=args.fold,
            )
        )

    if args.command == "eval-stage":
        raise SystemExit(
            evaluate(
                checkpoint=args.checkpoint,
                fold_id=args.fold,
                batch_size=args.batch_size,
                device=args.device,
            )
        )

    if args.command == "train-baselines":
        from baseline_runner import train_baselines
        raise SystemExit(
            train_baselines(
                root=ROOT,
                models=args.models,
                epochs=args.epochs,
                batch_size=args.batch_size,
                fold_id=args.fold,
            )
        )

    if args.command == "eval-baselines":
        from baseline_runner import eval_baselines
        raise SystemExit(
            eval_baselines(
                root=ROOT,
                models=args.models,
                checkpoint_type=args.checkpoint,
                fold_id=args.fold,
                batch_size=args.batch_size,
                device=args.device,
            )
        )

    if args.command == "train-ablations":
        from ablation_runner import train_ablations
        raise SystemExit(
            train_ablations(
                root=ROOT,
                models=args.models,
                epochs=args.epochs,
                batch_size=args.batch_size,
                fold_id=args.fold,
            )
        )

    if args.command == "eval-ablations":
        from ablation_runner import eval_ablations
        raise SystemExit(
            eval_ablations(
                root=ROOT,
                models=args.models,
                checkpoint_type=args.checkpoint,
                fold_id=args.fold,
                batch_size=args.batch_size,
                device=args.device,
            )
        )

    if args.command == "eval-robustness":
        raise SystemExit(
            eval_robustness(
                checkpoint=args.checkpoint,
                fold_id=args.fold,
                batch_size=args.batch_size,
                device=args.device,
            )
        )

    if args.command == "explain-stage":
        raise SystemExit(
            explain(
                checkpoint=args.checkpoint,
                limit=args.limit,
                top_k=args.top_k,
                device=args.device,
            )
        )

    if args.command == "make-tables":
        raise SystemExit(tables(include_llm=args.include_llm))

    if args.command == "prepare-llm-input":
        raise SystemExit(llm())


if __name__ == "__main__":
    main()
