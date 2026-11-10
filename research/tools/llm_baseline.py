#!/usr/bin/env python
"""
LLM Baseline: 只给全图统计特征，不给 Top-K 证据
用于对比实验：证明 STAGE 证据能减少幻觉
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, 'src')

from stage.llm.exam import ExamDataPreparer, LLMExamEngine
from stage.llm.grader import run_evaluation


def baseline_build_prompt(self, data, top_k_nodes, importance_scores, metadata):
    """全图统计摘要 Prompt（要求输出节点 ID）"""
    num_nodes = data.num_nodes
    num_edges = data.edge_index.shape[1]
    avg_degree = num_edges / max(1, num_nodes)

    # 提取前10个节点的度（作为示例）
    node_degrees = []
    for i in range(min(10, num_nodes)):
        degree = (data.edge_index[0] == i).sum().item() + (data.edge_index[1] == i).sum().item()
        node_degrees.append(f"node_{i}: degree={degree}")
    node_degree_str = ", ".join(node_degrees)

    prompt = f"""You are an APT attack chain analyst. Based on the statistical summary of a network traffic graph, reconstruct the attack chain step by step.

**Graph Summary**:
- Total nodes: {num_nodes}
- Total edges: {num_edges}
- Average node degree: {avg_degree:.2f}
- First 10 nodes and their degrees: {node_degree_str}

**Instructions**:
1. Identify possible attack phases: Reconnaissance, Initial Access, Delivery, C2, Exfiltration.
2. **CRITICAL**: For each phase you identify, you MUST output the specific node IDs that support your conclusion, in the exact format `node_123`.
3. If you cannot identify specific nodes for a phase, state "No specific nodes identified for this phase".
4. Format your response as a structured analysis with clear phase labels and node IDs.
"""
    return prompt


def main():
    parser = argparse.ArgumentParser(description="LLM Baseline - Full Stats only")
    parser.add_argument("--task", required=True, choices=["prepare-exam", "take-exam", "grade"])
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--max-samples", type=int, default=100)
    parser.add_argument("--model-name", default="qwen2.5-7b")
    parser.add_argument("--deploy-type", default="local", choices=["local", "api"])
    parser.add_argument("--model-dir", default=None)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--llm-root", default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    print("LLM Baseline Pipeline (Full Stats, no evidence)")
    print(f"task: {args.task}")

    # 设置根目录
    llm_root = args.llm_root or "outputs/llm_reports_baseline/"
    exam_dir = Path(llm_root) / "exam_papers"
    result_dir = Path(llm_root) / f"exam_results_{args.model_name}"

    if args.task == "prepare-exam":
        print(f"top_k: {args.top_k}")
        print(f"output: {exam_dir}")
        if args.dry_run:
            return

        preparer = ExamDataPreparer()
        # 替换 Prompt 构造方法
        preparer._build_prompt = baseline_build_prompt.__get__(preparer, ExamDataPreparer)
        # 修改输出目录
        preparer.exam_dir = exam_dir
        preparer.exam_dir.mkdir(parents=True, exist_ok=True)
        preparer.run_sampling(top_k=args.top_k)

    elif args.task == "take-exam":
        print(f"model_name: {args.model_name}")
        print(f"deploy_type: {args.deploy_type}")
        print(f"max_samples: {args.max_samples}")
        print(f"input: {exam_dir}")
        print(f"output: {result_dir}")
        if args.dry_run:
            return

        # 使用现有 LLMExamEngine，复用模型加载逻辑
        engine = LLMExamEngine(
            model_name=args.model_name,
            deploy_type=args.deploy_type,
            api_key=args.api_key,
            base_url=args.base_url,
            model_dir=args.model_dir,
            load_model=True,
        )
        # 修改输入输出目录
        engine.exam_dir = exam_dir
        engine.result_dir = result_dir
        engine.result_dir.mkdir(parents=True, exist_ok=True)
        engine.take_exam(max_samples=args.max_samples)

    elif args.task == "grade":
        print(f"llm_root: {llm_root}")
        if args.dry_run:
            return
        df = run_evaluation(
            save=True,
            output_dir=args.output_dir or "outputs/metrics/paper_tables/table10_llm_baseline.csv",
            llm_root=llm_root,
        )
        print(df.to_string(index=False))


if __name__ == "__main__":
    main()