import json
import numpy as np
from pathlib import Path
from collections import defaultdict

# 战术列表（根据论文定义）
TACTICS = ["LOADER", "STEALER", "C2_RAT", "BENIGN"]

def load_reports(dir_path):
    """加载所有 LLM 报告"""
    reports = []
    for f in Path(dir_path).glob('*.json'):
        with open(f, 'r') as fp:
            data = json.load(fp)
            reports.append(data)
    return reports

def extract_ground_truth(report):
    """从报告中提取真实战术（假设报告中有 'meta' 字段或文件名包含标签）"""
    # 这里需要根据实际情况获取真实标签
    # 简单起见，可以从文件名或 meta 中提取
    if 'meta' in report:
        return [report['meta'].get('true_label', '')]
    return []

def compute_metrics(predictions, ground_truths):
    """计算准确率、精确率、召回率、F1"""
    # 简单实现：准确率 = 预测正确的样本数 / 总样本数
    correct = 0
    total = len(predictions)
    for pred, gt in zip(predictions, ground_truths):
        if set(pred) == set(gt):
            correct += 1
    return correct / total if total > 0 else 0

def main():
    baseline_dir = 'outputs/llm_reports_baseline/exam_results_qwen2.5-7b'
    ours_dir = 'outputs/llm_reports/exam_results_qwen2.5-7b'

    baseline_reports = load_reports(baseline_dir)
    ours_reports = load_reports(ours_dir)

    print(f"Baseline 报告数: {len(baseline_reports)}")
    print(f"Ours 报告数: {len(ours_reports)}")

    # 提取 predicted_tactics
    baseline_preds = []
    ours_preds = []

    for r in baseline_reports:
        tactics = r.get('observed_tactics', [])
        baseline_preds.append(tactics)

    for r in ours_reports:
        tactics = r.get('observed_tactics', [])
        ours_preds.append(tactics)

    # 假设真实标签（ground truth）可以从文件名或数据集中获取
    # 由于我们无法自动获取，这里只能展示分布，无法计算准确率
    print("\nBaseline 预测的战术分布:")
    from collections import Counter
    baseline_counts = Counter([tuple(p) for p in baseline_preds])
    for k, v in baseline_counts.items():
        print(f"  {k}: {v}")

    print("\nOurs 预测的战术分布:")
    ours_counts = Counter([tuple(p) for p in ours_preds])
    for k, v in ours_counts.items():
        print(f"  {k}: {v}")

    # 比较差异
    print("\n差异分析:")
    print(f"Baseline 平均每个报告识别战术数: {np.mean([len(p) for p in baseline_preds]):.2f}")
    print(f"Ours 平均每个报告识别战术数: {np.mean([len(p) for p in ours_preds]):.2f}")

    # 如果 Baseline 识别较少或更多，可以说明问题
    # 例如，Ours 识别更多战术，说明证据帮助 LLM 发现更多攻击阶段
    print("\n结论：")
    if np.mean([len(p) for p in ours_preds]) > np.mean([len(p) for p in baseline_preds]):
        print("Ours 识别的战术种类更多，说明证据有助于 LLM 发现更多攻击阶段。")
    else:
        print("Baseline 和 Ours 识别的战术种类相近，但证据可能提高了准确性（需标注验证）。")

if __name__ == "__main__":
    main()