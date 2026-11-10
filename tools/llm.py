import argparse

from stage.llm.exam import ExamDataPreparer, LLMExamEngine
from stage.llm.grader import run_evaluation


def main():
    parser = argparse.ArgumentParser(description="Run STAGE LLM attribution pipeline.")
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        choices=["prepare-exam", "take-exam", "grade"],
        help="LLM pipeline task.",
    )

    parser.add_argument("--top-k", type=int, default=100, help="Number of top samples for exam preparation.")
    parser.add_argument("--max-samples", type=int, default=100, help="Maximum number of exam papers to run.")

    parser.add_argument("--model-name", type=str, default="qwen2.5-7b")
    parser.add_argument("--deploy-type", type=str, default="local", choices=["local", "api"])
    parser.add_argument("--model-dir", type=str, default=None, help="Local model directory.")
    parser.add_argument("--api-key", type=str, default=None, help="Optional API key. Prefer environment variables.")
    parser.add_argument("--base-url", type=str, default=None, help="Optional API base URL.")

    parser.add_argument(
        "--llm-root",
        type=str,
        default=None,
        help="Optional LLM report root containing exam_papers and exam_results_<model>.",
    )
    parser.add_argument(
        "--exam-dir",
        type=str,
        default=None,
        help="Optional exam paper directory.",
    )
    parser.add_argument(
        "--result-dir",
        type=str,
        default=None,
        help="Optional result directory for take-exam.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Optional output directory for LLM grading table.",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the plan without loading models or calling APIs.",
    )

    args = parser.parse_args()

    print("LLM pipeline plan")
    print("task:", args.task)

    if args.task == "prepare-exam":
        print("top_k:", args.top_k)

        if args.exam_dir is not None:
            print("output:", args.exam_dir)
        elif args.llm_root is not None:
            print("output:", f"{args.llm_root}/exam_papers/")
        else:
            print("output: outputs/llm_reports/exam_papers/")

        if args.dry_run:
            print("dry-run only; exam papers are not generated.")
            return

        preparer = ExamDataPreparer(
            exam_dir=args.exam_dir,
            llm_root=args.llm_root,
        )
        preparer.run_sampling(top_k=args.top_k)

    elif args.task == "take-exam":
        print("model_name:", args.model_name)
        print("deploy_type:", args.deploy_type)
        print("max_samples:", args.max_samples)

        if args.exam_dir is not None:
            print("input:", args.exam_dir)
        elif args.llm_root is not None:
            print("input:", f"{args.llm_root}/exam_papers/")
        else:
            print("input: outputs/llm_reports/exam_papers/")

        if args.result_dir is not None:
            print("output:", args.result_dir)
        elif args.llm_root is not None:
            print("output:", f"{args.llm_root}/exam_results_{args.model_name}/")
        else:
            print("output: outputs/llm_reports/exam_results_<model_name>/")

        if args.dry_run:
            print("dry-run only; LLM is not loaded and API is not called.")
            return

        engine = LLMExamEngine(
            model_name=args.model_name,
            deploy_type=args.deploy_type,
            api_key=args.api_key,
            base_url=args.base_url,
            model_dir=args.model_dir,
            load_model=True,
            exam_dir=args.exam_dir,
            result_dir=args.result_dir,
            llm_root=args.llm_root,
        )
        engine.take_exam(max_samples=args.max_samples)

    elif args.task == "grade":
        if args.llm_root is None:
            print("input: outputs/llm_reports/exam_results_*/")
        else:
            print("input:", f"{args.llm_root}/exam_results_*/")

        if args.output_dir is None:
            print("output: outputs/metrics/paper_tables/table10_llm.csv")
        else:
            print("output:", args.output_dir)

        if args.dry_run:
            print("dry-run only; grading is not executed.")
            return

        df = run_evaluation(
            save=True,
            output_dir=args.output_dir,
            llm_root=args.llm_root,
        )
        print(df.to_string(index=False))


if __name__ == "__main__":
    main()
