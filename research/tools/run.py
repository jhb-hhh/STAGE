"""Simple user-facing entry point for STAGE reproducibility.

This script intentionally exposes only simple commands:

    python tools/run.py verify

    python tools/run.py reproduce detection
    python tools/run.py reproduce robustness
    python tools/run.py reproduce explanation
    python tools/run.py reproduce sensitivity
    python tools/run.py reproduce llm
    python tools/run.py reproduce all

    python tools/run.py rebuild graphs
    python tools/run.py rebuild splits
    python tools/run.py rebuild train
    python tools/run.py rebuild detection
    python tools/run.py rebuild robustness
    python tools/run.py rebuild explanation
    python tools/run.py rebuild sensitivity
    python tools/run.py rebuild llm-prepare
    python tools/run.py rebuild llm-answer
    python tools/run.py rebuild llm-grade
    python tools/run.py rebuild all

Advanced settings are intentionally kept as constants in this file or in
configs/*.yaml, rather than exposed as command-line arguments.
"""

import argparse
import os
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

PATH_PROFILE_ENV = "STAGE_PATH_PROFILE"
REFERENCE_PROFILE = "reference"
WORKSPACE_PROFILE = "workspace"

# Internal defaults. Change them here or in configs/*.yaml if needed.
DEFAULT_BATCH_SIZE = 16
DEFAULT_TRAIN_EPOCHS = 20
DEFAULT_ROBUSTNESS_FOLD = 2
DEFAULT_LLM_TOP_K = 100
DEFAULT_LLM_MAX_SAMPLES = 100
DEFAULT_LLM_MODEL_NAME = "qwen2.5-7b"
DEFAULT_LLM_DEPLOY_TYPE = "local"

# Rebuild training covers the final STAGE model, all ablation variants,
# and all registered baselines.
STAGE_TRAIN_MODE = "stage"
BASELINE_TRAIN_NAMES = [
    "TS-IDS",
    "FlowTransformer",
    "E-T-GraphSAGE",
    "DIDS-MFL",
    "GraphIDS",
]
ABLATION_TRAIN_MODES = [
    "gcn_only",
    "lstm_only",
    "inter_serial",
    "fusion_add",
    "fusion_concat",
    "fusion_cross_attn",
    "gcn_trans_rvg",
    "gat_lstm_rvg",
]

REPRODUCE_ROOT = PROJECT_ROOT / "outputs" / "reproduce"
WORKSPACE_ROOT = PROJECT_ROOT / "outputs" / "workspace"

REPRODUCE_SOURCE_METRICS = REPRODUCE_ROOT / "metrics" / "source_metrics"
REPRODUCE_PAPER_TABLES = REPRODUCE_ROOT / "metrics" / "paper_tables"
REPRODUCE_BENCHMARKS = REPRODUCE_ROOT / "metrics" / "benchmarks"
REPRODUCE_SENSITIVITY = REPRODUCE_ROOT / "metrics" / "sensitivity"

WORKSPACE_SOURCE_METRICS = WORKSPACE_ROOT / "metrics" / "source_metrics"
WORKSPACE_PAPER_TABLES = WORKSPACE_ROOT / "metrics" / "paper_tables"
WORKSPACE_BENCHMARKS = WORKSPACE_ROOT / "metrics" / "benchmarks"
WORKSPACE_SENSITIVITY = WORKSPACE_ROOT / "metrics" / "sensitivity"
WORKSPACE_CHECKPOINTS = WORKSPACE_ROOT / "checkpoints"
WORKSPACE_STAGE_CHECKPOINTS = WORKSPACE_CHECKPOINTS / "stage"
WORKSPACE_LLM_REPORTS = WORKSPACE_ROOT / "llm_reports"


def add_src_to_path():
    if str(SRC_DIR) not in sys.path:
        sys.path.insert(0, str(SRC_DIR))


@contextmanager
def active_profile(profile):
    old_value = os.environ.get(PATH_PROFILE_ENV)
    os.environ[PATH_PROFILE_ENV] = profile

    try:
        yield
    finally:
        if old_value is None:
            os.environ.pop(PATH_PROFILE_ENV, None)
        else:
            os.environ[PATH_PROFILE_ENV] = old_value


def command_env(profile):
    env = os.environ.copy()
    env[PATH_PROFILE_ENV] = profile

    old_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(SRC_DIR) if not old_pythonpath else f"{SRC_DIR}:{old_pythonpath}"

    return env


def py_cmd(*args):
    return [sys.executable, *map(str, args)]


def run_step(label, cmd, profile):
    print()
    print("=" * 72)
    print(f"[RUN] {label}")
    print(f"[PROFILE] {profile}")
    print("=" * 72)

    subprocess.run(
        list(map(str, cmd)),
        cwd=PROJECT_ROOT,
        env=command_env(profile),
        check=True,
    )


def mkdir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def count(path, pattern):
    path = Path(path)
    if not path.exists():
        return 0
    return len(list(path.glob(pattern)))


def rcount(path, pattern):
    path = Path(path)
    if not path.exists():
        return 0
    return len(list(path.rglob(pattern)))


def count_llm_result_jsons(llm_dir):
    """Count JSON files under exam_results_* directories.

    The result directories may be symlinks, so we intentionally glob each
    result directory directly instead of relying on recursive traversal from
    the LLM root.
    """
    llm_dir = Path(llm_dir)

    if not llm_dir.exists():
        return 0

    total = 0
    for result_dir in llm_dir.glob("exam_results_*"):
        if result_dir.exists() or result_dir.is_symlink():
            total += count(result_dir, "*.json")

    return total


def require_dir(path, hint):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Missing directory: {path}\n{hint}")


def require_file(path, hint):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Missing file: {path}\n{hint}")


def require_files(path, pattern, hint):
    path = Path(path)
    if not path.exists() or not list(path.glob(pattern)):
        raise FileNotFoundError(f"Missing files: {path}/{pattern}\n{hint}")


def require_recursive_files(path, pattern, hint):
    path = Path(path)
    if not path.exists() or not list(path.rglob(pattern)):
        raise FileNotFoundError(f"Missing files: {path}/**/{pattern}\n{hint}")


def get_path_func():
    add_src_to_path()
    from stage.utils.config import get_path

    return get_path


def verify_imports():
    add_src_to_path()

    modules = [
        "stage",
        "stage.utils.config",
        "stage.data.dataset",
        "stage.models.stage",
        "stage.training.checkpoint",
        "stage.evaluation.detection",
        "stage.evaluation.robustness",
        "stage.explanation.benchmark",
        "stage.explanation.sensitivity",
        "stage.llm.exam",
        "stage.llm.grader",
        "stage.analysis.paper_tables",
    ]

    print("Import check")
    for module in modules:
        __import__(module)
        print(f"[OK] {module}")


def verify_reference():
    get_path = get_path_func()

    with active_profile(REFERENCE_PROFILE):
        graph_dir = get_path("data", "graph_dir")
        shifted_graph_dir = get_path("data", "shifted_graph_dir")
        split_dir = get_path("data", "split_dir")
        stage_ckpt_dir = get_path("outputs", "stage_checkpoint_dir")
        ablation_ckpt_dir = get_path("outputs", "ablation_checkpoint_dir")
        baseline_ckpt_dir = get_path("outputs", "baseline_checkpoint_dir")
        explainer_ckpt_dir = get_path("outputs", "explainer_checkpoint_dir")
        explanation_dir = get_path("outputs", "explanation_dir")
        llm_dir = get_path("outputs", "llm_report_dir")
        metrics_dir = get_path("outputs", "metrics_dir")

    items = [
        ("graphs", graph_dir),
        ("shifted graphs", shifted_graph_dir),
        ("splits", split_dir),
        ("stage checkpoints", stage_ckpt_dir),
        ("ablation checkpoints", ablation_ckpt_dir),
        ("baseline checkpoints", baseline_ckpt_dir),
        ("PGExplainer checkpoint dir", explainer_ckpt_dir),
        ("explanations", explanation_dir),
        ("llm reports", llm_dir),
        ("metrics", metrics_dir),
    ]

    print()
    print("Reference path check")
    for name, path in items:
        status = "OK" if Path(path).exists() else "MISSING"
        print(f"[{status}] {name}: {path}")

    print()
    print("Reference artifact counts")
    print("graph chunks:", count(graph_dir, "graphs_chunk_*.pt"))
    print("shifted fold dirs:", count(shifted_graph_dir, "fold_*"))
    print("split files:", count(split_dir, "fold_*.json"))
    print("stage checkpoints:", count(stage_ckpt_dir, "best_model_fold_*.pth"))
    print("ablation checkpoints:", rcount(ablation_ckpt_dir, "best_model_fold_*.pth"))
    print("baseline checkpoints:", rcount(baseline_ckpt_dir, "*.pth"))
    print("PGExplainer checkpoints:", count(explainer_ckpt_dir, "pgexplainer_weights.pt"))
    print("exam papers:", count(llm_dir / "exam_papers", "*.json"))
    print("llm result json files:", count_llm_result_jsons(llm_dir))
    print("source metric csv files:", count(metrics_dir / "source_metrics", "*.csv"))


def verify():
    verify_imports()
    verify_reference()
    print()
    print("[DONE] verify")


def ensure_reference_core():
    get_path = get_path_func()

    with active_profile(REFERENCE_PROFILE):
        graph_dir = get_path("data", "graph_dir")
        split_dir = get_path("data", "split_dir")
        ckpt_dir = get_path("outputs", "stage_checkpoint_dir")

    require_files(graph_dir, "graphs_chunk_*.pt", "Check data/reference/graphs.")
    require_files(split_dir, "fold_*.json", "Check data/reference/splits.")
    require_files(ckpt_dir, "best_model_fold_*.pth", "Check outputs/reference/checkpoints/stage.")


def ensure_reference_pgexplainer_checkpoint():
    get_path = get_path_func()

    with active_profile(REFERENCE_PROFILE):
        explainer_dir = get_path("outputs", "explainer_checkpoint_dir")

    require_file(
        explainer_dir / "pgexplainer_weights.pt",
        "Check outputs/reference/checkpoints/explainers/pgexplainer_weights.pt.",
    )


def ensure_workspace_pgexplainer_checkpoint():
    get_path = get_path_func()

    with active_profile(WORKSPACE_PROFILE):
        explainer_dir = get_path("outputs", "explainer_checkpoint_dir")

    require_file(
        explainer_dir / "pgexplainer_weights.pt",
        "Prepare outputs/workspace/checkpoints/explainers/pgexplainer_weights.pt before running explanation or sensitivity.",
    )


def ensure_reference_detection_inputs(tasks):
    """Check reference inputs required by selected detection tasks."""
    get_path = get_path_func()

    with active_profile(REFERENCE_PROFILE):
        graph_dir = get_path("data", "graph_dir")
        split_dir = get_path("data", "split_dir")
        stage_checkpoint_dir = get_path("outputs", "stage_checkpoint_dir")
        baseline_checkpoint_dir = get_path("outputs", "baseline_checkpoint_dir")
        ablation_checkpoint_dir = get_path("outputs", "ablation_checkpoint_dir")

    require_files(graph_dir, "graphs_chunk_*.pt", "Check data/reference/graphs.")
    require_files(split_dir, "fold_*.json", "Check data/reference/splits.")

    if "stage" in tasks:
        require_files(
            stage_checkpoint_dir,
            "best_model_fold_*.pth",
            "Check outputs/reference/checkpoints/stage.",
        )

    if "baselines" in tasks:
        require_files(
            baseline_checkpoint_dir,
            "*.pth",
            "Check outputs/reference/checkpoints/baselines.",
        )

    if "ablations" in tasks:
        require_recursive_files(
            ablation_checkpoint_dir,
            "best_model_fold_*.pth",
            "Check outputs/reference/checkpoints/ablation.",
        )


def ensure_workspace_graphs():
    require_files(
        WORKSPACE_ROOT / "graphs",
        "graphs_chunk_*.pt",
        "Run first: python tools/run.py rebuild graphs",
    )


def ensure_workspace_splits():
    require_files(
        WORKSPACE_ROOT / "splits",
        "fold_*.json",
        "Run first: python tools/run.py rebuild splits",
    )


def ensure_workspace_stage_checkpoints():
    require_files(
        WORKSPACE_STAGE_CHECKPOINTS,
        "best_model_fold_*.pth",
        "Run first: python tools/run.py rebuild train",
    )


def require_at_least_k_files(path, pattern, expected_count, hint):
    path = Path(path)
    files = list(path.glob(pattern)) if path.exists() else []

    if len(files) < expected_count:
        raise FileNotFoundError(
            f"Missing files: {path}/{pattern}\n"
            f"Expected at least {expected_count}, found {len(files)}.\n"
            f"{hint}"
        )


def ensure_workspace_ablation_checkpoints():
    ablation_root = WORKSPACE_CHECKPOINTS / "ablation"

    for mode in ABLATION_TRAIN_MODES:
        require_at_least_k_files(
            ablation_root / f"kfold_experiment_{mode}",
            "best_model_fold_*.pth",
            5,
            "Run first: python tools/run.py rebuild train",
        )


def ensure_workspace_baseline_checkpoints():
    baseline_root = WORKSPACE_CHECKPOINTS / "baselines"

    for name in BASELINE_TRAIN_NAMES:
        require_at_least_k_files(
            baseline_root,
            f"{name}_fold_*.pth",
            5,
            "Run first: python tools/run.py rebuild train",
        )


def ensure_workspace_shifted_graphs():
    require_dir(
        WORKSPACE_ROOT / "graphs_shifted",
        "Generate outputs/workspace/graphs_shifted before running robustness.",
    )


def make_table(root, table, profile):
    source_dir = root / "metrics" / "source_metrics"
    output_dir = root / "metrics" / "paper_tables"
    mkdir(output_dir)

    run_step(
        f"make paper table {table}",
        py_cmd(
            "tools/make_paper_tables.py",
            "--table",
            table,
            "--source-dir",
            source_dir,
            "--output-dir",
            output_dir,
        ),
        profile,
    )


def make_all_tables(root, profile):
    """Generate all paper-ready tables from source metrics."""
    for table in ["table5", "table6", "table8", "table9", "table10"]:
        make_table(root, table, profile)


def reproduce_detection(model="all", fold="all"):
    tasks, model_name = resolve_detection_selection(model)
    scope, fold_id = resolve_detection_scope(fold)

    ensure_reference_detection_inputs(tasks)
    mkdir(REPRODUCE_SOURCE_METRICS)

    cmd_args = [
        "tools/evaluate_all.py",
        "--scope",
        scope,
    ]

    if fold_id is not None:
        cmd_args.extend(["--fold", str(fold_id)])

    cmd_args.extend(
        [
            "--batch-size",
            str(DEFAULT_BATCH_SIZE),
            "--output-dir",
            str(REPRODUCE_SOURCE_METRICS),
            "--tasks",
            *tasks,
        ]
    )

    if model_name is not None:
        cmd_args.extend(["--model-name", model_name])

    run_step(
        f"reproduce detection model={model} fold={fold}",
        py_cmd(*cmd_args),
        REFERENCE_PROFILE,
    )


def reproduce_robustness():
    ensure_reference_core()
    mkdir(REPRODUCE_SOURCE_METRICS)

    run_step(
        "reproduce robustness",
        py_cmd(
            "tools/evaluate_all.py",
            "--scope",
            "fold",
            "--fold",
            DEFAULT_ROBUSTNESS_FOLD,
            "--batch-size",
            DEFAULT_BATCH_SIZE,
            "--output-dir",
            REPRODUCE_SOURCE_METRICS,
            "--tasks",
            "robustness",
        ),
        REFERENCE_PROFILE,
    )



def reproduce_explanation():
    ensure_reference_core()
    ensure_reference_pgexplainer_checkpoint()
    mkdir(REPRODUCE_BENCHMARKS)
    mkdir(REPRODUCE_SOURCE_METRICS)

    run_step(
        "reproduce explanation benchmark",
        py_cmd(
            "tools/explain.py",
            "--task",
            "benchmark",
            "--run",
            "--results-dir",
            REPRODUCE_BENCHMARKS,
            "--source-metrics-dir",
            REPRODUCE_SOURCE_METRICS,
        ),
        REFERENCE_PROFILE,
    )



def reproduce_sensitivity():
    ensure_reference_core()
    ensure_reference_pgexplainer_checkpoint()
    mkdir(REPRODUCE_SENSITIVITY)

    run_step(
        "reproduce sensitivity",
        py_cmd(
            "tools/explain.py",
            "--task",
            "sensitivity",
            "--run",
            "--results-dir",
            REPRODUCE_SENSITIVITY,
        ),
        REFERENCE_PROFILE,
    )


def reproduce_llm():
    get_path = get_path_func()

    with active_profile(REFERENCE_PROFILE):
        llm_root = get_path("outputs", "llm_report_dir")

    mkdir(REPRODUCE_SOURCE_METRICS)

    run_step(
        "reproduce LLM grading",
        py_cmd(
            "tools/llm.py",
            "--task",
            "grade",
            "--llm-root",
            llm_root,
            "--output-dir",
            REPRODUCE_SOURCE_METRICS,
        ),
        REFERENCE_PROFILE,
    )



def rebuild_graphs():
    run_step(
        "rebuild graphs",
        py_cmd("tools/build_graphs.py", "--run"),
        WORKSPACE_PROFILE,
    )


def rebuild_splits():
    run_step(
        "rebuild splits",
        py_cmd("tools/make_splits.py"),
        WORKSPACE_PROFILE,
    )


def parse_train_fold(value):
    """Parse --fold for rebuild train."""
    value = str(value).strip().lower()

    if value == "all":
        return -1

    try:
        fold = int(value)
    except ValueError as error:
        raise ValueError("--fold must be one of: all, 0, 1, 2, 3, 4") from error

    if fold < 0 or fold > 4:
        raise ValueError("--fold must be one of: all, 0, 1, 2, 3, 4")

    return fold


def normalize_train_model(value):
    """Normalize --model for rebuild train."""
    value = str(value).strip()

    aliases = {
        "all": "all",
        "stage": "stage",
        "stage_final": "stage",
        "ablations": "ablations",
        "ablation": "ablations",
        "baselines": "baselines",
        "baseline": "baselines",
    }

    lower = value.lower()

    if lower in aliases:
        return aliases[lower]

    if value in ABLATION_TRAIN_MODES:
        return value

    if value in BASELINE_TRAIN_NAMES:
        return value

    valid = [
        "all",
        "stage",
        "ablations",
        *ABLATION_TRAIN_MODES,
        "baselines",
        *BASELINE_TRAIN_NAMES,
    ]

    raise ValueError(
        f"Unknown train model: {value}. "
        f"Valid values: {', '.join(valid)}"
    )


def selected_ablation_modes(model):
    """Return ablation modes selected by --model."""
    if model in ("all", "ablations"):
        return list(ABLATION_TRAIN_MODES)

    if model in ABLATION_TRAIN_MODES:
        return [model]

    return []


def selected_baseline_model(model):
    """Return baseline selector passed to tools/train_baselines.py."""
    if model in ("all", "baselines"):
        return "baselines"

    if model in BASELINE_TRAIN_NAMES:
        return model

    return None


def should_train_stage(model):
    """Return whether STAGE should be trained for this selector."""
    return model in ("all", "stage")


DETECTION_MODEL_TASKS = {
    "all": ["stage", "baselines", "ablations"],
    "stage": ["stage"],
    "baseline": ["baselines"],
    "baselines": ["baselines"],
    "ablation": ["ablations"],
    "ablations": ["ablations"],
}


def resolve_detection_selection(model):
    """Resolve --model for detection tasks.

    Returns:
        tasks: evaluate_all task names.
        model_name: optional single baseline or ablation model filter.
    """
    value = str(model).strip()
    lower = value.lower()

    if lower in DETECTION_MODEL_TASKS:
        return list(DETECTION_MODEL_TASKS[lower]), None

    for baseline_name in BASELINE_TRAIN_NAMES:
        if value == baseline_name or lower == baseline_name.lower():
            return ["baselines"], baseline_name

    for ablation_name in ABLATION_TRAIN_MODES:
        if value == ablation_name or lower == ablation_name.lower():
            return ["ablations"], ablation_name

    valid = [
        "all",
        "stage",
        "baselines",
        *BASELINE_TRAIN_NAMES,
        "ablations",
        *ABLATION_TRAIN_MODES,
    ]

    raise ValueError(
        f"--model {model!r} is not valid for detection. "
        f"Valid values are: {', '.join(valid)}."
    )


def resolve_detection_scope(fold):
    """Resolve --fold for detection tasks."""
    fold_id = parse_train_fold(fold)

    if fold_id == -1:
        return "all", None

    return "fold", fold_id


def link_stage_checkpoint_dir_if_needed():
    generated = WORKSPACE_CHECKPOINTS / "kfold_experiment_stage_final"
    target = WORKSPACE_STAGE_CHECKPOINTS

    if target.exists() or target.is_symlink():
        return

    if not generated.exists():
        return

    target.parent.mkdir(parents=True, exist_ok=True)
    relative = os.path.relpath(generated, target.parent)
    target.symlink_to(relative, target_is_directory=True)
    print(f"[LINK] {target} -> {relative}")


def rebuild_train(model="all", fold="all", dataset="ID-APT"):
    model = normalize_train_model(model)
    fold_id = parse_train_fold(fold)

    if dataset == "ID-APT":
        ensure_workspace_graphs()
        ensure_workspace_splits()
    elif model not in ("DIDS-MFL", "GraphIDS"):
        raise ValueError(
            "External datasets are supported only for DIDS-MFL or GraphIDS."
        )
    mkdir(WORKSPACE_CHECKPOINTS)

    if should_train_stage(model):
        run_step(
            "rebuild train STAGE",
            py_cmd(
                "tools/train.py",
                "--mode",
                STAGE_TRAIN_MODE,
                "--fold",
                fold_id,
                "--batch-size",
                DEFAULT_BATCH_SIZE,
                "--epochs",
                DEFAULT_TRAIN_EPOCHS,
                "--output-root",
                WORKSPACE_CHECKPOINTS,
            ),
            WORKSPACE_PROFILE,
        )

        link_stage_checkpoint_dir_if_needed()

    ablation_modes = selected_ablation_modes(model)

    if ablation_modes:
        ablation_root = WORKSPACE_CHECKPOINTS / "ablation"
        mkdir(ablation_root)

        for mode in ablation_modes:
            run_step(
                f"rebuild train ablation {mode}",
                py_cmd(
                    "tools/train.py",
                    "--mode",
                    mode,
                    "--fold",
                    fold_id,
                    "--batch-size",
                    DEFAULT_BATCH_SIZE,
                    "--epochs",
                    DEFAULT_TRAIN_EPOCHS,
                    "--output-root",
                    ablation_root,
                ),
                WORKSPACE_PROFILE,
            )

    baseline_model = selected_baseline_model(model)

    if baseline_model is not None:
        baseline_root = WORKSPACE_CHECKPOINTS / "baselines"
        mkdir(baseline_root)

        run_step(
            "rebuild train baselines",
            py_cmd(
                "tools/train_baselines.py",
                "--model",
                baseline_model,
                "--fold",
                "all" if fold_id == -1 else fold_id,
                "--dataset",
                dataset,
                "--batch-size",
                DEFAULT_BATCH_SIZE,
            ),
            WORKSPACE_PROFILE,
        )


def rebuild_detection(model="all", fold="all", dataset="ID-APT"):
    tasks, model_name = resolve_detection_selection(model)
    scope, fold_id = resolve_detection_scope(fold)

    if dataset == "ID-APT":
        ensure_workspace_graphs()
        ensure_workspace_splits()
    elif model not in ("DIDS-MFL", "GraphIDS"):
        raise ValueError(
            "External datasets are supported only for DIDS-MFL or GraphIDS."
        )

    if "stage" in tasks:
        ensure_workspace_stage_checkpoints()

    if "ablations" in tasks:
        ensure_workspace_ablation_checkpoints()

    if "baselines" in tasks and dataset == "ID-APT" and model in ("all", "baselines"):
        ensure_workspace_baseline_checkpoints()

    mkdir(WORKSPACE_SOURCE_METRICS)

    cmd_args = [
        "tools/evaluate_all.py",
        "--scope",
        scope,
    ]

    if fold_id is not None:
        cmd_args.extend(["--fold", str(fold_id)])

    cmd_args.extend(
        [
            "--batch-size",
            str(DEFAULT_BATCH_SIZE),
            "--output-dir",
            str(WORKSPACE_SOURCE_METRICS),
            "--tasks",
            *tasks,
            "--dataset",
            dataset,
            "--include-advanced",
        ]
    )

    if model_name is not None:
        cmd_args.extend(["--model-name", model_name])

    run_step(
        f"rebuild detection model={model} fold={fold}",
        py_cmd(*cmd_args),
        WORKSPACE_PROFILE,
    )


def rebuild_robustness():
    ensure_workspace_graphs()
    ensure_workspace_shifted_graphs()
    ensure_workspace_splits()
    ensure_workspace_stage_checkpoints()
    ensure_workspace_baseline_checkpoints()
    mkdir(WORKSPACE_SOURCE_METRICS)

    run_step(
        "rebuild robustness",
        py_cmd(
            "tools/evaluate_all.py",
            "--scope",
            "fold",
            "--fold",
            DEFAULT_ROBUSTNESS_FOLD,
            "--batch-size",
            DEFAULT_BATCH_SIZE,
            "--output-dir",
            WORKSPACE_SOURCE_METRICS,
            "--tasks",
            "robustness",
        ),
        WORKSPACE_PROFILE,
    )



def rebuild_explanation():
    ensure_workspace_graphs()
    ensure_workspace_splits()
    ensure_workspace_stage_checkpoints()
    ensure_workspace_pgexplainer_checkpoint()
    mkdir(WORKSPACE_BENCHMARKS)
    mkdir(WORKSPACE_SOURCE_METRICS)

    run_step(
        "rebuild explanation benchmark",
        py_cmd(
            "tools/explain.py",
            "--task",
            "benchmark",
            "--run",
            "--results-dir",
            WORKSPACE_BENCHMARKS,
            "--source-metrics-dir",
            WORKSPACE_SOURCE_METRICS,
        ),
        WORKSPACE_PROFILE,
    )



def rebuild_sensitivity():
    ensure_workspace_graphs()
    ensure_workspace_splits()
    ensure_workspace_stage_checkpoints()
    ensure_workspace_pgexplainer_checkpoint()
    mkdir(WORKSPACE_SENSITIVITY)

    run_step(
        "rebuild sensitivity",
        py_cmd(
            "tools/explain.py",
            "--task",
            "sensitivity",
            "--run",
            "--results-dir",
            WORKSPACE_SENSITIVITY,
        ),
        WORKSPACE_PROFILE,
    )


def rebuild_llm_prepare():
    mkdir(WORKSPACE_LLM_REPORTS)

    run_step(
        "rebuild LLM exam papers",
        py_cmd(
            "tools/llm.py",
            "--task",
            "prepare-exam",
            "--top-k",
            DEFAULT_LLM_TOP_K,
            "--llm-root",
            WORKSPACE_LLM_REPORTS,
        ),
        WORKSPACE_PROFILE,
    )


def rebuild_llm_answer():
    require_files(
        WORKSPACE_LLM_REPORTS / "exam_papers",
        "*.json",
        "Run first: python tools/run.py rebuild llm-prepare",
    )

    run_step(
        "rebuild LLM answers",
        py_cmd(
            "tools/llm.py",
            "--task",
            "take-exam",
            "--max-samples",
            DEFAULT_LLM_MAX_SAMPLES,
            "--model-name",
            DEFAULT_LLM_MODEL_NAME,
            "--deploy-type",
            DEFAULT_LLM_DEPLOY_TYPE,
            "--llm-root",
            WORKSPACE_LLM_REPORTS,
        ),
        WORKSPACE_PROFILE,
    )


def rebuild_llm_grade():
    require_recursive_files(
        WORKSPACE_LLM_REPORTS,
        "exam_results_*/*.json",
        "Run first: python tools/run.py rebuild llm-answer",
    )
    mkdir(WORKSPACE_SOURCE_METRICS)

    run_step(
        "rebuild LLM grading",
        py_cmd(
            "tools/llm.py",
            "--task",
            "grade",
            "--llm-root",
            WORKSPACE_LLM_REPORTS,
            "--output-dir",
            WORKSPACE_SOURCE_METRICS,
        ),
        WORKSPACE_PROFILE,
    )



def reproduce_tables():
    make_all_tables(REPRODUCE_ROOT, REFERENCE_PROFILE)


def rebuild_tables():
    make_all_tables(WORKSPACE_ROOT, WORKSPACE_PROFILE)


REPRODUCE_TASKS = {
    "detection": reproduce_detection,
    "robustness": reproduce_robustness,
    "explanation": reproduce_explanation,
    "sensitivity": reproduce_sensitivity,
    "llm": reproduce_llm,
    "tables": reproduce_tables,
}

REBUILD_TASKS = {
    "graphs": rebuild_graphs,
    "splits": rebuild_splits,
    "train": rebuild_train,
    "detection": rebuild_detection,
    "robustness": rebuild_robustness,
    "explanation": rebuild_explanation,
    "sensitivity": rebuild_sensitivity,
    "llm-prepare": rebuild_llm_prepare,
    "llm-answer": rebuild_llm_answer,
    "llm-grade": rebuild_llm_grade,
    "tables": rebuild_tables,
}


def accepts_model_and_fold(mode, task):
    """Return whether --model/--fold are valid for this command."""
    return (
        mode == "rebuild" and task == "train"
    ) or (
        mode in ("reproduce", "rebuild") and task == "detection"
    )


def reproduce_all():
    for task in ["detection", "robustness", "explanation", "sensitivity", "llm"]:
        REPRODUCE_TASKS[task]()


def rebuild_all():
    for task in [
        "graphs",
        "splits",
        "train",
        "detection",
        "robustness",
        "explanation",
        "sensitivity",
        "llm-prepare",
        "llm-answer",
        "llm-grade",
    ]:
        REBUILD_TASKS[task]()


def build_parser():
    parser = argparse.ArgumentParser(
        description="Simple STAGE reproducibility entry point."
    )
    parser.add_argument(
        "mode",
        choices=["verify", "reproduce", "rebuild"],
        help="Execution mode.",
    )
    parser.add_argument(
        "task",
        nargs="?",
        help="Task name for reproduce or rebuild.",
    )
    parser.add_argument(
        "--model",
        default="all",
        help="Model selector for rebuild train and detection tasks.",
    )
    parser.add_argument(
        "--fold",
        default="all",
        help="Fold selector for rebuild train and detection tasks: all, 0, 1, 2, 3, or 4.",
    )
    parser.add_argument(
        "--dataset",
        default="ID-APT",
        help="Dataset for advanced baselines: ID-APT, DAPT2020, or CIC-IDS-2018+Sandworm.",
    )
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.mode == "verify":
        if args.task is not None:
            parser.error("verify does not accept a task.")
        if args.model != "all" or args.fold != "all" or args.dataset != "ID-APT":
            parser.error(
                "--model and --fold are only valid for: "
                "rebuild train, reproduce detection, rebuild detection"
            )
        verify()
        return

    if args.task is None:
        parser.error(f"{args.mode} requires a task.")

    if not accepts_model_and_fold(args.mode, args.task):
        if args.model != "all" or args.fold != "all" or args.dataset != "ID-APT":
            parser.error(
                "--model, --fold, and --dataset are only valid for: "
                "rebuild train, reproduce detection, rebuild detection"
            )

    if args.mode == "reproduce":
        if args.dataset != "ID-APT":
            parser.error(
                "External advanced-baseline checkpoints are workspace artifacts; "
                "use rebuild detection instead of reproduce detection."
            )
        if args.task == "all":
            reproduce_all()
            return

        if args.task not in REPRODUCE_TASKS:
            valid = sorted(REPRODUCE_TASKS.keys()) + ["all"]
            parser.error(f"Unknown reproduce task: {args.task}. Valid tasks: {valid}")

        if args.task == "detection":
            REPRODUCE_TASKS[args.task](model=args.model, fold=args.fold)
        else:
            REPRODUCE_TASKS[args.task]()
        return

    if args.mode == "rebuild":
        if args.task == "all":
            rebuild_all()
            return

        if args.task not in REBUILD_TASKS:
            valid = sorted(REBUILD_TASKS.keys()) + ["all"]
            parser.error(f"Unknown rebuild task: {args.task}. Valid tasks: {valid}")

        if args.task in ("train", "detection"):
            REBUILD_TASKS[args.task](
                model=args.model,
                fold=args.fold,
                dataset=args.dataset,
            )
        else:
            REBUILD_TASKS[args.task]()
        return


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, ValueError) as error:
        print()
        print(f"[ERROR] {error}", file=sys.stderr)
        sys.exit(1)
    except subprocess.CalledProcessError as error:
        print()
        print(
            f"[ERROR] Step failed with exit code {error.returncode}.",
            file=sys.stderr,
        )
        sys.exit(error.returncode)
    except KeyboardInterrupt:
        print()
        print("[ERROR] Interrupted by user.", file=sys.stderr)
        sys.exit(130)
