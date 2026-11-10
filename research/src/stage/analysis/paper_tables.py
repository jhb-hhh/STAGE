"""Generate paper-ready CSV tables from neutral source metrics."""

from pathlib import Path

import pandas as pd

from stage.llm.grader import run_evaluation
from stage.utils.config import get_path

METRIC_COLUMNS = ["Accuracy", "Precision", "Recall", "F1_Macro", "FPR", "MCC"]

MODEL_DISPLAY_NAMES = {
    "stage_final": "STAGE",
    "gcn_only": "GCN only",
    "lstm_only": "LSTM only",
    "inter_serial": "Serial GCN-LSTM",
    "gcn_trans_rvg": "GCN+Transformer+RVG",
    "gat_lstm_rvg": "GAT+LSTM+RVG",
    "fusion_add": "Add fusion",
    "fusion_concat": "Concat fusion",
    "fusion_cross_attn": "Cross-attention fusion",
}

BASELINE_ORDER = ["TS-IDS", "FlowTransformer", "E-T-GraphSAGE"]


SOURCE_DIR_OVERRIDE = None
PAPER_DIR_OVERRIDE = None


def set_table_dirs(source_dir=None, output_dir=None):
    """Override source/output directories for paper-table generation."""
    global SOURCE_DIR_OVERRIDE, PAPER_DIR_OVERRIDE
    SOURCE_DIR_OVERRIDE = Path(source_dir) if source_dir is not None else None
    PAPER_DIR_OVERRIDE = Path(output_dir) if output_dir is not None else None



def _metrics_dir() -> Path:
    """Return the metrics output directory."""
    return get_path("outputs", "metrics_dir")


def _source_dir() -> Path:
    """Return the source-metric directory."""
    return _metrics_dir() / "source_metrics"


def _raw_dir() -> Path:
    if SOURCE_DIR_OVERRIDE is not None:
        return SOURCE_DIR_OVERRIDE
    return _metrics_dir() / "source_metrics"


def _paper_dir() -> Path:
    if PAPER_DIR_OVERRIDE is not None:
        output_dir = PAPER_DIR_OVERRIDE
    else:
        output_dir = _metrics_dir() / "paper_tables"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def _source_paths():
    """Return source CSV paths used to build paper tables."""
    source_dir = _raw_dir()

    return {
        "detection_baselines": source_dir / "baseline_detection.csv",
        "ablation_summary": source_dir / "ablation_summary.csv",
        "robustness_baselines": source_dir / "robustness_baselines.csv",
        "robustness_stage": source_dir / "robustness_stage.csv",
        "explanation_benchmark": source_dir / "explanation_benchmark.csv",
    }


def _clean_value(value):
    """Normalize string-like CSV cells."""
    if pd.isna(value):
        return value

    return str(value).strip().lstrip("'").strip()


def _read_csv(path: Path) -> pd.DataFrame:
    """Read and normalize a source CSV file."""
    if not path.exists():
        raise FileNotFoundError(f"Required source CSV not found: {path}")

    df = pd.read_csv(path, encoding="utf-8-sig")

    for column in df.columns:
        if df[column].dtype == object:
            df[column] = df[column].map(_clean_value)

    return df


def _require_columns(df: pd.DataFrame, columns, source_name: str):
    """Validate that a dataframe contains required columns."""
    missing = [column for column in columns if column not in df.columns]

    if missing:
        raise ValueError(f"{source_name} is missing required columns: {missing}")


def _single_row(df: pd.DataFrame, condition, source_name: str) -> pd.Series:
    """Return exactly one row matching a condition."""
    rows = df[condition]

    if len(rows) != 1:
        raise ValueError(f"{source_name} expected exactly one matching row, got {len(rows)}")

    return rows.iloc[0]


def _format_float(value, digits=4):
    """Format a numeric value with fixed decimals."""
    return f"{float(value):.{digits}f}"


def _assert_no_empty_table(df: pd.DataFrame, table_name: str):
    """Reject empty generated tables."""
    if df.empty:
        raise ValueError(f"{table_name} is empty")


def _write_table(df: pd.DataFrame, filename: str):
    """Write one paper-ready table."""
    _assert_no_empty_table(df, filename)

    output_path = _paper_dir() / filename
    df.to_csv(output_path, index=False)

    return output_path, df


def make_table5_detection():
    """Generate Table 5: detection comparison."""
    sources = _source_paths()

    baseline_df = _read_csv(sources["detection_baselines"])
    ablation_df = _read_csv(sources["ablation_summary"])

    _require_columns(baseline_df, ["Model"] + METRIC_COLUMNS, "detection_baselines")
    _require_columns(ablation_df, ["Mode"] + METRIC_COLUMNS, "ablation_summary")

    baseline_df = baseline_df[baseline_df["Model"].isin(BASELINE_ORDER)].copy()
    baseline_df["Model"] = pd.Categorical(
        baseline_df["Model"],
        categories=BASELINE_ORDER,
        ordered=True,
    )
    baseline_df = baseline_df.sort_values("Model")
    baseline_df["Model"] = baseline_df["Model"].astype(str)

    stage_row = _single_row(
        ablation_df,
        ablation_df["Mode"] == "stage_final",
        "ablation_summary: stage_final",
    )

    stage_data = {"Model": "STAGE"}

    for column in METRIC_COLUMNS:
        stage_data[column] = stage_row[column]

    output_df = pd.concat(
        [baseline_df[["Model"] + METRIC_COLUMNS], pd.DataFrame([stage_data])],
        ignore_index=True,
    )

    return _write_table(output_df, "table5_detection.csv")


def make_table6_robustness():
    """Generate Table 6: robustness under shifted-domain graphs."""
    sources = _source_paths()

    baseline_df = _read_csv(sources["robustness_baselines"])
    stage_df = _read_csv(sources["robustness_stage"])

    _require_columns(
        baseline_df,
        ["Model", "Test Set", "F1", "Rec", "Prec", "Acc", "Drop%"],
        "robustness_baselines",
    )
    _require_columns(
        stage_df,
        ["Test Set", "F1-Score", "Recall", "Precision", "Accuracy", "F1 Drop"],
        "robustness_stage",
    )

    rows = []

    for model_name in BASELINE_ORDER:
        subset = baseline_df[baseline_df["Model"] == model_name]
        original = _single_row(
            subset,
            subset["Test Set"] == "Original",
            f"robustness_baselines: {model_name} original",
        )
        shifted = _single_row(
            subset,
            subset["Test Set"] == "Shifted",
            f"robustness_baselines: {model_name} shifted",
        )

        rows.append(
            {
                "Model": model_name,
                "Original_F1": _format_float(original["F1"]),
                "Shifted_F1": _format_float(shifted["F1"]),
                "Original_Recall": _format_float(original["Rec"]),
                "Shifted_Recall": _format_float(shifted["Rec"]),
                "Original_Precision": _format_float(original["Prec"]),
                "Shifted_Precision": _format_float(shifted["Prec"]),
                "Original_Accuracy": _format_float(original["Acc"]),
                "Shifted_Accuracy": _format_float(shifted["Acc"]),
                "F1_Drop": shifted["Drop%"],
            }
        )

    stage_original = _single_row(
        stage_df,
        stage_df["Test Set"] == "Original",
        "robustness_stage: original",
    )
    stage_shifted = _single_row(
        stage_df,
        stage_df["Test Set"] == "Shifted",
        "robustness_stage: shifted",
    )

    rows.append(
        {
            "Model": "STAGE",
            "Original_F1": _format_float(stage_original["F1-Score"]),
            "Shifted_F1": _format_float(stage_shifted["F1-Score"]),
            "Original_Recall": _format_float(stage_original["Recall"]),
            "Shifted_Recall": _format_float(stage_shifted["Recall"]),
            "Original_Precision": _format_float(stage_original["Precision"]),
            "Shifted_Precision": _format_float(stage_shifted["Precision"]),
            "Original_Accuracy": _format_float(stage_original["Accuracy"]),
            "Shifted_Accuracy": _format_float(stage_shifted["Accuracy"]),
            "F1_Drop": stage_shifted["F1 Drop"],
        }
    )

    output_df = pd.DataFrame(rows)
    return _write_table(output_df, "table6_robustness.csv")


def make_table8_ablation():
    """Generate Table 8: architecture ablation."""
    sources = _source_paths()
    df = _read_csv(sources["ablation_summary"])

    _require_columns(df, ["Mode"] + METRIC_COLUMNS, "ablation_summary")

    rows = []

    for mode, display_name in MODEL_DISPLAY_NAMES.items():
        row = _single_row(
            df,
            df["Mode"] == mode,
            f"ablation_summary: {mode}",
        )

        item = {"Mode": display_name}

        for column in METRIC_COLUMNS:
            item[column] = row[column]

        rows.append(item)

    output_df = pd.DataFrame(rows)
    return _write_table(output_df, "table8_ablation.csv")


def make_table9_explainers():
    """Generate Table 9: explanation benchmark."""
    sources = _source_paths()
    df = _read_csv(sources["explanation_benchmark"])

    source_renames = {
        "iou": "IoU",
        "precision": "Precision",
        "fid+": "FidPlus",
        "fid-": "FidMinus",
        "rob_fid+": "RobFidPlus",
        "rob_fid-": "RobFidMinus",
        "consistency": "Consistency",
        "time": "TimeSec",
    }

    df = df.rename(columns=source_renames)

    columns = [
        "Method",
        "Count",
        "IoU",
        "Precision",
        "FidPlus",
        "FidMinus",
        "RobFidPlus",
        "RobFidMinus",
        "Consistency",
        "TimeSec",
    ]

    _require_columns(df, columns, "explanation_benchmark")

    output_df = df[columns].copy()
    numeric_cols = [column for column in columns if column not in {"Method", "Count"}]

    for column in numeric_cols:
        output_df[column] = output_df[column].astype(float).map(lambda value: _format_float(value))

    return _write_table(output_df, "table9_explainers.csv")


def make_table10_llm():
    """Generate Table 10: LLM attribution."""
    output_df = run_evaluation(save=False)
    return _write_table(output_df, "table10_llm.csv")


def make_all_tables(include_llm=True):
    """Generate all paper-ready tables."""
    makers = [
        make_table5_detection,
        make_table6_robustness,
        make_table8_ablation,
        make_table9_explainers,
    ]

    results = [maker() for maker in makers]

    if include_llm:
        results.append(make_table10_llm())

    return results
