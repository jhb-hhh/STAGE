# Code Structure

## Top-level directories

    configs/
        Configuration files.

    docs/
        Documentation.

    manifest/
        File manifest and checksums.

    reference_artifacts/
        Reference checkpoints, source metrics, and paper-table materials.

    sample_data/
        Sanitized sample graph data, splits, and label mapping.

    src/stage/
        Main source package.

    tools/
        Command-line runners.

    outputs/
        Generated outputs.

## Source modules

    src/stage/data/
        Dataset loading.

    src/stage/models/
        STAGE model and ablation variants.

    src/stage/baselines/
        Baseline models.

    src/stage/training/
        Training utilities.

    src/stage/evaluation/
        Detection, baseline, ablation, and robustness evaluation.

    src/stage/explanation/
        Endogenous explanation utilities.

    src/stage/llm/
        LLM-related utilities. The artifact only prepares offline LLM input.

    src/stage/analysis/
        Paper-table generation.

    src/stage/utils/
        Configuration and path helpers.

## Main tools

    tools/run.py
        Unified artifact runner.

    tools/baseline_runner.py
        Baseline training and evaluation runner.

    tools/ablation_runner.py
        Ablation training and evaluation runner.

    tools/table_utils.py
        Console table preview helper.

    tools/train.py
        Lower-level STAGE training entry.

    tools/train_baselines.py
        Lower-level baseline training entry.

    tools/explain.py
        Lower-level explanation entry.

    tools/make_paper_tables.py
        Lower-level paper-table generation entry.
