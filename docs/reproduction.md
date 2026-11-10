# Reproduction Guide

Before running the commands below, follow the environment setup in README.md.

Run all commands from the artifact root.

On Linux or macOS, use:

    PYTHONPATH=src python tools/run.py <command>

On Windows CMD, first run:

    set PYTHONPATH=src

Then use:

    python tools\run.py <command>

## 1. Verify

    PYTHONPATH=src python tools/run.py verify

Checks whether the artifact package is complete. It verifies required directories, sample data, label mapping, reference checkpoints, source metrics, and key Python imports.

This is the first command to run after installing dependencies.

## 2. Reference-checkpoint workflow

    PYTHONPATH=src python tools/run.py run-reference

Runs the main executable workflow using provided reference checkpoints on sample_data.

This command validates:

    STAGE evaluation
    baseline evaluation
    ablation evaluation
    robustness evaluation
    endogenous explanation
    offline LLM-input preparation

It does not train models.


The results are sample-data execution results. They validate that the workflow runs correctly, but they are not full-paper metrics.

## 3. Paper-table regeneration

    PYTHONPATH=src python tools/run.py make-tables

Regenerates paper-ready tables from reference source metrics.

It reads:

    reference_artifacts/metrics/source_metrics/

and writes:

    outputs/reproduce/paper_tables/

This is the command for reproducing paper-scale result tables from the provided source metrics.

## 4. Training workflow

    PYTHONPATH=src python tools/run.py run-training

Trains selected models on sample_data and evaluates the trained checkpoints.

This command validates that the training and re-evaluation code paths work. The resulting metrics may be low because the sample data is small and training is short.

It is not intended to reproduce full-paper metrics.

## 5. STAGE commands

### Train STAGE

    PYTHONPATH=src python tools/run.py train-stage

Trains the STAGE model on sample_data.

The trained checkpoint is written under:

    outputs/workspace/checkpoints/training/

### Evaluate the reference STAGE checkpoint

    PYTHONPATH=src python tools/run.py eval-stage

Loads the provided STAGE reference checkpoint and evaluates it on sample_data.

The output is written under:

    outputs/reproduce/evaluation/

### Evaluate the locally trained STAGE checkpoint

    PYTHONPATH=src python tools/run.py eval-stage --checkpoint trained

Loads the checkpoint produced by train-stage and evaluates it on sample_data.

Run train-stage before using this command.

## 6. Baseline commands

### Train all baseline models

    PYTHONPATH=src python tools/run.py train-baselines

Trains the baseline models on sample_data.

This can take longer than training a single baseline. For a shorter training-path check, train one model.

### Train one baseline model

    PYTHONPATH=src python tools/run.py train-baselines --models TS-IDS

Trains only the selected baseline model.

Available baseline names include:

    TS-IDS
    FlowTransformer
    E-T-GraphSAGE

### Evaluate reference baseline checkpoints

    PYTHONPATH=src python tools/run.py eval-baselines

Loads the provided baseline reference checkpoints and evaluates them on sample_data.

The output is written under:

    outputs/reproduce/baselines/

### Evaluate a locally trained baseline checkpoint

    PYTHONPATH=src python tools/run.py eval-baselines --checkpoint trained --models TS-IDS

Loads the checkpoint produced by train-baselines for the selected model and evaluates it on sample_data.

Run train-baselines first before using --checkpoint trained.

## 7. Ablation commands

### Train all ablation variants

    PYTHONPATH=src python tools/run.py train-ablations

Trains ablation variants on sample_data.

This can take longer than training selected variants.

### Train selected ablation variants

    PYTHONPATH=src python tools/run.py train-ablations --models stage_final,gcn_only

Trains only the selected variants.

Common variant names include:

    stage_final
    gcn_only
    lstm_only
    inter_serial
    fusion_add
    fusion_concat
    fusion_cross_attn
    gcn_trans_rvg
    gat_lstm_rvg

### Evaluate reference ablation checkpoints

    PYTHONPATH=src python tools/run.py eval-ablations

Loads provided ablation reference checkpoints and evaluates them on sample_data.

The output is written under:

    outputs/reproduce/ablations/

### Evaluate locally trained ablation checkpoints

    PYTHONPATH=src python tools/run.py eval-ablations --checkpoint trained --models stage_final,gcn_only

Loads checkpoints produced by train-ablations for the selected variants and evaluates them on sample_data.

Run train-ablations first before using --checkpoint trained.

## 8. Robustness commands

### Evaluate robustness with the reference STAGE checkpoint

    PYTHONPATH=src python tools/run.py eval-robustness

Loads the STAGE reference checkpoint, evaluates original sample validation graphs, applies perturbations at runtime, and evaluates the perturbed graphs.

The output is written under:

    outputs/reproduce/robustness/

### Evaluate robustness with the locally trained STAGE checkpoint

    PYTHONPATH=src python tools/run.py eval-robustness --checkpoint trained

Uses the checkpoint produced by train-stage.

Run train-stage before using this command.

## 9. Explanation commands

### Run explanation with the reference STAGE checkpoint

    PYTHONPATH=src python tools/run.py explain-stage

Runs lightweight endogenous explanation on sample_data using the provided STAGE reference checkpoint.

The output is written under:

    outputs/reproduce/explanations/

### Run explanation with the locally trained STAGE checkpoint

    PYTHONPATH=src python tools/run.py explain-stage --checkpoint trained

Runs explanation using the checkpoint produced by train-stage.

Run train-stage before using this command.

## 10. Offline LLM-input preparation

    PYTHONPATH=src python tools/run.py prepare-llm-input

Converts explanation output into an offline LLM input sample.

This command does not execute LLM generation and does not require an API key or local LLM weights.

The output is written under:

    outputs/reproduce/llm/

## 11. Recommended review order

For a quick artifact review, run:

    PYTHONPATH=src python tools/run.py verify
    PYTHONPATH=src python tools/run.py run-reference
    PYTHONPATH=src python tools/run.py make-tables

To additionally validate training paths, run:

    PYTHONPATH=src python tools/run.py run-training
