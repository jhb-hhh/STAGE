# Data and LLM Notes

## Data

The full original dataset used to derive the large-scale experiments is larger than 500GB and is not included in this compact artifact.

The artifact includes a sanitized sample graph subset:

    sample_data/

This sample data is used for executable workflow validation, including training, evaluation, baseline comparison, ablation evaluation, robustness testing, explanation, and LLM-input preparation.

Sample-data results are not full-paper metrics.

Paper-scale tables are regenerated from:

    reference_artifacts/metrics/source_metrics/

using:

    PYTHONPATH=src python tools/run.py make-tables

## Robustness data

The artifact does not require a separate shifted dataset.

The robustness command applies perturbations to sample validation graphs at runtime:

    PYTHONPATH=src python tools/run.py eval-robustness

## LLM

The artifact does not execute LLM generation.

LLM generation would require external API access or local LLM weights, which can introduce cost, availability, versioning, hardware, and reproducibility issues.

The artifact provides the executable evidence-to-LLM input preparation step:

    PYTHONPATH=src python tools/run.py prepare-llm-input

This command generates an offline LLM input sample from extracted evidence.

No API key, local LLM weight, or LLM-specific GPU environment is required.
