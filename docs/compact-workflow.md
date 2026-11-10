# STAGE Artifact

This artifact provides a compact executable package for evaluating the STAGE workflow.

It includes source code, sample graph data, reference checkpoints, source metrics, and command-line workflows for artifact evaluation.

The full original dataset used to derive the large-scale experiments is not included because of its size.

## 1. Environment Setup

Python 3.9 is required for this artifact. We recommend using an isolated Python environment rather than the system Python or the Conda base environment.

You can use either Conda or Python venv.

### Option A: Conda

If Conda or Miniconda is available, create a new environment:

    conda create -n stage_artifact python=3.9
    conda activate stage_artifact

### Option B: Python venv

If Conda is not available, use Python's built-in venv module.

On Linux or macOS:

    python3.9 -m venv .venv
    source .venv/bin/activate

On Windows CMD:

    py -3.9 -m venv .venv
    .venv\Scripts\activate

On Windows PowerShell:

    py -3.9 -m venv .venv
    .venv\Scripts\Activate.ps1

If the py launcher is not available on Windows, install Python 3.9 first and make sure it is available from the terminal.

Check the active Python version with:

    python --version

After activating the environment, install dependencies from the artifact root:

    python -m pip install --upgrade pip
    pip install -r requirements.txt

CPU execution is supported by default. CUDA can be used by passing:

    --device cuda

## 2. Verify the Artifact

Run:

    PYTHONPATH=src python tools/run.py verify

This checks required files, sample data, checkpoints, source metrics, and Python imports.

On Windows CMD, use:

    set PYTHONPATH=src
    python tools\run.py verify

## 3. Recommended Workflows

Run the reference-checkpoint workflow:

    PYTHONPATH=src python tools/run.py run-reference

This uses provided reference checkpoints on sample_data and validates the executable evaluation workflow.

Regenerate paper tables:

    PYTHONPATH=src python tools/run.py make-tables

This regenerates paper-ready tables from reference source metrics.

Validate training paths:

    PYTHONPATH=src python tools/run.py run-training

This trains selected models on sample_data and evaluates the trained checkpoints.

## 4. Outputs

Generated outputs are written to:

    outputs/reproduce/
    outputs/workspace/

Reference checkpoints and source metrics are stored in:

    reference_artifacts/

Sample graph data is stored in:

    sample_data/

Most evaluation commands save CSV/JSON files and print compact table previews in the terminal.

## 5. More Documentation

For detailed reproduction commands, see:

    docs/reproduction.md

For data and LLM scope, see:

    docs/data_and_llm.md

For code structure, see:

    docs/code_structure.md

For common warnings and issues, see:

    docs/troubleshooting.md
