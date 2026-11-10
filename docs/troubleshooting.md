# Troubleshooting

## torch-scatter warning

A PyTorch Geometric warning about torch-scatter acceleration may appear.

This warning does not stop execution.

## Nested tensor warning

E-T-GraphSAGE may trigger a PyTorch nested tensor warning.

This warning does not stop execution.

## Checkpoint key warnings

Some ablation variants may report checkpoint key differences.

These warnings are expected for compatibility paths and do not stop evaluation.

## Low metrics from trained checkpoints

Training commands use sample_data and short training runs.

Low metrics from locally trained checkpoints are expected.

## Missing trained checkpoint

If a command with --checkpoint trained fails, run the corresponding training command first.

STAGE:

    PYTHONPATH=src python tools/run.py train-stage

Baselines:

    PYTHONPATH=src python tools/run.py train-baselines

Ablations:

    PYTHONPATH=src python tools/run.py train-ablations

## LLM generation

The artifact does not execute LLM generation.

Run:

    PYTHONPATH=src python tools/run.py prepare-llm-input

to generate the offline LLM input sample.
