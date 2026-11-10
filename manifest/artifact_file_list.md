# Artifact File List

This manifest summarizes the major files and directories included in the STAGE artifact.

## Code

- src/stage/
- tools/run.py
- tools/baseline_runner.py
- tools/ablation_runner.py
- tools/table_utils.py
- tools/train.py
- tools/train_baselines.py
- tools/explain.py
- tools/make_paper_tables.py

## Configuration

- configs/paths.yaml
- configs/model.yaml
- configs/train.yaml
- configs/baselines.yaml
- configs/explainers.yaml
- configs/llm.yaml
- configs/dataset.yaml

## Documentation

- README.md
- docs/reproduction.md
- docs/data_and_llm.md
- docs/code_structure.md
- docs/troubleshooting.md

## Sample data

- sample_data/graphs/graphs_chunk_000.pt
- sample_data/splits/
- sample_data/label_mapping.json
- sample_data/README.md
- manifest/sample_data_manifest.json

## Reference artifacts

- reference_artifacts/checkpoints/stage/stage_reference_checkpoint.pth
- reference_artifacts/checkpoints/baselines/
- reference_artifacts/checkpoints/ablation/
- reference_artifacts/metrics/source_metrics/
- reference_artifacts/paper_tables/

## Generated outputs

The following directories are generated or updated by artifact commands:

- outputs/reproduce/
- outputs/workspace/

Generated outputs can be recreated by running the commands in README.md.
