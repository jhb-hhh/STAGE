# STAGE

STAGE detects APT activity in windows of network flows and ranks individual flows as evidence. Its LLM workflows produce a forensic report for interpretation and standardized behavior labels for evaluation.

## Datasets

| Dataset | Public access | Contents |
| --- | --- | --- |
| ID-APT | [Full graph release](https://github.com/jhb-hhh/STAGE/releases/tag/id-apt-v1) · [Installation](datasets/ID-APT.md) · [65-graph sample](sample_data) | 23,719 original window graphs, 12 attack scenarios and benign traffic, five fixed splits, and fold-specific perturbed graphs. |
| DAPT 2020 | [Authors' dataset](https://gitlab.com/asu22/dapt2020) · [Reproduction instructions](datasets/DAPT2020.md) | Ten labeled flow CSVs used for the within-dataset binary experiment. |
| Four external cases | [Case-by-case capture links](datasets/real-cases.md) | QakBot BB12, Rig EK/RedLine, IcedID/Keyhole VNC, and malvertising/IcedID. |

## Quick start

The root directory follows the original compact artifact layout. Use Python 3.9 and run these commands from the repository root.

~~~sh
python -m pip install -r requirements.txt
python tools/run.py verify
python tools/run.py run-reference
python tools/run.py make-tables
~~~

The sample workflow evaluates supplied reference weights on 65 graphs. The compact table command reads the archived source metrics. See the [compact workflow guide](docs/compact-workflow.md) for training and output paths.

## Current experiments

Use [research/](research/README.md) for the current detector, five baseline adapters, graph construction, perturbation studies and LLM code. The compact runner and the research runner each resolve their own configuration relative to their project root.

| Output | Location |
| --- | --- |
| Archived compact checkpoints and original tables | [checkpoints](reference_artifacts/checkpoints), [tables](reference_artifacts/paper_tables) |
| Six-model DAPT 2020 evaluation | [five-fold summary](reference_artifacts/metrics/dapt2020/checked_summary.csv), [fold metrics and confusion counts](reference_artifacts/metrics/dapt2020/checked_fold_metrics.csv), [selected checkpoints](reference_artifacts/checkpoints/dapt2020-selected) |
| Additional ID-APT baselines | [metrics](reference_artifacts/metrics/id-apt-new-baselines), [checkpoints](reference_artifacts/checkpoints/id-apt-new-baselines) |
| ID-APT identity experiments | [fold results and summaries](reference_artifacts/metrics/id-apt-perturbation) |

The DAPT experiment trains and tests on DAPT. The ID-APT and DAPT results use their respective protocols. See the [model source notes](third_party/README.md) and the [compact artifact manifest](manifest/artifact_file_list.md).
