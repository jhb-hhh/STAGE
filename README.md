# STAGE

STAGE is a research implementation for graph-based APT detection, flow-evidence extraction, and evidence-grounded behavior interpretation. This anonymous review copy contains source code, configuration, dataset descriptions, a sanitized graph sample, and fixed ID-APT split indices. It does not contain trained checkpoints, recorded predictions, experiment metrics, or paper tables.

## Datasets

| Dataset | Access and description |
| --- | --- |
| ID-APT | [Description](datasets/ID-APT.md), [65-graph sanitized sample](sample_data/), [full-data split indices](research/data/reference/splits/), and [archive manifest](manifest/idapt-manifest.json). The complete graph archives are distributed separately; the owner-identifying download URL is omitted from this double-blind copy. |
| DAPT 2020 | [Authors' download](https://gitlab.com/asu22/dapt2020) and [preparation instructions](datasets/DAPT2020.md). |
| Four external cases | [Original capture pages and case descriptions](datasets/real-cases.md). |

## Code

The [research implementation](research/README.md) contains the current detector, baseline adapters, graph construction, training, evaluation, and LLM code. The root-level [compact implementation](docs/compact-workflow.md) supports a local smoke test on the sanitized sample graphs. See the [code map](docs/code_structure.md) for both layouts.

## Run the sample workflow

Use Python 3.9 from the repository root. These commands create new outputs locally; results from 65 sample graphs are not paper-scale results.

~~~sh
python -m pip install -r requirements.txt
PYTHONPATH=src python tools/run.py train-stage --epochs 1 --fold 2
PYTHONPATH=src python tools/run.py eval-stage --checkpoint trained --fold 2
~~~

On Windows PowerShell, set `$env:PYTHONPATH="src"` before running the Python commands. See [sample workflow details](docs/compact-workflow.md) and [dataset/LLM notes](docs/data_and_llm.md).
