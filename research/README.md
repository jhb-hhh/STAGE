# Full research implementation

This directory contains the current STAGE detector, baseline adapters, graph construction, evaluation, and LLM code. Run commands from `research/` with `src` on the Python import path.

~~~sh
python -m pip install -r requirements.txt
export PYTHONPATH=src
python tools/run.py --help
~~~

The [ID-APT description](../datasets/ID-APT.md) explains the graph format and split indices. The [DAPT 2020 instructions](../datasets/DAPT2020.md) use the original authors' CSVs. The [external-case page](../datasets/real-cases.md) links the source captures. With full graph archives installed under `data/reference/`, the current runner supports, for example, `python tools/run.py rebuild train --model GraphIDS --fold 0` and `python tools/run.py rebuild detection --model GraphIDS --fold 0`; substitute `DIDS-MFL` for the second adapter. Generated checkpoints and results belong under `outputs/workspace/` and are not bundled here.

The [baseline registry](src/stage/baselines/registry.py), [forensic-report code](src/stage/llm/analyst.py), and [behavior-identification code](src/stage/llm/exam.py) provide the corresponding implementation entry points. The [historical manual](docs/reproduction.md) describes additional server-side paths that may require resources beyond this anonymous source copy.
