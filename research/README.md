# Full experiment workflows

This directory contains the research implementation and configuration used on the STAGE server. Run commands from this directory with `src` on the Python import path.

~~~sh
python -m pip install -r requirements.txt
export PYTHONPATH=src
python tools/run.py --help
~~~

* [ID-APT graph release and splits](../datasets/ID-APT.md). The release installs under `data/reference`. On Linux, create the expected workspace paths once from this directory with the commands below. Model and evaluation output belongs under `outputs/workspace`.
* [DAPT benchmark](../datasets/DAPT2020.md). Use the standalone benchmark runner for the reported six-model, within-DAPT comparison.
* [Four external case studies](../datasets/real-cases.md). The LLM [forensic report](src/stage/llm/analyst.py) and [behavior identification](src/stage/llm/exam.py) workflows use different prompts. For local generation, configure the local model directory. API-backed generation reads the configured provider credentials from the environment.
* [Additional baseline adapters](src/stage/baselines/registry.py). For example, with full-data paths configured, run `python tools/run.py rebuild train --model GraphIDS --fold 0` and `python tools/run.py rebuild detection --model GraphIDS --fold 0`. Substitute `DIDS-MFL` for the second adapter.

The historical [reproduction manual](docs/reproduction.md) describes the original server's full reference tree. Selected public checkpoints and source metrics are indexed in the [repository README](../README.md). The research runner's full reference verification expects additional files from that tree. Use the compact runner at the repository root for a self-contained smoke test.

## Use the released ID-APT folds for a rebuild

From this directory, after unpacking the full release, run these commands in a fresh workspace:

~~~sh
mkdir -p outputs/workspace
ln -s ../../data/reference/graphs outputs/workspace/graphs
ln -s ../../data/reference/graphs_shifted outputs/workspace/graphs_shifted
ln -s ../../data/reference/splits outputs/workspace/splits
~~~