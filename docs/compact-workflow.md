# Compact sample workflow

This directory contains a sanitized 65-graph sample, source code, and configuration for a smoke test. No pre-trained weights, recorded predictions, metrics, or paper tables are included.

From the repository root, install `requirements.txt`, set `PYTHONPATH=src`, and run:

~~~sh
python tools/run.py train-stage --epochs 1 --fold 2
python tools/run.py eval-stage --checkpoint trained --fold 2
~~~

The training command writes a checkpoint under `outputs/workspace/`. Evaluation reads that locally trained checkpoint. Results on this small sample do not reproduce paper-scale estimates. For the full implementation and datasets, see [research/](../research/README.md) and the [dataset descriptions](../README.md#datasets).
