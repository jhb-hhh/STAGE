# Data and LLM notes

The repository includes [sanitized sample graphs](../sample_data/README.md) and [ID-APT dataset documentation](../datasets/ID-APT.md). Raw source traffic and full graph archives are separate from this source tree. The sample is intended for checking data loading and training code, not for reproducing paper-scale metrics.

The [forensic-report code](../research/src/stage/llm/analyst.py) and [behavior-identification code](../research/src/stage/llm/exam.py) contain the two LLM workflows. Running these workflows requires a separately configured local model or API provider. No API credentials or generated LLM outputs are included.
