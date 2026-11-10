# ID-APT

ID-APT comprises 23,719 graph windows across 12 attack scenarios and a benign class. One node represents one flow. The [sample](../sample_data/) contains 65 sanitized graphs with separate sample indices. The [full five-fold indices](../research/data/reference/splits/) refer to the full graph collection. Keep the original graph chunk filenames and loader ordering when using these indices.

## Full graph release

Download the [50 data archives and manifest](https://github.com/jhb-hhh/STAGE/releases/tag/id-apt-v1) (11.03 GB compressed). The data distribution is named `id-apt-v1`. From a clone of this repository, use GitHub CLI to fetch the manifest and archives, then verify and extract them:

~~~sh
gh release download id-apt-v1 --dir idapt-download
python tools/unpack_idapt.py idapt-download research
~~~

The archives unpack under `research/data/reference/`. They contain the original graph chunks, fold-specific shifted chunks, label map and five split files. The manifest includes byte counts and SHA-256 digests for each archive and member. Shifted fold 1 was regenerated with seed 43. Folds 0, 2, 3 and 4 come from the preserved shifted-graph runs.

## Source traffic

[CIC-IDS2017](https://www.unb.ca/cic/datasets/ids-2017.html) supplies benign background traffic. See the [seed registry](../research/src/stage/data/seed_registry.py) for the 12 input capture names and endpoint mappings and [Malware-Traffic-Analysis.net](https://www.malware-traffic-analysis.net/) for the original capture archive. This graph release contains derived graphs. Overlapping windows may share flows, and windows from the same attack scenario may share source traces.
