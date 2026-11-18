# ID-APT

ID-APT comprises 23,719 graph windows across 12 attack scenarios and a benign class. One graph node represents one network flow. The [sanitized sample](../sample_data/) contains 65 graphs with separate sample indices. The [five full-data split files](../research/data/reference/splits/) refer to the complete graph collection; they cannot be applied directly to the sample.

## Data availability during anonymous review

The [archive manifest](../manifest/idapt-manifest.json) describes the separately distributed full graph archives and their checksums. The complete archives are not stored in this source repository. A direct link to the original release would reveal the repository owner's identity, so it is omitted from this double-blind copy. The sanitized graph sample and split metadata remain available here. Once the archives have been obtained through an anonymous distribution channel, `python tools/unpack_idapt.py <archive-directory> research` verifies and extracts them under `research/data/reference/`.

## Source traffic

[CIC-IDS2017](https://www.unb.ca/cic/datasets/ids-2017.html) supplies benign background traffic. The [seed registry](../research/src/stage/data/seed_registry.py) lists the 12 input capture names and endpoint mappings; the original captures are available from [Malware-Traffic-Analysis.net](https://www.malware-traffic-analysis.net/). The graph archive contains derived graphs. Overlapping windows may share flows, and windows from the same attack scenario may share source traces.
