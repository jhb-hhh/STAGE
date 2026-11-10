"""Convert DAPT2020 CICFlowMeter CSVs to the ID-APT graph schema."""

from __future__ import annotations

import argparse
import math
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

MAX_FLOWS = 3000
STRIDE = 1500
MAX_GROUP = 100
SAMPLED_EDGES = 200


ALIASES = {
    "src_ip": ("Src IP",),
    "dst_ip": ("Dst IP",),
    "src_port": ("Src Port",),
    "dst_port": ("Dst Port", "Dst Port"),
    "protocol": ("Protocol",),
    "timestamp": ("Timestamp",),
    "duration": ("Flow Duration",),
    "fwd_packets": ("Total Fwd Packet", "Tot Fwd Pkts"),
    "bwd_packets": ("Total Bwd packets", "Tot Bwd Pkts"),
    "fwd_bytes": ("Total Length of Fwd Packet", "TotLen Fwd Pkts"),
    "bwd_bytes": ("Total Length of Bwd Packet", "TotLen Bwd Pkts"),
    "length_mean": ("Packet Length Mean", "Pkt Len Mean"),
    "length_std": ("Packet Length Std", "Pkt Len Std"),
    "iat_mean": ("Flow IAT Mean",),
    "iat_std": ("Flow IAT Std",),
    "label": ("Stage", "Label"),
}


def column(frame, key):
    for name in ALIASES[key]:
        if name in frame.columns:
            return name
    raise KeyError(f"Missing {key}; accepted columns: {ALIASES[key]}")


def numeric(frame, key):
    return pd.to_numeric(frame[column(frame, key)], errors="coerce").fillna(0).clip(lower=0)


def add_group_edges(frame, key, relation, rng):
    edges, attrs = [], []
    for _, group in frame.groupby(key):
        indices = group.index.tolist()
        if len(indices) < 2:
            continue
        if len(indices) > MAX_GROUP:
            pairs = [tuple(rng.sample(indices, 2)) for _ in range(SAMPLED_EDGES)]
        else:
            pairs = [(u, v) for u in indices for v in indices if u != v]
        for u, v in pairs:
            delta = abs(float(frame.at[u, "start_time"]) - float(frame.at[v, "start_time"]))
            edges.append((u, v))
            attrs.append((float(relation), float(np.log1p(delta))))
    return edges, attrs


def is_attack(value):
    text = str(value).strip().lower()
    return text not in {"", "benign", "normal", "0", "nan"}


def make_graph(frame, rng):
    frame = frame.copy().reset_index(drop=True)
    duration = numeric(frame, "duration") / 1_000_000.0
    packets = numeric(frame, "fwd_packets") + numeric(frame, "bwd_packets")
    byte_count = numeric(frame, "fwd_bytes") + numeric(frame, "bwd_bytes")
    rate = packets / (duration + 1e-6)
    values = np.column_stack(
        [
            np.log1p(duration) / 10.0,
            np.log1p(packets) / 15.0,
            np.log1p(byte_count) / 25.0,
            np.log1p(rate) / 15.0,
            np.log1p(numeric(frame, "length_mean")) / 10.0,
            np.log1p(numeric(frame, "length_std")) / 10.0,
            np.log1p(numeric(frame, "iat_mean") / 1_000_000.0) / 5.0,
            np.log1p(numeric(frame, "iat_std") / 1_000_000.0) / 5.0,
        ]
    ).astype(np.float32)
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)

    frame["src_ip"] = frame[column(frame, "src_ip")].astype(str)
    frame["dst_ip"] = frame[column(frame, "dst_ip")].astype(str)
    edges1, attrs1 = add_group_edges(frame, "src_ip", 1, rng)
    edges2, attrs2 = add_group_edges(frame, "dst_ip", 2, rng)
    edges = edges1 + edges2 + [(i, i) for i in range(len(frame))]
    attrs = attrs1 + attrs2 + [(0.0, 0.0) for _ in range(len(frame))]

    protocol = numeric(frame, "protocol").astype(int)
    proto_idx = protocol.map({6: 0, 17: 1}).fillna(2).astype(int)
    label_name = column(frame, "label")
    graph = Data(
        x=torch.tensor(values),
        edge_index=torch.tensor(edges, dtype=torch.long).t().contiguous(),
        edge_attr=torch.tensor(attrs, dtype=torch.float32),
        y=torch.tensor([int(frame[label_name].map(is_attack).any())], dtype=torch.long),
        proto_idx=torch.tensor(proto_idx.to_numpy(), dtype=torch.long),
        sport_idx=torch.tensor(numeric(frame, "src_port").clip(0, 1024).astype(int).to_numpy(), dtype=torch.long),
        dport_idx=torch.tensor(numeric(frame, "dst_port").clip(0, 1024).astype(int).to_numpy(), dtype=torch.long),
        flags_idx=torch.zeros(len(frame), dtype=torch.long),
    )
    graph.num_nodes = len(frame)
    return graph


def windows(frame):
    frame = frame.sort_values("start_time").reset_index(drop=True)
    # Match ID-APT exactly: each source capture is one ordered flow sequence;
    # sequences longer than 3000 flows use a 1500-flow sliding stride.  Do not
    # introduce an external ten-minute boundary that ID-APT never used.
    total = len(frame)
    if total <= MAX_FLOWS:
        if total >= 5:
            yield frame
        return
    start = 0
    while start < total:
        part = frame.iloc[start : start + MAX_FLOWS]
        if len(part) >= 5:
            yield part
        if start + MAX_FLOWS >= total:
            break
        start += STRIDE


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv-dir", type=Path, default=Path("data/external/dapt2020/csv"))
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "workspace" / "cross_domain" / "dapt2020_idapt_aligned" / "graphs")
    parser.add_argument("--chunk-size", type=int, default=1000)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if list(args.output_dir.glob("graphs_chunk_*.pt")):
        raise FileExistsError(f"Refusing to overwrite existing chunks in {args.output_dir}")

    rng = random.Random(42)
    graphs, chunk_id, total = [], 0, 0
    for path in sorted(args.csv_dir.glob("*.csv")):
        frame = pd.read_csv(path, low_memory=False)
        frame.columns = [str(c).strip() for c in frame.columns]
        try:
            timestamp_column = column(frame, "timestamp")
            # Validate every field before accumulating graphs so malformed
            # auxiliary CSV files are skipped as a whole.
            for required in ALIASES:
                column(frame, required)
        except KeyError as exc:
            print(f"skipped {path.name}: {exc}", flush=True)
            continue
        timestamps = pd.to_datetime(frame[timestamp_column], errors="coerce", dayfirst=True)
        frame = frame.loc[timestamps.notna()].copy()
        frame["start_time"] = timestamps.loc[timestamps.notna()].astype("int64") / 1e9
        for window in windows(frame):
            graphs.append(make_graph(window, rng))
            total += 1
            if len(graphs) >= args.chunk_size:
                torch.save(graphs, args.output_dir / f"graphs_chunk_{chunk_id:04d}.pt")
                graphs, chunk_id = [], chunk_id + 1
        print(f"converted {path.name}: total_graphs={total}", flush=True)
    if graphs:
        torch.save(graphs, args.output_dir / f"graphs_chunk_{chunk_id:04d}.pt")
    print(f"saved {total} graphs to {args.output_dir}")


if __name__ == "__main__":
    main()
