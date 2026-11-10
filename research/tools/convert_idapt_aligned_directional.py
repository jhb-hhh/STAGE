"""Convert CICFlowMeter CSVs into the directional ID-APT graph protocol.

ID-APT extracts one node per directional five-tuple from PCAP. CICFlowMeter
stores forward and backward statistics in one bidirectional row. This converter
splits every usable row into forward/backward directional nodes before applying
the exact ID-APT fixed feature transforms and 3000/1500 graph window protocol.
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data

# PyTorch otherwise transfers every tensor through a separate POSIX file
# descriptor when a graph is returned by a multiprocessing worker.  Large CIC
# conversions can exhaust that channel and fail with ``received 0 items of
# ancdata``.  File-system sharing keeps the parallel conversion robust.
torch.multiprocessing.set_sharing_strategy("file_system")


MAX_FLOWS = 3000
STRIDE = 1500
MAX_GROUP = 100
SAMPLED_EDGES = 200

ALIASES = {
    "src_ip": ("Src IP",),
    "dst_ip": ("Dst IP",),
    "src_port": ("Src Port",),
    "dst_port": ("Dst Port",),
    "protocol": ("Protocol",),
    "timestamp": ("Timestamp",),
    "duration": ("Flow Duration",),
    "fwd_packets": ("Total Fwd Packet", "Tot Fwd Pkts"),
    "bwd_packets": ("Total Bwd packets", "Tot Bwd Pkts"),
    "fwd_bytes": ("Total Length of Fwd Packet", "TotLen Fwd Pkts"),
    "bwd_bytes": ("Total Length of Bwd Packet", "TotLen Bwd Pkts"),
    "fwd_len_mean": ("Fwd Packet Length Mean", "Fwd Pkt Len Mean"),
    "bwd_len_mean": ("Bwd Packet Length Mean", "Bwd Pkt Len Mean"),
    "fwd_len_std": ("Fwd Packet Length Std", "Fwd Pkt Len Std"),
    "bwd_len_std": ("Bwd Packet Length Std", "Bwd Pkt Len Std"),
    "fwd_iat_mean": ("Fwd IAT Mean",),
    "bwd_iat_mean": ("Bwd IAT Mean",),
    "fwd_iat_std": ("Fwd IAT Std",),
    "bwd_iat_std": ("Bwd IAT Std",),
    "label": ("Stage", "Label"),
}


def resolve_columns(frame):
    result = {}
    for key, choices in ALIASES.items():
        for name in choices:
            if name in frame.columns:
                result[key] = name
                break
        if key not in result:
            raise KeyError(f"missing {key}; accepted={choices}")
    return result


def numeric(frame, name):
    return pd.to_numeric(frame[name], errors="coerce").replace([np.inf, -np.inf], np.nan)


def attack_label(value):
    return str(value).strip().lower() not in {"", "benign", "normal", "0", "nan"}


def directional_rows(frame):
    frame = frame.copy()
    frame.columns = [str(c).strip() for c in frame.columns]
    c = resolve_columns(frame)
    timestamps = pd.to_datetime(frame[c["timestamp"]], errors="coerce", dayfirst=True)
    base = pd.DataFrame(
        {
            "src_ip": frame[c["src_ip"]].astype(str),
            "dst_ip": frame[c["dst_ip"]].astype(str),
            "src_port": numeric(frame, c["src_port"]),
            "dst_port": numeric(frame, c["dst_port"]),
            "proto": numeric(frame, c["protocol"]),
            "start_time": timestamps.astype("int64") / 1e9,
            "duration": numeric(frame, c["duration"]) / 1_000_000.0,
            "attack": frame[c["label"]].map(attack_label),
        }
    )
    base.loc[timestamps.isna(), "start_time"] = np.nan

    def side(prefix, reverse=False):
        packet_key = f"{prefix}_packets"
        byte_key = f"{prefix}_bytes"
        count = numeric(frame, c[packet_key])
        out = base.copy()
        if reverse:
            out[["src_ip", "dst_ip"]] = out[["dst_ip", "src_ip"]].to_numpy()
            out[["src_port", "dst_port"]] = out[["dst_port", "src_port"]].to_numpy()
        out["pkt_count"] = count
        out["byte_count"] = numeric(frame, c[byte_key])
        out["rate"] = count / (out["duration"] + 1e-6)
        out["pkt_len_mean"] = numeric(frame, c[f"{prefix}_len_mean"])
        out["pkt_len_std"] = numeric(frame, c[f"{prefix}_len_std"])
        out["iat_mean"] = numeric(frame, c[f"{prefix}_iat_mean"]) / 1_000_000.0
        out["iat_std"] = numeric(frame, c[f"{prefix}_iat_std"]) / 1_000_000.0
        return out

    forward = side("fwd")
    backward = side("bwd", reverse=True)
    expanded = pd.concat([forward, backward], ignore_index=True)
    required = [
        "src_port", "dst_port", "proto", "start_time", "duration", "pkt_count",
        "byte_count", "rate", "pkt_len_mean", "pkt_len_std", "iat_mean", "iat_std",
    ]
    expanded = expanded.replace([np.inf, -np.inf], np.nan).dropna(subset=required)
    expanded = expanded.loc[expanded["pkt_count"] > 0].copy()
    expanded.sort_values("start_time", kind="stable", inplace=True)
    expanded.reset_index(drop=True, inplace=True)
    return expanded


def add_edges(frame, group_col, relation, rng):
    edges, attrs = [], []
    for _, group in frame.groupby(group_col, sort=False):
        nodes = group.index.tolist()
        if len(nodes) < 2:
            continue
        if len(nodes) > MAX_GROUP:
            pairs = [tuple(rng.sample(nodes, 2)) for _ in range(SAMPLED_EDGES)]
        else:
            pairs = [(u, v) for u in nodes for v in nodes if u != v]
        for u, v in pairs:
            delta = abs(float(frame.at[u, "start_time"]) - float(frame.at[v, "start_time"]))
            edges.append((u, v))
            attrs.append((float(relation), float(np.log1p(delta))))
    return edges, attrs


def make_graph(frame, rng, source_name, window_index):
    frame = frame.copy().reset_index(drop=True)
    feature_names = [
        "duration", "pkt_count", "byte_count", "rate", "pkt_len_mean",
        "pkt_len_std", "iat_mean", "iat_std",
    ]
    divisors = np.asarray([10.0, 15.0, 25.0, 15.0, 10.0, 10.0, 5.0, 5.0])
    values = np.log1p(frame[feature_names].clip(lower=0).to_numpy(dtype=np.float64)) / divisors
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    edges1, attrs1 = add_edges(frame, "src_ip", 1, rng)
    edges2, attrs2 = add_edges(frame, "dst_ip", 2, rng)
    edges = edges1 + edges2 + [(i, i) for i in range(len(frame))]
    attrs = attrs1 + attrs2 + [(0.0, 0.0) for _ in range(len(frame))]
    proto = frame["proto"].astype(int).map({6: 0, 17: 1}).fillna(2).astype(int)

    graph = Data(
        x=torch.tensor(values, dtype=torch.float32),
        edge_index=torch.tensor(edges, dtype=torch.long).t().contiguous(),
        edge_attr=torch.tensor(attrs, dtype=torch.float32),
        y=torch.tensor([int(frame["attack"].any())], dtype=torch.long),
        proto_idx=torch.tensor(proto.to_numpy(), dtype=torch.long),
        sport_idx=torch.tensor(frame["src_port"].clip(0, 1024).astype(int).to_numpy(), dtype=torch.long),
        dport_idx=torch.tensor(frame["dst_port"].clip(0, 1024).astype(int).to_numpy(), dtype=torch.long),
        flags_idx=torch.zeros(len(frame), dtype=torch.long),
    )
    graph.num_nodes = len(frame)
    graph.filename = source_name
    graph.window_index = int(window_index)
    return graph


def make_graph_worker(task):
    frame, seed, source_name, window_index = task
    return make_graph(frame, random.Random(seed), source_name, window_index)


def graph_windows(frame):
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


def input_paths(dataset):
    if dataset == "dapt2020":
        return sorted(Path("data/external/dapt2020/csv").glob("*.csv"))
    return [
        Path("data/external/cicids2018/csv/Thuesday-20-02-2018_TrafficForML_CICFlowMeter.csv"),
        Path("data/external/sandworm/SandwormAPT_flow_labelled.csv"),
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("dapt2020", "cic-sandworm"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--chunk-size", type=int, default=500)
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if list(args.output_dir.glob("graphs_chunk_*.pt")):
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")

    rng = random.Random(42)
    saved, chunk_id, buffer = 0, 0, []
    for path in input_paths(args.dataset):
        try:
            raw = pd.read_csv(path, low_memory=False)
            flows = directional_rows(raw)
        except Exception as exc:
            print(f"skipped {path.name}: {exc}", flush=True)
            continue
        del raw
        tasks = (
            (window, rng.randrange(2**31), path.name, window_index)
            for window_index, window in enumerate(graph_windows(flows))
        )
        with mp.Pool(processes=max(1, args.workers)) as pool:
            for graph in pool.imap(make_graph_worker, tasks, chunksize=1):
                buffer.append(graph)
                saved += 1
                if len(buffer) >= args.chunk_size:
                    torch.save(buffer, args.output_dir / f"graphs_chunk_{chunk_id:04d}.pt")
                    buffer, chunk_id = [], chunk_id + 1
                if saved % 100 == 0:
                    print(f"saved_graphs={saved} source={path.name}", flush=True)
        print(f"converted {path.name}: directional_flows={len(flows)} total_graphs={saved}", flush=True)
    if buffer:
        torch.save(buffer, args.output_dir / f"graphs_chunk_{chunk_id:04d}.pt")
    print(f"done: saved {saved} graphs to {args.output_dir}")


if __name__ == "__main__":
    main()
