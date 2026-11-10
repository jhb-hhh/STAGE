import json
import shutil
from pathlib import Path

import torch

from shortcut_ablation.config import (
    GRAPH_OUTPUT_DIR,
    LABEL_FILE,
    PCAP_OUTPUT_DIR,
    OUTPUT_ROOT,
)


META_DIR = OUTPUT_ROOT / "meta"
IP_VOCAB_PATH = META_DIR / "ip_vocab.json"


def _reset_graph_outputs():
    if GRAPH_OUTPUT_DIR.exists():
        shutil.rmtree(GRAPH_OUTPUT_DIR)
    GRAPH_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    META_DIR.mkdir(parents=True, exist_ok=True)


def _collect_graph_chunks():
    chunks = sorted(GRAPH_OUTPUT_DIR.glob("graphs_chunk_*.pt"))
    if not chunks:
        raise FileNotFoundError(f"No graph chunks found in {GRAPH_OUTPUT_DIR}")
    return chunks


def _to_raw_port_tensor(raw_ports, num_nodes):
    values = []
    for p in raw_ports:
        try:
            v = int(p)
        except Exception:
            v = 0
        values.append(max(0, min(65535, v)))

    if len(values) != num_nodes:
        raise ValueError(f"raw port length mismatch: {len(values)} != {num_nodes}")

    return torch.tensor(values, dtype=torch.long)


def _build_ip_vocab(chunks):
    ip_set = set()

    for chunk_path in chunks:
        graphs = torch.load(chunk_path, map_location="cpu", weights_only=False)
        for data in graphs:
            ip_set.update(str(x) for x in getattr(data, "src_ip_list", []))
            ip_set.update(str(x) for x in getattr(data, "dst_ip_list", []))

    # 0 reserved for unknown/padding
    vocab = {ip: idx + 1 for idx, ip in enumerate(sorted(ip_set))}

    with IP_VOCAB_PATH.open("w", encoding="utf-8") as f:
        json.dump(vocab, f, indent=2, sort_keys=True)

    return vocab


def _ip_index_tensor(ip_list, vocab, num_nodes):
    values = [vocab.get(str(ip), 0) for ip in ip_list]

    if len(values) != num_nodes:
        raise ValueError(f"IP list length mismatch: {len(values)} != {num_nodes}")

    return torch.tensor(values, dtype=torch.long)


def _enhance_chunks_with_shortcut_fields(chunks, ip_vocab):
    total_graphs = 0

    for chunk_path in chunks:
        graphs = torch.load(chunk_path, map_location="cpu", weights_only=False)
        enhanced = []

        for data in graphs:
            num_nodes = int(data.num_nodes)

            data.src_ip_idx_global = _ip_index_tensor(data.src_ip_list, ip_vocab, num_nodes)
            data.dst_ip_idx_global = _ip_index_tensor(data.dst_ip_list, ip_vocab, num_nodes)

            data.sport_idx_raw = _to_raw_port_tensor(data.raw_src_port_list, num_nodes)
            data.dport_idx_raw = _to_raw_port_tensor(data.raw_dst_port_list, num_nodes)

            enhanced.append(data)
            total_graphs += 1

        torch.save(enhanced, chunk_path)

    return total_graphs


def run_build_graphs(reset=True):
    if not PCAP_OUTPUT_DIR.exists():
        raise FileNotFoundError(f"Shortcut PCAP dir not found: {PCAP_OUTPUT_DIR}")
    if not LABEL_FILE.exists():
        raise FileNotFoundError(f"Shortcut label file not found: {LABEL_FILE}")

    if reset:
        _reset_graph_outputs()
    else:
        GRAPH_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        META_DIR.mkdir(parents=True, exist_ok=True)

    # 复用主项目 graph_construction，但只 monkeypatch 输入/输出路径。
    # 不修改 src/stage/ 主逻辑。
    from stage.data import graph_construction as gc

    gc.RAW_PCAP_DIR = str(PCAP_OUTPUT_DIR)
    gc.LABEL_FILE_PATH = str(LABEL_FILE)

    # graph_construction.py computes these globals at import time,
    # so we must patch both the resolver functions and the resolved globals.
    gc.resolve_graph_save_dir = lambda: GRAPH_OUTPUT_DIR

    def _resolve_label_map_path(graph_save_dir):
        return Path(graph_save_dir) / "label_mapping.json"

    gc.resolve_label_map_path = _resolve_label_map_path
    gc.GRAPH_SAVE_DIR = str(GRAPH_OUTPUT_DIR)
    gc.LABEL_MAP_PATH = str(GRAPH_OUTPUT_DIR / "label_mapping.json")

    print("[INFO] Shortcut graph construction")
    print(f"[INFO] input_pcap_dir: {gc.RAW_PCAP_DIR}")
    print(f"[INFO] label_file: {gc.LABEL_FILE_PATH}")
    print(f"[INFO] output_graph_dir: {GRAPH_OUTPUT_DIR}")

    gc.process_all_pcaps()

    chunks = _collect_graph_chunks()
    print(f"[INFO] Graph chunks generated: {len(chunks)}")

    ip_vocab = _build_ip_vocab(chunks)
    print(f"[INFO] Global IP vocab size: {len(ip_vocab)}")
    print(f"[INFO] IP vocab saved to: {IP_VOCAB_PATH}")

    total_graphs = _enhance_chunks_with_shortcut_fields(chunks, ip_vocab)
    print(f"[OK] Enhanced {total_graphs} graphs with shortcut-visible fields.")
