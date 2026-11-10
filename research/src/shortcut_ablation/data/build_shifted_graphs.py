import json
import random
import shutil
from pathlib import Path

import torch

from shortcut_ablation.config import (
    GRAPH_OUTPUT_DIR,
    SHIFTED_GRAPH_OUTPUT_DIR,
    OUTPUT_ROOT,
)


META_DIR = OUTPUT_ROOT / "meta"
SHIFT_META_PATH = META_DIR / "shift_metadata.json"


def _load_chunks(graph_dir):
    chunks = sorted(Path(graph_dir).glob("graphs_chunk_*.pt"))
    if not chunks:
        raise FileNotFoundError(f"No graph chunks found in {graph_dir}")
    return chunks


def _make_permutation(values, rng):
    values = sorted(set(int(v) for v in values))
    if not values:
        return {}
    if len(values) == 1:
        only = values[0]
        return {only: only}

    shuffled = values[:]
    rng.shuffle(shuffled)

    # 避免大多数 identity 保持不变
    for i, v in enumerate(values):
        if shuffled[i] == v:
            j = (i + 1) % len(values)
            shuffled[i], shuffled[j] = shuffled[j], shuffled[i]

    return dict(zip(values, shuffled))


def _collect_identity_values(chunks):
    ip_indices = set()
    raw_ports = set()

    for chunk_path in chunks:
        graphs = torch.load(chunk_path, map_location="cpu", weights_only=False)
        for data in graphs:
            ip_indices.update(data.src_ip_idx_global.tolist())
            ip_indices.update(data.dst_ip_idx_global.tolist())
            raw_ports.update(data.sport_idx_raw.tolist())
            raw_ports.update(data.dport_idx_raw.tolist())

    ip_indices.discard(0)
    return ip_indices, raw_ports


def _apply_tensor_map(tensor, mapping):
    flat = tensor.view(-1).tolist()
    shifted = [mapping.get(int(v), int(v)) for v in flat]
    return torch.tensor(shifted, dtype=torch.long).view_as(tensor)


def _clip_port_tensor(raw_port_tensor):
    return raw_port_tensor.clamp(min=0, max=1024).long()


def _shift_ip_lists(ip_list, idx_tensor, index_to_ip):
    shifted = []
    for idx in idx_tensor.tolist():
        shifted.append(index_to_ip.get(int(idx), "0.0.0.0"))
    if len(shifted) != len(ip_list):
        raise ValueError("Shifted IP list length mismatch.")
    return shifted


def _load_index_to_ip():
    ip_vocab_path = META_DIR / "ip_vocab.json"
    if not ip_vocab_path.exists():
        raise FileNotFoundError(f"Missing IP vocab: {ip_vocab_path}")

    with ip_vocab_path.open("r", encoding="utf-8") as f:
        ip_vocab = json.load(f)

    return {int(idx): ip for ip, idx in ip_vocab.items()}


def run_build_shifted_graphs(seed=2026, noise_std=0.05, reset=True):
    rng = random.Random(seed)
    torch.manual_seed(seed)

    chunks = _load_chunks(GRAPH_OUTPUT_DIR)
    index_to_ip = _load_index_to_ip()

    if reset and SHIFTED_GRAPH_OUTPUT_DIR.exists():
        shutil.rmtree(SHIFTED_GRAPH_OUTPUT_DIR)
    SHIFTED_GRAPH_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    META_DIR.mkdir(parents=True, exist_ok=True)

    ip_indices, raw_ports = _collect_identity_values(chunks)
    ip_map = _make_permutation(ip_indices, rng)
    port_map = _make_permutation(raw_ports, rng)

    total_graphs = 0

    for chunk_path in chunks:
        graphs = torch.load(chunk_path, map_location="cpu", weights_only=False)
        shifted_graphs = []

        for data in graphs:
            original_edge_index = data.edge_index.clone()
            original_edge_attr = data.edge_attr.clone()
            original_y = data.y.clone()

            # 1. IP identity permutation: 拓扑保持，只改模型可见的 IP identity
            data.src_ip_idx_global = _apply_tensor_map(data.src_ip_idx_global, ip_map)
            data.dst_ip_idx_global = _apply_tensor_map(data.dst_ip_idx_global, ip_map)
            data.src_ip_list = _shift_ip_lists(data.src_ip_list, data.src_ip_idx_global, index_to_ip)
            data.dst_ip_list = _shift_ip_lists(data.dst_ip_list, data.dst_ip_idx_global, index_to_ip)

            # 2. Raw port reassignment
            data.sport_idx_raw = _apply_tensor_map(data.sport_idx_raw, port_map).clamp(0, 65535)
            data.dport_idx_raw = _apply_tensor_map(data.dport_idx_raw, port_map).clamp(0, 65535)
            data.sport_idx = _clip_port_tensor(data.sport_idx_raw)
            data.dport_idx = _clip_port_tensor(data.dport_idx_raw)
            data.raw_src_port_list = [int(v) for v in data.sport_idx_raw.tolist()]
            data.raw_dst_port_list = [int(v) for v in data.dport_idx_raw.tolist()]

            # 3. Numerical feature noise
            noise = torch.randn_like(data.x) * noise_std
            data.x = (data.x + noise).clamp(min=0.0)

            # Safety checks: shifted domain 不允许破坏图结构和标签
            if not torch.equal(data.edge_index, original_edge_index):
                raise RuntimeError("edge_index changed unexpectedly.")
            if not torch.equal(data.edge_attr, original_edge_attr):
                raise RuntimeError("edge_attr changed unexpectedly.")
            if not torch.equal(data.y, original_y):
                raise RuntimeError("label y changed unexpectedly.")

            shifted_graphs.append(data)
            total_graphs += 1

        out_path = SHIFTED_GRAPH_OUTPUT_DIR / chunk_path.name
        torch.save(shifted_graphs, out_path)

    # 复制 label mapping，保证类别映射一致
    label_map_src = GRAPH_OUTPUT_DIR / "label_mapping.json"
    label_map_dst = SHIFTED_GRAPH_OUTPUT_DIR / "label_mapping.json"
    if label_map_src.exists():
        shutil.copy(label_map_src, label_map_dst)

    metadata = {
        "seed": seed,
        "noise_std": noise_std,
        "ip_identity_shift": "consistent global permutation over src_ip_idx_global/dst_ip_idx_global",
        "port_shift": "consistent global permutation over sport_idx_raw/dport_idx_raw",
        "topology_preserved": True,
        "edge_index_preserved": True,
        "edge_attr_preserved": True,
        "num_ip_identities": len(ip_indices),
        "num_raw_ports": len(raw_ports),
        "num_graphs": total_graphs,
        "source_graph_dir": str(GRAPH_OUTPUT_DIR),
        "shifted_graph_dir": str(SHIFTED_GRAPH_OUTPUT_DIR),
    }

    with SHIFT_META_PATH.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"[OK] Shifted shortcut graphs saved to: {SHIFTED_GRAPH_OUTPUT_DIR}")
    print(f"[OK] Shift metadata saved to: {SHIFT_META_PATH}")
    print(f"[OK] Shifted graphs: {total_graphs}")
    print(f"[INFO] IP identities shifted: {len(ip_indices)}")
    print(f"[INFO] Raw ports shifted: {len(raw_ports)}")
