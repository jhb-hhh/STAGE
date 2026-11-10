"""PCAP-to-graph construction pipeline for ID-APT graph snapshots."""

import gc
import glob
import json
import logging
import multiprocessing
import os
import random
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scapy.all import IP, TCP, UDP, PcapReader
from sklearn.preprocessing import LabelEncoder
from torch_geometric.data import Data
from tqdm import tqdm

from stage.utils.config import get_active_profile, get_path

RAW_PCAP_DIR = str(get_path("data", "processed_dir") / "synthesized_pcaps")
LABEL_FILE_PATH = str(get_path("data", "label_file"))


def resolve_graph_save_dir():
    """Resolve graph-construction output directory.

    Graph rebuilding is only routed to outputs/workspace/graphs when the
    workspace path profile is active. The default profile keeps the historical
    generated-graph location to avoid overwriting reference paper artifacts.
    """
    if get_active_profile() == "workspace":
        return get_path("data", "graph_dir")

    return get_path("data", "processed_dir") / "graphs_generated"


def resolve_label_map_path(graph_save_dir):
    """Resolve the label-map path paired with generated graph chunks."""
    if get_active_profile() == "workspace":
        return get_path("data", "label_map")

    return Path(graph_save_dir) / "label_mapping.json"


GRAPH_SAVE_DIR = str(resolve_graph_save_dir())
LABEL_MAP_PATH = str(resolve_label_map_path(GRAPH_SAVE_DIR))

FLOW_TIMEOUT = 60
MAX_GROUP_SIZE = 100
SAMPLE_EDGES_PER_GROUP = 200
MAX_FLOWS_PER_GRAPH = 3000
WINDOW_STRIDE = 1500
CHUNK_SAVE_SIZE = 1000

PROTO_MAP = {6: 0, 17: 1}
FLAG_MAP = {
    "NONE": 0,
    "S": 1,
    "SA": 2,
    "A": 3,
    "F": 4,
    "FA": 5,
    "FSA": 6,
    "R": 7,
    "PA": 8,
    "FPA": 9,
}

EDGE_RELATION_TYPES = {
    "self": 0,
    "same_src": 1,
    "same_dst": 2,
}

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_task_list(label_file_path):
    """Load the graph-construction task list."""
    logger.info(f"Loading task list: {label_file_path}")

    try:
        df = pd.read_csv(label_file_path)
        df.dropna(subset=["filename"], inplace=True)
        df.replace("N/A", None, inplace=True)
        logger.info(f"Found {len(df)} tasks to process.")
        return df

    except Exception as exc:
        logger.error(f"Failed to read label file: {exc}")
        return None


def process_pcap_to_flows(pcap_path, flow_timeout=FLOW_TIMEOUT):
    """Extract five-tuple flow records from one PCAP file."""
    active_flows = {}
    completed_flows = []

    try:
        with PcapReader(pcap_path) as reader:
            for pkt in reader:
                if not (pkt.haslayer(IP) and (pkt.haslayer(TCP) or pkt.haslayer(UDP))):
                    continue

                try:
                    current_time = float(pkt.time)
                    src_ip = pkt[IP].src
                    dst_ip = pkt[IP].dst
                    proto = pkt[IP].proto
                    pkt_len = len(pkt[IP])

                    if proto == 6:
                        src_port = pkt[TCP].sport
                        dst_port = pkt[TCP].dport
                    elif proto == 17:
                        src_port = pkt[UDP].sport
                        dst_port = pkt[UDP].dport
                    else:
                        continue

                    flow_key = (src_ip, dst_ip, src_port, dst_port, proto)

                    if flow_key not in active_flows:
                        active_flows[flow_key] = {
                            "src_ip": src_ip,
                            "dst_ip": dst_ip,
                            "src_port": src_port,
                            "dst_port": dst_port,
                            "proto": proto,
                            "pkt_count": 1,
                            "byte_count": pkt_len,
                            "start_time": current_time,
                            "last_time": current_time,
                            "tcp_flags": set(),
                            "pkt_lengths": [pkt_len],
                            "timestamps": [current_time],
                        }

                        if proto == 6:
                            active_flows[flow_key]["tcp_flags"].add(str(pkt[TCP].flags))

                    else:
                        flow = active_flows[flow_key]

                        if current_time - flow["last_time"] > flow_timeout:
                            completed_flows.append(flow)
                            active_flows[flow_key] = {
                                "src_ip": src_ip,
                                "dst_ip": dst_ip,
                                "src_port": src_port,
                                "dst_port": dst_port,
                                "proto": proto,
                                "pkt_count": 1,
                                "byte_count": pkt_len,
                                "start_time": current_time,
                                "last_time": current_time,
                                "tcp_flags": set(),
                                "pkt_lengths": [pkt_len],
                                "timestamps": [current_time],
                            }

                            if proto == 6:
                                active_flows[flow_key]["tcp_flags"].add(str(pkt[TCP].flags))

                        else:
                            flow["pkt_count"] += 1
                            flow["byte_count"] += pkt_len
                            flow["last_time"] = current_time

                            if proto == 6:
                                flow["tcp_flags"].add(str(pkt[TCP].flags))

                            flow["pkt_lengths"].append(pkt_len)
                            flow["timestamps"].append(current_time)

                except Exception:
                    continue

        completed_flows.extend(active_flows.values())

        for flow in completed_flows:
            flow["duration"] = flow["last_time"] - flow["start_time"]

            if flow["duration"] < 0:
                flow["duration"] = 0.0

            flow["rate"] = flow["pkt_count"] / (flow["duration"] + 1e-6)
            flow["tcp_flags"] = ",".join(sorted(list(flow["tcp_flags"]))) if flow["tcp_flags"] else "NONE"

            lengths = np.array(flow["pkt_lengths"])
            timestamps = np.array(flow["timestamps"])

            flow["pkt_len_mean"] = np.mean(lengths)
            flow["pkt_len_std"] = np.std(lengths)

            if len(timestamps) > 1:
                iat = np.diff(timestamps)
                flow["iat_mean"] = np.mean(iat)
                flow["iat_std"] = np.std(iat)
            else:
                flow["iat_mean"] = 0.0
                flow["iat_std"] = 0.0

            del flow["pkt_lengths"]
            del flow["timestamps"]

        return completed_flows

    except Exception as exc:
        logger.warning(f"PCAP parsing failed: {exc}")
        return []


def add_edges_with_attributes(df, group_col, relation_type_val, max_size, sample_per_group):
    """Build graph edges and edge attributes for one relational grouping."""
    local_edges = []
    local_attrs = []

    try:
        groups = df.groupby(group_col)

        for _, group in groups:
            node_indices = group.index.tolist()

            if len(node_indices) < 2:
                continue

            if len(node_indices) > max_size:
                pairs = [tuple(random.sample(node_indices, 2)) for _ in range(sample_per_group)]
            else:
                pairs = [
                    (node_indices[i], node_indices[j])
                    for i in range(len(node_indices))
                    for j in range(len(node_indices))
                    if i != j
                ]

            for u, v in pairs:
                time_diff = abs(df.at[u, "start_time"] - df.at[v, "start_time"])
                time_diff_norm = np.log1p(time_diff)
                local_edges.append((u, v))
                local_attrs.append([float(relation_type_val), float(time_diff_norm)])

    except Exception:
        pass

    return local_edges, local_attrs


def process_and_save_worker(index, row, label_map_dict):
    """Process one PCAP file and save graph-window fragments."""
    try:
        temp_save_path = os.path.join(GRAPH_SAVE_DIR, f"temp_part_{index}.pt")

        if os.path.exists(temp_save_path):
            return "skipped"

        pcap_filename = row["filename"]
        attack_type_label_str = row["attack_type"]
        pcap_full_path = os.path.join(RAW_PCAP_DIR, pcap_filename)

        if not os.path.exists(pcap_full_path):
            return "missing"

        flows_list = process_pcap_to_flows(pcap_full_path)

        if not flows_list:
            return "empty"

        flows_df_all = pd.DataFrame(flows_list)
        del flows_list

        numerical_cols = [
            "duration",
            "pkt_count",
            "byte_count",
            "rate",
            "pkt_len_mean",
            "pkt_len_std",
            "iat_mean",
            "iat_std",
        ]

        flows_df_all.replace([np.inf, -np.inf], np.nan, inplace=True)
        flows_df_all.dropna(subset=numerical_cols, inplace=True)
        flows_df_all.sort_values(by="start_time", inplace=True)
        flows_df_all.reset_index(drop=True, inplace=True)

        if flows_df_all.empty:
            return "empty"

        flows_df_all["duration"] = np.log1p(flows_df_all["duration"]) / 10.0
        flows_df_all["pkt_count"] = np.log1p(flows_df_all["pkt_count"]) / 15.0
        flows_df_all["byte_count"] = np.log1p(flows_df_all["byte_count"]) / 25.0
        flows_df_all["rate"] = np.log1p(flows_df_all["rate"]) / 15.0
        flows_df_all["pkt_len_mean"] = np.log1p(flows_df_all["pkt_len_mean"]) / 10.0
        flows_df_all["pkt_len_std"] = np.log1p(flows_df_all["pkt_len_std"]) / 10.0
        flows_df_all["iat_mean"] = np.log1p(flows_df_all["iat_mean"]) / 5.0
        flows_df_all["iat_std"] = np.log1p(flows_df_all["iat_std"]) / 5.0

        total_flows = len(flows_df_all)
        slices = []

        if total_flows <= MAX_FLOWS_PER_GRAPH:
            slices.append((0, total_flows))
        else:
            start = 0
            while start < total_flows:
                end = min(start + MAX_FLOWS_PER_GRAPH, total_flows)
                slices.append((start, end))

                if end == total_flows:
                    break

                start += WINDOW_STRIDE

        graph_data_list = []

        for win_idx, (start_idx, end_idx) in enumerate(slices):
            sub_df = flows_df_all.iloc[start_idx:end_idx].copy().reset_index(drop=True)

            if len(sub_df) < 5:
                continue

            x = torch.tensor(sub_df[numerical_cols].values, dtype=torch.float)

            proto_indices = sub_df["proto"].map(lambda p: PROTO_MAP.get(p, 2)).values
            src_port_indices = sub_df["src_port"].clip(0, 1024).values
            dst_port_indices = sub_df["dst_port"].clip(0, 1024).values
            flag_indices = sub_df["tcp_flags"].map(lambda f: FLAG_MAP.get(f, 0)).values

            ip_le = LabelEncoder()
            all_ips = pd.concat([sub_df["src_ip"], sub_df["dst_ip"]]).unique()
            ip_le.fit(all_ips)
            sub_df["src_ip_idx"] = ip_le.transform(sub_df["src_ip"])
            sub_df["dst_ip_idx"] = ip_le.transform(sub_df["dst_ip"])

            edge_list = []
            edge_attrs_list = []

            e1, a1 = add_edges_with_attributes(
                sub_df,
                "src_ip_idx",
                EDGE_RELATION_TYPES["same_src"],
                MAX_GROUP_SIZE,
                SAMPLE_EDGES_PER_GROUP,
            )
            edge_list.extend(e1)
            edge_attrs_list.extend(a1)

            e2, a2 = add_edges_with_attributes(
                sub_df,
                "dst_ip_idx",
                EDGE_RELATION_TYPES["same_dst"],
                MAX_GROUP_SIZE,
                SAMPLE_EDGES_PER_GROUP,
            )
            edge_list.extend(e2)
            edge_attrs_list.extend(a2)

            num_nodes = len(sub_df)

            for i in range(num_nodes):
                edge_list.append((i, i))
                edge_attrs_list.append([float(EDGE_RELATION_TYPES["self"]), 0.0])

            if not edge_list:
                continue

            edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
            edge_attr = torch.tensor(edge_attrs_list, dtype=torch.float)

            y_int = label_map_dict.get(attack_type_label_str, 0)
            y = torch.tensor([y_int], dtype=torch.long)

            graph_data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
            graph_data.num_nodes = num_nodes

            graph_data.proto_idx = torch.tensor(proto_indices, dtype=torch.long)
            graph_data.sport_idx = torch.tensor(src_port_indices, dtype=torch.long)
            graph_data.dport_idx = torch.tensor(dst_port_indices, dtype=torch.long)
            graph_data.flags_idx = torch.tensor(flag_indices, dtype=torch.long)

            graph_data.filename = pcap_filename
            graph_data.attack_type_str = attack_type_label_str
            graph_data.window_index = win_idx
            graph_data.src_ip_list = sub_df["src_ip"].tolist()
            graph_data.dst_ip_list = sub_df["dst_ip"].tolist()
            graph_data.start_time_list = sub_df["start_time"].tolist()

            graph_data.raw_proto_list = sub_df["proto"].tolist()
            graph_data.raw_src_port_list = sub_df["src_port"].tolist()
            graph_data.raw_dst_port_list = sub_df["dst_port"].tolist()
            graph_data.raw_tcp_flags_list = sub_df["tcp_flags"].tolist()
            graph_data.raw_duration_list = np.expm1(sub_df["duration"] * 10.0).tolist()
            graph_data.raw_pkt_count_list = np.expm1(sub_df["pkt_count"] * 15.0).round().astype(int).tolist()
            graph_data.raw_byte_count_list = np.expm1(sub_df["byte_count"] * 25.0).round().astype(int).tolist()

            graph_data_list.append(graph_data)

        del flows_df_all
        gc.collect()

        if graph_data_list:
            torch.save(graph_data_list, temp_save_path)
            del graph_data_list
            gc.collect()
            return "saved"

        return "empty"

    except Exception:
        return "error"


def worker_wrapper(args):
    """Run one multiprocessing graph-construction task."""
    return process_and_save_worker(*args)


def merge_fragments_to_chunks():
    """Merge temporary graph fragments into graph chunk files."""
    logger.info("Merging graph-window fragments into chunk files.")

    temp_files = glob.glob(os.path.join(GRAPH_SAVE_DIR, "temp_part_*.pt"))

    if not temp_files:
        logger.warning("No fragment files found; nothing to merge.")
        return

    logger.info(f"Detected {len(temp_files)} fragment files to merge.")

    try:
        temp_files.sort(key=lambda x: int(os.path.basename(x).split("_")[2].split(".")[0]))
    except Exception:
        temp_files.sort()

    chunk_buffer = []
    chunk_index = 0
    target_chunk_size = 500

    for file_path in tqdm(temp_files, desc="Merging fragments"):
        try:
            graphs = torch.load(file_path, weights_only=False)
            chunk_buffer.extend(graphs)
            os.remove(file_path)

            if len(chunk_buffer) >= target_chunk_size:
                save_path = os.path.join(GRAPH_SAVE_DIR, f"graphs_chunk_{chunk_index}.pt")
                torch.save(chunk_buffer, save_path)
                chunk_buffer = []
                chunk_index += 1
                gc.collect()

        except Exception as exc:
            logger.error(f"Failed to merge file {file_path}: {exc}")

    if chunk_buffer:
        save_path = os.path.join(GRAPH_SAVE_DIR, f"graphs_chunk_{chunk_index}.pt")
        torch.save(chunk_buffer, save_path)
        logger.info(f"Final chunk {chunk_index} saved.")

    logger.info("Fragment merging completed and temporary files were removed.")


def process_all_pcaps():
    """Build graph chunks from all synthesized PCAP files."""
    logger.info("Starting graph construction.")

    task_list_df = load_task_list(LABEL_FILE_PATH)

    if task_list_df is None:
        return

    if os.path.exists(GRAPH_SAVE_DIR):
        logger.warning(f"Cleaning generated graph directory: {GRAPH_SAVE_DIR} before rebuilding graphs.")
        shutil.rmtree(GRAPH_SAVE_DIR)

    os.makedirs(GRAPH_SAVE_DIR, exist_ok=True)

    all_possible_labels = task_list_df["attack_type"].unique()
    global_label_encoder = LabelEncoder()
    global_label_encoder.fit(all_possible_labels)
    label_map_dict = {label: int(idx) for idx, label in enumerate(global_label_encoder.classes_)}

    with open(LABEL_MAP_PATH, "w", encoding="utf-8") as file:
        json.dump(label_map_dict, file, indent=2)

    logger.info(f"Label mapping saved to: {LABEL_MAP_PATH}")

    tasks = []

    for index, row in task_list_df.iterrows():
        tasks.append((index, row, label_map_dict))

    max_workers = 64
    logger.info(f"Starting stage 1 with {max_workers} worker processes.")

    start_time = time.time()
    stats = {"saved": 0, "skipped": 0, "empty": 0, "error": 0}

    with multiprocessing.Pool(processes=max_workers) as pool:
        for result in tqdm(
            pool.imap_unordered(worker_wrapper, tasks, chunksize=1),
            total=len(tasks),
            desc="Building graphs",
        ):
            stats[result] = stats.get(result, 0) + 1

    stage1_duration = time.time() - start_time
    logger.info(f"Stage 1 duration: {stage1_duration:.2f} seconds")

    merge_fragments_to_chunks()

    logger.info(f"Graph construction completed. Total duration: {time.time() - start_time:.2f} seconds")


if __name__ == "__main__":
    multiprocessing.freeze_support()
    process_all_pcaps()
