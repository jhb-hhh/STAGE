
import os
import sys
import pandas as pd
import torch
import numpy as np
from tqdm import tqdm
from torch_geometric.data import Data
from datetime import datetime, timedelta

# 添加项目路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# 配置路径
DAPT_CSV_DIR = "data/external/dapt2020/csv"
OUTPUT_DIR = "data/external/dapt2020/graphs_temporal"
os.makedirs(OUTPUT_DIR, exist_ok=True)

WINDOW_SECONDS = 600  # 10 分钟

def parse_datetime(dt_str):
    try:
        dt = datetime.strptime(dt_str, "%d/%m/%Y %I:%M:%S %p")
    except ValueError:
        dt = datetime.strptime(dt_str, "%d/%m/%Y %H:%M:%S")
    return dt

def convert_df_to_graph(df):
    if len(df) == 0:
        return None, 0.0

    all_ips = pd.concat([df['Src IP'], df['Dst IP']]).unique()
    ip_to_idx = {ip: i for i, ip in enumerate(all_ips)}
    num_nodes = len(all_ips)

    src_indices = [ip_to_idx[ip] for ip in df['Src IP']]
    dst_indices = [ip_to_idx[ip] for ip in df['Dst IP']]
    edge_index = torch.tensor([src_indices, dst_indices], dtype=torch.long)

    ignore_cols = ['Flow ID', 'Src IP', 'Dst IP', 'Src Port', 'Dst Port',
                   'Timestamp', 'Activity', 'Stage', 'Label']
    numeric_cols = [col for col in df.columns if col not in ignore_cols and pd.api.types.is_numeric_dtype(df[col])]
    selected_cols = numeric_cols[:15] if len(numeric_cols) > 15 else numeric_cols
    edge_attr_raw = df[selected_cols].fillna(0).values.astype(np.float32)
    max_vals = np.max(np.abs(edge_attr_raw), axis=0)
    max_vals[max_vals == 0] = 1
    edge_attr = edge_attr_raw / max_vals
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)

    out_degree = torch.zeros(num_nodes, dtype=torch.float)
    in_degree = torch.zeros(num_nodes, dtype=torch.float)
    for src, dst in zip(src_indices, dst_indices):
        out_degree[src] += 1
        in_degree[dst] += 1
    x = torch.stack([out_degree, in_degree, out_degree + in_degree], dim=1)
    if x.max() > 0:
        x = x / x.max()

    if 'Stage' in df.columns:
        y_np = df['Stage'].apply(lambda s: 0 if s.lower() == 'benign' else 1).values
        y = torch.tensor(y_np, dtype=torch.long)
        mal_ratio = (y_np == 1).mean()
    else:
        y = torch.zeros(len(df), dtype=torch.long)
        mal_ratio = 0.0

    data = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        y=y,
        proto_idx=torch.zeros(num_nodes, dtype=torch.long),
        sport_idx=torch.zeros(num_nodes, dtype=torch.long),
        dport_idx=torch.zeros(num_nodes, dtype=torch.long),
        flags_idx=torch.zeros(num_nodes, dtype=torch.long),
    )
    return data, mal_ratio

def main():
    csv_files = [os.path.join(DAPT_CSV_DIR, f) for f in os.listdir(DAPT_CSV_DIR) if f.endswith('.csv')]
    if not csv_files:
        print("❌ No CSV files found.")
        return

    print(f"✅ Found {len(csv_files)} CSV files.")
    total_chunks = 0
    chunk_stats = []

    for file_path in tqdm(csv_files, desc="Processing files"):
        print(f"\nProcessing: {os.path.basename(file_path)}")
        df = pd.read_csv(file_path, low_memory=False)

        # ========== 关键修复：检查列名 ==========
        if 'Timestamp' not in df.columns:
            print(f"  ⚠️ Skipping file: 'Timestamp' column not found.")
            continue

        df['_dt'] = df['Timestamp'].apply(parse_datetime)
        min_dt = df['_dt'].min()
        max_dt = df['_dt'].max()
        print(f"  Time range: {min_dt} -> {max_dt}")

        current_start = min_dt
        window_id = 0
        while current_start <= max_dt:
            current_end = current_start + timedelta(seconds=WINDOW_SECONDS)
            mask = (df['_dt'] >= current_start) & (df['_dt'] < current_end)
            sub_df = df[mask]
            if len(sub_df) > 50:
                data, mal_ratio = convert_df_to_graph(sub_df)
                if data is not None:
                    base_name = os.path.splitext(os.path.basename(file_path))[0]
                    save_path = os.path.join(OUTPUT_DIR, f"graphs_chunk_{base_name}_{window_id:04d}.pt")
                    torch.save([data], save_path)
                    total_chunks += 1
                    chunk_stats.append({
                        'file': base_name,
                        'window': window_id,
                        'edges': data.edge_index.shape[1],
                        'nodes': data.x.shape[0],
                        'mal_ratio': mal_ratio
                    })
                    if mal_ratio > 0.05:
                        print(f"    ⚠️ Window {window_id}: {len(sub_df)} flows, mal_ratio={mal_ratio:.2%}")
            current_start = current_end
            window_id += 1

    print("\n" + "="*60)
    print("Temporal Conversion Complete!")
    print(f"Total graph chunks generated: {total_chunks}")
    print(f"Output directory: {OUTPUT_DIR}")
    mal_chunks = [s for s in chunk_stats if s['mal_ratio'] > 0]
    print(f"Chunks with any malicious flow: {len(mal_chunks)}")
    print("="*60)

    # 保存统计信息
    stats_df = pd.DataFrame(chunk_stats)
    stats_df.to_csv(os.path.join(OUTPUT_DIR, "chunk_stats.csv"), index=False)
    print("Saved chunk statistics to chunk_stats.csv")

if __name__ == "__main__":
    main()