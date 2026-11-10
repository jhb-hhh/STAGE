import csv
import random
import shutil
import time
from pathlib import Path

from scapy.all import rdpcap, wrpcap

from shortcut_ablation.config import (
    BACKGROUND_CHUNKS_DIR,
    LABEL_FILE,
    LABEL_OUTPUT_DIR,
    PCAP_OUTPUT_DIR,
)
from shortcut_ablation.data.seed_catalog import get_seed_records, validate_seed_files


def load_background_chunks():
    chunks = sorted(
        p for p in BACKGROUND_CHUNKS_DIR.iterdir()
        if p.is_file() and p.suffix.lower() in {".pcap", ".pcapng"} and p.name.startswith("bg_chunk_")
    )
    if not chunks:
        raise FileNotFoundError(f"No background chunks found in {BACKGROUND_CHUNKS_DIR}")
    return chunks


def reset_outputs():
    if PCAP_OUTPUT_DIR.exists():
        shutil.rmtree(PCAP_OUTPUT_DIR)
    if LABEL_OUTPUT_DIR.exists():
        shutil.rmtree(LABEL_OUTPUT_DIR)


def init_outputs():
    PCAP_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    LABEL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if not LABEL_FILE.exists():
        with LABEL_FILE.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["filename", "attack_type", "victim_ip", "malicious_ips", "injection_time"],
            )
            writer.writeheader()


def count_existing_labels():
    if not LABEL_FILE.exists():
        return 0
    with LABEL_FILE.open("r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if not header:
            return 0
        return sum(1 for row in reader if row and row[0])


def append_label(row):
    with LABEL_FILE.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["filename", "attack_type", "victim_ip", "malicious_ips", "injection_time"],
            extrasaction="ignore",
        )
        writer.writerow(row)


def rescale_timestamps(packets, new_start_time):
    if not packets:
        return []

    try:
        original_start = float(packets[0].time)
    except Exception:
        return [pkt.copy() for pkt in packets]

    out = []
    for pkt in packets:
        pkt_copy = pkt.copy()
        try:
            pkt_copy.time = new_start_time + (float(pkt.time) - original_start)
        except Exception:
            pass
        out.append(pkt_copy)
    return out


def generate_benign(background_chunks, output_path):
    source = random.choice(background_chunks)
    shutil.copy(source, output_path)


def synthesize_malicious(background_chunks, seed_record, output_path):
    seed_path = Path(seed_record["pcap_path"])
    background_path = random.choice(background_chunks)

    attack_packets = list(rdpcap(str(seed_path)))
    background_packets = list(rdpcap(str(background_path)))

    if not attack_packets or not background_packets:
        raise RuntimeError(f"Empty PCAP encountered: seed={seed_path}, background={background_path}")

    # w/o anti-shortcut preprocessing:
    # 不执行 dynamic IP remapping，不修改 seed PCAP 内的原始 victim/malicious IP。
    injection_base_time = float(background_packets[0].time) + random.uniform(0.001, 0.01)
    rescaled_attack = rescale_timestamps(attack_packets, injection_base_time)

    injection_point = random.randint(0, len(background_packets))
    final_packets = (
        background_packets[:injection_point]
        + rescaled_attack
        + background_packets[injection_point:]
    )

    wrpcap(str(output_path), final_packets)
    return injection_base_time


def run_synthesis(max_files, benign_ratio, seed, reset):
    random.seed(seed)

    missing = validate_seed_files()
    if missing:
        raise FileNotFoundError("Missing seed PCAP files:\n" + "\n".join(missing))

    if reset:
        reset_outputs()

    init_outputs()

    background_chunks = load_background_chunks()
    seed_records = get_seed_records()

    start_index = count_existing_labels()
    if start_index >= max_files:
        print(f"[INFO] Existing labels: {start_index}. Target max_files={max_files}. Nothing to do.")
        return

    print(f"[INFO] Background chunks: {len(background_chunks)}")
    print(f"[INFO] Seed records: {len(seed_records)}")
    print(f"[INFO] Output PCAP dir: {PCAP_OUTPUT_DIR}")
    print(f"[INFO] Label file: {LABEL_FILE}")
    print(f"[INFO] Generating from index {start_index + 1} to {max_files}")
    print(f"[INFO] Shortcut mode: dynamic IP remapping is DISABLED.")

    t0 = time.time()

    for i in range(start_index, max_files):
        run_index = i + 1
        filename = f"shortcut_run_{run_index:05d}.pcap"
        output_path = PCAP_OUTPUT_DIR / filename

        print(f"\n--- Generating {run_index}/{max_files}: {filename} ---")

        if random.random() < benign_ratio:
            print("  Type: Benign")
            generate_benign(background_chunks, output_path)
            label = {
                "filename": filename,
                "attack_type": "Benign",
                "victim_ip": "N/A",
                "malicious_ips": "N/A",
                "injection_time": "N/A",
            }
        else:
            seed_record = random.choice(seed_records)
            print(f"  Type: Malicious | Seed: {seed_record['name']}")
            injection_time = synthesize_malicious(background_chunks, seed_record, output_path)

            label = {
                "filename": filename,
                "attack_type": seed_record["name"],
                "victim_ip": seed_record["victim"],
                "malicious_ips": ",".join(seed_record["malicious_ips"]),
                "injection_time": injection_time,
            }

        append_label(label)
        print(f"  [OK] Wrote: {output_path}")

    print(f"\n[OK] Done. Generated {max_files - start_index} files in {time.time() - t0:.2f}s.")
