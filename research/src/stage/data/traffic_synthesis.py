"""Traffic synthesis utilities for generating ID-APT PCAP samples."""

import csv
import ipaddress
import os
import random
import shutil
import time

from scapy.all import Ether, IP, rdpcap, wrpcap

from stage.data.seed_registry import (
    ATTACKER_IP_POOL_CIDR,
    BENIGN_RATIO,
    LABEL_FILE,
    OUTPUT_DIR,
    TOTAL_FILES_TO_GENERATE,
    VICTIM_IP_POOL_CIDR,
    seed_library,
)
from stage.utils.config import get_path

PROJECT_ROOT = str(get_path("project", "root"))
BACKGROUND_CHUNKS_DIR = str(
    get_path("project", "root") / "dataset" / "background" / "background_chunks"
)


def load_background_chunks(chunks_dir):
    """Return all available background PCAP chunk paths."""
    print(f"[INFO] Scanning for background chunks in: {chunks_dir}")

    if not os.path.exists(chunks_dir):
        print(f"[ERROR] Background chunks directory not found: {chunks_dir}")
        print("[INFO] Please run the background chunking command first.")
        return None

    try:
        chunk_files = [
            os.path.join(chunks_dir, filename)
            for filename in os.listdir(chunks_dir)
            if filename.endswith((".pcap", ".pcapng")) and filename.startswith("bg_chunk_")
        ]

        if not chunk_files:
            print(f"[ERROR] No background chunks found in {chunks_dir}.")
            return None

        print(f"[INFO] Found {len(chunk_files)} background chunks to use as canvas.")
        return chunk_files

    except Exception as exc:
        print(f"[ERROR] Error scanning background chunks directory: {exc}")
        return None


def generate_benign_chunk(available_chunks, output_path):
    """Create a benign sample by copying one background chunk."""
    try:
        source_chunk = random.choice(available_chunks)
        shutil.copy(source_chunk, output_path)
        print(
            f"    [INFO] Benign: Copied '{os.path.basename(source_chunk)}' "
            f"to '{os.path.basename(output_path)}'"
        )

    except Exception as exc:
        print(f"    [ERROR] Failed to copy benign chunk: {exc}")
        try:
            with open(output_path, "w", encoding="utf-8") as file:
                file.write("")
        except Exception:
            pass


def remap_addresses(packets, ip_map, mac_map=None):
    """Replace packet IP and MAC addresses according to mapping tables."""
    if mac_map is None:
        mac_map = {}

    print(f"    [INFO] Remapping addresses for {len(packets)} packets...")

    remapped_packets = []
    remap_count = 0

    for pkt in packets:
        pkt_copy = pkt.copy()
        remapped = False

        if IP in pkt_copy:
            original_src_ip = pkt_copy[IP].src
            original_dst_ip = pkt_copy[IP].dst

            if original_src_ip in ip_map:
                pkt_copy[IP].src = ip_map[original_src_ip]
                remapped = True

            if original_dst_ip in ip_map:
                pkt_copy[IP].dst = ip_map[original_dst_ip]
                remapped = True

        if mac_map and Ether in pkt_copy:
            if pkt_copy[Ether].src in mac_map:
                pkt_copy[Ether].src = mac_map[pkt_copy[Ether].src]
            if pkt_copy[Ether].dst in mac_map:
                pkt_copy[Ether].dst = mac_map[pkt_copy[Ether].dst]

        remapped_packets.append(pkt_copy)

        if remapped:
            remap_count += 1

    print(f"    [INFO] Address remapping complete. {remap_count} packets had IPs remapped.")
    return remapped_packets


def rescale_timestamps(packets, new_start_time):
    """Shift packet timestamps to a new start time."""
    print(
        f"    [INFO] Rescaling timestamps for {len(packets)} packets "
        f"to start near {new_start_time:.2f}..."
    )

    if not packets:
        return []

    try:
        original_start_time = float(packets[0].time)
    except (AttributeError, ValueError, IndexError):
        print("    [WARN] Could not determine original start time. Timestamp rescaling skipped.")
        return [pkt.copy() for pkt in packets]

    rescaled_packets = []

    for pkt in packets:
        try:
            time_delta = float(pkt.time) - original_start_time
            pkt_copy = pkt.copy()

            if time_delta < 0:
                print(f"    [WARN] Negative time delta detected ({time_delta:.4f}). Using original timestamp.")
            else:
                pkt_copy.time = new_start_time + time_delta

            rescaled_packets.append(pkt_copy)

        except (AttributeError, ValueError):
            print("    [WARN] Packet missing valid time. Skipping rescale for this packet.")
            rescaled_packets.append(pkt.copy())

    return rescaled_packets


def get_new_ip_randomly(network, assigned_ips_set, role, max_attempts=1000):
    """Sample an unused IP address from the given network."""
    try:
        network_address_int = int(network.network_address)
        broadcast_address_int = int(network.broadcast_address)

        if network_address_int > broadcast_address_int:
            raise ValueError("Network address > broadcast")

    except Exception as exc:
        print(f"    [ERROR] Invalid network range for {network}: {exc}")
        return None

    for _ in range(max_attempts):
        try:
            if network.num_addresses <= 2:
                if network.num_addresses == 1:
                    random_ip_int = network_address_int
                elif network.num_addresses == 2:
                    random_ip_int = random.choice([network_address_int, broadcast_address_int])
                else:
                    raise ValueError("No assignable addresses")
            else:
                random_ip_int = random.randint(network_address_int + 1, broadcast_address_int - 1)

        except ValueError:
            if network.num_addresses == 2:
                random_ip_int = random.choice([network_address_int, broadcast_address_int])
            elif network.num_addresses == 1:
                random_ip_int = network_address_int
            else:
                print(f"    [WARN] Cannot generate random IP in small network {network} for role '{role}'.")
                return None

        ip_str = str(ipaddress.ip_address(random_ip_int))

        if ip_str not in assigned_ips_set:
            assigned_ips_set.add(ip_str)
            return ip_str

    print(f"    [WARN] IP pool likely exhausted after {max_attempts} attempts for role '{role}'.")
    return None


def generate_dynamic_ip_map(ip_map_template, victim_pool_cidr, attacker_pool_cidr):
    """Generate a dynamic IP mapping from seed templates and address pools."""
    print("    [INFO] Generating dynamic IP map...")

    new_ip_map = {}
    assigned_ips = set()
    new_victim_ip_str = "N/A"

    try:
        victim_net = ipaddress.ip_network(victim_pool_cidr)
        attacker_net = ipaddress.ip_network(attacker_pool_cidr)
    except ValueError as exc:
        print(f"    [ERROR] Invalid CIDR format in config: {exc}. Cannot generate IP map.")
        return {}, "N/A", "N/A"

    original_victim_ip_in_template = ip_map_template.get("victim")

    for role, original_ip_or_list in ip_map_template.items():
        if role == "victim":
            if isinstance(original_ip_or_list, str):
                new_ip = get_new_ip_randomly(victim_net, assigned_ips, "victim")
                if new_ip:
                    new_ip_map[original_ip_or_list] = new_ip
                    new_victim_ip_str = new_ip
                else:
                    return {}, "N/A", "N/A"
            else:
                print(f"    [WARN] Victim IP in template is not a string: {original_ip_or_list}. Skipping.")

        elif role.startswith("malicious"):
            ip_list_to_process = []

            if isinstance(original_ip_or_list, str):
                ip_list_to_process.append(original_ip_or_list)
            elif isinstance(original_ip_or_list, list):
                ip_list_to_process = original_ip_or_list
            elif role.endswith("_domains"):
                continue
            else:
                print(f"    [WARN] Unknown format for role '{role}' in template: {original_ip_or_list}. Skipping.")
                continue

            for original_ip in ip_list_to_process:
                if isinstance(original_ip, str):
                    new_ip = get_new_ip_randomly(attacker_net, assigned_ips, role)
                    if new_ip:
                        new_ip_map[original_ip] = new_ip
                    else:
                        return {}, new_victim_ip_str, "N/A"
                else:
                    print(f"    [WARN] Item in malicious list is not a string: {original_ip}. Skipping.")

    new_malicious_ips_list = []

    for original_ip, new_ip in new_ip_map.items():
        if new_ip is not None and original_ip != original_victim_ip_in_template:
            new_malicious_ips_list.append(new_ip)

    new_malicious_ips_str = (
        ",".join(sorted(set(new_malicious_ips_list)))
        if new_malicious_ips_list
        else "N/A"
    )

    print(f"    [INFO] Dynamic IP map generated. New victim IP: {new_victim_ip_str}")
    print(f"    [DEBUG] New malicious IPs string: {new_malicious_ips_str}")

    return new_ip_map, new_victim_ip_str, new_malicious_ips_str


def synthesize_and_inject(background_chunk_path, seed_pcap_path, output_path, new_ip_map):
    """Inject a remapped attack seed into a selected background PCAP chunk."""
    print(f"    [INFO] Starting synthesis for {os.path.basename(output_path)}")

    try:
        start_load = time.time()
        attack_packets = rdpcap(seed_pcap_path)

        if not attack_packets:
            print(f"      [WARN] Seed PCAP file is empty: {seed_pcap_path}. Skipping injection.")
            with open(output_path, "w", encoding="utf-8") as file:
                file.write("")
            return 0

        print(
            f"      [OK] Loaded {len(attack_packets)} attack packets from seed "
            f"in {time.time() - start_load:.2f}s."
        )

    except Exception as exc:
        print(f"      [ERROR] Failed to load seed PCAP {seed_pcap_path}: {exc}")
        with open(output_path, "w", encoding="utf-8") as file:
            file.write("")
        return 0

    print(f"      [INFO] Reading background chunk: {os.path.basename(background_chunk_path)}...")

    try:
        start_load_bg = time.time()
        background_chunk = list(rdpcap(background_chunk_path))

        print(
            f"      [OK] Loaded {len(background_chunk)} background packets "
            f"in {time.time() - start_load_bg:.2f}s."
        )

        if not background_chunk:
            print("      [ERROR] Failed to read background chunk because it is empty. Aborting injection.")
            with open(output_path, "w", encoding="utf-8") as file:
                file.write("")
            return 0

    except Exception as exc:
        print(f"\n      [ERROR] Error reading background chunk: {exc}")
        return 0

    try:
        injection_base_time = float(background_chunk[0].time) + random.uniform(0.001, 0.01)
    except (AttributeError, ValueError, IndexError):
        print("      [WARN] Background chunk lacks valid time. Using current time for injection.")
        injection_base_time = time.time()

    print(f"      --- INJECTION START (Base Time: {injection_base_time:.4f}) ---")

    remapped_attack = remap_addresses(attack_packets, new_ip_map)
    rescaled_attack = rescale_timestamps(remapped_attack, injection_base_time)

    try:
        injection_point_in_chunk = random.randint(0, len(background_chunk))
        print(
            f"      [INFO] Injecting {len(rescaled_attack)} attack packets into "
            f"background chunk at index {injection_point_in_chunk}."
        )

        final_packets = (
            background_chunk[:injection_point_in_chunk]
            + rescaled_attack
            + background_chunk[injection_point_in_chunk:]
        )

        print(f"      [INFO] Writing {len(final_packets)} packets to disk...")
        start_write = time.time()

        wrpcap(output_path, final_packets)

        end_write = time.time()

        print(
            f"      [OK] Successfully wrote {len(final_packets)} total packets to "
            f"{os.path.basename(output_path)} in {end_write - start_write:.2f}s."
        )
        print("      --- INJECTION COMPLETE ---")

        return injection_base_time

    except Exception as exc:
        print(f"\n      [ERROR] Error during injection/write phase: {exc}")

        if os.path.exists(output_path):
            try:
                os.remove(output_path)
                print(f"      [INFO] Removed incomplete file: {output_path}")
            except OSError as remove_exc:
                print(f"      [WARN] Could not remove incomplete file: {remove_exc}")

        return 0


def append_to_labels_csv(label_filepath, label_info_dict):
    """Append one label record to the label CSV file."""
    file_exists = os.path.isfile(label_filepath) and os.path.getsize(label_filepath) > 0

    try:
        with open(label_filepath, "a", newline="", encoding="utf-8") as csvfile:
            fieldnames = ["filename", "attack_type", "victim_ip", "malicious_ips", "injection_time"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames, extrasaction="ignore")

            if not file_exists:
                writer.writeheader()

            if label_info_dict.get("filename"):
                writer.writerow(label_info_dict)

    except IOError as exc:
        print(f"[ERROR] Failed to write to label file {label_filepath}: {exc}")


def main():
    """Run the full PCAP synthesis pipeline."""
    label_file_initialized = False

    try:
        if os.path.exists(LABEL_FILE) and os.path.getsize(LABEL_FILE) > 0:
            print(f"[INFO] Label file found: {LABEL_FILE}. Will attempt to resume.")
            label_file_initialized = True
        else:
            print(f"[INFO] Label file not found or empty. Initializing: {LABEL_FILE}")
            os.makedirs(os.path.dirname(LABEL_FILE), exist_ok=True)

            with open(LABEL_FILE, "w", newline="", encoding="utf-8") as csvfile:
                fieldnames = ["filename", "attack_type", "victim_ip", "malicious_ips", "injection_time"]
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()

            label_file_initialized = True
            print("[INFO] Label file initialized successfully.")

    except Exception as exc:
        print(f"[ERROR] Failed during label file check/initialization {LABEL_FILE}: {exc}")
        exit(1)

    if not label_file_initialized:
        print("[ERROR] Could not initialize or access label file. Exiting.")
        exit(1)

    available_bg_chunks = load_background_chunks(BACKGROUND_CHUNKS_DIR)

    if not available_bg_chunks:
        print("[ERROR] No background chunks available. Exiting.")
        exit(1)

    start_index = 0

    try:
        line_count = 0

        with open(LABEL_FILE, "r", encoding="utf-8") as csvfile:
            reader = csv.reader(csvfile)
            header = next(reader, None)

            if header:
                for row in reader:
                    if row and row[0]:
                        line_count += 1

                start_index = line_count

                if start_index > 0:
                    print(
                        f"[INFO] Found {start_index} existing records in label file. "
                        f"Resuming from index {start_index}."
                    )
            else:
                print("[INFO] Label file is empty or only contains header. Starting from index 0.")

    except FileNotFoundError:
        print("[INFO] Label file not found. Starting from index 0.")
        start_index = 0

    except Exception as exc:
        print(f"[WARN] Error reading label file to determine progress: {exc}. Starting from index 0.")
        start_index = 0

    if start_index >= TOTAL_FILES_TO_GENERATE:
        print(
            f"[INFO] Found {start_index} records, which meets or exceeds the target "
            f"({TOTAL_FILES_TO_GENERATE}). Nothing to generate."
        )
        exit(0)

    print(f"[INFO] Starting dataset generation from index {start_index} up to {TOTAL_FILES_TO_GENERATE}...")
    print(f"[INFO] BENIGN_RATIO set to {BENIGN_RATIO}")
    print(f"[INFO] Total target for this run: {TOTAL_FILES_TO_GENERATE - start_index} files.")

    start_time = time.time()

    for i in range(start_index, TOTAL_FILES_TO_GENERATE):
        run_index = i + 1
        output_filename_base = f"synthetic_run_{run_index:05d}.pcap"
        output_filepath = os.path.join(OUTPUT_DIR, output_filename_base)
        label_info = {"filename": output_filename_base}

        print(f"\n--- Generating file {run_index}/{TOTAL_FILES_TO_GENERATE} ---")

        if random.random() < BENIGN_RATIO:
            print("  Type: Benign")
            label_info["attack_type"] = "Benign"
            label_info["victim_ip"] = "N/A"
            label_info["malicious_ips"] = "N/A"
            label_info["injection_time"] = "N/A"

            generate_benign_chunk(available_bg_chunks, output_filepath)

        else:
            print("  Type: Malicious")

            selected_seed = random.choice(seed_library)
            label_info["attack_type"] = selected_seed["name"]
            print(f"  Selected Seed: {selected_seed['name']}")

            new_ip_map, new_victim_ip, new_malicious_ips_string = generate_dynamic_ip_map(
                selected_seed["ip_map_template"],
                VICTIM_IP_POOL_CIDR,
                ATTACKER_IP_POOL_CIDR,
            )

            label_info["victim_ip"] = new_victim_ip
            label_info["malicious_ips"] = new_malicious_ips_string

            background_chunk_to_use = random.choice(available_bg_chunks)

            actual_injection_time = synthesize_and_inject(
                background_chunk_to_use,
                selected_seed["pcap_path"],
                output_filepath,
                new_ip_map,
            )

            label_info["injection_time"] = actual_injection_time if actual_injection_time > 0 else "N/A"

        try:
            append_to_labels_csv(LABEL_FILE, label_info)
            print(f"  Label appended for: {output_filename_base}")
        except Exception as exc:
            print(f"[ERROR] Failed to append label for {output_filename_base}: {exc}")

    end_time = time.time()

    print("\n--- Dataset generation complete ---")
    print(f"Generated {TOTAL_FILES_TO_GENERATE - start_index} new files in {end_time - start_time:.2f} seconds.")
    print(f"Total files in dataset: {TOTAL_FILES_TO_GENERATE}")
    print(f"Output PCAPs are in: {OUTPUT_DIR}")
    print(f"Label file is: {LABEL_FILE}")


if __name__ == "__main__":
    main()
