"""Seed traffic registry and synthesis parameters for ID-APT generation."""

import os
from pathlib import Path

PROJECT_ROOT = str(Path(__file__).resolve().parents[3])

BASE_DATA_DIR = os.path.join(PROJECT_ROOT, "dataset")
SEED_DIR = os.path.join(BASE_DATA_DIR, "seeds")
BACKGROUND_DIR = os.path.join(BASE_DATA_DIR, "background")

OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data/processed/synthesized_pcaps")
LABEL_DIR = os.path.join(PROJECT_ROOT, "data/processed/labels")
LABEL_FILE = os.path.join(LABEL_DIR, "labels.csv")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(LABEL_DIR, exist_ok=True)

seed_library = [
    {
        "name": "CVE-2017-0199_AgentTesla",
        "date": "2025-01-09",
        "pcap_path": os.path.join(SEED_DIR, "2025-01-09-CVE-2017-0199-XLS-to-DBatLoader-or-GuLoader-for-AgentTesla-variant.pcap"),
        "ip_map_template": {
            "victim": "10.1.9.101",
            "malicious_initial": ["192.3.27.144", "107.172.31.5", "14.103.79.10", "104.17.201.1", "104.17.202.1"],
            "malicious_exfil": "89.39.83.184",
        },
    },
    {
        "name": "SmartApeSG_FakeUpdate_RAT_StealC",
        "date": "2025-03-26",
        "pcap_path": os.path.join(SEED_DIR, "2025-03-26-SmartApeSG-leads-to-NetSupport-RAT-and-StealC.pcap"),
        "ip_map_template": {
            "victim": "10.3.26.44",
            "malicious_rat_c2": "194.180.191.168",
            "malicious_stealc_c2": "193.239.237.40",
            "malicious_initial": ["79.141.173.27", "172.67.150.76", "104.21.88.31"],
        },
    },
    {
        "name": "SmartApeSG_CAPTCHA_RAT_StealCv2",
        "date": "2025-08-20",
        "pcap_path": os.path.join(SEED_DIR, "2025-08-20-SmartAgeSG-Netsupport-RAT-with-StealCv2.pcap"),
        "ip_map_template": {
            "victim": "10.8.20.101",
            "malicious_rat_c2": "38.146.28.242",
            "malicious_stealc_c2": "79.141.165.202",
            "malicious_initial": ["23.23.49.179", "172.86.90.13", "45.61.150.28"],
        },
    },
    {
        "name": "Kongtuke_ClickFix_Lumma",
        "date": "2025-09-03",
        "pcap_path": os.path.join(SEED_DIR, "2025-09-03-Kongtuke-ClickFix-leading-to-Lumma-Stealer.pcap"),
        "ip_map_template": {
            "victim": "160.9.3.101",
            "malicious_download": "85.209.129.105",
            "malicious_lumma_c2": "91.212.166.160",
            "malicious_initial": ["192.153.57.201"],
            "malicious_misc": ["64.95.13.172", "104.16.231.132", "104.16.230.132"],
        },
    },
    {
        "name": "Lumma_Ghostsocks",
        "date": "2025-09-24",
        "pcap_path": os.path.join(SEED_DIR, "2025-09-24-traffic-from-running-Setup.exe.pcap"),
        "ip_map_template": {
            "victim": "10.9.24.101",
            "malicious_lumma_c2": "164.90.129.126",
            "malicious_backdoor_download": "86.54.24.25",
            "malicious_backdoor_c2": ["91.212.166.19", "91.84.106.132"],
        },
    },
    {
        "name": "ParrotTDS_SocGholish_AsyncRAT",
        "date": "2024-02-21",
        "pcap_path": os.path.join(SEED_DIR, "2024-02-21-SocGholish-AsyncRAT-infection.pcap"),
        "ip_map_template": {
            "victim": "10.2.21.101",
            "malicious_socgholish": ["66.135.17.87", "45.59.170.106"],
            "malicious_rat_download": ["49.13.65.235", "167.71.107.109"],
            "malicious_rat_c2": "5.161.113.150",
        },
    },
    {
        "name": "Macro_SSLoad_CobaltStrike",
        "date": "2024-04-18",
        "pcap_path": os.path.join(SEED_DIR, "2024-04-18-SSLoad-with-follow-up-Cobalt-Strike-DLL.pcap"),
        "ip_map_template": {
            "victim": "10.4.18.169",
            "malicious_ssload_c2": "85.239.53.219",
            "malicious_cs_download": "212.18.104.28",
            "malicious_cs_c2_attempt": "193.32.176.22",
        },
    },
    {
        "name": "GoogleAd_Matanbuchus_Danabot",
        "date": "2024-06-17",
        "pcap_path": os.path.join(SEED_DIR, "2024-06-17-Matanbuchus-infection-with-Danbot-traffic.pcap"),
        "ip_map_template": {
            "victim": "10.6.17.101",
            "malicious_matanbuchus_c2": "194.67.193.206",
            "malicious_danabot_c2": "34.130.217.52",
            "malicious_initial_download": ["47.89.157.126", "8.211.34.5", "194.67.193.204", "47.254.129.255"],
        },
    },
    {
        "name": "Latrodectus_VNC",
        "date": "2024-06-25",
        "pcap_path": os.path.join(SEED_DIR, "2024-06-25-Latrodectus-infection-with-BackConnect-and-Keyhole-VNC.pcap"),
        "ip_map_template": {
            "victim": "10.6.25.101",
            "malicious_latrodectus_c2": [
                "74.119.194.126",
                "94.232.249.102",
                "94.232.249.103",
                "184.174.96.179",
                "104.129.20.167",
                "198.244.224.83",
            ],
            "malicious_vnc": "64.7.198.158",
            "malicious_download": ["85.208.108.63", "193.32.177.192", "85.208.108.30"],
        },
    },
    {
        "name": "GuLoader_RemcosRAT",
        "date": "2024-08-26",
        "pcap_path": os.path.join(SEED_DIR, "2024-08-26-GuLoader-for-Remcos-RAT.pcap"),
        "ip_map_template": {
            "victim": "10.8.26.101",
            "malicious_guloader_download": "194.36.140.235",
            "malicious_remcos_c2": "206.123.148.197",
        },
    },
    {
        "name": "GoogleAd_FakeBat_Redline_Gozi",
        "date": "2023-02-03",
        "pcap_path": os.path.join(SEED_DIR, "2023-02-03-DEV-0569-part-2-FakeBat-download-and-infection-traffic.pcap"),
        "ip_map_template": {
            "victim": "10.2.3.101",
            "malicious_redline": ["62.204.41.176", "62.204.41.175"],
            "malicious_gozi": ["80.77.23.77", "80.77.25.114", "80.77.25.109", "170.130.165.188", "45.11.182.208", "176.113.115.177"],
            "malicious_vnc": "194.87.216.194",
            "malicious_initial_download": ["47.251.52.170", "81.177.136.237", "81.177.6.46", "199.101.134.238", "204.155.149.140", "199.101.133.166"],
        },
    },
    {
        "name": "TA577_Pikabot_CobaltStrike",
        "date": "2023-12-18",
        "pcap_path": os.path.join(SEED_DIR, "2023-12-18-TA577-Pikabot-infection-with-Cobalt-Strike.pcap"),
        "ip_map_template": {
            "victim": "10.0.0.101",
            "malicious_pikabot_c2": ["45.33.15.215", "45.56.71.218"],
            "malicious_cs_c2": "207.246.99.159",
            "malicious_download": "143.95.249.177",
            "malicious_initial": ["41.185.8.61"],
        },
    },
]

BACKGROUND_PCAP_PATH = os.path.join(BACKGROUND_DIR, "background_traffic.pcap")
BACKGROUND_TEST_PCAP_PATH = os.path.join(BACKGROUND_DIR, "background_test.pcap")

TOTAL_FILES_TO_GENERATE = 10000
BENIGN_RATIO = 0.5
VICTIM_IP_POOL_CIDR = "192.168.0.0/16"
ATTACKER_IP_POOL_CIDR = "100.64.0.0/10"
USE_TEST_BACKGROUND = False
