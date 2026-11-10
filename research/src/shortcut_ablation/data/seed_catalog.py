from pathlib import Path
from shortcut_ablation.config import SEED_DIR


SEED_LIBRARY = [
    {
        "name": "CVE-2017-0199_AgentTesla",
        "pcap_filename": "2025-01-09-CVE-2017-0199-XLS-to-DBatLoader-or-GuLoader-for-AgentTesla-variant.pcap",
        "victim": "10.1.9.101",
        "malicious_ips": ["192.3.27.144", "107.172.31.5", "14.103.79.10", "104.17.201.1", "104.17.202.1", "89.39.83.184"],
    },
    {
        "name": "SmartApeSG_FakeUpdate_RAT_StealC",
        "pcap_filename": "2025-03-26-SmartApeSG-leads-to-NetSupport-RAT-and-StealC.pcap",
        "victim": "10.3.26.44",
        "malicious_ips": ["194.180.191.168", "193.239.237.40", "79.141.173.27", "172.67.150.76", "104.21.88.31"],
    },
    {
        "name": "SmartApeSG_CAPTCHA_RAT_StealCv2",
        "pcap_filename": "2025-08-20-SmartAgeSG-Netsupport-RAT-with-StealCv2.pcap",
        "victim": "10.8.20.101",
        "malicious_ips": ["38.146.28.242", "79.141.165.202", "23.23.49.179", "172.86.90.13", "45.61.150.28"],
    },
    {
        "name": "Kongtuke_ClickFix_Lumma",
        "pcap_filename": "2025-09-03-Kongtuke-ClickFix-leading-to-Lumma-Stealer.pcap",
        "victim": "160.9.3.101",
        "malicious_ips": ["85.209.129.105", "91.212.166.160", "192.153.57.201", "64.95.13.172", "104.16.231.132", "104.16.230.132"],
    },
    {
        "name": "Lumma_Ghostsocks",
        "pcap_filename": "2025-09-24-traffic-from-running-Setup.exe.pcap",
        "victim": "10.9.24.101",
        "malicious_ips": ["164.90.129.126", "86.54.24.25", "91.212.166.19", "91.84.106.132"],
    },
    {
        "name": "ParrotTDS_SocGholish_AsyncRAT",
        "pcap_filename": "2024-02-21-SocGholish-AsyncRAT-infection.pcap",
        "victim": "10.2.21.101",
        "malicious_ips": ["66.135.17.87", "45.59.170.106", "49.13.65.235", "167.71.107.109", "5.161.113.150"],
    },
    {
        "name": "Macro_SSLoad_CobaltStrike",
        "pcap_filename": "2024-04-18-SSLoad-with-follow-up-Cobalt-Strike-DLL.pcap",
        "victim": "10.4.18.169",
        "malicious_ips": ["85.239.53.219", "212.18.104.28", "193.32.176.22"],
    },
    {
        "name": "GoogleAd_Matanbuchus_Danabot",
        "pcap_filename": "2024-06-17-Matanbuchus-infection-with-Danbot-traffic.pcap",
        "victim": "10.6.17.101",
        "malicious_ips": ["194.67.193.206", "34.130.217.52", "47.89.157.126", "8.211.34.5", "194.67.193.204", "47.254.129.255"],
    },
    {
        "name": "Latrodectus_VNC",
        "pcap_filename": "2024-06-25-Latrodectus-infection-with-BackConnect-and-Keyhole-VNC.pcap",
        "victim": "10.6.25.101",
        "malicious_ips": ["74.119.194.126", "94.232.249.102", "94.232.249.103", "184.174.96.179", "104.129.20.167", "198.244.224.83", "64.7.198.158", "85.208.108.63", "193.32.177.192", "85.208.108.30"],
    },
    {
        "name": "GuLoader_RemcosRAT",
        "pcap_filename": "2024-08-26-GuLoader-for-Remcos-RAT.pcap",
        "victim": "10.8.26.101",
        "malicious_ips": ["194.36.140.235", "206.123.148.197"],
    },
    {
        "name": "GoogleAd_FakeBat_Redline_Gozi",
        "pcap_filename": "2023-02-03-DEV-0569-part-2-FakeBat-download-and-infection-traffic.pcap",
        "victim": "10.2.3.101",
        "malicious_ips": ["62.204.41.176", "62.204.41.175", "80.77.23.77", "80.77.25.114", "80.77.25.109", "170.130.165.188", "45.11.182.208", "176.113.115.177", "194.87.216.194", "47.251.52.170", "81.177.136.237", "81.177.6.46", "199.101.134.238", "204.155.149.140", "199.101.133.166"],
    },
    {
        "name": "TA577_Pikabot_CobaltStrike",
        "pcap_filename": "2023-12-18-TA577-Pikabot-infection-with-Cobalt-Strike.pcap",
        "victim": "10.0.0.101",
        "malicious_ips": ["45.33.15.215", "45.56.71.218", "207.246.99.159", "143.95.249.177", "41.185.8.61"],
    },
]


def get_seed_records():
    records = []
    for item in SEED_LIBRARY:
        record = dict(item)
        record["pcap_path"] = SEED_DIR / item["pcap_filename"]
        records.append(record)
    return records


def validate_seed_files():
    missing = []
    for record in get_seed_records():
        if not Path(record["pcap_path"]).exists():
            missing.append(str(record["pcap_path"]))
    return missing
