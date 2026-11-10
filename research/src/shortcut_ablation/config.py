from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# 原始 PCAP 标准入口：刚才已经通过软链接修复
DATASET_DIR = PROJECT_ROOT / "dataset"
SEED_DIR = DATASET_DIR / "seeds"
BACKGROUND_DIR = DATASET_DIR / "background"
BACKGROUND_CHUNKS_DIR = BACKGROUND_DIR / "background_chunks"

# shortcut ablation 独立输出目录
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "shortcut_ablation"
PCAP_OUTPUT_DIR = OUTPUT_ROOT / "pcaps"
LABEL_OUTPUT_DIR = OUTPUT_ROOT / "labels"
LABEL_FILE = LABEL_OUTPUT_DIR / "labels.csv"

GRAPH_OUTPUT_DIR = OUTPUT_ROOT / "graphs"
SHIFTED_GRAPH_OUTPUT_DIR = OUTPUT_ROOT / "graphs_shifted"
SPLIT_OUTPUT_DIR = OUTPUT_ROOT / "splits"
CHECKPOINT_OUTPUT_DIR = OUTPUT_ROOT / "checkpoints"
METRIC_OUTPUT_DIR = OUTPUT_ROOT / "metrics"

DEFAULT_TOTAL_FILES = 10000
DEFAULT_BENIGN_RATIO = 0.5
