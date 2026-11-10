"""Build PyG graph chunks from synthesized PCAP files."""

import argparse

from stage.data import graph_construction
from stage.utils.config import get_active_profile


def build_parser():
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(
        description="Build PyG graph chunks from synthesized PCAP files."
    )
    parser.add_argument(
        "--run",
        action="store_true",
        help="Actually run graph construction. Without this flag, only prints paths.",
    )

    return parser


def main():
    """Print the graph-construction plan and optionally run it."""
    parser = build_parser()
    args = parser.parse_args()

    print("Graph construction plan")
    print("path_profile:", get_active_profile() or "default")
    print("input_pcap_dir:", graph_construction.RAW_PCAP_DIR)
    print("label_file:", graph_construction.LABEL_FILE_PATH)
    print("output_graph_dir:", graph_construction.GRAPH_SAVE_DIR)

    if get_active_profile() != "workspace":
        print("note: default graph construction writes to a generated-graph directory.")
        print("note: use python tools/run.py rebuild graphs for workspace rebuilds.")

    if not args.run:
        print("dry-run only; graph construction is not started.")
        return

    graph_construction.process_all_pcaps()


if __name__ == "__main__":
    main()
