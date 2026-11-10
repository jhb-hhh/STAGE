"""Dataset loader for pre-built ID-APT graph snapshots."""

import glob
import logging
from pathlib import Path

import torch
from torch_geometric.data import InMemoryDataset
from tqdm import tqdm

from stage.utils.config import get_path

logger = logging.getLogger(__name__)


class IDAPTDataset(InMemoryDataset):
    """In-memory PyG dataset backed by pre-built graph chunk files."""

    def __init__(self, root_dir=None, transform=None, pre_transform=None):
        self.graph_dir = Path(root_dir) if root_dir is not None else get_path("data", "graph_dir")
        super().__init__(str(self.graph_dir), transform, pre_transform)
        self.data, self.slices = self._load_and_merge_in_memory()
        logger.info(f"Loaded {len(self)} graph snapshots from {self.graph_dir}")

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return []

    def download(self):
        pass

    def process(self):
        pass

    def _load_and_merge_in_memory(self):
        """Load all graph chunks and collate them into one in-memory dataset."""
        pattern = str(self.graph_dir / "graphs_chunk_*.pt")
        chunk_files = sorted(glob.glob(pattern))

        if not chunk_files:
            raise FileNotFoundError(f"No graphs_chunk_*.pt files found in {self.graph_dir}")

        data_list = []

        for file_path in tqdm(chunk_files, desc="Loading graph chunks"):
            chunk_data = torch.load(file_path, weights_only=False)

            if self.pre_filter is not None:
                chunk_data = [d for d in chunk_data if self.pre_filter(d)]

            if self.pre_transform is not None:
                chunk_data = [self.pre_transform(d) for d in chunk_data]

            data_list.extend(chunk_data)

        return self.collate(data_list)


NIDSDataset = IDAPTDataset


if __name__ == "__main__":
    dataset = IDAPTDataset()
    print(f"Dataset length: {len(dataset)}")
    print(dataset[0])
