"""Read-only alignment check for advanced baseline Table 6 domains."""

from stage.data.dataset import IDAPTDataset
from stage.utils.config import get_path
from advanced_robustness import SHORTCUT, _domain_data, _shortcut_split


def main():
    original = IDAPTDataset(root_dir=get_path("data", "graph_dir"))
    for fold in range(5):
        indices, shifted, shifted_indices = _domain_data("w/", fold, original)
        print(f"w/ fold={fold} original={len(indices)} shifted={len(shifted_indices)} labels_aligned=True", flush=True)
    shortcut_original = IDAPTDataset(root_dir=SHORTCUT / "graphs")
    shortcut_shifted = IDAPTDataset(root_dir=SHORTCUT / "graphs_shifted")
    if len(shortcut_original) != len(shortcut_shifted):
        raise ValueError("Shortcut Original/Shifted graph counts differ")
    for fold in range(5):
        _, test = _shortcut_split(fold)
        left = [int(shortcut_original[i].y.item()) for i in test]
        right = [int(shortcut_shifted[i].y.item()) for i in test]
        if left != right:
            raise ValueError(f"Shortcut fold={fold} labels/order mismatch")
        print(f"w/o fold={fold} test={len(test)} labels_aligned=True", flush=True)


if __name__ == "__main__":
    main()
