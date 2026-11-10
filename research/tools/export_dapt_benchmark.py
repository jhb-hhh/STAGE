"""Recompute each test metric from predictions and export a complete paper row block."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from dapt_benchmark import MODELS, METRICS, metrics


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    folder = args.output
    splits = json.loads((folder / "splits.json").read_text())
    rows, fold_rows = [], []
    missing = []
    for name in MODELS:
        model_metrics = []
        for fold in range(5):
            path = folder / "results" / f"{name}_fold_{fold}.json"
            if not path.exists():
                missing.append(f"{name}/fold_{fold}")
                continue
            result = json.loads(path.read_text())
            pred = result["predictions"]
            assert sorted(r["graph_id"] for r in pred) == sorted(splits[fold]["test"])
            assert all(r["pred"] == int(r["score"] > result["selection"]["threshold"]) for r in pred)
            computed = metrics([r["true"] for r in pred], [r["pred"] for r in pred])
            for metric in METRICS:
                assert abs(computed[metric] - result["metrics"][metric]) < 1e-12
            choices = [json.loads(p.read_text()) for p in (folder / "runs" / name / f"fold_{fold}").glob("candidate_*/complete.json")]
            assert len(choices) == 2
            assert result["selection"]["validation_macro_f1"] == max(c["validation_macro_f1"] for c in choices)
            model_metrics.append(computed)
            fold_rows.append({"Model": name, "Fold": fold, **computed})
        row = {"Model": name, "Folds": len(model_metrics)}
        for metric in METRICS:
            values = [m[metric] for m in model_metrics]
            row[metric + "_mean"] = np.mean(values) if values else np.nan
            row[metric + "_std"] = np.std(values, ddof=1) if len(values) > 1 else np.nan
        rows.append(row)
    pd.DataFrame(fold_rows).to_csv(folder / "checked_fold_metrics.csv", index=False)
    pd.DataFrame(rows).to_csv(folder / "checked_summary.csv", index=False)
    if missing:
        print("Not final; incomplete: " + ", ".join(missing))
        return
    best = {m: (min if m == "FPR" else max)(r[m + "_mean"] for r in rows) for m in METRICS}
    header = ["Dataset", "Model", "Accuracy", "Precision", "Recall", "F1 (Macro)", "FPR", "MCC"]
    markdown = ["# DAPT2020: five blocked folds, mean ± sample std", "",
                "| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    latex = []
    for row in rows:
        cells, tex_cells = [], []
        for metric in METRICS:
            avg, std = row[metric + "_mean"], row[metric + "_std"]
            s = f"{avg:.4f} ± {std:.4f}"
            tex = f"{avg:.4f} \\pm {std:.4f}"
            if abs(avg - best[metric]) < 1e-12:
                s = "**" + s + "**"
                tex = "\\mathbf{" + tex + "}"
            cells.append(s)
            tex_cells.append("$" + tex + "$")
        model = "STAGE (Ours)" if row["Model"] == "STAGE" else row["Model"]
        markdown.append("| DAPT2020 | " + model + " | " + " | ".join(cells) + " |")
        latex.append("DAPT2020 & " + model + " & " + " & ".join(tex_cells) + " \\\\")
    markdown += ["", "Precision/Recall: attack-positive binary; F1: two-class macro average. "
                 "All methods use common input fields and validation-only selection. "
                 "Existing STAGE graph-window adaptations; same-campaign within-domain evaluation. "
                 "Test folds contain only 18–23 graph windows, including overlapping windows within each fold. "
                 "These results do not establish unseen-campaign or cross-domain performance.",
                 "", "Bold marks the best measured mean in each column; it does not indicate statistical significance."]
    (folder / "paper_dapt_table.md").write_text("\n".join(markdown) + "\n", encoding="utf-8")
    (folder / "paper_dapt_rows.tex").write_text("\n".join(latex) + "\n", encoding="utf-8")
    print("\n".join(markdown))


if __name__ == "__main__":
    main()
