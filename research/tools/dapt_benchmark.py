"""DAPT binary benchmark: split raw flows BEFORE graph construction.

Standalone versioned workspace experiment. No reference artifacts are changed.
Uses the project's graph-window adapters, not byte-for-byte official baselines.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch_geometric.loader import DataLoader
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             matthews_corrcoef, precision_score, recall_score)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from stage.baselines.registry import build_baseline
from stage.models.stage import StageModel
from convert_idapt_aligned_directional import directional_rows, graph_windows, make_graph

MODELS = ["TS-IDS", "FlowTransformer", "E-T-GraphSAGE", "GraphIDS", "DIDS-MFL", "STAGE"]
LRS = {"TS-IDS": 1e-3, "FlowTransformer": 1e-3, "E-T-GraphSAGE": 1e-4,
       "GraphIDS": 7.423525164428174e-5, "DIDS-MFL": 1e-4, "STAGE": 3e-4}
METRICS = ["Accuracy", "Precision", "Recall", "F1_Macro", "FPR", "MCC"]


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def dump(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    tmp.replace(path)


def save(path, obj):
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    torch.save(obj, tmp)
    tmp.replace(path)


def parse_time(series):
    first = pd.to_datetime(series, format="%d/%m/%Y %I:%M:%S %p", errors="coerce")
    missing = first.isna()
    first.loc[missing] = pd.to_datetime(series[missing], format="%d/%m/%Y %H:%M:%S", errors="coerce")
    if first.isna().any():
        raise ValueError("Unparsed timestamps: " + str(series[first.isna()].head().tolist()))
    return first


def prepare(args):
    out = args.output
    if (out / "manifest.json").exists() or (out / "graphs.pt").exists():
        raise FileExistsError("Use a fresh experiment directory; prepared datasets are immutable")
    out.mkdir(parents=True, exist_ok=True)
    paths = sorted(args.raw_dir.glob("*.csv"))
    if len(paths) != 10:
        raise ValueError(f"Expected 10 DAPT captures, found {len(paths)}")
    schema = None
    for p in paths:
        columns = pd.read_csv(p, nrows=0).columns.str.strip().tolist()
        if "Timestamp" in columns and "Stage" in columns:
            schema = columns
            break
    frames, sources = [], []
    for p in paths:
        df = pd.read_csv(p, low_memory=False)
        df.columns = df.columns.str.strip()
        repaired = "Timestamp" not in df
        if repaired:
            df = pd.read_csv(p, header=None, names=schema, low_memory=False)
        if df.shape[1] != 85:
            raise ValueError(f"Unexpected schema {p}")
        df["_source"] = p.name
        df["_row"] = np.arange(len(df))
        ts = parse_time(df["Timestamp"])
        df["_time"] = ts.astype("int64") / 1e9
        df["_day"] = ts.dt.strftime("%Y-%m-%d")
        duration = pd.to_numeric(df["Flow Duration"], errors="coerce")
        if duration.isna().any() or (duration < 0).any():
            raise ValueError(f"Invalid flow durations {p}")
        df["_end"] = df["_time"] + duration / 1e6
        # Exclude identifiers/annotations which may differ between duplicate exports.
        keys = [c for c in schema if c not in {"Flow ID", "Stage", "Activity", "Label"}]
        df["_fingerprint"] = pd.util.hash_pandas_object(df[keys].astype(str), index=False).astype(str)
        sources.append({"name": p.name, "sha256": sha(p), "rows": len(df), "header_restored": repaired})
        frames.append(df)
    all_rows = pd.concat(frames, ignore_index=True)
    all_rows["_attack"] = ~all_rows.Stage.str.strip().str.lower().eq("benign")
    if all_rows.groupby("_fingerprint")._attack.nunique().max() != 1:
        raise ValueError("Conflicting duplicate labels")
    raw_count = len(all_rows)
    all_rows = all_rows.drop_duplicates("_fingerprint").copy()
    all_rows["_block"] = -1
    boundaries, purged = {}, 0
    # Common day boundaries across public/private captures keep contemporaneous
    # observations together. Split is label-independent and is not future-only.
    for day, group in all_rows.groupby("_day", sort=True):
        cuts = np.quantile(group._time, [.2, .4, .6, .8])
        if len(set(cuts)) != 4:
            raise ValueError("Insufficient unique temporal boundaries")
        start_block = np.searchsorted(cuts, group._time, side="right")
        end_block = np.searchsorted(cuts, group._end, side="right")
        keep = start_block == end_block
        # Purge flows touching a boundary (timestamps have one-second precision).
        for cut in cuts:
            keep &= ~((group._time <= cut + 1) & (group._end >= cut - 1))
        all_rows.loc[group.index[keep], "_block"] = start_block[keep]
        purged += int((~keep).sum())
        boundaries[day] = cuts.tolist()
    kept = all_rows[all_rows._block >= 0].copy()
    graphs, records, flow_audit = [], [], []
    for (source, block), group in kept.groupby(["_source", "_block"], sort=True):
        # Carry record IDs through the converter using its original row indices.
        group = group.sort_values("_time", kind="stable").reset_index(drop=True)
        flows = directional_rows(group)
        expected_ts = set(group._time.tolist())
        if not set(flows.start_time.tolist()).issubset(expected_ts):
            raise AssertionError("Timestamp conversion disagrees with split parser")
        for wi, window in enumerate(graph_windows(flows)):
            graph_id = len(graphs)
            graph = make_graph(window, random.Random(42 + graph_id), source, wi)
            graph.graph_id = graph_id
            graph.block_id = int(block)
            graphs.append(graph)
            records.append({"id": graph_id, "source": source, "block": int(block),
                            "window": wi, "nodes": graph.num_nodes,
                            "label": int(graph.y.item()),
                            "start": float(window.start_time.min()),
                            "end": float((window.start_time + window.duration).max())})
        flow_audit.extend(group[["_source", "_row", "_fingerprint", "_day", "_block", "_time", "_end"]].to_dict("records"))
        print(f"PREPARE {source} block={block} raw={len(group)} directional={len(flows)} graphs_total={len(graphs)}", flush=True)
    splits = []
    for fold in range(5):
        split = {"fold": fold, "train": [], "val": [], "test": []}
        for rec in records:
            partition = "test" if rec["block"] == fold else "val" if rec["block"] == (fold + 1) % 5 else "train"
            split[partition].append(rec["id"])
        split["class_counts"] = {}
        for key in ("train", "val", "test"):
            counts = np.bincount([records[i]["label"] for i in split[key]], minlength=2).tolist()
            if min(counts) == 0:
                raise ValueError(f"Fold {fold} {key} lacks a class: {counts}")
            split["class_counts"][key] = counts
        splits.append(split)
    # Fingerprints cannot cross temporal blocks; graph overlap is within blocks only.
    audit = pd.DataFrame(flow_audit)
    assert audit.groupby("_fingerprint")._block.nunique().max() == 1
    pd.DataFrame(records).to_csv(out / "graph_index.csv", index=False)
    audit.to_csv(out / "raw_assignment.csv", index=False)
    save(out / "graphs.pt", graphs)
    dump(out / "splits.json", splits)
    manifest = {"protocol": "dapt-within-domain-purged-time-block-v1", "sources": sources,
                "raw_rows": raw_count, "deduplicated_rows": len(all_rows), "boundary_purged_rows": purged,
                "retained_raw_rows": len(kept), "graphs": len(graphs), "boundaries_unix_seconds": boundaries,
                "max_window_flows": 3000, "stride": 1500, "classes": {"0": "benign", "1": "any_attack_in_window"},
                "split": "test block k, validation block (k+1)%5, remaining blocks training, within each day",
                "features": "8 fixed-log numeric + protocol + capped ports; flags unavailable (zero)",
                "feature_parity": "all baselines receive the same learned categorical embeddings as STAGE",
                "limitations": ["same-campaign within-domain evaluation, not unseen-campaign generalization",
                                "windows overlap within each partition; not independent attack instances",
                                "flow endpoints/stats reconstructed from bidirectional CSVs, not raw PCAP",
                                "uses existing STAGE graph-window baseline adaptations, not official exact reproductions"],
                "hashes": {p: sha(out / p) for p in ("graphs.pt", "splits.json", "graph_index.csv", "raw_assignment.csv")},
                "code_hashes": {"runner": sha(__file__), "converter": sha(Path(__file__).with_name("convert_idapt_aligned_directional.py"))}}
    dump(out / "manifest.json", manifest)
    print(json.dumps({"graphs": len(graphs), "purged": purged, "splits": [s["class_counts"] for s in splits]}), flush=True)


class CommonInput(nn.Module):
    """Expose the SAME input fields to each existing baseline backbone."""
    def __init__(self, name):
        super().__init__()
        self.proto_emb = nn.Embedding(256, 8)
        self.port_emb = nn.Embedding(1026, 16)
        self.flag_emb = nn.Embedding(64, 8)
        self.core = build_baseline(name, num_classes=2, in_channels=56, edge_dim=2)

    def inputs(self, batch):
        out = batch.clone()
        out.x = torch.cat([batch.x, self.proto_emb(batch.proto_idx), self.port_emb(batch.sport_idx),
                           self.port_emb(batch.dport_idx), self.flag_emb(batch.flags_idx)], dim=-1)
        return out

    def forward(self, batch, **kwargs):
        return self.core(self.inputs(batch), **kwargs)

    def reconstruction_loss(self, batch):
        return self.core.reconstruction_loss(self.inputs(batch))

    def anomaly_score(self, batch):
        return self.core.anomaly_score(self.inputs(batch))


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_model(name, device):
    return (StageModel(num_classes=2) if name == "STAGE" else CommonInput(name)).to(device)


def loader(graphs, indices, shuffle=False, balanced=False):
    selected = [graphs[i] for i in indices]
    sampler = None
    if balanced:
        labels = np.array([g.y.item() for g in selected], dtype=int)
        counts = np.bincount(labels, minlength=2)
        weights = torch.tensor(1 / counts[labels], dtype=torch.double)
        sampler = torch.utils.data.WeightedRandomSampler(weights, len(weights), replacement=True)
    # Batch one prevents padding effects in recurrent encoders. Accumulate 4.
    return DataLoader(selected, batch_size=1, shuffle=shuffle and sampler is None,
                      sampler=sampler, num_workers=0)


@torch.no_grad()
def predict(model, data_loader, name, device):
    model.eval()
    scores, labels, ids = [], [], []
    for batch in data_loader:
        batch = batch.to(device)
        value = model.anomaly_score(batch) if name == "GraphIDS" else model(batch).softmax(-1)[:, 1]
        scores.extend(value.cpu().tolist())
        labels.extend(batch.y.cpu().tolist())
        ids.extend(batch.graph_id.cpu().tolist())
    values = np.array(scores, dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Non-finite evaluation scores")
    return np.array(labels, dtype=int), values, ids


def calibrate(labels, scores):
    # Exact distinct predictions, validation ONLY; deterministic closest-median tie break.
    unique = np.unique(scores)
    candidates = np.r_[np.nextafter(unique[0], -np.inf), unique]
    best, threshold = -1., float(np.median(scores))
    for t in candidates:
        value = f1_score(labels, scores > t, labels=[0, 1], average="macro", zero_division=0)
        if value > best or (value == best and abs(t - np.median(scores)) < abs(threshold - np.median(scores))):
            best, threshold = float(value), float(t)
    return best, threshold


def metrics(y, pred):
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {"Accuracy": float(accuracy_score(y, pred)),
            "Precision": float(precision_score(y, pred, zero_division=0)),
            "Recall": float(recall_score(y, pred, zero_division=0)),
            "F1_Macro": float(f1_score(y, pred, labels=[0, 1], average="macro", zero_division=0)),
            "FPR": float(fp / (fp + tn)) if fp + tn else None,
            "MCC": float(matthews_corrcoef(y, pred)),
            "Precision_Macro": float(precision_score(y, pred, labels=[0, 1], average="macro", zero_division=0)),
            "Recall_Macro": float(recall_score(y, pred, labels=[0, 1], average="macro", zero_division=0)),
            "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}


def train_candidate(args, name, fold, candidate, graphs, split, device):
    folder = args.output / "runs" / name / f"fold_{fold}" / f"candidate_{candidate}"
    folder.mkdir(parents=True, exist_ok=True)
    if (folder / "complete.json").exists():
        return json.loads((folder / "complete.json").read_text())
    seed_all(20261105 + fold)
    model = make_model(name, device)
    lr = LRS[name] * (1 if candidate == 0 else 3)
    if name == "GraphIDS":
        optimizer = torch.optim.AdamW([
            {"params": model.core.encoder.parameters(), "weight_decay": .6},
            {"params": model.core.transformer.parameters(), "weight_decay": .011263928281120002},
            {"params": [*model.proto_emb.parameters(), *model.port_emb.parameters(), *model.flag_emb.parameters()], "weight_decay": .6},
        ], lr=lr)
    elif name == "DIDS-MFL":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = None if name in {"DIDS-MFL", "GraphIDS"} else torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    train_indices = split["train"]
    if name == "GraphIDS":
        train_indices = [i for i in train_indices if graphs[i].y.item() == 0]
    train_loader = loader(graphs, train_indices, shuffle=True, balanced=name != "GraphIDS")
    val_loader = loader(graphs, split["val"])
    start, best, wait, chosen = 0, -1., 0, None
    last = folder / "last.pt"
    if last.exists():
        state = torch.load(last, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        if scheduler:
            scheduler.load_state_dict(state["scheduler"])
        start, best, wait, chosen = state["epoch"], state["best"], state["wait"], state["chosen"]
        random.setstate(state["rng_python"])
        np.random.set_state(state["rng_numpy"])
        torch.set_rng_state(state["rng_torch"].cpu())
        if device.type == "cuda":
            torch.cuda.set_rng_state_all([x.cpu() for x in state["rng_cuda"]])
    for epoch in range(start, args.epochs):
        if wait >= args.patience and epoch >= args.min_epochs:
            break
        t0 = time.time()
        model.train()
        optimizer.zero_grad(set_to_none=True)
        losses = []
        for step, batch in enumerate(train_loader):
            batch = batch.to(device)
            if name == "GraphIDS":
                loss = model.reconstruction_loss(batch)
            elif name == "DIDS-MFL":
                logits, aux = model(batch, return_aux=True)
                loss = nn.functional.cross_entropy(logits, batch.y)
                loss = loss + .1 * (aux["orthogonality"] + aux["decorrelation"] + aux["smoothness"])
            else:
                loss = nn.functional.cross_entropy(model(batch), batch.y, label_smoothing=.1)
            if not torch.isfinite(loss):
                raise ValueError(f"Nonfinite loss {name} fold {fold}")
            group_start = (step // 4) * 4
            divisor = min(4, len(train_loader) - group_start)
            (loss / divisor).backward()
            losses.append(float(loss.detach()))
            if (step + 1) % 4 == 0 or step + 1 == len(train_loader):
                nn.utils.clip_grad_norm_(model.parameters(), 1.)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
        if scheduler:
            scheduler.step()
        y, score, _ = predict(model, val_loader, name, device)
        value, threshold = calibrate(y, score)
        if value > best + 1e-12:
            best, wait = value, 0
            chosen = {"model": name, "fold": fold, "candidate": candidate, "lr": lr,
                      "epoch": epoch + 1, "validation_macro_f1": best, "threshold": threshold,
                      "checkpoint": str(folder / "best.pt")}
            save(folder / "best.pt", {"model": model.state_dict(), "selection": chosen})
        else:
            wait += 1
        record = {"model": name, "fold": fold, "candidate": candidate, "epoch": epoch + 1,
                  "loss": float(np.mean(losses)), "val_macro_f1": value, "best_val_macro_f1": best,
                  "seconds": time.time() - t0, "wait": wait}
        with open(folder / "epochs.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
        print(json.dumps(record), flush=True)
        dump(args.output / f"status_{args.lane}.json", {**record, "state": "training", "updated": time.time()})
        save(last, {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict() if scheduler else None,
                    "epoch": epoch + 1, "best": best, "wait": wait, "chosen": chosen,
                    "rng_python": random.getstate(), "rng_numpy": np.random.get_state(),
                    "rng_torch": torch.get_rng_state(), "rng_cuda": torch.cuda.get_rng_state_all() if device.type == "cuda" else []})
    dump(folder / "complete.json", chosen)
    del model, optimizer
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return chosen


def run(args):
    torch.set_num_threads(2)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_per_process_memory_fraction(args.memory_fraction, device)
    manifest = json.loads((args.output / "manifest.json").read_text(encoding="utf-8"))
    for p, digest in manifest["hashes"].items():
        if sha(args.output / p) != digest:
            raise ValueError(f"Dataset hash changed: {p}")
    config = {"epochs": args.epochs, "patience": args.patience, "min_epochs": args.min_epochs,
              "candidates": args.candidates, "learning_rates": LRS, "candidate_multipliers": [1, 3],
              "selection": "validation macro-F1 with validation-only threshold for ALL methods",
              "seed": "20261105 + fold, identical seed for LR candidates", "batch_size": 1,
              "gradient_accumulation": 4, "binary_positive": "attack", "std_ddof": 1,
              "runner_sha256": sha(__file__), "manifest_sha256": sha(args.output / "manifest.json"),
              "source_hashes": {str(p.relative_to(ROOT)): sha(p) for p in sorted((ROOT / "src" / "stage" / "baselines").rglob("*.py"))},
              "stage_source_sha256": sha(ROOT / "src" / "stage" / "models" / "stage.py")}
    config_path = args.output / "training_config.json"
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise ValueError("Training configuration changed; use a new experiment directory")
    dump(config_path, config)
    graphs = torch.load(args.output / "graphs.pt", weights_only=False)
    splits = json.loads((args.output / "splits.json").read_text())
    # Finish all methods' fold 0 before moving to later folds for early visibility.
    for fold in args.folds:
        split = splits[fold]
        for name in args.models:
            dest = args.output / "results" / f"{name}_fold_{fold}.json"
            if dest.exists():
                continue
            candidates = [train_candidate(args, name, fold, c, graphs, split, device) for c in range(args.candidates)]
            chosen = max(candidates, key=lambda c: c["validation_macro_f1"])
            model = make_model(name, device)
            model.load_state_dict(torch.load(chosen["checkpoint"], map_location=device, weights_only=False)["model"])
            y, score, ids = predict(model, loader(graphs, split["test"]), name, device)
            pred = (score > chosen["threshold"]).astype(int)
            result = {"model": name, "fold": fold, "selection": chosen, "metrics": metrics(y, pred),
                      "predictions": [{"graph_id": i, "true": int(t), "score": float(s), "pred": int(p)}
                                      for i, t, s, p in zip(ids, y, score, pred)]}
            dump(dest, result)
            print("TEST_COMPLETE " + json.dumps({"model": name, "fold": fold, **result["metrics"]}), flush=True)
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
    dump(args.output / f"status_{args.lane}.json", {"state": "complete", "models": args.models, "folds": args.folds, "updated": time.time()})


def summarize(args):
    rows = []
    for name in MODELS:
        found = [json.loads(p.read_text()) for p in sorted((args.output / "results").glob(f"{name}_fold_*.json"))]
        row = {"Model": name, "completed_folds": len(found), "status": "complete" if len(found) == 5 else "partial"}
        for metric in METRICS:
            values = [r["metrics"][metric] for r in found]
            row[metric + "_mean"] = float(np.mean(values)) if values else None
            row[metric + "_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else None
        rows.append(row)
    dump(args.output / "summary.json", rows)
    with open(args.output / "summary.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(rows, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["prepare", "run", "summarize"])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--raw-dir", type=Path, default=Path("data/external/dapt2020/csv"))
    parser.add_argument("--models", nargs="+", choices=MODELS, default=MODELS)
    parser.add_argument("--folds", nargs="+", type=int, choices=range(5), default=list(range(5)))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--memory-fraction", type=float, default=.20)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--min-epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--candidates", type=int, choices=[1, 2], default=2)
    parser.add_argument("--lane", default="main")
    args = parser.parse_args()
    try:
        {"prepare": prepare, "run": run, "summarize": summarize}[args.command](args)
    except Exception as exc:
        if args.command == "run":
            dump(args.output / f"status_{args.lane}.json", {"state": "failed", "error": repr(exc), "updated": time.time()})
        raise


if __name__ == "__main__":
    main()
