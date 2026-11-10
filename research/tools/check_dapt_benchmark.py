"""Integration checks using real prepared graphs, without changing experiment weights."""
import argparse
import json
import time
import numpy as np
import pandas as pd
import torch
from dapt_benchmark import MODELS, make_model, loader, metrics, calibrate, dump, sha


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=__import__("pathlib").Path, required=True)
    p.add_argument("--device", default="cuda:3")
    args = p.parse_args()
    torch.set_num_threads(2)
    device = torch.device(args.device)
    torch.cuda.set_per_process_memory_fraction(.20, device)
    manifest = json.loads((args.output / "manifest.json").read_text())
    for filename, expected in manifest["hashes"].items():
        assert sha(args.output / filename) == expected
    graphs = torch.load(args.output / "graphs.pt", weights_only=False)
    splits = json.loads((args.output / "splits.json").read_text())
    all_test = []
    audit = pd.read_csv(args.output / "raw_assignment.csv")
    assert not audit._fingerprint.duplicated().any()
    for day, cuts in manifest["boundaries_unix_seconds"].items():
        rows = audit[audit._day == day]
        for cut in cuts:
            assert not ((rows._time <= cut + 1) & (rows._end >= cut - 1)).any()
    for s in splits:
        train, val, test = [set(s[k]) for k in ["train", "val", "test"]]
        assert not (train & val or train & test or val & test)
        assert train | val | test == set(range(len(graphs)))
        all_test.extend(test)
        for key, ids in [("train", train), ("val", val), ("test", test)]:
            assert len({int(graphs[i].y.item()) for i in ids}) == 2
            raw_ids = set(audit[audit._block.isin({int(graphs[i].block_id) for i in ids})]._fingerprint)
            assert raw_ids
    assert sorted(all_test) == list(range(len(graphs)))
    for graph in graphs:
        assert 5 <= graph.num_nodes <= 3000
        assert torch.isfinite(graph.x).all() and torch.isfinite(graph.edge_attr).all()
        assert int(graph.edge_index.max()) < graph.num_nodes
    reference = metrics([0, 0, 1, 1], [0, 1, 0, 1])
    for key in ["Accuracy", "Precision", "Recall", "F1_Macro", "FPR"]:
        assert reference[key] == .5
    assert reference["MCC"] == 0
    f1, threshold = calibrate(np.array([0, 0, 1, 1]), np.array([.1, .2, .8, .9]))
    assert f1 == 1 and .2 <= threshold < .8
    samples = sorted({max(range(len(graphs)), key=lambda i: graphs[i].num_nodes),
                      max(range(len(graphs)), key=lambda i: graphs[i].edge_index.shape[1])})
    results = []
    for name in MODELS:
        torch.cuda.reset_peak_memory_stats(device)
        model = make_model(name, device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        start = time.time()
        for batch in loader(graphs, samples):
            batch = batch.to(device)
            model.train()
            optimizer.zero_grad()
            if name == "GraphIDS":
                loss = model.reconstruction_loss(batch)
            elif name == "DIDS-MFL":
                logits, aux = model(batch, return_aux=True)
                loss = torch.nn.functional.cross_entropy(logits, batch.y) + .1 * sum(aux.values())
            else:
                loss = torch.nn.functional.cross_entropy(model(batch), batch.y)
            assert torch.isfinite(loss)
            loss.backward()
            emb = model.spatial_encoder.port_emb if name == "STAGE" else model.port_emb
            assert emb.weight.grad is not None and torch.isfinite(emb.weight.grad).all()
            assert emb.weight.grad.abs().sum() > 0
            optimizer.step()
            model.eval()
            with torch.no_grad():
                score = model.anomaly_score(batch) if name == "GraphIDS" else model(batch)
                assert torch.isfinite(score).all()
                assert score.shape == ((1,) if name == "GraphIDS" else (1, 2))
        torch.cuda.synchronize(device)
        record = {"model": name, "state": "passed", "graphs": samples, "seconds": time.time() - start,
                  "peak_allocated_mib": torch.cuda.max_memory_allocated(device) / 1024**2}
        results.append(record)
        print(json.dumps(record), flush=True)
        del model, optimizer, batch, loss, score, emb
        if name == "DIDS-MFL":
            del logits, aux
        torch.cuda.empty_cache()
    dump(args.output / "verification.json", {"split_audit": "passed", "metric_test": "passed", "models": results})


if __name__ == "__main__":
    main()
