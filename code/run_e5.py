"""E5: three new application datasets (Amazon Photo, Amazon Computers, LastFM Asia), 60/20/20.

Same models, settings and alpha grid as run_main.py; no new tuning.
Usage: python run_e5.py [--seeds 5] [--time-one] [--gcn-epochs 1000]
--time-one prints dataset statistics and times one split per dataset to estimate the total.
--gcn-epochs N re-runs ONLY the GCN with N epochs and merges it into the E5 CSVs; the old GCN rows
  are kept, relabelled "GCN (300 epochs)" (fair-GCN re-run: the 300-epoch GCN under-fitted Computers).
"""
import argparse
import time
from pathlib import Path

import pandas as pd
import psutil
import torch

from data import edge_homophily, load_dataset
from run_main import run_dataset, summarise

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
NEW = ["Photo", "Computers", "LastFMAsia"]
PUBLISHED_SIZE = {"Photo": (7650, 745), "Computers": (13752, 767), "LastFMAsia": (7624, 128)}  # (nodes, features)
MEMORY_FACTOR = 10        # rough allowance: dense X plus normalised copy, gradients and activations
GROUPS = ["LP", "MLP", "GCN"]                                   # the MLP group also yields MLP + S and MLP + C&S
REPORT = ["MLP", "LP", "GCN (300 epochs)", "GCN", "MLP + S", "MLP + C&S"]


def fits_in_memory(name):
    """Check, BEFORE loading, that the dense feature matrix fits comfortably in free RAM."""
    n, p = PUBLISHED_SIZE[name]
    need_gb = MEMORY_FACTOR * n * p * 4 / 1e9                     # float32 = 4 bytes
    free_gb = psutil.virtual_memory().available / 1e9
    print(f"{name}: dense X {n} x {p} -> about {need_gb:.2f} GB needed (x{MEMORY_FACTOR}), {free_gb:.1f} GB free")
    return need_gb < free_gb


def statistics(names):
    """Nodes, undirected edges, features, classes, edge homophily; also merged into results/homophily.csv."""
    rows = []
    for name in names:
        X, y, edge_index = load_dataset(name, normalise=False)
        degree = torch.bincount(edge_index[0], minlength=X.size(0))
        rows.append(dict(dataset=name, nodes=X.size(0), edges=edge_index.size(1) // 2, features=X.size(1),
                         classes=int(y.max()) + 1, edge_homophily=edge_homophily(edge_index, y),
                         isolated=int((degree == 0).sum()), feature_min=X.min().item(), feature_max=X.max().item(),
                         zero_feature_rows=int((X.abs().sum(1) == 0).sum())))
    stats = pd.DataFrame(rows)
    print(stats.round(3).to_string(index=False))
    h = pd.read_csv(RESULTS_DIR / "homophily.csv", index_col="dataset").edge_homophily
    for r in rows:
        h[r["dataset"]] = r["edge_homophily"]
    h.to_csv(RESULTS_DIR / "homophily.csv")
    return stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--time-one", action="store_true")
    parser.add_argument("--gcn-epochs", type=int)
    args = parser.parse_args()

    names = [d for d in NEW if fits_in_memory(d)]
    for d in sorted(set(NEW) - set(names)):
        print(f"SKIPPED {d}: too large for this machine")
    for d in list(names):
        try:
            load_dataset(d)
        except OSError as err:                                  # e.g. the download host no longer exists
            print(f"SKIPPED {d}: could not be downloaded ({err})")
            names.remove(d)
    statistics(names)

    if args.time_one:
        times = {}
        for d in names:
            start = time.perf_counter()
            run_dataset(d, [0], GROUPS)
            times[d] = time.perf_counter() - start
        print(f"One split: { {d: round(t) for d, t in times.items()} } s; estimated total for 5 seeds: "
              f"{5 * sum(times.values()) / 60:.1f} min, for 3 seeds: {3 * sum(times.values()) / 60:.1f} min")
        return

    start, rows = time.perf_counter(), []
    groups = ["GCN"] if args.gcn_epochs else GROUPS
    for d in names:
        rows += run_dataset(d, range(args.seeds), groups, gcn_epochs=args.gcn_epochs or 300)
        print(f"{d} done (total {time.perf_counter() - start:.0f} s)", flush=True)
    df = pd.DataFrame(rows)
    if args.gcn_epochs:                                         # merge: keep old GCN rows under a new name
        old = pd.read_csv(RESULTS_DIR / "e5_runs.csv")
        if (old.method == "GCN (300 epochs)").any():           # already merged once: replace the long-run rows
            old = old[old.method != "GCN"]
        else:
            old.loc[old.method == "GCN", "method"] = "GCN (300 epochs)"
        df = pd.concat([old, df], ignore_index=True)
        gcn = df[df.method.str.startswith("GCN")].groupby(["dataset", "method"])
        print(gcn[["base_train_acc", "test_acc", "train_time_s"]].mean().round(2).to_string())
    df.to_csv(RESULTS_DIR / "e5_runs.csv", index=False)
    summary = summarise(df).reset_index()
    summary[summary.method.isin(REPORT)].to_csv(RESULTS_DIR / "e5_summary.csv", index=False)

    h = pd.read_csv(RESULTS_DIR / "homophily.csv", index_col="dataset").edge_homophily
    print(f"\nE5: test accuracy (mean +- std over {args.seeds} splits)")
    print(f"{'Method':<11}" + "".join(f"{d + f' (h={h[d]:.2f})':>24}" for d in names))
    for m in [m for m in REPORT if m in set(df.method)]:
        cells = [df[(df.dataset == d) & (df.method == m)].test_acc for d in names]
        print(f"{m:<11}" + "".join(f"{c.mean():>16.2f} +- {c.std():4.2f}" for c in cells))
    print(f"\nTotal time {time.perf_counter() - start:.0f} s")


if __name__ == "__main__":
    main()
