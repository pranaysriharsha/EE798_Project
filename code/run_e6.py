"""E6: per-node analysis of MLP vs MLP + S vs MLP + C&S, grouped by degree and by local homophily.

Cora, Citeseer, Texas, Chameleon; 60/20/20; seeds 0-4. Same models and alpha grid as run_main.py.
Local homophily of node i = share of its neighbours with the same TRUE label (an analysis quantity only).
"""
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from data import build_S, load_dataset, random_split
from models import MLP
from run_main import post_process
from train import train_model

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
DATASETS = ["Cora", "Citeseer", "Texas", "Chameleon"]
SEEDS = range(5)
DEGREE_GROUPS = ["0", "1", "2-3", "4-7", "8+"]
HOMOPHILY_GROUPS = ["isolated", "0-0.25", "0.25-0.5", "0.5-0.75", "0.75-1"]


def node_properties(edge_index, y, n):
    """Degree and local homophily of every node (local homophily is NaN for isolated nodes)."""
    src, dst = edge_index
    degree = torch.bincount(src, minlength=n)
    same = torch.zeros(n).scatter_add_(0, src, (y[src] == y[dst]).float())   # same-label neighbours
    local_h = same / degree.clamp(min=1)
    local_h[degree == 0] = float("nan")
    return degree.numpy(), local_h.numpy()


def run(name):
    """Per-node test predictions of MLP, MLP + S, MLP + C&S for every seed."""
    X, y, edge_index = load_dataset(name)
    n, p, c = X.size(0), X.size(1), int(y.max()) + 1
    S = build_S(edge_index, n).to_sparse_csr()
    degree, local_h = node_properties(edge_index, y, n)
    frames = []
    for seed in SEEDS:
        tr, va, te = split = random_split(n, 0.6, 0.2, seed)
        _, Z, t, k = train_model(MLP(p, c), X, y, split, seed=seed)
        out = {r["method"]: r for r in post_process(S, Z, y, tr, va, "MLP", t, k)}
        frames.append(pd.DataFrame({
            "dataset": name, "seed": seed, "node": te.numpy(), "degree": degree[te], "local_h": local_h[te],
            "y": y[te].numpy(), "pred_mlp": out["MLP"]["scores"][te].argmax(1).numpy(),
            "pred_s": out["MLP + S"]["scores"][te].argmax(1).numpy(),
            "pred_cs": out["MLP + C&S"]["scores"][te].argmax(1).numpy()}))
    return pd.concat(frames)


def add_groups(nodes):
    """Degree group and local-homophily group of every test node."""
    nodes["degree_group"] = pd.cut(nodes.degree, [-1, 0, 1, 3, 7, np.inf], labels=DEGREE_GROUPS)
    h_group = pd.cut(nodes.local_h, [0, 0.25, 0.5, 0.75, 1.0001], right=False, labels=HOMOPHILY_GROUPS[1:])
    nodes["homophily_group"] = h_group.cat.add_categories("isolated").fillna("isolated")
    return nodes


def summarise_groups(nodes, column, grouping):
    """Per (dataset, group): node count, accuracies, and nodes fixed / broken by C&S relative to the MLP."""
    ok_mlp, ok_cs = nodes.pred_mlp == nodes.y, nodes.pred_cs == nodes.y
    flags = nodes.assign(mlp_ok=ok_mlp, s_ok=nodes.pred_s == nodes.y, cs_ok=ok_cs,
                         fixed=~ok_mlp & ok_cs, broken=ok_mlp & ~ok_cs)
    out = flags.groupby(["dataset", column], observed=True).agg(
        n_nodes=("y", "size"), mlp_acc=("mlp_ok", "mean"), s_acc=("s_ok", "mean"), cs_acc=("cs_ok", "mean"),
        fixed=("fixed", "sum"), broken=("broken", "sum")).reset_index().rename(columns={column: "group"})
    out[["mlp_acc", "s_acc", "cs_acc"]] *= 100
    return out.assign(grouping=grouping)


def main():
    start = time.perf_counter()
    nodes = []
    for name in DATASETS:
        nodes.append(run(name))
        print(f"{name} done (total {time.perf_counter() - start:.0f} s)", flush=True)
    nodes = add_groups(pd.concat(nodes, ignore_index=True))
    nodes.to_csv(RESULTS_DIR / "e6_nodes.csv", index=False)

    summary = pd.concat([summarise_groups(nodes, "degree_group", "degree"),
                         summarise_groups(nodes, "homophily_group", "local_homophily")])
    summary = summary[["grouping", "dataset", "group", "n_nodes", "mlp_acc", "s_acc", "cs_acc", "fixed", "broken"]]
    summary.round(2).to_csv(RESULTS_DIR / "e6_summary.csv", index=False)

    print("\nE6: test nodes pooled over 5 splits; accuracy in %; fixed/broken = C&S vs MLP")
    for grouping in ["degree", "local_homophily"]:
        print(f"\nBy {grouping}")
        print(summary[summary.grouping == grouping].drop(columns="grouping").round(1).to_string(index=False))
    print(f"\nTotal time {time.perf_counter() - start:.0f} s")


if __name__ == "__main__":
    main()
