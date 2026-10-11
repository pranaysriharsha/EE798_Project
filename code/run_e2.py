"""E2: failure case on low-homophily graphs (Texas, Chameleon), 60/20/20, seeds 0-9.

Same models, settings and alpha grid as run_main.py; no dataset-specific tuning.
Also sweeps the Smooth strength alpha_smooth for MLP + S on all five datasets.
"""
import time
from pathlib import Path

import pandas as pd

from data import build_S, edge_homophily, load_dataset, random_split
from models import MLP
from propagate import smooth
from run_main import run_dataset, summarise
from train import accuracy, train_model

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
LOW_HOMOPHILY = ["Texas", "Chameleon"]
ALL = ["Cora", "Citeseer", "Pubmed", "Texas", "Chameleon"]
REPORT = ["Linear", "MLP", "LP", "GCN", "MLP + S", "MLP + C&S", "Linear + C&S"]
SWEEP = [0.0, 0.1, 0.3, 0.5, 0.7, 0.8, 0.9, 0.99]               # 0.0 = no smoothing, i.e. the plain MLP
SEEDS = range(10)


def smooth_sweep(name):
    """MLP + S test accuracy at each fixed alpha_smooth (no validation choice), one row per (seed, alpha)."""
    X, y, edge_index = load_dataset(name)
    n, p, c = X.size(0), X.size(1), int(y.max()) + 1
    S = build_S(edge_index, n).to_sparse_csr()
    rows = []
    for seed in SEEDS:
        tr, va, te = split = random_split(n, 0.6, 0.2, seed)
        _, Z, _, _ = train_model(MLP(p, c), X, y, split, seed=seed)
        for a in SWEEP:
            Y_hat = smooth(S, Z, y, tr, a)
            rows.append(dict(dataset=name, seed=seed, alpha_s=a, test_acc=100 * accuracy(Y_hat, y, te),
                             val_acc=100 * accuracy(Y_hat, y, va)))
    return rows


def main():
    start = time.perf_counter()
    homophily = {}
    for name in ALL:
        _, y, edge_index = load_dataset(name)
        homophily[name] = edge_homophily(edge_index, y)
    pd.Series(homophily, name="edge_homophily").rename_axis("dataset").to_csv(RESULTS_DIR / "homophily.csv")

    rows = []
    for name in LOW_HOMOPHILY:
        rows += run_dataset(name, SEEDS)
        print(f"{name} done (total {time.perf_counter() - start:.0f} s)", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / "e2_runs.csv", index=False)
    summary = summarise(df).reset_index()
    summary = summary[summary.method.isin(REPORT)]
    summary["edge_homophily"] = summary.dataset.map(homophily)
    summary.to_csv(RESULTS_DIR / "e2_summary.csv", index=False)

    sweep = []
    for name in ALL:
        sweep += smooth_sweep(name)
        print(f"sweep {name} done (total {time.perf_counter() - start:.0f} s)", flush=True)
    sweep = pd.DataFrame(sweep)
    sweep.to_csv(RESULTS_DIR / "e2_smooth_sweep.csv", index=False)

    print("\nE2: test accuracy (mean +- std over 10 splits)")
    print(f"{'Method':<14}" + "".join(f"{d + f' (h={homophily[d]:.2f})':>22}" for d in LOW_HOMOPHILY))
    for m in REPORT:
        cells = [df[(df.dataset == d) & (df.method == m)].test_acc for d in LOW_HOMOPHILY]
        print(f"{m:<14}" + "".join(f"{c.mean():>14.2f} +- {c.std():4.2f}" for c in cells))

    print("\nMLP + S test accuracy vs fixed alpha_smooth (alpha 0 = plain MLP)")
    table = sweep.groupby(["dataset", "alpha_s"]).test_acc.mean().unstack().loc[ALL]
    print(table.round(2).to_string())
    print(f"\nTotal time {time.perf_counter() - start:.0f} s")


if __name__ == "__main__":
    main()
