"""Step 2 check: Linear, MLP and GCN on Cora, 60/20/20 splits, seeds 0-4."""
from pathlib import Path

import pandas as pd

from data import build_S_tilde, load_dataset, random_split
from models import GCN, MLP, Linear
from train import train_model

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
PAPER = {"Linear": 73.85, "MLP": 74.21, "GCN": 85.77}           # paper Tables 2 and 6, Cora
SEEDS = range(5)


def main():
    X, y, edge_index = load_dataset("Cora")
    n, p, c = X.size(0), X.size(1), int(y.max()) + 1
    S_tilde = build_S_tilde(edge_index, n)
    models = {"Linear": (Linear(p, c), None), "MLP": (MLP(p, c), None), "GCN": (GCN(p, c), S_tilde)}

    rows = []
    for name, (model, adj) in models.items():
        for seed in SEEDS:
            split = random_split(n, 0.6, 0.2, seed)
            acc, _, t, n_params = train_model(model, X, y, split, adj=adj, seed=seed)
            rows.append({"method": name, "seed": seed, "test_acc": 100 * acc, "time_s": t, "params": n_params})
    df = pd.DataFrame(rows)
    RESULTS_DIR.mkdir(exist_ok=True)
    df.to_csv(RESULTS_DIR / "step2_cora.csv", index=False)

    print(f"\nCora, 60/20/20, {len(SEEDS)} splits")
    print(f"{'Method':<8} {'Ours (%)':>15} {'Paper':>7} {'Diff':>6} {'Time (s)':>9} {'Params':>8}")
    for name, g in df.groupby("method", sort=False):
        m, s = g.test_acc.mean(), g.test_acc.std()
        print(f"{name:<8} {m:>7.2f} +- {s:<4.2f} {PAPER[name]:>7.2f} {m - PAPER[name]:>+6.2f} "
              f"{g.time_s.mean():>9.2f} {g.params.iloc[0]:>8,}")


if __name__ == "__main__":
    main()
