"""Validation grid for the Linear base model. Test accuracy is never used to choose.

Commands used in this project:
python tune_linear.py                    -> full grid on Cora, seeds 0-4 (Step 2/3)
python tune_linear.py Pubmed --reduced   -> reduced grid, seeds 0-1 (Step 4 rule: Linear > 3 points below paper)
python tune_linear.py Pubmed --lrs 0.1 0.5 1.0 --wds 0 5e-5 --seeds 2 --out linear_grid_pubmed_fix.csv
                                         -> Step 4 fix: weaker regularisation for the under-fitted Pubmed Linear
"""
import argparse
import time
from itertools import product
from pathlib import Path

import pandas as pd

from data import load_dataset, random_split
from models import Linear
from train import accuracy, train_model

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
FULL = dict(lrs=[0.01, 0.1, 0.5], wds=[0.0, 5e-4, 5e-3], seeds=range(5))
REDUCED = dict(lrs=[0.1, 0.5], wds=[5e-4, 5e-3], seeds=range(2))
EPOCHS = 1000


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", nargs="?", default="Cora")
    parser.add_argument("--reduced", action="store_true")
    parser.add_argument("--lrs", type=float, nargs="+")        # explicit grid overrides the presets
    parser.add_argument("--wds", type=float, nargs="+")
    parser.add_argument("--seeds", type=int)
    parser.add_argument("--out")
    args = parser.parse_args()
    grid = dict(REDUCED if args.reduced else FULL)
    if args.lrs:
        grid.update(lrs=args.lrs, wds=args.wds, seeds=range(args.seeds))

    start, rows = time.perf_counter(), []
    for normalise in (True, False):
        X, y, _ = load_dataset(args.dataset, normalise=normalise)
        p, c = X.size(1), int(y.max()) + 1
        for lr, wd in product(grid["lrs"], grid["wds"]):
            for seed in grid["seeds"]:
                split = random_split(X.size(0), 0.6, 0.2, seed)
                test_acc, Z, _, _ = train_model(Linear(p, c), X, y, split, lr=lr, weight_decay=wd,
                                                epochs=EPOCHS, seed=seed)
                rows.append({"features": "row-norm" if normalise else "raw", "lr": lr, "wd": wd, "seed": seed,
                             "val_acc": 100 * accuracy(Z, y, split[1]), "test_acc": 100 * test_acc})
    df = pd.DataFrame(rows)
    RESULTS_DIR.mkdir(exist_ok=True)
    df.to_csv(RESULTS_DIR / (args.out or f"linear_grid_{args.dataset.lower()}.csv"), index=False)

    summary = df.groupby(["features", "lr", "wd"]).agg(
        val=("val_acc", "mean"), test=("test_acc", "mean"), test_std=("test_acc", "std"))
    summary = summary.sort_values("val", ascending=False)       # rank by validation only
    print(summary.round(2).to_string())
    best = summary.index[0]
    print(f"\n{args.dataset}: chosen by validation: features={best[0]}, lr={best[1]}, wd={best[2]} -> "
          f"test {summary.iloc[0].test:.2f} +- {summary.iloc[0].test_std:.2f}")
    print(f"Grid time: {time.perf_counter() - start:.0f} s")


if __name__ == "__main__":
    main()
