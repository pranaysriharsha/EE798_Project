"""E8: wrong training labels. Cora and Citeseer, 60/20/20, seeds 0-2.

A fraction p of the TRAINING labels is flipped to a different class chosen uniformly at random
(data.flip_labels). Simplification: validation and test labels stay clean, so model selection and
alpha selection still see correct labels. Same models, settings and alpha grid as run_main.py.
"""
import time
from pathlib import Path

import pandas as pd

from run_main import run_dataset

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
DATASETS = ["Cora", "Citeseer"]
NOISE = [0.0, 0.1, 0.2, 0.3, 0.4]
SEEDS = range(3)
GROUPS = ["LP", "MLP", "GCN"]                                   # the MLP group also yields MLP + S and MLP + C&S
REPORT = ["MLP", "LP", "GCN", "MLP + S", "MLP + C&S"]


def main():
    start, rows = time.perf_counter(), []
    for d in DATASETS:
        for p in NOISE:
            rows += run_dataset(d, SEEDS, GROUPS, label_noise=p)
            print(f"{d} p={p:.1f} done (total {time.perf_counter() - start:.0f} s)", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / "e8_runs.csv", index=False)
    summary = df[df.method.isin(REPORT)].groupby(["dataset", "label_noise", "method"]).agg(
        test_mean=("test_acc", "mean"), test_std=("test_acc", "std"), val_mean=("val_acc", "mean"),
        base_train_acc=("base_train_acc", "mean"), sigma=("sigma", "mean"))
    summary.round(4).to_csv(RESULTS_DIR / "e8_summary.csv")

    print(f"\nE8: clean-test accuracy (mean +- std over {len(SEEDS)} splits) vs share p of flipped training labels")
    for d in DATASETS:
        print(f"\n{d}\n{'Method':<11}" + "".join(f"{f'p={p:.1f}':>15}" for p in NOISE) + f"{'drop 0->0.4':>13}")
        drops = {}
        for m in REPORT:
            g = df[(df.dataset == d) & (df.method == m)].groupby("label_noise").test_acc
            mean, std = g.mean(), g.std()
            drops[m] = mean[0.0] - mean[0.4]
            print(f"{m:<11}" + "".join(f"{mu:>8.2f} +- {sd:4.2f}" for mu, sd in zip(mean, std)) + f"{drops[m]:>13.2f}")
        worst = max(drops, key=drops.get)
        print(f"Largest loss on {d}: {worst} ({drops[worst]:.2f} points)")
    print(f"\nTotal time {time.perf_counter() - start:.0f} s")


if __name__ == "__main__":
    main()
