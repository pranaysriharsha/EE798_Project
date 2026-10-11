"""E1: label scarcity on Cora and Citeseer. Training share 1-60%, validation fixed at 20%, rest test.

Usage: python run_e1.py [--seeds 5] [--time-one]
--time-one runs a single split per dataset (1% share) to estimate the total time.
"""
import argparse
import time
from pathlib import Path

import pandas as pd

from run_main import run_dataset

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
DATASETS = ["Cora", "Citeseer"]
SHARES = [0.01, 0.02, 0.05, 0.10, 0.30, 0.60]
REPORT = ["MLP", "LP", "GCN", "MLP + C&S"]
GROUPS = ["LP", "MLP", "GCN"]                                   # MLP group also yields MLP + C&S


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--time-one", action="store_true")
    args = parser.parse_args()

    if args.time_one:
        start = time.perf_counter()
        for d in DATASETS:
            run_dataset(d, [0], GROUPS, train_frac=0.01)
        per_seed = (time.perf_counter() - start) * len(SHARES)    # one split of each dataset, times 6 shares
        print(f"One split per dataset: {per_seed / len(SHARES):.0f} s. "
              f"Estimated total for {args.seeds} seeds: {args.seeds * per_seed / 60:.1f} min")
        return

    start, rows = time.perf_counter(), []
    for d in DATASETS:
        for share in SHARES:
            rows += run_dataset(d, range(args.seeds), GROUPS, train_frac=share)
            print(f"{d} {share:.0%} done (total {time.perf_counter() - start:.0f} s)", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / "e1_runs.csv", index=False)
    summary = df[df.method.isin(REPORT)].groupby(["dataset", "train_frac", "method"]).agg(
        test_mean=("test_acc", "mean"), test_std=("test_acc", "std"), val_mean=("val_acc", "mean"),
        train_time_s=("train_time_s", "mean"), prop_time_s=("prop_time_s", "mean"))
    summary.round(4).to_csv(RESULTS_DIR / "e1_summary.csv")

    print(f"\nE1: test accuracy (mean +- std over {args.seeds} splits)")
    for d in DATASETS:
        print(f"\n{d}\n{'Method':<11}" + "".join(f"{s:>15.0%}" for s in SHARES))
        for m in REPORT:
            g = df[(df.dataset == d) & (df.method == m)].groupby("train_frac").test_acc
            print(f"{m:<11}" + "".join(f"{mu:>8.2f} +- {sd:4.2f}" for mu, sd in zip(g.mean(), g.std())))
    print(f"\nTotal time {time.perf_counter() - start:.0f} s")


if __name__ == "__main__":
    main()
