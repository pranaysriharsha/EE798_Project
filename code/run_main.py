"""Step 4: full reproduction on Cora, Citeseer, Pubmed (60/20/20, seeds 0-9), incl. the E4 step ablation.

Usage: python run_main.py [--datasets Cora Citeseer Pubmed] [--seeds 10] [--only Linear MLP LP GCN]
With --only, just those model groups are re-run and merged into the existing results CSVs
(used once: python run_main.py --datasets Pubmed --only Linear, after re-tuning Pubmed Linear).
"""
import argparse
import time
from pathlib import Path

import pandas as pd
import torch

from data import build_S, build_S_tilde, flip_labels, load_dataset, random_split
from models import GCN, MLP, Linear
from propagate import correct, error_matrix, label_propagation, smooth
from train import accuracy, train_model

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
DATASETS = ["Cora", "Citeseer", "Pubmed"]
GROUPS = ["LP", "Linear", "MLP", "GCN"]
ALPHAS = [0.5, 0.7, 0.8, 0.9, 0.95, 0.99]
# Linear setting chosen on Cora validation (tune_linear.py), used everywhere except where overridden.
DEFAULT_LINEAR = dict(normalise=False, lr=0.5, weight_decay=5e-4, epochs=1000)
# Pubmed: Cora setting under-fitted (train acc 84%); re-chosen on validation (tune_linear.py, Step 4 fix).
LINEAR_CFG = {"Pubmed": dict(normalise=True, lr=1.0, weight_decay=0.0, epochs=1000)}
# Paper test accuracies (Tables 2, 5, 6), order: Cora, Citeseer, Pubmed
PAPER = {"Linear": (73.85, 70.27, 87.10), "MLP": (74.21, 69.34, 86.73), "LP": (86.50, 70.64, 83.74),
         "GCN": (85.77, 73.68, 88.13), "Linear + C": (79.56, 72.56, 88.56),
         "Linear + C&S": (88.62, 76.31, 89.99), "MLP + C&S": (87.56, 76.42, 89.29)}
METHODS = ["Linear", "MLP", "LP", "GCN", "Linear + C", "Linear + S", "Linear + C&S",
           "MLP + C", "MLP + S", "MLP + C&S"]


def select(candidates, y, val_idx):
    """candidates = [(alphas, scores), ...]; return the pair with best validation accuracy (first on ties)."""
    return max(candidates, key=lambda cand: accuracy(cand[1], y, val_idx))


def timed(fn):
    """Run fn() once and return its wall-clock time in seconds."""
    start = time.perf_counter()
    fn()
    return time.perf_counter() - start


def post_process(S, Z, y, tr, va, base, train_time, n_params, alphas_c=ALPHAS, alphas_s=ALPHAS, max_scale=None):
    """Base model plus its three post-processing variants (C only, S only, C&S), alphas tuned on validation.

    alphas_s and max_scale differ from the defaults only in E7.
    """
    Z_rs = {a: correct(S, Z, y, tr, a, max_scale) for a in alphas_c}   # Correct once per alpha_correct
    a_c, Z_r = select(list(Z_rs.items()), y, va)
    a_s, Y_s = select([(a, smooth(S, Z, y, tr, a)) for a in alphas_s], y, va)        # Smooth on raw Z
    (a_c2, a_s2), Y_cs = select([((ac, a), smooth(S, Z_rs[ac], y, tr, a)) for ac in alphas_c for a in alphas_s], y, va)

    sigma = error_matrix(Z, y, tr)[tr].abs().sum(dim=1).mean().item()   # Autoscale's target error size
    info = dict(train_time_s=train_time, params=n_params,       # post-processing adds no parameters
                base_train_acc=100 * accuracy(Z, y, tr), sigma=sigma)
    return [
        dict(method=base, scores=Z, **info),
        dict(method=f"{base} + C", scores=Z_r, alpha_c=a_c,
             prop_time_s=timed(lambda: correct(S, Z, y, tr, a_c, max_scale)), **info),
        dict(method=f"{base} + S", scores=Y_s, alpha_s=a_s,
             prop_time_s=timed(lambda: smooth(S, Z, y, tr, a_s)), **info),
        dict(method=f"{base} + C&S", scores=Y_cs, alpha_c=a_c2, alpha_s=a_s2,
             prop_time_s=timed(lambda: smooth(S, correct(S, Z, y, tr, a_c2, max_scale), y, tr, a_s2)), **info),
    ]


def run_dataset(name, seeds, groups=GROUPS, train_frac=0.6, label_noise=0.0, gcn_epochs=300):
    """Selected model groups on one dataset for the given seeds; returns one row per (seed, method).

    Split: train_frac training, 20% validation, the rest test.
    label_noise (E8): share of TRAINING labels flipped to a wrong class; validation/test labels stay clean.
    gcn_epochs: 300 everywhere except the fair-GCN re-run on the Amazon graphs (1000).
    """
    X, y, edge_index = load_dataset(name)
    n, p, c = X.size(0), X.size(1), int(y.max()) + 1
    lin = LINEAR_CFG.get(name, DEFAULT_LINEAR)
    X_lin = X if lin["normalise"] or "Linear" not in groups else load_dataset(name, normalise=False)[0]
    lin_kw = {k: v for k, v in lin.items() if k != "normalise"}
    S = build_S(edge_index, n).to_sparse_csr()                  # CSR: fast sparse matmul for propagation
    S_tilde = build_S_tilde(edge_index, n)
    has_neighbour = torch.bincount(edge_index[0], minlength=n) > 0

    rows = []
    for seed in seeds:
        start = time.perf_counter()
        tr, va, te = split = random_split(n, train_frac, 0.2, seed)
        y_obs = flip_labels(y, tr, label_noise, c, seed) if label_noise > 0 else y   # labels the methods see
        runs = []
        if "LP" in groups:
            a, lp_scores = select([(a, label_propagation(S, y_obs, tr, c, a)) for a in ALPHAS], y_obs, va)
            runs.append(dict(method="LP", scores=lp_scores, alpha_s=a, train_time_s=0.0, params=0,
                             prop_time_s=timed(lambda: label_propagation(S, y_obs, tr, c, a))))
        if "Linear" in groups:
            _, Z, t, k = train_model(Linear(p, c), X_lin, y_obs, split, seed=seed, **lin_kw)
            runs += post_process(S, Z, y_obs, tr, va, "Linear", t, k)
        if "MLP" in groups:
            _, Z, t, k = train_model(MLP(p, c), X, y_obs, split, seed=seed)
            runs += post_process(S, Z, y_obs, tr, va, "MLP", t, k)
        if "GCN" in groups:
            _, Z, t, k = train_model(GCN(p, c), X, y_obs, split, adj=S_tilde, seed=seed, epochs=gcn_epochs)
            runs.append(dict(method="GCN", scores=Z, train_time_s=t, params=k, base_train_acc=100 * accuracy(Z, y_obs, tr)))

        te_conn = te[has_neighbour[te]]                         # test nodes with at least one neighbour
        for r in runs:
            scores = r.pop("scores")
            rows.append(dict(dataset=name, seed=seed, train_frac=train_frac, label_noise=label_noise, **r,
                             test_acc=100 * accuracy(scores, y, te),
                             val_acc=100 * accuracy(scores, y, va),
                             test_acc_connected=100 * accuracy(scores, y, te_conn),
                             n_test=len(te), n_isolated_test=len(te) - len(te_conn)))
        accs = "  ".join(f"{r['method']} {r['test_acc']:.2f}" for r in rows[-len(runs):]
                         if r["method"] in ("LP", "GCN") or r["method"].endswith("C&S"))
        print(f"  {name} seed {seed}: {time.perf_counter() - start:5.1f} s | {accs}", flush=True)
    return rows


def summarise(df):
    """Mean +- std over splits for every (dataset, method)."""
    return df.groupby(["dataset", "method"], sort=False).agg(
        test_mean=("test_acc", "mean"), test_std=("test_acc", "std"), val_mean=("val_acc", "mean"),
        test_connected_mean=("test_acc_connected", "mean"), base_train_acc=("base_train_acc", "mean"),
        sigma=("sigma", "mean"), train_time_s=("train_time_s", "mean"),
        prop_time_s=("prop_time_s", "mean"), params=("params", "first")).round(4)


def fmt(g, col="test_acc"):
    return f"{g[col].mean():6.2f} +- {g[col].std():4.2f}"


def print_tables(df, datasets):
    by = {(d, m): g for (d, m), g in df.groupby(["dataset", "method"])}

    print("\nA. Reproduction: ours (mean +- std over splits) vs paper; '!' marks |gap| > 3 points")
    print(f"{'Method':<14}" + "".join(f"{d:>33}" for d in datasets))
    for m in ["Linear", "MLP", "LP", "GCN", "Linear + C", "Linear + C&S", "MLP + C&S"]:
        cells = []
        for d in datasets:
            ours, paper = by[d, m].test_acc.mean(), PAPER[m][DATASETS.index(d)]
            flag = "!" if abs(ours - paper) > 3 else " "
            cells.append(f"{fmt(by[d, m])} | {paper:6.2f} {ours - paper:+6.2f}{flag}")
        print(f"{m:<14}" + "".join(f"{c:>33}" for c in cells))

    print("\nB. Step ablation (E4): test accuracy, mean +- std")
    print(f"{'Method':<14}" + "".join(f"{d:>16}" for d in datasets))
    for base in ["Linear", "MLP"]:
        for m in [base, f"{base} + C", f"{base} + S", f"{base} + C&S"]:
            print(f"{m:<14}" + "".join(f"{fmt(by[d, m]):>16}" for d in datasets))

    print("\nC. Cost: parameters | base-model training time (s) | propagation time (ms), means over splits")
    print(f"{'Method':<14}" + "".join(f"{d:>28}" for d in datasets))
    for m in METHODS:
        cells = []
        for d in datasets:
            g = by[d, m]
            prop = f"{1000 * g.prop_time_s.mean():6.1f}" if g.prop_time_s.notna().any() else "     -"
            cells.append(f"{int(g.params.iloc[0]):>8,} | {g.train_time_s.mean():6.2f} | {prop}")
        print(f"{m:<14}" + "".join(f"{c:>28}" for c in cells))

    print("\nIsolated test nodes (no neighbours) and accuracy on the remaining test nodes")
    for d in datasets:
        for m in ["LP", "Linear + C&S", "MLP + C&S"]:
            g = by[d, m]
            print(f"  {d:<9} {m:<13} isolated test nodes {g.n_isolated_test.mean():4.1f} of {g.n_test.iloc[0]}"
                  f" | all test {fmt(g)} | connected only {fmt(g, 'test_acc_connected')}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=DATASETS)
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--only", nargs="+", choices=GROUPS)
    args = parser.parse_args()

    rows, start = [], time.perf_counter()
    for name in args.datasets:
        t0 = time.perf_counter()
        rows += run_dataset(name, range(args.seeds), args.only or GROUPS)
        print(f"{name} done in {time.perf_counter() - t0:.0f} s (total {time.perf_counter() - start:.0f} s)", flush=True)

    df = pd.DataFrame(rows)
    if args.only:                                               # merge: replace only the re-run rows
        old = pd.read_csv(RESULTS_DIR / "reproduction_runs.csv")
        group = old.method.str.split(" + ", regex=False).str[0]
        old = old[~(old.dataset.isin(args.datasets) & group.isin(args.only))]
        df = pd.concat([old, df], ignore_index=True)
    RESULTS_DIR.mkdir(exist_ok=True)
    df.to_csv(RESULTS_DIR / "reproduction_runs.csv", index=False)
    summarise(df).to_csv(RESULTS_DIR / "reproduction_summary.csv")
    print_tables(df, [d for d in DATASETS if d in set(df.dataset)])


if __name__ == "__main__":
    main()
