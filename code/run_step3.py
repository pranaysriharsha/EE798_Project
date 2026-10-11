"""Step 3 check: LP, Linear + C, Linear + C&S, MLP + C&S on Cora, 60/20/20 splits, seeds 0-4."""
import time
from pathlib import Path

import pandas as pd
import torch

from data import build_S, load_dataset, random_split
from models import MLP, Linear
from propagate import correct, error_matrix, label_propagation, smooth
from train import accuracy, train_model

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
ALPHAS = [0.5, 0.7, 0.8, 0.9, 0.95, 0.99]
SEEDS = range(5)
LINEAR_CFG = dict(normalise=False, lr=0.5, weight_decay=5e-4, epochs=1000)  # chosen on validation by tune_linear.py
PAPER = {"LP": 86.50, "Linear": 73.85, "Linear + C": 79.56, "Linear + C&S": 88.62,
         "MLP": 74.21, "MLP + C&S": 87.56}


def select(candidates, y, val_idx):
    """candidates = [(alphas, scores), ...]; return the pair with best validation accuracy (first on ties)."""
    return max(candidates, key=lambda cand: accuracy(cand[1], y, val_idx))


def check_S(S):
    """Sanity check: S is symmetric and has no self-loops."""
    asym = (S - S.t()).coalesce().values().abs().max().item()
    loops = (S.indices()[0] == S.indices()[1]).sum().item()
    print(f"[check] S: max |S - S^T| = {asym:.1e}, self-loops = {loops}")


def run_cs(S, Z, y, split, c, name, seed):
    """Tune alphas on validation for C-only and C&S on top of base predictions Z; return result rows."""
    tr, va, te = split
    Z_rs = {a: correct(S, Z, y, tr, a) for a in ALPHAS}         # Correct once per alpha_correct
    a_c, Z_r = select(list(Z_rs.items()), y, va)
    (a_c2, a_s), Y_hat = select([((ac, a), smooth(S, Z_rs[ac], y, tr, a)) for ac in ALPHAS for a in ALPHAS], y, va)

    start = time.perf_counter()                                 # time one C&S pass at the chosen alphas
    smooth(S, correct(S, Z, y, tr, a_c2), y, tr, a_s)
    t_prop = time.perf_counter() - start

    base_val, c_val = accuracy(Z, y, va), accuracy(Z_r, y, va)
    print(f"[check] {name} seed {seed}: E max |.| on val+test = {error_matrix(Z, y, tr)[torch.cat([va, te])].abs().max():.1f}; "
          f"val acc base {100 * base_val:.2f} -> after Correct {100 * c_val:.2f}" + ("  <-- LOWER" if c_val < base_val else ""))
    return [
        {"method": name, "seed": seed, "test_acc": 100 * accuracy(Z, y, te), "val_acc": 100 * base_val},
        {"method": f"{name} + C", "seed": seed, "test_acc": 100 * accuracy(Z_r, y, te), "val_acc": 100 * c_val,
         "alpha_c": a_c},
        {"method": f"{name} + C&S", "seed": seed, "test_acc": 100 * accuracy(Y_hat, y, te),
         "val_acc": 100 * accuracy(Y_hat, y, va), "alpha_c": a_c2, "alpha_s": a_s, "prop_time_s": t_prop},
    ]


def main():
    X, y, edge_index = load_dataset("Cora")
    X_lin = X if LINEAR_CFG["normalise"] else load_dataset("Cora", normalise=False)[0]
    n, p, c = X.size(0), X.size(1), int(y.max()) + 1
    S = build_S(edge_index, n)
    check_S(S)
    S = S.to_sparse_csr()                                       # CSR: ~10x faster sparse matmul on CPU

    rows = []
    for seed in SEEDS:
        split = random_split(n, 0.6, 0.2, seed)
        tr, va, te = split
        (a, scores) = select([(a, label_propagation(S, y, tr, c, a)) for a in ALPHAS], y, va)
        start = time.perf_counter()
        label_propagation(S, y, tr, c, a)
        rows.append({"method": "LP", "seed": seed, "test_acc": 100 * accuracy(scores, y, te),
                     "val_acc": 100 * accuracy(scores, y, va), "alpha_s": a, "prop_time_s": time.perf_counter() - start})

        lin_cfg = {k: v for k, v in LINEAR_CFG.items() if k != "normalise"}
        _, Z_lin, _, _ = train_model(Linear(p, c), X_lin, y, split, seed=seed, **lin_cfg)
        _, Z_mlp, _, _ = train_model(MLP(p, c), X, y, split, seed=seed)
        rows += run_cs(S, Z_lin, y, split, c, "Linear", seed)
        rows += run_cs(S, Z_mlp, y, split, c, "MLP", seed)

    df = pd.DataFrame(rows)
    RESULTS_DIR.mkdir(exist_ok=True)
    df.to_csv(RESULTS_DIR / "step3_cora.csv", index=False)

    print(f"\nCora, 60/20/20, {len(SEEDS)} splits")
    print(f"{'Method':<14} {'Ours (%)':>15} {'Paper':>7} {'Diff':>6}  {'alpha_c':<26} {'alpha_s':<26} {'prop. time (ms)':>15}")
    for name in ["LP", "Linear", "Linear + C", "Linear + C&S", "MLP", "MLP + C&S"]:
        g = df[df.method == name]
        m, s = g.test_acc.mean(), g.test_acc.std()
        ac = str(g.alpha_c.tolist()) if g.alpha_c.notna().any() else "-"
        as_ = str(g.alpha_s.tolist()) if g.alpha_s.notna().any() else "-"
        t = f"{1000 * g.prop_time_s.mean():.1f}" if g.prop_time_s.notna().any() else "-"
        print(f"{name:<14} {m:>7.2f} +- {s:<4.2f} {PAPER[name]:>7.2f} {m - PAPER[name]:>+6.2f}  {ac:<26} {as_:<26} {t:>15}")


if __name__ == "__main__":
    main()
