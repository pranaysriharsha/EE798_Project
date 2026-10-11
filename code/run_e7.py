"""E7: two safeguards for C&S (our additions, not in the paper).

Part 1: cap the Autoscale factor (max_scale) for Linear + C&S, seeds 0-4, on
        Pubmed (under-fitted Linear), Pubmed (re-tuned Linear), Cora (standard Linear).
Part 2: allow "no smoothing" (alpha_smooth = 0) in the validation grid for MLP + S and MLP + C&S
        on Texas and Chameleon, seeds 0-9 (same splits as E2).
"""
import time
from pathlib import Path

import pandas as pd

from data import build_S, load_dataset, random_split
from models import MLP, Linear
from propagate import autoscale_factor, error_matrix, propagate
from run_main import ALPHAS, DEFAULT_LINEAR, LINEAR_CFG, post_process
from train import accuracy, train_model

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
MAX_SCALES = [None, 1, 2, 5, 10]
SETTINGS = {  # label: (dataset, Linear setting)
    "Pubmed, under-fitted Linear": ("Pubmed", dict(normalise=False, lr=0.5, weight_decay=5e-4, epochs=1000)),
    "Pubmed, re-tuned Linear": ("Pubmed", LINEAR_CFG["Pubmed"]),
    "Cora, standard Linear": ("Cora", DEFAULT_LINEAR),
}
EXTENDED_ALPHAS_S = [0.0, 0.1, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 0.99]   # 0 = no smoothing; first wins ties


def result_rows(out, y, va, te, **tags):
    """Turn post_process output into rows with test/val accuracy (scores dropped)."""
    return [dict(**tags, method=r["method"], alpha_c=r.get("alpha_c"), alpha_s=r.get("alpha_s"),
                 sigma=r["sigma"], test_acc=100 * accuracy(r["scores"], y, te),
                 val_acc=100 * accuracy(r["scores"], y, va)) for r in out]


def part1_max_scale():
    rows = []
    for label, (name, cfg) in SETTINGS.items():
        X, y, edge_index = load_dataset(name, normalise=cfg["normalise"])
        n, p, c = X.size(0), X.size(1), int(y.max()) + 1
        S = build_S(edge_index, n).to_sparse_csr()
        kw = {k: v for k, v in cfg.items() if k != "normalise"}
        for seed in range(5):
            tr, va, te = split = random_split(n, 0.6, 0.2, seed)
            _, Z, t, k = train_model(Linear(p, c), X, y, split, seed=seed, **kw)
            E = error_matrix(Z, y, tr)
            factor = autoscale_factor(E, propagate(S, E, 0.9), tr)[te].squeeze()   # uncapped, at alpha 0.9
            for ms in MAX_SCALES:
                out = [r for r in post_process(S, Z, y, tr, va, "Linear", t, k, max_scale=ms)
                       if r["method"] in ("Linear", "Linear + C", "Linear + C&S")]
                rows += result_rows(out, y, va, te, part="max_scale", setting=label, seed=seed,
                                    max_scale="none" if ms is None else str(ms),
                                    median_factor=factor.median().item(),
                                    share_factor_gt_2=100 * (factor > 2).float().mean().item())
        print(f"{label} done", flush=True)
    return rows


def part2_no_smoothing():
    rows = []
    for name in ["Texas", "Chameleon"]:
        X, y, edge_index = load_dataset(name)
        n, p, c = X.size(0), X.size(1), int(y.max()) + 1
        S = build_S(edge_index, n).to_sparse_csr()
        for seed in range(10):
            tr, va, te = split = random_split(n, 0.6, 0.2, seed)
            _, Z, t, k = train_model(MLP(p, c), X, y, split, seed=seed)
            for grid, alphas_s in [("paper grid", ALPHAS), ("with alpha 0", EXTENDED_ALPHAS_S)]:
                out = [r for r in post_process(S, Z, y, tr, va, "MLP", t, k, alphas_s=alphas_s)
                       if r["method"] in ("MLP", "MLP + S", "MLP + C&S")]
                rows += result_rows(out, y, va, te, part="no_smoothing", setting=name, seed=seed, grid=grid)
        print(f"{name} done", flush=True)
    return rows


def main():
    start = time.perf_counter()
    runs = pd.DataFrame(part1_max_scale() + part2_no_smoothing())
    runs.to_csv(RESULTS_DIR / "e7_runs.csv", index=False)

    p1 = runs[runs.part == "max_scale"].groupby(["setting", "max_scale", "method"], sort=False).agg(
        test_mean=("test_acc", "mean"), test_std=("test_acc", "std"), val_mean=("val_acc", "mean"),
        sigma=("sigma", "mean"), median_factor=("median_factor", "mean"),
        share_factor_gt_2=("share_factor_gt_2", "mean")).reset_index()
    p2 = runs[runs.part == "no_smoothing"].groupby(["setting", "grid", "method"], sort=False).agg(
        test_mean=("test_acc", "mean"), test_std=("test_acc", "std"), val_mean=("val_acc", "mean"),
        share_alpha_s_0=("alpha_s", lambda a: 100 * (a == 0).mean())).reset_index()
    pd.concat([p1.assign(part="max_scale"), p2.assign(part="no_smoothing")]).round(3).to_csv(
        RESULTS_DIR / "e7_summary.csv", index=False)

    print("\nE7 part 1: Linear + C&S (and + C) with the Autoscale factor capped at max_scale; 5 splits")
    for label in SETTINGS:
        g = p1[p1.setting == label]
        base = g[g.method == "Linear"].iloc[0]
        print(f"\n{label}: Linear {base.test_mean:.2f}, sigma {base.sigma:.3f}, uncapped factor on test nodes "
              f"(alpha 0.9): median {base.median_factor:.1f}, {base.share_factor_gt_2:.0f}% above 2")
        print(f"{'max_scale':<10} {'Linear + C':>16} {'Linear + C&S':>16}")
        for ms in ["none"] + [str(m) for m in MAX_SCALES[1:]]:
            cells = [g[(g.max_scale == ms) & (g.method == m)].iloc[0] for m in ("Linear + C", "Linear + C&S")]
            print(f"{ms:<10}" + "".join(f"{r.test_mean:>9.2f} +- {r.test_std:4.2f}" for r in cells))

    print("\nE7 part 2: alpha_smooth grid with and without 0 (= no smoothing), chosen on validation; 10 splits")
    print(f"{'Dataset':<10} {'Grid':<13} {'MLP':>14} {'MLP + S':>14} {'MLP + C&S':>14} {'S: alpha 0 chosen':>18} {'C&S: alpha 0 chosen':>20}")
    for name in ["Texas", "Chameleon"]:
        for grid in ["paper grid", "with alpha 0"]:
            g = p2[(p2.setting == name) & (p2.grid == grid)].set_index("method")
            print(f"{name:<10} {grid:<13}" + "".join(f"{g.test_mean[m]:>7.2f} +- {g.test_std[m]:4.2f}"
                                                    for m in ("MLP", "MLP + S", "MLP + C&S"))
                  + f"{g.share_alpha_s_0['MLP + S']:>17.0f}%{g.share_alpha_s_0['MLP + C&S']:>19.0f}%")
    for name in ["Texas", "Chameleon"]:
        g = p2[p2.setting == name]
        acc = {(r.grid, r.method): r.test_mean for r in g.itertuples()}
        for m in ("MLP + S", "MLP + C&S"):
            loss = acc["paper grid", "MLP"] - acc["paper grid", m]             # what smoothing cost before
            gain = acc["with alpha 0", m] - acc["paper grid", m]
            share = f"{100 * gain / loss:.0f}% of the loss recovered" if loss > 0 else "no loss to recover"
            print(f"{name} {m}: {acc['paper grid', m]:.2f} -> {acc['with alpha 0', m]:.2f} "
                  f"(MLP {acc['paper grid', 'MLP']:.2f}); {share}")
    print(f"\nTotal time {time.perf_counter() - start:.0f} s")


if __name__ == "__main__":
    main()
