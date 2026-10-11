"""Q: qualitative results for one split (seed 0) of Cora and Texas.

For the test nodes: t-SNE embeddings of (a) the MLP output Z, (b) Z after Correct, (c) after Smooth,
using the alphas that MLP + C&S chose on validation for this split; plus each node's outcome
(stayed right / fixed / broken / stayed wrong, MLP vs MLP + C&S). Writes results/q_points.csv.
"""
from pathlib import Path

import pandas as pd
import torch
from sklearn.manifold import TSNE

from data import build_S, load_dataset, random_split
from models import MLP
from propagate import correct, smooth
from run_main import post_process
from train import accuracy, train_model

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
PERPLEXITY = {"Cora": 30, "Texas": 10}                          # Texas has only 38 test nodes
SEED = 0
STAGES = ["Z", "after Correct", "after Smooth"]


def stages(name):
    """MLP output and the two C&S stages for seed 0, restricted to test nodes."""
    X, y, edge_index = load_dataset(name)
    n, p, c = X.size(0), X.size(1), int(y.max()) + 1
    S = build_S(edge_index, n).to_sparse_csr()
    tr, va, te = split = random_split(n, 0.6, 0.2, SEED)
    _, Z, t, k = train_model(MLP(p, c), X, y, split, seed=SEED)
    cs = {r["method"]: r for r in post_process(S, Z, y, tr, va, "MLP", t, k)}["MLP + C&S"]
    Z_r = correct(S, Z, y, tr, cs["alpha_c"])
    Y_hat = smooth(S, Z_r, y, tr, cs["alpha_s"])
    assert torch.equal(Y_hat.argmax(1), cs["scores"].argmax(1))  # same pipeline as the reported C&S
    print(f"{name} seed {SEED}: alpha_c {cs['alpha_c']}, alpha_s {cs['alpha_s']}; test acc "
          + ", ".join(f"{s} {100 * accuracy(M, y, te):.2f}" for s, M in zip(STAGES, (Z, Z_r, Y_hat))))
    return y[te], te, dict(zip(STAGES, (Z[te], Z_r[te], Y_hat[te])))


def outcome(right_before, right_after):
    """Label each node by how C&S changed it relative to the MLP."""
    return [("stayed right" if a else "broken") if b else ("fixed" if a else "stayed wrong")
            for b, a in zip(right_before.tolist(), right_after.tolist())]


def main():
    frames = []
    for name, perplexity in PERPLEXITY.items():
        y_te, te, mats = stages(name)
        right_mlp = mats["Z"].argmax(1) == y_te
        right_cs = mats["after Smooth"].argmax(1) == y_te
        for stage, M in mats.items():
            # same t-SNE settings and the same fixed random seed for every panel
            xy = TSNE(n_components=2, perplexity=perplexity, init="pca", random_state=0).fit_transform(M.numpy())
            frames.append(pd.DataFrame({"dataset": name, "stage": stage, "node": te.numpy(), "label": y_te.numpy(),
                                        "pred": M.argmax(1).numpy(), "x": xy[:, 0], "y": xy[:, 1],
                                        "outcome": outcome(right_mlp, right_cs)}))
        print(f"{name} done", flush=True)
    pd.concat(frames).to_csv(RESULTS_DIR / "q_points.csv", index=False)
    print(pd.concat(frames).query("stage == 'Z'").groupby(["dataset", "outcome"]).size().to_string())


if __name__ == "__main__":
    main()
