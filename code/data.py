"""Dataset loading, graph matrices and random splits.

PyTorch Geometric is used ONLY to download/read the raw datasets.
Everything after that (cleaning edges, building S, splits) is plain PyTorch.
"""
from pathlib import Path

import torch
from torch_geometric.datasets import Amazon, LastFMAsia, Planetoid, WebKB, WikipediaNetwork

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def load_dataset(name, normalise=True):
    """Return (X, y, edge_index) for one dataset.

    X: (n, p) features; if normalise, each row is scaled to sum to 1.
    y: (n,) integer labels.
    edge_index: (2, 2m) undirected edges, stored in both directions, no self-loops, no duplicates.
    """
    if name in ("Cora", "Citeseer", "Pubmed"):
        g = Planetoid(DATA_DIR / "Planetoid", name)[0]
    elif name == "Texas":
        g = WebKB(DATA_DIR / "WebKB", name)[0]
    elif name == "Chameleon":
        # geom_gcn_preprocess=True gives the 5-class version used in heterophily papers
        g = WikipediaNetwork(DATA_DIR / "Wikipedia", "chameleon", geom_gcn_preprocess=True)[0]
    elif name in ("Photo", "Computers"):                       # Amazon co-purchase graphs (E5)
        g = Amazon(DATA_DIR / "Amazon", name)[0]
    elif name == "LastFMAsia":                                  # LastFM Asia social network (E5)
        g = LastFMAsia(DATA_DIR / "LastFMAsia")[0]
    else:
        raise ValueError(f"unknown dataset {name}")

    X = g.x
    if normalise:
        X = X / X.sum(dim=1, keepdim=True).clamp(min=1e-12)     # row-normalise; all-zero rows stay zero
    return X, g.y, make_undirected(g.edge_index)


def make_undirected(edge_index):
    """Drop self-loops and duplicates, then store every edge as (i,j) and (j,i)."""
    src, dst = edge_index
    keep = src != dst                                           # drop self-loops
    lo = torch.minimum(src[keep], dst[keep])
    hi = torch.maximum(src[keep], dst[keep])
    pairs = torch.unique(torch.stack([lo, hi]), dim=1)          # one column per undirected edge
    return torch.cat([pairs, pairs.flip(0)], dim=1)             # add the reverse direction


def edge_homophily(edge_index, y):
    """Share of edges whose two endpoints have the same label (each direction counted; same ratio)."""
    return (y[edge_index[0]] == y[edge_index[1]]).float().mean().item()


def sym_normalise(edge_index, n):
    """Sparse D^(-1/2) A D^(-1/2) for the 0/1 adjacency A given by edge_index."""
    src, dst = edge_index
    deg = torch.zeros(n).scatter_add_(0, src, torch.ones(src.size(0)))   # degree of each node
    d_inv_sqrt = deg.pow(-0.5)
    d_inv_sqrt[deg == 0] = 0.0                                  # isolated node: row of zeros, no 1/0
    values = d_inv_sqrt[src] * d_inv_sqrt[dst]                  # entry (i,j) = 1 / sqrt(d_i d_j)
    return torch.sparse_coo_tensor(edge_index, values, (n, n)).coalesce()


def build_S(edge_index, n):
    """S = D^(-1/2) A D^(-1/2), no self-loops. Used by label propagation, Correct and Smooth."""
    return sym_normalise(edge_index, n)


def build_S_tilde(edge_index, n):
    """S~ = D~^(-1/2) (A + I) D~^(-1/2), with self-loops. Used by the GCN (Kipf & Welling)."""
    loops = torch.arange(n).repeat(2, 1)                        # (i,i) for every node
    return sym_normalise(torch.cat([edge_index, loops], dim=1), n)


def flip_labels(y, idx, p, n_classes, seed):
    """Copy of y in which a fraction p of the nodes in idx get a different class, uniformly at random (E8)."""
    gen = torch.Generator().manual_seed(seed)
    y = y.clone()
    chosen = idx[torch.randperm(len(idx), generator=gen)[:int(round(p * len(idx)))]]
    shift = torch.randint(1, n_classes, (len(chosen),), generator=gen)   # 1..c-1, so never the true class
    y[chosen] = (y[chosen] + shift) % n_classes
    return y


def random_split(n, train_frac=0.6, val_frac=0.2, seed=0):
    """Random disjoint train/val/test node indices; test gets whatever is left."""
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(seed))
    n_train, n_val = int(train_frac * n), int(val_frac * n)
    return perm[:n_train], perm[n_train:n_train + n_val], perm[n_train + n_val:]
