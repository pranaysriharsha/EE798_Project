"""Step 1 check: download the 5 datasets and print basic statistics.

PyTorch Geometric is used here ONLY to download and load the raw data.
"""
from pathlib import Path

import torch
from torch_geometric.datasets import Planetoid, WebKB, WikipediaNetwork

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def load(name):
    """Return the single graph object for a dataset name."""
    if name in ("Cora", "Citeseer", "Pubmed"):
        return Planetoid(DATA_DIR / "Planetoid", name)[0]
    if name == "Texas":
        return WebKB(DATA_DIR / "WebKB", name)[0]
    if name == "Chameleon":
        # geom_gcn_preprocess=True gives the 5-class version used in heterophily papers
        return WikipediaNetwork(DATA_DIR / "Wikipedia", "chameleon", geom_gcn_preprocess=True)[0]
    raise ValueError(name)


def undirected_edges(edge_index):
    """Unique undirected edges (i < j), self-loops removed. Shape (2, m)."""
    src, dst = edge_index
    keep = src != dst                                   # drop self-loops
    lo = torch.minimum(src[keep], dst[keep])
    hi = torch.maximum(src[keep], dst[keep])
    return torch.unique(torch.stack([lo, hi]), dim=1)   # (i,j) and (j,i) become one edge


def edge_homophily(edges, y):
    """Share of edges whose two endpoints have the same label."""
    return (y[edges[0]] == y[edges[1]]).float().mean().item()


def main():
    print(f"{'Dataset':<10} {'Nodes':>7} {'Edges(PyG,dir)':>15} {'Edges(undir)':>13} "
          f"{'Features':>9} {'Classes':>8} {'Homophily':>10}")
    for name in ["Cora", "Citeseer", "Pubmed", "Texas", "Chameleon"]:
        g = load(name)
        und = undirected_edges(g.edge_index)
        print(f"{name:<10} {g.num_nodes:>7} {g.edge_index.size(1):>15} {und.size(1):>13} "
              f"{g.num_features:>9} {int(g.y.max()) + 1:>8} {edge_homophily(und, g.y):>10.3f}")


if __name__ == "__main__":
    main()
