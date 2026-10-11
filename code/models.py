"""Base predictors (Linear, MLP) and the GCN baseline, written in plain PyTorch.

Every model has the same call signature model(X, adj). Linear and MLP ignore adj
(they never see the graph); the GCN uses adj = S~.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class Linear(nn.Module):
    """Multinomial logistic regression: one linear layer, outputs class scores (logits)."""

    def __init__(self, in_dim, n_classes):
        super().__init__()
        self.lin = nn.Linear(in_dim, n_classes)

    def forward(self, x, adj=None):
        return self.lin(x)


class MLP(nn.Module):
    """3 linear layers; each hidden layer is followed by BatchNorm, ReLU, Dropout(0.5) (paper App. A)."""

    def __init__(self, in_dim, n_classes, hidden=64, n_layers=3, dropout=0.5):
        super().__init__()
        dims = [in_dim] + [hidden] * (n_layers - 1) + [n_classes]   # e.g. [1433, 64, 64, 7]
        self.lins = nn.ModuleList(nn.Linear(a, b) for a, b in zip(dims[:-1], dims[1:]))
        self.norms = nn.ModuleList(nn.BatchNorm1d(hidden) for _ in range(n_layers - 1))
        self.dropout = dropout

    def forward(self, x, adj=None):
        for lin, norm in zip(self.lins[:-1], self.norms):
            x = F.dropout(F.relu(norm(lin(x))), self.dropout, self.training)
        return self.lins[-1](x)                                     # last layer: plain logits


def dropout_nonzero(x, p, training):
    """Dropout applied only to the non-zero entries of x (as in Kipf & Welling's code).

    Same result in distribution as F.dropout (dropping a zero changes nothing), but ~15x faster
    on bag-of-words features, where ~99% of entries are zero.
    """
    if not training:
        return x
    rows, cols = x.nonzero(as_tuple=True)
    keep = torch.rand(rows.size(0)) >= p                        # keep each non-zero with prob. 1-p
    out = torch.zeros_like(x)
    out[rows[keep], cols[keep]] = x[rows[keep], cols[keep]] / (1 - p)   # rescale survivors
    return out


class GCNLayer(nn.Module):
    """One graph convolution: H_out = S~ (H W + b)  (neighbourhood average of transformed features)."""

    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim)

    def forward(self, x, adj):
        return torch.sparse.mm(adj, self.lin(x))                    # transform first: cheaper when out_dim < in_dim


class GCN(nn.Module):
    """Plain 2-layer GCN (Kipf & Welling 2017): dropout -> GCN -> ReLU -> dropout -> GCN."""

    def __init__(self, in_dim, n_classes, hidden=64, dropout=0.5):
        super().__init__()
        self.conv1 = GCNLayer(in_dim, hidden)
        self.conv2 = GCNLayer(hidden, n_classes)
        self.dropout = dropout

    def forward(self, x, adj):
        x = dropout_nonzero(x, self.dropout, self.training)       # input dropout on sparse features
        x = F.relu(self.conv1(x, adj))
        x = F.dropout(x, self.dropout, self.training)
        return self.conv2(x, adj)
