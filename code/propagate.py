"""Label propagation, Correct, Smooth (Huang et al., ICLR 2021, Sec. 2.2-2.3), in plain PyTorch.

All three steps use the same iteration: M(t+1) = (1 - alpha) * M0 + alpha * S @ M(t).
"""
import torch
import torch.nn.functional as F


def propagate(S, M0, alpha, iters=50):
    """Spread the rows of M0 over the graph. alpha near 1: trust neighbours; near 0: keep M0."""
    M = M0
    for _ in range(iters):
        M = (1 - alpha) * M0 + alpha * torch.sparse.mm(S, M)    # restart term + neighbour average
    return M


def one_hot_on(y, idx, n, c):
    """(n, c) matrix: one-hot label rows for nodes in idx, zero rows for everyone else."""
    Y = torch.zeros(n, c)
    Y[idx] = F.one_hot(y[idx], c).float()
    return Y


def label_propagation(S, y, train_idx, c, alpha):
    """Zhou et al. (2004): propagate the training labels only, ignoring features.

    Returns an (n, c) score matrix; the predicted class of node i is its row's argmax.
    """
    return propagate(S, one_hot_on(y, train_idx, S.size(0), c), alpha)


def error_matrix(Z, y, train_idx):
    """Eq. 1: E = Y - Z on training rows, exactly zero on validation and test rows."""
    E = torch.zeros_like(Z)
    E[train_idx] = F.one_hot(y[train_idx], Z.size(1)).float() - Z[train_idx]
    return E


def autoscale_factor(E, E_hat, train_idx, max_scale=None):
    """Per-node Autoscale factor sigma / ||e_hat_i||_1, shape (n, 1).

    max_scale (E7 safeguard, not in the paper): cap the factor so weak propagated errors
    are not blown up. None keeps the paper's behaviour.
    """
    sigma = E[train_idx].abs().sum(dim=1).mean()               # average L1 error size on training nodes
    norms = E_hat.abs().sum(dim=1, keepdim=True)                # L1 size of each propagated error row
    scale = torch.where(norms > 0, sigma / norms.clamp(min=1e-12), torch.zeros_like(norms))  # 0 if no error arrived
    return scale if max_scale is None else scale.clamp(max=max_scale)


def correct(S, Z, y, train_idx, alpha, max_scale=None):
    """Correct step with Autoscale: add propagated training errors to the base predictions Z."""
    E = error_matrix(Z, y, train_idx)
    E_hat = propagate(S, E, alpha)
    scale = autoscale_factor(E, E_hat, train_idx, max_scale)

    non_train = torch.ones(Z.size(0), dtype=torch.bool)
    non_train[train_idx] = False
    Z_r = Z.clone()
    Z_r[non_train] = Z[non_train] + scale[non_train] * E_hat[non_train]   # rescale every error row to size sigma
    return Z_r


def smooth(S, Z_r, y, train_idx, alpha):
    """Smooth step, Eq. 4: true labels on training nodes, corrected scores elsewhere, then propagate."""
    H = Z_r.clone()
    H[train_idx] = F.one_hot(y[train_idx], Z_r.size(1)).float()
    return propagate(S, H, alpha)


def correct_and_smooth(S, Z, y, train_idx, alpha_correct, alpha_smooth, max_scale=None):
    """Full C&S. The two alphas are separate hyperparameters."""
    return smooth(S, correct(S, Z, y, train_idx, alpha_correct, max_scale), y, train_idx, alpha_smooth)
