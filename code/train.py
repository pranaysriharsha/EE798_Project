"""Full-batch training with best-validation model selection."""
import copy
import random
import time

import numpy as np
import torch
import torch.nn.functional as F


def set_seed(seed):
    """Fix every random number generator we use, from one integer."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def accuracy(scores, y, idx):
    """Share of nodes in idx whose highest-scoring class equals the true label."""
    return (scores[idx].argmax(dim=1) == y[idx]).float().mean().item()


def train_model(model, X, y, split, adj=None, lr=0.01, weight_decay=0.0, epochs=300, seed=0):
    """Train on split[0] with cross-entropy; keep the weights with the best validation accuracy.

    split = (train_idx, val_idx, test_idx). adj is S~ for the GCN, None otherwise.
    Returns (test accuracy, Z = softmax output for all nodes, training time in s, parameter count).
    """
    train_idx, val_idx, test_idx = split
    set_seed(seed)
    for m in model.modules():                                   # re-initialise weights under this seed
        if m is not model and hasattr(m, "reset_parameters"):
            m.reset_parameters()
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_val, best_state = -1.0, None
    start = time.perf_counter()
    for _ in range(epochs):
        model.train()                                           # dropout on, batch norm uses batch stats
        opt.zero_grad()
        loss = F.cross_entropy(model(X, adj)[train_idx], y[train_idx])   # loss on training nodes only
        loss.backward()
        opt.step()

        model.eval()                                            # dropout off for evaluation
        with torch.no_grad():
            val_acc = accuracy(model(X, adj), y, val_idx)
        if val_acc > best_val:                                  # remember the best epoch so far
            best_val, best_state = val_acc, copy.deepcopy(model.state_dict())
    train_time = time.perf_counter() - start

    model.load_state_dict(best_state)                           # roll back to the best-validation epoch
    model.eval()
    with torch.no_grad():
        Z = F.softmax(model(X, adj), dim=1)
    n_params = sum(p.numel() for p in model.parameters())
    return accuracy(Z, y, test_idx), Z, train_time, n_params
