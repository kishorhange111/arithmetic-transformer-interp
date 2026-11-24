"""Linear probes: where in the residual stream is each carry linearly decodable?

For every (layer, position) we fit a logistic-regression probe on train-set
residuals and report held-out accuracy. A probe that succeeds at a position
*before* the information is needed — and fails at layer 0 — is evidence that
the model computes the carry internally rather than reading it off the input.
Probes are correlational; activation patching tests causality.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from .data import Batch
from .model import Transformer


@torch.no_grad()
def residuals(model: Transformer, batch: Batch) -> list[torch.Tensor]:
    _, resid = model(batch.tokens[:, :-1], return_resid=True)
    return [r.detach() for r in resid]


def fit_probe(
    x_tr: torch.Tensor,
    y_tr: torch.Tensor,
    x_te: torch.Tensor,
    y_te: torch.Tensor,
    n_classes: int,
    steps: int = 300,
    lr: float = 0.05,
    wd: float = 1e-3,
) -> float:
    """Multinomial logistic regression with standardised inputs; returns test accuracy."""
    mu, sd = x_tr.mean(0, keepdim=True), x_tr.std(0, keepdim=True) + 1e-6
    x_tr, x_te = (x_tr - mu) / sd, (x_te - mu) / sd
    W = torch.zeros(x_tr.shape[1], n_classes, requires_grad=True)
    b = torch.zeros(n_classes, requires_grad=True)
    opt = torch.optim.Adam([W, b], lr=lr, weight_decay=wd)
    for _ in range(steps):
        loss = F.cross_entropy(x_tr @ W + b, y_tr)
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        return ((x_te @ W + b).argmax(1) == y_te).float().mean().item()


def probe_grid(
    model: Transformer,
    train: Batch,
    test: Batch,
    labels: dict[str, tuple[np.ndarray, np.ndarray, int]],
    positions: list[int] | None = None,
) -> dict[str, np.ndarray]:
    """labels: name -> (y_train, y_test, n_classes). Returns name -> acc[layer, pos]."""
    r_tr, r_te = residuals(model, train), residuals(model, test)
    positions = positions if positions is not None else list(range(r_tr[0].shape[1]))
    out = {}
    for name, (ytr, yte, k) in labels.items():
        grid = np.full((len(r_tr), len(positions)), np.nan)
        ytr_t, yte_t = torch.as_tensor(ytr), torch.as_tensor(yte)
        for li in range(len(r_tr)):
            for pi, p in enumerate(positions):
                grid[li, pi] = fit_probe(r_tr[li][:, p], ytr_t, r_te[li][:, p], yte_t, k)
        out[name] = grid
    return out


def majority_baseline(y: np.ndarray) -> float:
    return float(np.bincount(y).max() / len(y))
