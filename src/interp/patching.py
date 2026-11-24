"""Activation patching (causal tracing; Meng et al., 2022; Heimersheim & Nanda, 2024).

For a *clean* prompt and a *corrupted* prompt that differs only in the units
digits so that the units carry flips, we run the corrupted prompt while
restoring the clean residual stream at one ``(layer, pos)``. The normalised
logit-difference recovery

    (LD_patched - LD_corrupt) / (LD_clean - LD_corrupt)

measures how much of the carry-dependent behaviour that single location
carries: 1 = fully restores clean behaviour, 0 = no effect.
"""

from __future__ import annotations

import numpy as np
import torch

from .data import ANSWER_POS, Batch, make_batch
from .model import Transformer


def carry_flip_pairs(n: int, rng: np.random.Generator) -> tuple[Batch, Batch]:
    """Pairs with identical tens digits whose units digits flip the units carry.

    Tens digits are drawn so that the tens column sum is 9 in both runs: then the
    units carry alone decides the hundreds digit s2 (and s1 changes too), which
    isolates the carry pathway.
    """
    a1 = rng.integers(0, 10, n)
    b1 = 9 - a1
    # clean: units carry = 1; corrupt: units carry = 0
    a0c = rng.integers(1, 10, n)
    b0c = np.array([rng.integers(10 - x, 10) for x in a0c])
    a0x = rng.integers(0, 10, n)
    b0x = np.array([rng.integers(0, 10 - x) for x in a0x])
    clean = make_batch(10 * a1 + a0c, 10 * b1 + b0c)
    corrupt = make_batch(10 * a1 + a0x, 10 * b1 + b0x)
    assert (clean.carry0 == 1).all() and (corrupt.carry0 == 0).all()
    return clean, corrupt


def logit_diff(logits: torch.Tensor, clean: Batch, corrupt: Batch, digit: int) -> torch.Tensor:
    """Mean logit(clean answer digit) - logit(corrupt answer digit) at that digit's position."""
    pos = ANSWER_POS[digit]
    lg = logits[:, pos, :]
    yc = clean.tokens[:, 7 + digit]
    yx = corrupt.tokens[:, 7 + digit]
    return (lg.gather(1, yc[:, None]) - lg.gather(1, yx[:, None])).squeeze(1).mean()


@torch.no_grad()
def patch_grid(model: Transformer, clean: Batch, corrupt: Batch, digit: int = 0) -> np.ndarray:
    """Recovery[layer, pos] for restoring the clean residual into the corrupted run."""
    _, clean_resid = model(clean.tokens[:, :-1], return_resid=True)
    ld_clean = logit_diff(model(clean.tokens[:, :-1]), clean, corrupt, digit).item()
    ld_corr = logit_diff(model(corrupt.tokens[:, :-1]), clean, corrupt, digit).item()
    denom = ld_clean - ld_corr
    n_layers, T = len(clean_resid), clean.tokens.shape[1] - 1
    grid = np.zeros((n_layers, T))
    for layer in range(n_layers):
        for p in range(T):

            def patch(layer_idx, x, layer=layer, p=p):
                if layer_idx != layer:
                    return x
                x = x.clone()
                x[:, p] = clean_resid[layer][:, p]
                return x

            ld = logit_diff(model(corrupt.tokens[:, :-1], patch=patch), clean, corrupt, digit)
            grid[layer, p] = (ld.item() - ld_corr) / denom if abs(denom) > 1e-6 else np.nan
    return grid
