"""TopK sparse autoencoder (Gao et al., 2024, "Scaling and evaluating sparse autoencoders").

``z = TopK(W_enc (x - b_pre) + b_enc)``, ``x_hat = W_dec z + b_pre`` with
unit-norm decoder columns. TopK fixes L0 = k exactly, so there is no L1
coefficient to tune and no shrinkage bias.

Evaluation follows standard practice:

* **FVU** — fraction of variance unexplained by the reconstruction
* **dead fraction** — latents that never fire on the evaluation set
* **spliced CE** — task loss when the SAE reconstruction replaces the residual
  stream, compared with the clean model and with zero-ablation
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from .data import Batch
from .model import Transformer, loss_fn


class TopKSAE(nn.Module):
    def __init__(self, d_in: int, n_latents: int, k: int):
        super().__init__()
        self.k = k
        self.b_pre = nn.Parameter(torch.zeros(d_in))
        self.W_enc = nn.Parameter(torch.randn(n_latents, d_in) / d_in**0.5)
        self.b_enc = nn.Parameter(torch.zeros(n_latents))
        self.W_dec = nn.Parameter(self.W_enc.data.T.clone())
        self.normalize_decoder()

    @torch.no_grad()
    def normalize_decoder(self) -> None:
        self.W_dec.data /= self.W_dec.data.norm(dim=0, keepdim=True) + 1e-8

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        pre = (x - self.b_pre) @ self.W_enc.T + self.b_enc
        vals, idx = pre.topk(self.k, dim=-1)
        return torch.zeros_like(pre).scatter_(-1, idx, torch.relu(vals))

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return z @ self.W_dec.T + self.b_pre

    def forward(self, x):
        z = self.encode(x)
        return self.decode(z), z


def fvu(x: torch.Tensor, x_hat: torch.Tensor) -> float:
    return ((x - x_hat).pow(2).sum() / (x - x.mean(0)).pow(2).sum()).item()


@dataclass
class SAEReport:
    fvu: float
    dead_frac: float
    ce_clean: float
    ce_spliced: float
    ce_zero_ablate: float

    @property
    def ce_recovered(self) -> float:
        """1 = splicing costs nothing, 0 = as bad as zero-ablating the stream."""
        return (self.ce_zero_ablate - self.ce_spliced) / (self.ce_zero_ablate - self.ce_clean)


def collect(model: Transformer, batch: Batch, layer: int, positions: list[int]) -> torch.Tensor:
    with torch.no_grad():
        _, resid = model(batch.tokens[:, :-1], return_resid=True)
    return resid[layer][:, positions].reshape(-1, resid[layer].shape[-1])


def train_sae(
    acts: torch.Tensor,
    n_latents: int = 256,
    k: int = 8,
    steps: int = 3000,
    bs: int = 1024,
    lr: float = 1e-3,
    seed: int = 0,
) -> TopKSAE:
    torch.manual_seed(seed)
    sae = TopKSAE(acts.shape[1], n_latents, k)
    with torch.no_grad():
        sae.b_pre.copy_(acts.median(0).values)  # geometric-median-ish init
    opt = torch.optim.Adam(sae.parameters(), lr=lr)
    g = torch.Generator().manual_seed(seed)
    for _ in range(steps):
        x = acts[torch.randint(0, len(acts), (bs,), generator=g)]
        x_hat, _ = sae(x)
        loss = (x - x_hat).pow(2).sum(-1).mean()
        opt.zero_grad()
        loss.backward()
        # Remove the gradient component parallel to each decoder column (keeps unit norm).
        with torch.no_grad():
            Wd, gd = sae.W_dec.data, sae.W_dec.grad
            gd -= (gd * Wd).sum(0, keepdim=True) * Wd
        opt.step()
        sae.normalize_decoder()
    return sae.eval()


@torch.no_grad()
def evaluate_sae(
    sae: TopKSAE, model: Transformer, batch: Batch, layer: int, positions: list[int]
) -> SAEReport:
    acts = collect(model, batch, layer, positions)
    x_hat, z = sae(acts)
    dead = ((z > 0).sum(0) == 0).float().mean().item()

    def splice(fn):
        def patch(layer_idx, x):
            if layer_idx != layer:
                return x
            x = x.clone()
            x[:, positions] = fn(x[:, positions])
            return x

        return patch

    ce_clean = loss_fn(model, batch).item()
    ce_spliced = loss_fn(model, batch, patch=splice(lambda h: sae(h)[0])).item()
    ce_zero = loss_fn(model, batch, patch=splice(torch.zeros_like)).item()
    return SAEReport(fvu(acts, x_hat), dead, ce_clean, ce_spliced, ce_zero)


@torch.no_grad()
def feature_concept_correlation(
    sae: TopKSAE,
    model: Transformer,
    batch: Batch,
    layer: int,
    position: int,
    concepts: dict[str, np.ndarray],
) -> dict[str, list[tuple[int, float]]]:
    """For each concept, latents ranked by |Pearson r| between activation and the concept."""
    z = sae.encode(collect(model, batch, layer, [position])).numpy()
    out = {}
    for name, y in concepts.items():
        y = (y - y.mean()) / (y.std() + 1e-8)
        zs = (z - z.mean(0)) / (z.std(0) + 1e-8)
        r = (zs * y[:, None]).mean(0)
        order = np.argsort(-np.abs(r))[:5]
        out[name] = [(int(i), float(r[i])) for i in order]
    return out
