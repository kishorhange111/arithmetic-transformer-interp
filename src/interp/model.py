"""A small attention-only-friendly transformer with hookable residual stream.

``forward(tokens, return_resid=True)`` returns the residual stream *before* each
block and after the last one: ``resid[l]`` has shape ``[B, T, d]`` for
``l = 0..n_layer`` (``resid[0]`` = embeddings).

``patch`` lets callers overwrite the residual stream at chosen ``(layer, pos)``
locations — the primitive behind activation patching and SAE splicing.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .data import ANSWER_POS, SEQ_LEN, VOCAB, Batch

PatchFn = Callable[[int, torch.Tensor], torch.Tensor]


@dataclass
class Config:
    n_layer: int = 2
    n_head: int = 4
    d_model: int = 128
    d_mlp: int = 512
    vocab: int = len(VOCAB)
    seq_len: int = SEQ_LEN


class Attention(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.h = cfg.n_head
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model)
        self.o = nn.Linear(cfg.d_model, cfg.d_model)
        self.last_pattern: torch.Tensor | None = None

    def forward(self, x, keep_pattern: bool = False):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        q, k, v = (t.view(B, T, self.h, C // self.h).transpose(1, 2) for t in (q, k, v))
        if keep_pattern:
            att = (q @ k.transpose(-1, -2)) / (C // self.h) ** 0.5
            mask = torch.ones(T, T, dtype=torch.bool, device=x.device).tril()
            att = att.masked_fill(~mask, float("-inf")).softmax(-1)
            self.last_pattern = att.detach()
            y = att @ v
        else:
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.o(y.transpose(1, 2).reshape(B, T, C))


class Block(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(cfg.d_model), nn.LayerNorm(cfg.d_model)
        self.attn = Attention(cfg)
        self.mlp = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_mlp), nn.GELU(), nn.Linear(cfg.d_mlp, cfg.d_model)
        )

    def forward(self, x, keep_pattern=False):
        x = x + self.attn(self.ln1(x), keep_pattern)
        return x + self.mlp(self.ln2(x))


class Transformer(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.tok = nn.Embedding(cfg.vocab, cfg.d_model)
        self.pos = nn.Embedding(cfg.seq_len, cfg.d_model)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layer))
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.unembed = nn.Linear(cfg.d_model, cfg.vocab, bias=False)

    def forward(
        self,
        tokens: torch.Tensor,
        return_resid: bool = False,
        patch: PatchFn | None = None,
        keep_pattern: bool = False,
    ):
        T = tokens.shape[1]
        x = self.tok(tokens) + self.pos(torch.arange(T, device=tokens.device))
        resid = []
        for layer, blk in enumerate(self.blocks):
            if patch is not None:
                x = patch(layer, x)
            resid.append(x)
            x = blk(x, keep_pattern)
        if patch is not None:
            x = patch(self.cfg.n_layer, x)
        resid.append(x)
        logits = self.unembed(self.ln_f(x))
        return (logits, resid) if return_resid else logits


def answer_logits(logits: torch.Tensor) -> torch.Tensor:
    """Logits that predict s2, s1, s0 — shape ``[B, 3, vocab]``."""
    return logits[:, list(ANSWER_POS), :]


def loss_fn(model: Transformer, batch: Batch, patch: PatchFn | None = None) -> torch.Tensor:
    logits = answer_logits(model(batch.tokens[:, :-1], patch=patch))
    targets = batch.tokens[:, 7:10]
    return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))


@torch.no_grad()
def accuracy(model: Transformer, batch: Batch) -> dict[str, float]:
    """Teacher-forced per-digit and whole-answer accuracy (equals greedy decoding when exact)."""
    pred = answer_logits(model(batch.tokens[:, :-1])).argmax(-1)
    correct = pred == batch.tokens[:, 7:10]
    return {
        "exact": correct.all(1).float().mean().item(),
        "s2": correct[:, 0].float().mean().item(),
        "s1": correct[:, 1].float().mean().item(),
        "s0": correct[:, 2].float().mean().item(),
    }


def train(
    cfg: Config,
    train_batch: Batch,
    steps: int = 3000,
    bs: int = 512,
    lr: float = 1e-3,
    wd: float = 0.1,
    seed: int = 0,
    log_every: int = 0,
) -> Transformer:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = Transformer(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.1)
    n = len(train_batch.a)
    for step in range(steps):
        idx = rng.integers(0, n, bs)
        b = Batch(train_batch.tokens[idx], train_batch.a[idx], train_batch.b[idx])
        loss = loss_fn(model, b)
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        if log_every and step % log_every == 0:
            print(f"step {step} loss {loss.item():.4f}", flush=True)
    return model.eval()
