"""2-digit addition with fixed-length sequences, so every token has a fixed position.

Sequence layout (length 10)::

    pos:   0     1  2  3  4  5  6  7  8  9
    tok: <bos>  a1 a0  +  b1 b0  = s2 s1 s0

The model predicts ``s2`` at position 6, ``s1`` at 7 and ``s0`` at 8. Because the
model emits the most significant digit first, it must resolve both carries before
writing ``s2`` — which makes carries a natural target for probing and patching.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

VOCAB = [str(d) for d in range(10)] + ["+", "=", "<bos>"]
PLUS, EQ, BOS = 10, 11, 12
SEQ_LEN = 10
ANSWER_POS = (6, 7, 8)  # positions whose next-token prediction is s2, s1, s0
POS_NAMES = ["bos", "a1", "a0", "+", "b1", "b0", "=", "s2", "s1", "s0"]


@dataclass
class Batch:
    tokens: torch.Tensor  # [B, 10]
    a: np.ndarray
    b: np.ndarray

    @property
    def carry0(self) -> np.ndarray:
        """Units carry: a0 + b0 >= 10."""
        return ((self.a % 10 + self.b % 10) >= 10).astype(np.int64)

    @property
    def carry1(self) -> np.ndarray:
        """Tens carry (the hundreds digit): a + b >= 100."""
        return ((self.a + self.b) >= 100).astype(np.int64)

    @property
    def sum_digits(self) -> np.ndarray:
        s = self.a + self.b
        return np.stack([s // 100, (s // 10) % 10, s % 10], axis=1)


def encode(a: np.ndarray, b: np.ndarray) -> torch.Tensor:
    s = a + b
    cols = [
        np.full_like(a, BOS),
        a // 10,
        a % 10,
        np.full_like(a, PLUS),
        b // 10,
        b % 10,
        np.full_like(a, EQ),
        s // 100,
        (s // 10) % 10,
        s % 10,
    ]
    return torch.as_tensor(np.stack(cols, axis=1), dtype=torch.long)


def make_batch(a, b) -> Batch:
    a, b = np.asarray(a, dtype=np.int64), np.asarray(b, dtype=np.int64)
    return Batch(encode(a, b), a, b)


def random_batch(n: int, rng: np.random.Generator) -> Batch:
    return make_batch(rng.integers(0, 100, n), rng.integers(0, 100, n))


def all_pairs() -> Batch:
    """All 10,000 problems — small enough to evaluate exhaustively."""
    a, b = np.meshgrid(np.arange(100), np.arange(100), indexing="ij")
    return make_batch(a.ravel(), b.ravel())


def split(seed: int = 0, test_frac: float = 0.2) -> tuple[Batch, Batch]:
    """Disjoint train/test split over the 10k problems (no test problem seen in training)."""
    full = all_pairs()
    perm = np.random.default_rng(seed).permutation(len(full.a))
    n_test = int(test_frac * len(perm))
    te, tr = perm[:n_test], perm[n_test:]
    return make_batch(full.a[tr], full.b[tr]), make_batch(full.a[te], full.b[te])
