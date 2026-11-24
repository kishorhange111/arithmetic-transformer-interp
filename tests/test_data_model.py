import numpy as np
import torch

from interp.data import SEQ_LEN, all_pairs, make_batch, split
from interp.model import Config, Transformer, accuracy, train


def test_encoding_layout():
    b = make_batch([47, 99], [58, 1])
    assert b.tokens.shape == (2, SEQ_LEN)
    assert b.tokens[0].tolist() == [12, 4, 7, 10, 5, 8, 11, 1, 0, 5]  # 47+58=105
    assert b.carry0.tolist() == [1, 1] and b.carry1.tolist() == [1, 1]
    assert make_batch([12], [34]).carry0.tolist() == [0]


def test_split_is_disjoint_and_complete():
    tr, te = split(seed=0)
    pairs = lambda bt: set(zip(bt.a.tolist(), bt.b.tolist(), strict=True))  # noqa: E731
    assert not pairs(tr) & pairs(te)
    assert len(pairs(tr)) + len(pairs(te)) == 10_000
    assert len(all_pairs().a) == 10_000


def test_resid_and_patch_identity():
    torch.manual_seed(0)
    m = Transformer(Config(n_layer=2, d_model=32, d_mlp=64))
    tok = make_batch(np.arange(4), np.arange(4)).tokens[:, :-1]
    logits, resid = m(tok, return_resid=True)
    assert len(resid) == 3 and resid[0].shape == (4, 9, 32)
    torch.testing.assert_close(m(tok, patch=lambda layer, x: x), logits)


def test_patch_changes_output_when_resid_replaced():
    torch.manual_seed(0)
    m = Transformer(Config(n_layer=1, d_model=32, d_mlp=64))
    tok = make_batch([1], [2]).tokens[:, :-1]
    base = m(tok)
    zeroed = m(tok, patch=lambda layer, x: torch.zeros_like(x) if layer == 1 else x)
    assert not torch.allclose(base, zeroed)


def test_training_learns_addition():
    tr, te = split(seed=0)
    m = train(Config(n_layer=2, d_model=64, d_mlp=256), tr, steps=1000, bs=256, lr=3e-3)
    acc = accuracy(m, te)
    assert acc["exact"] > 0.8, acc
