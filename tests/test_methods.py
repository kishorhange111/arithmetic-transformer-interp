import numpy as np
import pytest
import torch

from interp.data import make_batch, random_batch
from interp.model import Config, Transformer
from interp.patching import carry_flip_pairs, patch_grid
from interp.probes import fit_probe, majority_baseline
from interp.sae import TopKSAE, evaluate_sae, feature_concept_correlation, fvu, train_sae


def small_model():
    torch.manual_seed(0)
    return Transformer(Config(n_layer=1, d_model=32, d_mlp=64)).eval()


def test_probe_recovers_linear_signal():
    g = torch.Generator().manual_seed(0)
    x = torch.randn(600, 8, generator=g)
    y = (x[:, 3] > 0).long()
    assert fit_probe(x[:400], y[:400], x[400:], y[400:], 2) > 0.95
    noise = torch.randint(0, 2, (600,), generator=g)
    assert fit_probe(x[:400], noise[:400], x[400:], noise[400:], 2) < 0.65


def test_majority_baseline():
    assert majority_baseline(np.array([0, 0, 0, 1])) == 0.75


def test_carry_flip_pairs_isolate_units_carry():
    clean, corrupt = carry_flip_pairs(50, np.random.default_rng(0))
    assert (clean.a // 10 == corrupt.a // 10).all()
    assert (clean.carry1 == 1).all() and (corrupt.carry1 == 0).all()


def test_patch_grid_restoring_final_resid_everywhere_recovers_fully():
    m = small_model()
    clean, corrupt = carry_flip_pairs(32, np.random.default_rng(0))
    grid = patch_grid(m, clean, corrupt, digit=0)
    assert grid.shape == (2, 9)
    # Patching the final residual at the answer position fully restores that logit.
    assert grid[-1, 6] == pytest.approx(1.0, abs=1e-4)
    # Positions after the readout position cannot influence it (causal mask).
    assert np.allclose(grid[:, 7:], 0.0, atol=1e-5)


def test_topk_sae_has_exact_l0_and_unit_decoder():
    sae = TopKSAE(16, 64, k=4)
    _, z = sae(torch.randn(10, 16))
    assert ((z > 0).sum(1) <= 4).all()
    torch.testing.assert_close(sae.W_dec.norm(dim=0), torch.ones(64), atol=1e-5, rtol=0)


def test_sae_training_reduces_fvu():
    g = torch.Generator().manual_seed(0)
    basis = torch.randn(32, 16, generator=g)
    codes = (torch.rand(4000, 32, generator=g) < 0.1).float() * torch.rand(4000, 32, generator=g)
    acts = codes @ basis
    sae0 = TopKSAE(16, 64, k=4)
    before = fvu(acts, sae0(acts)[0].detach())
    sae = train_sae(acts, n_latents=64, k=4, steps=400, bs=256, lr=3e-3)
    after = fvu(acts, sae(acts)[0].detach())
    assert after < 0.5 * before and after < 0.3


def test_evaluate_sae_and_correlation_shapes():
    m = small_model()
    batch = random_batch(256, np.random.default_rng(0))
    from interp.sae import collect

    acts = collect(m, batch, 1, [6])
    sae = train_sae(acts, n_latents=32, k=4, steps=50, bs=128)
    rep = evaluate_sae(sae, m, batch, 1, [6])
    assert 0 <= rep.dead_frac <= 1 and rep.fvu >= 0
    corr = feature_concept_correlation(sae, m, batch, 1, 6, {"carry0": batch.carry0.astype(float)})
    assert len(corr["carry0"]) == 5


def test_make_batch_accepts_lists():
    assert make_batch([1, 2], [3, 4]).tokens.shape == (2, 10)
