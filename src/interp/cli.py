"""Full analysis pipeline: train -> probes -> activation patching -> SAEs -> report.

Usage::

    interp --out results --seed 0
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from .data import ANSWER_POS, POS_NAMES, split
from .model import Config, accuracy, train
from .patching import carry_flip_pairs, patch_grid
from .probes import majority_baseline, probe_grid
from .sae import collect, evaluate_sae, feature_concept_correlation, train_sae


def run(
    seed: int,
    steps: int,
    d_model: int,
    n_layer: int,
    sae_k: int,
    sae_latents: int,
    sae_steps: int = 3000,
) -> dict:
    t0 = time.perf_counter()
    tr, te = split(seed=seed)
    cfg = Config(n_layer=n_layer, d_model=d_model, d_mlp=4 * d_model)
    model = train(cfg, tr, steps=steps, bs=512, lr=1e-3, seed=seed)
    acc = accuracy(model, te)
    print(f"[seed {seed}] test accuracy {acc}", flush=True)

    labels = {
        "carry0": (tr.carry0, te.carry0, 2),
        "carry1": (tr.carry1, te.carry1, 2),
        "s1_digit": (tr.sum_digits[:, 1], te.sum_digits[:, 1], 10),
    }
    probes = probe_grid(model, tr, te, labels)
    baselines = {k: majority_baseline(v[1]) for k, v in labels.items()}
    print(f"[seed {seed}] probes done", flush=True)

    clean, corrupt = carry_flip_pairs(512, np.random.default_rng(seed + 7))
    patch = {f"s{2 - d}": patch_grid(model, clean, corrupt, digit=d) for d in (0, 1)}
    print(f"[seed {seed}] patching done", flush=True)

    sae_results = {}
    for layer in range(1, n_layer + 1):
        acts = collect(model, tr, layer, list(ANSWER_POS))
        sae = train_sae(acts, n_latents=sae_latents, k=sae_k, steps=sae_steps, seed=seed)
        rep = evaluate_sae(sae, model, te, layer, list(ANSWER_POS))
        corr = feature_concept_correlation(
            sae,
            model,
            te,
            layer,
            ANSWER_POS[0],
            {"carry0": te.carry0.astype(float), "carry1": te.carry1.astype(float)},
        )
        sae_results[f"layer{layer}"] = {
            "fvu": rep.fvu,
            "dead_frac": rep.dead_frac,
            "ce_clean": rep.ce_clean,
            "ce_spliced": rep.ce_spliced,
            "ce_zero_ablate": rep.ce_zero_ablate,
            "ce_recovered": rep.ce_recovered,
            "top_features_at_eq": corr,
        }
    print(f"[seed {seed}] SAEs done", flush=True)
    return {
        "seed": seed,
        "config": {"n_layer": n_layer, "d_model": d_model, "steps": steps},
        "test_accuracy": acc,
        "probe_acc": {k: v.tolist() for k, v in probes.items()},
        "probe_majority_baseline": baselines,
        "patch_recovery": {k: v.tolist() for k, v in patch.items()},
        "sae": sae_results,
        "wall_seconds": round(time.perf_counter() - t0, 1),
    }


def heatmap(ax, grid, title, xlabels, vmin=0.0, vmax=1.0, cmap="viridis"):
    im = ax.imshow(np.asarray(grid), vmin=vmin, vmax=vmax, cmap=cmap, aspect="auto")
    ax.set_xticks(range(len(xlabels)), xlabels, fontsize=7)
    ax.set_yticks(range(len(grid)), [f"L{i}" for i in range(len(grid))], fontsize=7)
    ax.set_title(title, fontsize=8)
    for (i, j), v in np.ndenumerate(np.asarray(grid)):
        ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=5.5, color="w")
    return im


def figures(res: dict, out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    xl = POS_NAMES[:9]
    fig, axes = plt.subplots(1, 3, figsize=(12, 2.8))
    for ax, k in zip(axes, ["carry0", "carry1", "s1_digit"], strict=True):
        heatmap(
            ax,
            res["probe_acc"][k],
            f"probe acc: {k} (majority {res['probe_majority_baseline'][k]:.2f})",
            xl,
        )
    fig.tight_layout()
    fig.savefig(out / "probes.png", dpi=140)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(9, 2.8))
    for ax, k in zip(axes, ["s2", "s1"], strict=True):
        heatmap(
            ax,
            res["patch_recovery"][k],
            f"patching recovery for {k} (units-carry flip)",
            xl,
            vmin=-0.2,
            vmax=1.0,
            cmap="magma",
        )
    fig.tight_layout()
    fig.savefig(out / "patching.png", dpi=140)
    plt.close(fig)


def main(argv=None) -> None:
    p = argparse.ArgumentParser(prog="interp")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--steps", type=int, default=4000)
    p.add_argument("--d-model", type=int, default=128)
    p.add_argument("--n-layer", type=int, default=2)
    p.add_argument("--sae-k", type=int, default=8)
    p.add_argument("--sae-latents", type=int, default=256)
    p.add_argument("--sae-steps", type=int, default=3000)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--out", type=Path, default=Path("results"))
    p.add_argument("--no-plots", action="store_true")
    a = p.parse_args(argv)
    torch.set_num_threads(a.threads)
    res = run(a.seed, a.steps, a.d_model, a.n_layer, a.sae_k, a.sae_latents, a.sae_steps)
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / f"seed{a.seed}.json").write_text(json.dumps(res, indent=1))
    if not a.no_plots:
        figures(res, a.out)
    print(json.dumps({k: res[k] for k in ("test_accuracy", "wall_seconds")}, indent=1))


if __name__ == "__main__":
    main()
