import json

from interp.cli import main


def test_pipeline_tiny(tmp_path):
    main(
        [
            "--steps",
            "30",
            "--d-model",
            "32",
            "--n-layer",
            "1",
            "--sae-latents",
            "32",
            "--sae-k",
            "4",
            "--sae-steps",
            "20",
            "--threads",
            "1",
            "--out",
            str(tmp_path),
            "--no-plots",
        ]
    )
    res = json.loads((tmp_path / "seed0.json").read_text())
    assert set(res["probe_acc"]) == {"carry0", "carry1", "s1_digit"}
    assert len(res["probe_acc"]["carry0"]) == 2  # embeddings + 1 layer
    assert set(res["patch_recovery"]) == {"s2", "s1"}
    assert "layer1" in res["sae"]
