"""Tests for PerplexityEngine."""

from pathlib import Path
import pytest

from quantforge.core.config import PerplexityConfig
from quantforge.engines.perplexity import PerplexityEngine


def test_perplexity_build_command(tmp_path: Path):
    model = tmp_path / "model.gguf"
    model.write_bytes(b"GGUF")
    data = tmp_path / "eval.txt"
    data.write_text("Test evaluation corpus")

    cfg = PerplexityConfig(
        model_path=model,
        data_path=data,
        ctx_size=1024,
        batch_size=256,
        n_gpu_layers=20,
        threads=8,
    )

    engine = PerplexityEngine()
    cmd = engine.build_command(cfg)

    assert any("llama-perplexity" in arg for arg in cmd)
    assert "-m" in cmd and str(model) in cmd
    assert "-f" in cmd and str(data) in cmd
    assert "-c" in cmd and "1024" in cmd
    assert "-b" in cmd and "256" in cmd
    assert "-ngl" in cmd and "20" in cmd
    assert "-t" in cmd and "8" in cmd
