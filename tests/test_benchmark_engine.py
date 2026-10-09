"""Tests for BenchmarkEngine."""

from pathlib import Path
import pytest

from quantforge.core.config import BenchmarkConfig
from quantforge.engines.benchmark import BenchmarkEngine


def test_bench_build_command(tmp_path: Path):
    model = tmp_path / "model.gguf"
    model.write_bytes(b"GGUF")

    cfg = BenchmarkConfig(
        model_path=model,
        prompt_tokens=256,
        gen_tokens=64,
        n_gpu_layers=33,
        threads=6,
        repetitions=2,
    )

    engine = BenchmarkEngine()
    cmd = engine.build_command(cfg)

    assert any("llama-bench" in arg for arg in cmd)
    assert "-m" in cmd and str(model) in cmd
    assert "-p" in cmd and "256" in cmd
    assert "-n" in cmd and "64" in cmd
    assert "-r" in cmd and "2" in cmd
    assert "-ngl" in cmd and "33" in cmd
    assert "-t" in cmd and "6" in cmd


def test_bench_regex_parsing():
    engine = BenchmarkEngine()
    line = "| llama-7b | Q4_K_M | 4.07 GB | 234.50 +/- 2.10 | 45.80 +/- 0.50 |"
    m = engine.RE_SPEED.search(line)
    assert m is not None
    assert float(m.group("pp")) == 234.50
    assert float(m.group("tg")) == 45.80
