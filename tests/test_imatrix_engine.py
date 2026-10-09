"""Tests for ImatrixEngine."""

from pathlib import Path
import pytest

from quantforge.core.config import ImatrixConfig
from quantforge.engines.imatrix import ImatrixEngine


def test_imatrix_build_command(tmp_path: Path):
    in_file = tmp_path / "model.gguf"
    in_file.write_bytes(b"GGUF")
    data_file = tmp_path / "corpus.txt"
    data_file.write_text("Hello text")
    out_file = tmp_path / "imatrix.dat"

    cfg = ImatrixConfig(
        input_path=in_file,
        data_path=data_file,
        output_path=out_file,
        ctx_size=1024,
        chunks=32,
        n_gpu_layers=10,
        threads=4,
    )

    engine = ImatrixEngine()
    cmd = engine.build_command(cfg)

    assert any("llama-imatrix" in arg for arg in cmd)
    assert "-m" in cmd and str(in_file) in cmd
    assert "-f" in cmd and str(data_file) in cmd
    assert "-o" in cmd and str(out_file) in cmd
    assert "-c" in cmd and "1024" in cmd
    assert "--chunks" in cmd and "32" in cmd
    assert "--output-format" in cmd and "dat" in cmd
    assert "-ngl" in cmd and "10" in cmd
    assert "-t" in cmd and "4" in cmd


def test_generate_sample_corpus(tmp_path: Path):
    target = tmp_path / "sample_calib.txt"
    engine = ImatrixEngine()
    engine.generate_sample_calibration_corpus(target, repeats=5)

    assert target.is_file()
    content = target.read_text(encoding="utf-8")
    assert "topological space" in content
    assert len(content) > 200
