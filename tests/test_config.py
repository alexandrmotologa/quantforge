"""Tests for QuantForge configuration schemas and enums."""

import pytest
from pathlib import Path

from quantforge.core.config import (
    BenchmarkConfig,
    ImatrixConfig,
    JobStatus,
    JobType,
    MatrixQuantConfig,
    PerplexityConfig,
    PipelineRecipe,
    QuantizationConfig,
    QuantType,
    settings,
)


def test_quant_type_properties():
    assert QuantType.Q4_K_M.value == "Q4_K_M"
    assert QuantType.Q4_K_M.estimated_bpw == 4.5
    assert QuantType.Q4_K_M.family == "K-Quant"
    assert QuantType.Q4_K_M.is_imatrix_recommended is True
    assert QuantType.Q4_K_M.is_imatrix_required is False

    assert QuantType.IQ1_S.is_imatrix_required is True
    assert QuantType.IQ2_M.is_imatrix_required is True
    assert QuantType.IQ2_M.family == "I-Quant"
    assert QuantType.F16.family == "Float/Identity"
    assert QuantType.Q8_0.family == "Legacy Quant"
    assert QuantType.Q8_0.is_imatrix_required is False


def test_quantization_config_validation(tmp_path: Path):
    in_file = tmp_path / "model.gguf"
    in_file.write_bytes(b"GGUF_TEST")
    out_file = tmp_path / "model-q4km.gguf"

    cfg = QuantizationConfig(
        input_path=in_file,
        output_path=out_file,
        quant_type=QuantType.Q4_K_M,
    )
    cfg.validate_prerequisites()
    assert cfg.input_path.is_file()

    # Missing input file
    bad_cfg = QuantizationConfig(
        input_path=tmp_path / "nonexistent.gguf",
        output_path=out_file,
        quant_type=QuantType.Q4_K_M,
    )
    with pytest.raises(FileNotFoundError):
        bad_cfg.validate_prerequisites()

    # IQ1_S requires imatrix
    iq_cfg = QuantizationConfig(
        input_path=in_file,
        output_path=out_file,
        quant_type=QuantType.IQ1_S,
        imatrix_path=None,
    )
    with pytest.raises(ValueError, match="requires an importance matrix"):
        iq_cfg.validate_prerequisites()


def test_imatrix_config_validation(tmp_path: Path):
    model = tmp_path / "base.gguf"
    model.write_bytes(b"GGUF")
    data = tmp_path / "corpus.txt"
    data.write_text("sample calibration text")
    out = tmp_path / "imatrix.dat"

    cfg = ImatrixConfig(
        input_path=model,
        data_path=data,
        output_path=out,
        ctx_size=1024,
        chunks=32,
    )
    cfg.validate_prerequisites()
    assert cfg.ctx_size == 1024


def test_perplexity_config(tmp_path: Path):
    cfg = PerplexityConfig(
        model_path=tmp_path / "model.gguf",
        data_path=tmp_path / "wiki.test.raw",
        ctx_size=2048,
        batch_size=256,
    )
    assert cfg.ctx_size == 2048
    assert cfg.batch_size == 256


def test_benchmark_config(tmp_path: Path):
    cfg = BenchmarkConfig(
        model_path=tmp_path / "model.gguf",
        prompt_tokens=256,
        gen_tokens=64,
        repetitions=2,
    )
    assert cfg.prompt_tokens == 256
    assert cfg.repetitions == 2


def test_settings_directories(tmp_path: Path):
    orig = settings.data_dir
    try:
        settings.data_dir = tmp_path / "qf_data"
        settings.ensure_directories()
        assert settings.data_dir.is_dir()
        assert settings.binaries_path.is_dir()
        assert settings.cache_path.is_dir()
    finally:
        settings.data_dir = orig
