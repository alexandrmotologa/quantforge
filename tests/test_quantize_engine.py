"""Tests for QuantizeEngine with unit and real binary integration tests."""

from pathlib import Path
import pytest

from quantforge.core.binary_manager import BinaryManager
from quantforge.core.config import MatrixQuantConfig, QuantizationConfig, QuantType
from quantforge.engines.quantize import QuantizeEngine
from .helpers import write_minimal_llama_model


def test_build_command(tmp_path: Path):
    in_file = tmp_path / "model.gguf"
    in_file.write_bytes(b"GGUF")
    out_file = tmp_path / "model-q4km.gguf"
    im_file = tmp_path / "imatrix.dat"
    im_file.write_bytes(b"DATA")

    cfg = QuantizationConfig(
        input_path=in_file,
        output_path=out_file,
        quant_type=QuantType.Q4_K_M,
        imatrix_path=im_file,
        leave_output_tensor=True,
        pure=True,
        threads=8,
    )

    engine = QuantizeEngine()
    cmd = engine.build_command(cfg)

    assert any("llama-quantize" in arg for arg in cmd)
    assert "--imatrix" in cmd
    assert str(im_file) in cmd
    assert "--leave-output-tensor" in cmd
    assert "--pure" in cmd
    assert str(in_file) in cmd
    assert str(out_file) in cmd
    assert "Q4_K_M" in cmd
    assert "8" in cmd


@pytest.mark.asyncio
async def test_quantize_prerequisites_fail(tmp_path: Path):
    cfg = QuantizationConfig(
        input_path=tmp_path / "missing.gguf",
        output_path=tmp_path / "out.gguf",
        quant_type=QuantType.Q4_K_M,
    )
    engine = QuantizeEngine()
    with pytest.raises(FileNotFoundError):
        await engine.quantize(cfg)


@pytest.mark.asyncio
async def test_real_quantization_execution(tmp_path: Path):
    """End-to-end integration test with real native llama-quantize binary."""
    in_model = tmp_path / "base.gguf"
    out_model = tmp_path / "out_q80.gguf"
    write_minimal_llama_model(in_model)

    engine = QuantizeEngine()
    cfg = QuantizationConfig(
        input_path=in_model,
        output_path=out_model,
        quant_type=QuantType.Q8_0,
    )

    res = await engine.quantize(cfg)
    assert res.success is True
    assert res.output_path.is_file()
    assert res.quantized_size_bytes > 0
    assert res.compression_ratio > 0.0
    assert res.tensor_count >= 1
