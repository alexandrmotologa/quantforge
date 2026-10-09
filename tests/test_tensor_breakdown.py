"""Tests for tensor architectural breakdown and custom overrides."""

from pathlib import Path
from quantforge.formats.gguf_reader import GGUFReader
from quantforge.core.config import QuantizationConfig, QuantType
from quantforge.engines.quantize import QuantizeEngine
from .helpers import write_minimal_llama_model


def test_tensor_breakdown_categorization(tmp_path: Path):
    model_path = tmp_path / "model.gguf"
    write_minimal_llama_model(model_path)

    reader = GGUFReader(model_path)
    info = reader.read_model_info(load_tensors=True)
    breakdown = reader.get_tensor_breakdown(info)

    assert "Embeddings" in breakdown
    assert "Output Head" in breakdown
    assert breakdown["Embeddings"]["count"] >= 1
    assert breakdown["Output Head"]["count"] >= 1
    assert breakdown["Embeddings"]["param_pct"] > 0.0


def test_quantize_with_tensor_type_overrides(tmp_path: Path):
    in_model = tmp_path / "model.gguf"
    in_model.write_bytes(b"GGUF")
    out_model = tmp_path / "out.gguf"

    cfg = QuantizationConfig(
        input_path=in_model,
        output_path=out_model,
        quant_type=QuantType.Q4_K_M,
        tensor_type_overrides={
            "output.weight": "Q8_0",
            "token_embd.weight": "Q8_0",
        },
    )

    engine = QuantizeEngine()
    cmd = engine.build_command(cfg)

    assert "--tensor-type" in cmd
    assert "output.weight=Q8_0" in cmd
    assert "token_embd.weight=Q8_0" in cmd
