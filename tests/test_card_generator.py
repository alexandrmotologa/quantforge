"""Tests for ModelCardGenerator."""

from pathlib import Path
from quantforge.formats.card_generator import ModelCardGenerator
from quantforge.formats.gguf_reader import GGUFModelInfo
from quantforge.engines.pareto import QuantPoint, ParetoAnalyzer


def test_model_card_generator():
    gen = ModelCardGenerator()
    vram = gen.estimate_vram_requirement_gb(4.1, ctx_length=4096)
    assert vram > 4.1

    info = GGUFModelInfo(
        file_path=Path("mistral-7b.gguf"),
        file_size_bytes=14 * 1024 * 1024 * 1024,
        version=3,
        tensor_count=291,
        metadata_kv_count=20,
        architecture="llama",
        context_length=8192,
        estimated_parameters=7240000000,
    )

    pts = [
        QuantPoint(quant_type="Q4_K_M", file_path=Path("q4km.gguf"), file_size_gb=4.07, perplexity=5.25),
        QuantPoint(quant_type="Q8_0", file_path=Path("q80.gguf"), file_size_gb=7.70, perplexity=5.12),
    ]

    report = ParetoAnalyzer().compute_frontier(pts)
    card = gen.generate_card("Mistral-7B-v0.3", model_info=info, pareto_report=report)

    assert "# Mistral-7B-v0.3 - GGUF Quantizations" in card
    assert "Base Architecture" in card
    assert "llama" in card
    assert "Parameters" in card
    assert "7.2B" in card
    assert "Context Length" in card
    assert "`Q4_K_M`" in card
    assert "ollama create" in card
