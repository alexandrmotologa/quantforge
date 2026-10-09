"""Tests for VRAM and KV cache calculator."""

import pytest
from pathlib import Path
from quantforge.core.vram_calc import calculate_vram, calculate_vram_from_gguf, KV_BYTES_PER_ELEMENT


def test_calculate_vram_7b_fp16():
    # Mistral 7B standard dimensions: 32 layers, 8 KV heads, 128 head dim (GQA)
    model_bytes = 4_000_000_000  # ~4GB Q4_K_M
    estimate = calculate_vram(
        model_size_bytes=model_bytes,
        layers=32,
        kv_heads=8,
        head_dim=128,
        context_size=8192,
        kv_quant="fp16",
        embedding_length=4096,
    )

    assert estimate.model_size_gb > 3.7
    # 2 * 32 * 8 * 128 * 8192 * 2 bytes = 1,073,741,824 bytes = 1.0 GB
    assert 0.95 <= estimate.kv_cache_gb <= 1.05
    assert estimate.total_vram_gb < 8.0
    # Should fit comfortably on a 12GB GPU
    assert estimate.fits_gpus["12GB (RTX 3060 / 4070)"] is True


def test_calculate_vram_quantized_kv_cache():
    model_bytes = 4_000_000_000
    est_fp16 = calculate_vram(
        model_size_bytes=model_bytes,
        layers=32,
        kv_heads=8,
        head_dim=128,
        context_size=32768,
        kv_quant="fp16",
    )
    est_q4 = calculate_vram(
        model_size_bytes=model_bytes,
        layers=32,
        kv_heads=8,
        head_dim=128,
        context_size=32768,
        kv_quant="q4_0",
    )

    # Q4_0 KV cache should be approximately 28% of FP16 KV cache (0.5625 / 2.0)
    assert est_q4.kv_cache_mb < est_fp16.kv_cache_mb * 0.35
    assert est_q4.total_vram_mb < est_fp16.total_vram_mb


def test_calculate_vram_from_real_gguf(tmp_path: Path):
    from tests.helpers import write_synthetic_gguf
    model_path = tmp_path / "test_vram.gguf"
    write_synthetic_gguf(model_path)

    estimate = calculate_vram_from_gguf(model_path, context_size=2048, kv_quant="q8_0")
    assert estimate.context_size == 2048
    assert estimate.kv_quant == "q8_0"
    assert estimate.total_vram_mb > 0
    assert len(estimate.fits_gpus) > 0
    assert "8GB (RTX 4060 / Apple M-series 8G)" in estimate.fits_gpus
