"""Interactive VRAM and KV cache memory estimation engine for GGUF models."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Union

from quantforge.formats.gguf_reader import GGUFReader

# Standard consumer and enterprise GPU VRAM tiers (in GB)
GPU_TIERS: Dict[str, float] = {
    "8GB (RTX 4060 / Apple M-series 8G)": 8.0,
    "12GB (RTX 3060 / 4070)": 12.0,
    "16GB (RTX 4080 / Apple 16G)": 16.0,
    "24GB (RTX 3090 / 4090)": 24.0,
    "48GB (RTX 6000 Ada / A6000)": 48.0,
    "80GB (A100 / H100 80G)": 80.0,
}

# Bytes per element for KV cache quantization types in llama.cpp
# Q8_0: 34 bytes per block of 32 (1.0625 bytes/element)
# Q4_0: 18 bytes per block of 32 (0.5625 bytes/element)
KV_BYTES_PER_ELEMENT: Dict[str, float] = {
    "fp16": 2.0,
    "f16": 2.0,
    "q8_0": 1.0625,
    "q4_0": 0.5625,
}


@dataclass
class VRAMEstimate:
    """Detailed breakdown of VRAM requirements for a model at a specific context size."""

    model_size_mb: float
    model_size_gb: float
    kv_cache_mb: float
    kv_cache_gb: float
    activation_scratch_mb: float
    cuda_overhead_mb: float
    total_vram_mb: float
    total_vram_gb: float
    context_size: int
    kv_quant: str
    fits_gpus: Dict[str, bool] = field(default_factory=dict)
    max_context_per_gpu: Dict[str, int] = field(default_factory=dict)

    def summary(self) -> str:
        """Formatted human-readable summary of VRAM allocation."""
        return (
            f"VRAM Breakdown @ {self.context_size} ctx ({self.kv_quant.upper()} KV):\n"
            f"  • Weights:    {self.model_size_gb:.2f} GB ({self.model_size_mb:.1f} MB)\n"
            f"  • KV Cache:   {self.kv_cache_gb:.2f} GB ({self.kv_cache_mb:.1f} MB)\n"
            f"  • Scratch/Act:{self.activation_scratch_mb:.1f} MB\n"
            f"  • CUDA Runtime:{self.cuda_overhead_mb:.1f} MB\n"
            f"  ----------------------------------------\n"
            f"  Total VRAM:   {self.total_vram_gb:.2f} GB ({self.total_vram_mb:.1f} MB)"
        )


def calculate_vram(
    model_size_bytes: int,
    layers: int,
    kv_heads: int,
    head_dim: int,
    context_size: int,
    kv_quant: str = "fp16",
    embedding_length: Optional[int] = None,
    cuda_overhead_mb: float = 512.0,
) -> VRAMEstimate:
    """Calculates comprehensive VRAM footprint and GPU compatibility."""
    model_size_mb = model_size_bytes / (1024 * 1024)
    model_size_gb = model_size_bytes / (1024 * 1024 * 1024)

    quant_norm = kv_quant.lower()
    bpe = KV_BYTES_PER_ELEMENT.get(quant_norm, 2.0)

    # KV Cache bytes = 2 (Key + Value) * layers * kv_heads * head_dim * context_size * bytes_per_elem
    kv_bytes = 2 * layers * kv_heads * head_dim * context_size * bpe
    kv_cache_mb = kv_bytes / (1024 * 1024)
    kv_cache_gb = kv_bytes / (1024 * 1024 * 1024)

    # Activation & scratchpad buffers scale with context length and hidden dimension
    hidden_dim = embedding_length if embedding_length else (kv_heads * head_dim)
    # Scratchpad estimation: approx context * hidden_dim * 4 bytes + base working memory
    act_bytes = (context_size * hidden_dim * 4) + (64 * 1024 * 1024)
    act_mb = act_bytes / (1024 * 1024)

    total_mb = model_size_mb + kv_cache_mb + act_mb + cuda_overhead_mb
    total_gb = total_mb / 1024.0

    # Determine GPU fit
    fits: Dict[str, bool] = {}
    max_ctx: Dict[str, int] = {}

    # One token of KV cache in MB
    kv_per_token_mb = (2 * layers * kv_heads * head_dim * bpe) / (1024 * 1024)
    act_per_token_mb = (hidden_dim * 4) / (1024 * 1024)
    mb_per_token = kv_per_token_mb + act_per_token_mb

    base_mb = model_size_mb + cuda_overhead_mb + 64.0

    for name, vram_gb in GPU_TIERS.items():
        vram_limit_mb = vram_gb * 1024.0
        fits[name] = total_mb <= vram_limit_mb

        if vram_limit_mb > base_mb and mb_per_token > 0:
            remaining_mb = vram_limit_mb - base_mb
            tokens = int(remaining_mb / mb_per_token)
            # Round down to nearest 512
            max_ctx[name] = max(0, (tokens // 512) * 512)
        else:
            max_ctx[name] = 0

    return VRAMEstimate(
        model_size_mb=round(model_size_mb, 1),
        model_size_gb=round(model_size_gb, 2),
        kv_cache_mb=round(kv_cache_mb, 1),
        kv_cache_gb=round(kv_cache_gb, 2),
        activation_scratch_mb=round(act_mb, 1),
        cuda_overhead_mb=round(cuda_overhead_mb, 1),
        total_vram_mb=round(total_mb, 1),
        total_vram_gb=round(total_gb, 2),
        context_size=context_size,
        kv_quant=quant_norm,
        fits_gpus=fits,
        max_context_per_gpu=max_ctx,
    )


def calculate_vram_from_gguf(
    file_path: Union[str, Path],
    context_size: Optional[int] = None,
    kv_quant: str = "fp16",
    cuda_overhead_mb: float = 512.0,
) -> VRAMEstimate:
    """Reads GGUF architectural metadata directly and estimates required VRAM."""
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"GGUF model file not found: {path}")

    reader = GGUFReader(path)
    info = reader.read_model_info(load_tensors=False)

    ctx = context_size if context_size is not None else (info.context_length or 4096)
    layers = info.block_count or 32
    arch = info.architecture or "llama"

    # Attention parameters
    heads = info.metadata.get(f"{arch}.attention.head_count", 32)
    kv_heads = info.metadata.get(f"{arch}.attention.head_count_kv", heads)
    key_len = info.metadata.get(f"{arch}.attention.key_length")

    if key_len:
        head_dim = int(key_len)
    elif info.embedding_length and heads:
        head_dim = int(info.embedding_length) // int(heads)
    else:
        head_dim = 128

    return calculate_vram(
        model_size_bytes=info.file_size_bytes,
        layers=int(layers),
        kv_heads=int(kv_heads),
        head_dim=int(head_dim),
        context_size=int(ctx),
        kv_quant=kv_quant,
        embedding_length=info.embedding_length,
        cuda_overhead_mb=cuda_overhead_mb,
    )
