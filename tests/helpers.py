"""Test helper utilities for QuantForge."""

import struct
from pathlib import Path
import numpy as np
import gguf

from quantforge.formats.gguf_reader import GGMLType, GGUFValueType


def write_synthetic_gguf(path: Path) -> Path:
    """Creates a minimal valid GGUF v3 file on disk for unit testing."""
    with open(path, "wb") as f:
        f.write(b"GGUF")
        f.write(struct.pack("<I", 3))
        f.write(struct.pack("<Q", 1))
        f.write(struct.pack("<Q", 2))

        key1 = "general.architecture".encode("utf-8")
        f.write(struct.pack("<Q", len(key1)))
        f.write(key1)
        f.write(struct.pack("<I", GGUFValueType.STRING))
        val1 = "llama".encode("utf-8")
        f.write(struct.pack("<Q", len(val1)))
        f.write(val1)

        key2 = "llama.context_length".encode("utf-8")
        f.write(struct.pack("<Q", len(key2)))
        f.write(key2)
        f.write(struct.pack("<I", GGUFValueType.UINT32))
        f.write(struct.pack("<I", 4096))

        t_name = "blk.0.attn_q.weight".encode("utf-8")
        f.write(struct.pack("<Q", len(t_name)))
        f.write(t_name)
        f.write(struct.pack("<I", 2))
        f.write(struct.pack("<Q", 128))
        f.write(struct.pack("<Q", 64))
        f.write(struct.pack("<I", GGMLType.Q4_K))
        f.write(struct.pack("<Q", 0))

        f.write(b"\x00" * 1024)

    return path


def write_minimal_llama_model(path: Path) -> Path:
    """Creates a complete minimal valid llama GGUF model that native llama-quantize can quantize."""
    writer = gguf.GGUFWriter(str(path), "llama")
    writer.add_context_length(512)
    writer.add_embedding_length(256)
    writer.add_block_count(1)
    writer.add_feed_forward_length(512)
    writer.add_head_count(4)
    writer.add_head_count_kv(4)
    writer.add_layer_norm_rms_eps(1e-5)
    writer.add_tokenizer_model("llama")

    tensor_data = np.ones((256, 256), dtype=np.float32)
    writer.add_tensor("token_embd.weight", tensor_data)
    writer.add_tensor("output.weight", tensor_data)

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    return path
