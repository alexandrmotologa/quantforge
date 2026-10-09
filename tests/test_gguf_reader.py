"""Tests for GGUFReader parsing headers, metadata, and tensor info."""

import struct
from pathlib import Path
import pytest

from quantforge.formats.gguf_reader import GGMLType, GGUFReader, GGUFValueType


def write_synthetic_gguf(path: Path) -> Path:
    """Creates a minimal valid GGUF v3 file on disk for unit testing."""
    with open(path, "wb") as f:
        # Magic "GGUF"
        f.write(b"GGUF")
        # Version 3
        f.write(struct.pack("<I", 3))
        # Tensor count = 1
        f.write(struct.pack("<Q", 1))
        # Metadata KV count = 2
        f.write(struct.pack("<Q", 2))

        # KV 1: "general.architecture" -> STRING "llama"
        key1 = "general.architecture".encode("utf-8")
        f.write(struct.pack("<Q", len(key1)))
        f.write(key1)
        f.write(struct.pack("<I", GGUFValueType.STRING))
        val1 = "llama".encode("utf-8")
        f.write(struct.pack("<Q", len(val1)))
        f.write(val1)

        # KV 2: "llama.context_length" -> UINT32 4096
        key2 = "llama.context_length".encode("utf-8")
        f.write(struct.pack("<Q", len(key2)))
        f.write(key2)
        f.write(struct.pack("<I", GGUFValueType.UINT32))
        f.write(struct.pack("<I", 4096))

        # Tensor 1: "blk.0.attn_q.weight", 2 dims: [128, 64], type=Q4_K (12), offset=0
        t_name = "blk.0.attn_q.weight".encode("utf-8")
        f.write(struct.pack("<Q", len(t_name)))
        f.write(t_name)
        f.write(struct.pack("<I", 2)) # n_dims
        f.write(struct.pack("<Q", 128))
        f.write(struct.pack("<Q", 64))
        f.write(struct.pack("<I", GGMLType.Q4_K)) # type
        f.write(struct.pack("<Q", 0)) # offset

        # Raw tensor payload (dummy 1024 bytes)
        f.write(b"\x00" * 1024)

    return path


def test_gguf_reader_parses_synthetic_file(tmp_path: Path):
    model_path = tmp_path / "synthetic.gguf"
    write_synthetic_gguf(model_path)

    reader = GGUFReader(model_path)
    info = reader.read_model_info(load_tensors=True)

    assert info.version == 3
    assert info.architecture == "llama"
    assert info.context_length == 4096
    assert info.tensor_count == 1
    assert len(info.tensors) == 1
    assert info.tensors[0].name == "blk.0.attn_q.weight"
    assert info.tensors[0].dimensions == [128, 64]
    assert info.tensors[0].type_name == "Q4_K"
    assert info.estimated_parameters == 128 * 64
    assert info.dominant_quant == "Q4_K"


def test_gguf_reader_invalid_magic(tmp_path: Path):
    bad_file = tmp_path / "bad.gguf"
    bad_file.write_bytes(b"INVALID_MAGIC_HEADER_TEST")

    reader = GGUFReader(bad_file)
    with pytest.raises(ValueError, match="Invalid GGUF magic header"):
        reader.read_model_info()
