"""Pure Python GGUF header, metadata, and tensor info parser."""

from __future__ import annotations

import os
import struct
from dataclasses import dataclass, field
from enum import IntEnum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


class GGUFValueType(IntEnum):
    """GGUF metadata value data types according to specification."""

    UINT8 = 0
    INT8 = 1
    UINT16 = 2
    INT16 = 3
    UINT32 = 4
    INT32 = 5
    FLOAT32 = 6
    BOOL = 7
    STRING = 8
    ARRAY = 9
    UINT64 = 10
    INT64 = 11
    FLOAT64 = 12


class GGMLType(IntEnum):
    """GGML tensor quantization numerical types."""

    F32 = 0
    F16 = 1
    Q4_0 = 2
    Q4_1 = 3
    Q5_0 = 6
    Q5_1 = 7
    Q8_0 = 8
    Q8_1 = 9
    Q2_K = 10
    Q3_K = 11
    Q4_K = 12
    Q5_K = 13
    Q6_K = 14
    Q8_K = 15
    IQ2_XXS = 16
    IQ2_XS = 17
    IQ3_XXS = 18
    IQ1_S = 19
    IQ4_NL = 20
    IQ3_S = 21
    IQ2_S = 22
    IQ4_XS = 23
    I8 = 24
    I16 = 25
    I32 = 26
    I64 = 27
    F64 = 28
    IQ1_M = 29
    BF16 = 30
    Q4_0_4_4 = 31
    Q4_0_4_8 = 32
    Q4_0_8_8 = 33
    TQ1_0 = 34
    TQ2_0 = 35


@dataclass
class TensorInfo:
    """Descriptor for an individual tensor within the GGUF file."""

    name: str
    n_dimensions: int
    dimensions: List[int]
    tensor_type: int
    type_name: str
    offset: int

    @property
    def element_count(self) -> int:
        count = 1
        for dim in self.dimensions:
            count *= dim
        return count


@dataclass
class GGUFModelInfo:
    """Parsed high-level model metadata from GGUF format."""

    file_path: Path
    file_size_bytes: int
    version: int
    tensor_count: int
    metadata_kv_count: int
    architecture: str
    context_length: Optional[int] = None
    embedding_length: Optional[int] = None
    block_count: Optional[int] = None
    quant_version: Optional[int] = None
    dominant_quant: Optional[str] = None
    estimated_parameters: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    tensors: List[TensorInfo] = field(default_factory=list)


class GGUFReader:
    """Reads and validates binary GGUF files directly."""

    GGUF_MAGIC = b"GGUF"

    def __init__(self, file_path: Path) -> None:
        self.file_path = Path(file_path).resolve()
        if not self.file_path.is_file():
            raise FileNotFoundError(f"GGUF file does not exist: {self.file_path}")

    def read_model_info(self, load_tensors: bool = True) -> GGUFModelInfo:
        """Parses headers, metadata key-values, and tensor list from the file."""
        file_size = self.file_path.stat().st_size

        with open(self.file_path, "rb") as f:
            magic = f.read(4)
            if magic != self.GGUF_MAGIC:
                raise ValueError(
                    f"Invalid GGUF magic header in {self.file_path}. Expected {self.GGUF_MAGIC!r}, got {magic!r}"
                )

            version = struct.unpack("<I", f.read(4))[0]
            if version not in (2, 3):
                raise ValueError(f"Unsupported GGUF version: {version}. Expected 2 or 3.")

            tensor_count = struct.unpack("<Q", f.read(8))[0]
            metadata_kv_count = struct.unpack("<Q", f.read(8))[0]

            metadata: Dict[str, Any] = {}
            for _ in range(metadata_kv_count):
                key = self._read_string(f)
                val_type = struct.unpack("<I", f.read(4))[0]
                val = self._read_value(f, val_type)
                metadata[key] = val

            architecture = str(metadata.get("general.architecture", "unknown"))
            ctx_len = metadata.get(f"{architecture}.context_length") or metadata.get("context_length")
            emb_len = metadata.get(f"{architecture}.embedding_length")
            block_count = metadata.get(f"{architecture}.block_count")
            quant_version = metadata.get("general.quantization_version")

            tensors: List[TensorInfo] = []
            type_distribution: Dict[str, int] = {}
            total_elements = 0

            if load_tensors and tensor_count > 0:
                for _ in range(tensor_count):
                    t_name = self._read_string(f)
                    n_dims = struct.unpack("<I", f.read(4))[0]
                    dims = [struct.unpack("<Q", f.read(8))[0] for _ in range(n_dims)]
                    t_type = struct.unpack("<I", f.read(4))[0]
                    offset = struct.unpack("<Q", f.read(8))[0]

                    type_name = self._get_type_name(t_type)
                    t_info = TensorInfo(
                        name=t_name,
                        n_dimensions=n_dims,
                        dimensions=dims,
                        tensor_type=t_type,
                        type_name=type_name,
                        offset=offset,
                    )
                    tensors.append(t_info)
                    total_elements += t_info.element_count
                    type_distribution[type_name] = type_distribution.get(type_name, 0) + t_info.element_count

            # Determine dominant quantization type by weight count
            dominant_quant = None
            if type_distribution:
                dominant_quant = max(type_distribution.items(), key=lambda kv: kv[1])[0]

            return GGUFModelInfo(
                file_path=self.file_path,
                file_size_bytes=file_size,
                version=version,
                tensor_count=tensor_count,
                metadata_kv_count=metadata_kv_count,
                architecture=architecture,
                context_length=ctx_len,
                embedding_length=emb_len,
                block_count=block_count,
                quant_version=quant_version,
                dominant_quant=dominant_quant,
                estimated_parameters=total_elements,
                metadata=metadata,
                tensors=tensors,
            )

    def _read_string(self, f) -> str:
        length = struct.unpack("<Q", f.read(8))[0]
        data = f.read(length)
        return data.decode("utf-8", errors="replace")

    def _read_value(self, f, val_type: int) -> Any:
        if val_type == GGUFValueType.UINT8:
            return struct.unpack("<B", f.read(1))[0]
        elif val_type == GGUFValueType.INT8:
            return struct.unpack("<b", f.read(1))[0]
        elif val_type == GGUFValueType.UINT16:
            return struct.unpack("<H", f.read(2))[0]
        elif val_type == GGUFValueType.INT16:
            return struct.unpack("<h", f.read(2))[0]
        elif val_type == GGUFValueType.UINT32:
            return struct.unpack("<I", f.read(4))[0]
        elif val_type == GGUFValueType.INT32:
            return struct.unpack("<i", f.read(4))[0]
        elif val_type == GGUFValueType.FLOAT32:
            return struct.unpack("<f", f.read(4))[0]
        elif val_type == GGUFValueType.BOOL:
            return struct.unpack("<?", f.read(1))[0]
        elif val_type == GGUFValueType.STRING:
            return self._read_string(f)
        elif val_type == GGUFValueType.UINT64:
            return struct.unpack("<Q", f.read(8))[0]
        elif val_type == GGUFValueType.INT64:
            return struct.unpack("<q", f.read(8))[0]
        elif val_type == GGUFValueType.FLOAT64:
            return struct.unpack("<d", f.read(8))[0]
        elif val_type == GGUFValueType.ARRAY:
            sub_type = struct.unpack("<I", f.read(4))[0]
            arr_len = struct.unpack("<Q", f.read(8))[0]
            # Cap array parsing to avoid huge memory consumption on tokenizers
            max_read = min(arr_len, 500)
            items = [self._read_value(f, sub_type) for _ in range(max_read)]
            if arr_len > max_read:
                # Skip remainder
                for _ in range(arr_len - max_read):
                    self._skip_value(f, sub_type)
            return items
        else:
            raise ValueError(f"Unknown GGUF value type code: {val_type}")

    def _skip_value(self, f, val_type: int) -> None:
        size_map = {
            GGUFValueType.UINT8: 1,
            GGUFValueType.INT8: 1,
            GGUFValueType.UINT16: 2,
            GGUFValueType.INT16: 2,
            GGUFValueType.UINT32: 4,
            GGUFValueType.INT32: 4,
            GGUFValueType.FLOAT32: 4,
            GGUFValueType.BOOL: 1,
            GGUFValueType.UINT64: 8,
            GGUFValueType.INT64: 8,
            GGUFValueType.FLOAT64: 8,
        }
        if val_type in size_map:
            f.seek(size_map[val_type], os.SEEK_CUR)
        elif val_type == GGUFValueType.STRING:
            length = struct.unpack("<Q", f.read(8))[0]
            f.seek(length, os.SEEK_CUR)
        else:
            # Fallback read
            self._read_value(f, val_type)

    def _get_type_name(self, type_id: int) -> str:
        try:
            return GGMLType(type_id).name
        except ValueError:
            return f"TYPE_{type_id}"
