"""Configuration schemas, typed enums, and runtime settings for QuantForge."""

from __future__ import annotations

import os
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class QuantType(str, Enum):
    """Supported llama.cpp GGUF quantization formats."""

    # Unquantized / Baseline
    F32 = "F32"
    F16 = "F16"
    BF16 = "BF16"

    # Legacy uniform quants
    Q4_0 = "Q4_0"
    Q4_1 = "Q4_1"
    Q5_0 = "Q5_0"
    Q5_1 = "Q5_1"
    Q8_0 = "Q8_0"

    # K-Quants (Mixed precision)
    Q2_K = "Q2_K"
    Q3_K_S = "Q3_K_S"
    Q3_K_M = "Q3_K_M"
    Q3_K_L = "Q3_K_L"
    Q4_K_S = "Q4_K_S"
    Q4_K_M = "Q4_K_M"
    Q5_K_S = "Q5_K_S"
    Q5_K_M = "Q5_K_M"
    Q6_K = "Q6_K"

    # Importance matrix quants (I-Quants)
    IQ1_S = "IQ1_S"
    IQ1_M = "IQ1_M"
    IQ2_XXS = "IQ2_XXS"
    IQ2_XS = "IQ2_XS"
    IQ2_S = "IQ2_S"
    IQ2_M = "IQ2_M"
    IQ3_XXS = "IQ3_XXS"
    IQ3_XS = "IQ3_XS"
    IQ3_S = "IQ3_S"
    IQ3_M = "IQ3_M"
    IQ4_XS = "IQ4_XS"
    IQ4_NL = "IQ4_NL"

    # Pass-through
    COPY = "COPY"

    @property
    def is_imatrix_required(self) -> bool:
        """Returns True if this quant requires an importance matrix."""
        return self in {
            QuantType.IQ1_S,
            QuantType.IQ1_M,
            QuantType.IQ2_XXS,
            QuantType.IQ2_XS,
            QuantType.IQ2_S,
            QuantType.IQ2_M,
        }

    @property
    def is_imatrix_recommended(self) -> bool:
        """Returns True if this quant strongly benefits from an importance matrix."""
        return self.is_imatrix_required or self in {
            QuantType.IQ3_XXS,
            QuantType.IQ3_XS,
            QuantType.IQ3_S,
            QuantType.IQ3_M,
            QuantType.IQ4_XS,
            QuantType.IQ4_NL,
            QuantType.Q2_K,
            QuantType.Q3_K_S,
            QuantType.Q3_K_M,
            QuantType.Q4_K_S,
            QuantType.Q4_K_M,
        }

    @property
    def estimated_bpw(self) -> float:
        """Estimated bits per weight (BPW) for file sizing heuristics."""
        bpw_map = {
            QuantType.F32: 32.0,
            QuantType.F16: 16.0,
            QuantType.BF16: 16.0,
            QuantType.Q8_0: 8.5,
            QuantType.Q6_K: 6.56,
            QuantType.Q5_K_M: 5.5,
            QuantType.Q5_K_S: 5.3,
            QuantType.Q5_1: 5.5,
            QuantType.Q5_0: 5.0,
            QuantType.Q4_K_M: 4.5,
            QuantType.Q4_K_S: 4.3,
            QuantType.IQ4_NL: 4.5,
            QuantType.IQ4_XS: 4.25,
            QuantType.Q4_1: 4.5,
            QuantType.Q4_0: 4.2,
            QuantType.IQ3_M: 3.7,
            QuantType.IQ3_S: 3.44,
            QuantType.IQ3_XS: 3.3,
            QuantType.IQ3_XXS: 3.06,
            QuantType.Q3_K_L: 3.82,
            QuantType.Q3_K_M: 3.5,
            QuantType.Q3_K_S: 3.2,
            QuantType.IQ2_M: 2.7,
            QuantType.IQ2_S: 2.5,
            QuantType.IQ2_XS: 2.31,
            QuantType.IQ2_XXS: 2.06,
            QuantType.Q2_K: 2.56,
            QuantType.IQ1_M: 1.75,
            QuantType.IQ1_S: 1.56,
            QuantType.COPY: 16.0,
        }
        return bpw_map.get(self, 4.5)

    @property
    def family(self) -> str:
        """Returns the quant category family name."""
        name = self.value
        if name.startswith("IQ"):
            return "I-Quant"
        if "_K" in name:
            return "K-Quant"
        if name in {"F32", "F16", "BF16", "COPY"}:
            return "Float/Identity"
        return "Legacy Quant"


class JobStatus(str, Enum):
    """Execution status of an asynchronous job."""

    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class JobType(str, Enum):
    """Categorized type of QuantForge task."""

    QUANTIZE = "quantize"
    MATRIX = "matrix"
    IMATRIX = "imatrix"
    PERPLEXITY = "perplexity"
    BENCHMARK = "benchmark"
    PIPELINE = "pipeline"


class QuantizationConfig(BaseModel):
    """Configuration for a single GGUF quantization command."""

    input_path: Path
    output_path: Path
    quant_type: QuantType = QuantType.Q4_K_M
    imatrix_path: Optional[Path] = None
    leave_output_tensor: bool = False
    pure: bool = False
    threads: Optional[int] = None
    tensor_type_overrides: Dict[str, str] = Field(default_factory=dict)
    extra_args: List[str] = Field(default_factory=list)

    @field_validator("input_path", "output_path", "imatrix_path", mode="after")
    @classmethod
    def resolve_path(cls, v: Optional[Path]) -> Optional[Path]:
        if v is not None:
            return v.resolve()
        return None

    def validate_prerequisites(self) -> None:
        """Checks if input model and required imatrix exist."""
        if not self.input_path.is_file():
            raise FileNotFoundError(f"Input model not found: {self.input_path}")
        if self.quant_type.is_imatrix_required and not self.imatrix_path:
            raise ValueError(
                f"Quantization type {self.quant_type.value} requires an importance matrix. "
                "Provide an --imatrix file."
            )
        if self.imatrix_path and not self.imatrix_path.is_file():
            raise FileNotFoundError(f"Imatrix file not found: {self.imatrix_path}")


class MatrixQuantConfig(BaseModel):
    """Configuration for generating a set of quant variants in batch."""

    input_path: Path
    output_dir: Path
    quant_types: List[QuantType] = Field(
        default_factory=lambda: [
            QuantType.Q4_K_M,
            QuantType.Q5_K_M,
            QuantType.Q8_0,
        ]
    )
    imatrix_path: Optional[Path] = None
    leave_output_tensor: bool = False
    pure: bool = False
    threads: Optional[int] = None


class ImatrixConfig(BaseModel):
    """Configuration for importance matrix calibration runs."""

    input_path: Path
    data_path: Path
    output_path: Path
    ctx_size: int = Field(default=2048, ge=128, le=131072)
    chunks: int = Field(default=64, ge=1, le=4096)
    n_gpu_layers: int = Field(default=0, ge=0)
    threads: Optional[int] = None
    extra_args: List[str] = Field(default_factory=list)

    def validate_prerequisites(self) -> None:
        """Checks if model and calibration files exist."""
        if not self.input_path.is_file():
            raise FileNotFoundError(f"Base model not found: {self.input_path}")
        if not self.data_path.is_file():
            raise FileNotFoundError(f"Calibration dataset not found: {self.data_path}")


class PerplexityConfig(BaseModel):
    """Configuration for perplexity evaluation against a test corpus."""

    model_path: Path
    data_path: Path
    ctx_size: int = Field(default=2048, ge=128)
    n_gpu_layers: int = Field(default=0, ge=0)
    batch_size: int = Field(default=512, ge=1)
    threads: Optional[int] = None
    extra_args: List[str] = Field(default_factory=list)


class BenchmarkConfig(BaseModel):
    """Configuration for inference throughput measurement with llama-bench."""

    model_path: Path
    prompt_tokens: int = Field(default=512, ge=1)
    gen_tokens: int = Field(default=128, ge=1)
    n_gpu_layers: int = Field(default=0, ge=0)
    threads: Optional[int] = None
    repetitions: int = Field(default=3, ge=1)


class PipelineRecipe(BaseModel):
    """YAML recipe describing an end-to-end quantization and evaluation run."""

    name: str = "quantforge-pipeline"
    input_model: Path
    output_dir: Path
    calibration: Optional[Dict[str, Any]] = None
    quants: List[QuantType] = Field(
        default_factory=lambda: [QuantType.Q4_K_M, QuantType.Q5_K_M, QuantType.Q8_0]
    )
    evaluation: Optional[Dict[str, Any]] = None
    export_model_card: bool = True


class QuantForgeSettings(BaseSettings):
    """Application-wide configuration and paths."""

    model_config = SettingsConfigDict(
        env_prefix="QUANTFORGE_",
        env_file=".env",
        extra="ignore",
    )

    data_dir: Path = Field(
        default_factory=lambda: Path(os.environ.get("QUANTFORGE_DATA_DIR", Path.home() / ".quantforge"))
    )
    binary_dir: Optional[Path] = None
    db_filename: str = "quantforge.sqlite"
    default_threads: int = Field(default_factory=lambda: max(1, (os.cpu_count() or 4) - 1))
    web_host: str = "127.0.0.1"
    web_port: int = 8000

    @property
    def database_path(self) -> Path:
        """Absolute path to SQLite database."""
        return self.data_dir / self.db_filename

    @property
    def binaries_path(self) -> Path:
        """Directory holding downloaded or detected native executables."""
        if self.binary_dir is not None:
            return self.binary_dir
        return self.data_dir / "bin"

    @property
    def cache_path(self) -> Path:
        """Directory holding temporary files, calibration chunks, and logs."""
        return self.data_dir / "cache"

    def ensure_directories(self) -> None:
        """Creates storage directories if they do not exist."""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.binaries_path.mkdir(parents=True, exist_ok=True)
        self.cache_path.mkdir(parents=True, exist_ok=True)


# Global settings singleton
settings = QuantForgeSettings()
