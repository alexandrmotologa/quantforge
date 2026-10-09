"""Quantization engine executing llama-quantize commands and matrix batches."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional
import uuid

from quantforge.core.binary_manager import BinaryManager
from quantforge.core.config import (
    JobStatus,
    JobType,
    MatrixQuantConfig,
    QuantizationConfig,
    QuantType,
    settings,
)
from quantforge.core.db import db
from quantforge.core.executor import (
    ExecutionProgress,
    ExecutionResult,
    ProgressCallback,
    SubprocessExecutor,
)
from quantforge.formats.gguf_reader import GGUFReader


@dataclass
class QuantizationResult:
    """Outcome and metrics of a completed quantization process."""

    job_id: str
    input_path: Path
    output_path: Path
    quant_type: QuantType
    success: bool
    duration_seconds: float
    original_size_bytes: int
    quantized_size_bytes: int
    compression_ratio: float
    tensor_count: int = 0
    error_message: Optional[str] = None
    stdout: str = ""
    stderr: str = ""


class QuantizeEngine:
    """Orchestrates llama-quantize execution with real-time feedback."""

    def __init__(self, binary_manager: Optional[BinaryManager] = None) -> None:
        self.binary_manager = binary_manager or BinaryManager()
        self.executor = SubprocessExecutor()

    def build_command(self, config: QuantizationConfig) -> List[str]:
        """Constructs llama-quantize CLI invocation arguments."""
        exe = self.binary_manager.require_binary("quantize")
        cmd = [str(exe)]

        # Apply flags before or with positional arguments
        if config.imatrix_path:
            cmd.extend(["--imatrix", str(config.imatrix_path)])
        if config.leave_output_tensor:
            cmd.append("--leave-output-tensor")
        if config.pure:
            cmd.append("--pure")

        # Tensor type overrides
        for tensor_pat, target_q in config.tensor_type_overrides.items():
            cmd.extend(["--tensor-type", f"{tensor_pat}={target_q}"])

        # Positional arguments: input, output, type, [threads]
        cmd.append(str(config.input_path))
        cmd.append(str(config.output_path))
        cmd.append(config.quant_type.value)

        threads = config.threads or settings.default_threads
        if threads:
            cmd.append(str(threads))

        # Additional user arguments
        if config.extra_args:
            cmd.extend(config.extra_args)

        return cmd

    async def quantize(
        self,
        config: QuantizationConfig,
        job_id: Optional[str] = None,
        on_progress: Optional[ProgressCallback] = None,
    ) -> QuantizationResult:
        """Executes a single quantization job."""
        config.validate_prerequisites()
        config.output_path.parent.mkdir(parents=True, exist_ok=True)

        job_uuid = job_id or f"job-q-{uuid.uuid4().hex[:8]}"
        orig_size = config.input_path.stat().st_size

        # Register in database
        db.create_job(
            job_id=job_uuid,
            job_type=JobType.QUANTIZE.value,
            input_model=str(config.input_path),
            output_model=str(config.output_path),
            quant_type=config.quant_type.value,
        )

        cmd = self.build_command(config)

        async def _internal_progress(p: ExecutionProgress) -> None:
            db.update_job_progress(
                job_id=job_uuid,
                progress_pct=p.percent,
                current_step=p.current_item,
                log_line=p.raw_line,
            )
            if on_progress:
                res = on_progress(p)
                if asyncio.iscoroutine(res):
                    await res

        exec_res = await self.executor.execute(
            command=cmd,
            on_progress=_internal_progress,
        )

        if not exec_res.success:
            db.finalize_job(
                job_id=job_uuid,
                status=JobStatus.FAILED.value,
                return_code=exec_res.return_code,
                error_message=exec_res.error_message,
                full_logs=exec_res.stderr + "\n" + exec_res.stdout,
            )
            return QuantizationResult(
                job_id=job_uuid,
                input_path=config.input_path,
                output_path=config.output_path,
                quant_type=config.quant_type,
                success=False,
                duration_seconds=exec_res.duration_seconds,
                original_size_bytes=orig_size,
                quantized_size_bytes=0,
                compression_ratio=0.0,
                error_message=exec_res.error_message,
                stdout=exec_res.stdout,
                stderr=exec_res.stderr,
            )

        # Verify output exists and is a valid GGUF file
        if not config.output_path.is_file():
            err = f"Expected output file not found on disk: {config.output_path}"
            db.finalize_job(
                job_id=job_uuid,
                status=JobStatus.FAILED.value,
                return_code=1,
                error_message=err,
            )
            return QuantizationResult(
                job_id=job_uuid,
                input_path=config.input_path,
                output_path=config.output_path,
                quant_type=config.quant_type,
                success=False,
                duration_seconds=exec_res.duration_seconds,
                original_size_bytes=orig_size,
                quantized_size_bytes=0,
                compression_ratio=0.0,
                error_message=err,
            )

        quant_size = config.output_path.stat().st_size
        comp_ratio = orig_size / quant_size if quant_size > 0 else 0.0

        # Validate with GGUFReader
        tensor_count = 0
        try:
            reader = GGUFReader(config.output_path)
            model_info = reader.read_model_info(load_tensors=False)
            tensor_count = model_info.tensor_count
        except Exception:
            pass

        metrics = {
            "original_size_mb": round(orig_size / (1024 * 1024), 2),
            "quantized_size_mb": round(quant_size / (1024 * 1024), 2),
            "compression_ratio": round(comp_ratio, 2),
            "duration_seconds": exec_res.duration_seconds,
            "tensor_count": tensor_count,
        }

        db.finalize_job(
            job_id=job_uuid,
            status=JobStatus.COMPLETED.value,
            return_code=0,
            metrics=metrics,
            full_logs=exec_res.stderr + "\n" + exec_res.stdout,
        )

        return QuantizationResult(
            job_id=job_uuid,
            input_path=config.input_path,
            output_path=config.output_path,
            quant_type=config.quant_type,
            success=True,
            duration_seconds=exec_res.duration_seconds,
            original_size_bytes=orig_size,
            quantized_size_bytes=quant_size,
            compression_ratio=round(comp_ratio, 2),
            tensor_count=tensor_count,
            stdout=exec_res.stdout,
            stderr=exec_res.stderr,
        )

    async def quantize_matrix(
        self,
        config: MatrixQuantConfig,
        on_item_finish: Optional[Callable[[QuantizationResult], None]] = None,
    ) -> List[QuantizationResult]:
        """Quantizes an input model into multiple formats sequentially."""
        if not config.input_path.is_file():
            raise FileNotFoundError(f"Input model not found: {config.input_path}")

        config.output_dir.mkdir(parents=True, exist_ok=True)
        base_name = config.input_path.stem
        # Strip existing quant tags if present (e.g. model-f16 -> model)
        for tag in ["-f16", "-fp16", "-f32", "-fp32"]:
            if base_name.lower().endswith(tag):
                base_name = base_name[: -len(tag)]

        results: List[QuantizationResult] = []
        for q_type in config.quant_types:
            # Check imatrix requirements
            imatrix_to_use = config.imatrix_path
            if q_type.is_imatrix_required and not imatrix_to_use:
                # Skip if required imatrix is missing
                continue

            out_file = config.output_dir / f"{base_name}-{q_type.value.lower()}.gguf"
            single_cfg = QuantizationConfig(
                input_path=config.input_path,
                output_path=out_file,
                quant_type=q_type,
                imatrix_path=imatrix_to_use,
                leave_output_tensor=config.leave_output_tensor,
                pure=config.pure,
                threads=config.threads,
            )

            res = await self.quantize(single_cfg)
            results.append(res)
            if on_item_finish:
                on_item_finish(res)

        return results
