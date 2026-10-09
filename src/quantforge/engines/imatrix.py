"""Importance matrix calibration engine wrapping llama-imatrix."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional
import uuid

from quantforge.core.binary_manager import BinaryManager
from quantforge.core.config import ImatrixConfig, JobStatus, JobType, settings
from quantforge.core.db import db
from quantforge.core.executor import (
    ExecutionProgress,
    ExecutionResult,
    ProgressCallback,
    SubprocessExecutor,
)


@dataclass
class ImatrixResult:
    """Outcome of an importance matrix calibration run."""

    job_id: str
    output_path: Path
    dataset_path: Path
    success: bool
    duration_seconds: float
    file_size_bytes: int
    chunks_processed: int
    error_message: Optional[str] = None
    stdout: str = ""
    stderr: str = ""


class ImatrixEngine:
    """Manages importance matrix computation with llama-imatrix."""

    DEFAULT_CALIBRATION_PROMPTS = [
        "In mathematics, a topological space is a set endowed with a structure, called a topology.",
        "The quick brown fox jumps over the lazy dog. Programming in Python emphasizes readability.",
        "Differential equations describe the relationship between an unknown function and its derivatives.",
        "Neural networks learn representations through backpropagation and stochastic gradient descent.",
        "Hardware acceleration via GPU architectures enables parallel matrix multiplication at scale.",
    ]

    def __init__(self, binary_manager: Optional[BinaryManager] = None) -> None:
        self.binary_manager = binary_manager or BinaryManager()
        self.executor = SubprocessExecutor()

    def build_command(self, config: ImatrixConfig) -> List[str]:
        """Constructs llama-imatrix CLI arguments."""
        exe = self.binary_manager.require_binary("imatrix")
        cmd = [
            str(exe),
            "-m", str(config.input_path),
            "-f", str(config.data_path),
            "-o", str(config.output_path),
            "-c", str(config.ctx_size),
            "--chunks", str(config.chunks),
        ]

        if str(config.output_path).lower().endswith(".dat"):
            cmd.extend(["--output-format", "dat"])

        if config.n_gpu_layers > 0:
            cmd.extend(["-ngl", str(config.n_gpu_layers)])

        threads = config.threads or settings.default_threads
        if threads:
            cmd.extend(["-t", str(threads)])

        if config.extra_args:
            cmd.extend(config.extra_args)

        return cmd

    def generate_sample_calibration_corpus(self, target_path: Path, repeats: int = 40) -> Path:
        """Generates a starter calibration text file if none is provided."""
        target_path.parent.mkdir(parents=True, exist_ok=True)
        content = "\n\n".join(self.DEFAULT_CALIBRATION_PROMPTS * repeats)
        target_path.write_text(content, encoding="utf-8")
        return target_path

    async def calibrate(
        self,
        config: ImatrixConfig,
        job_id: Optional[str] = None,
        on_progress: Optional[ProgressCallback] = None,
    ) -> ImatrixResult:
        """Runs the imatrix computation pipeline."""
        config.validate_prerequisites()
        config.output_path.parent.mkdir(parents=True, exist_ok=True)

        job_uuid = job_id or f"job-imatrix-{uuid.uuid4().hex[:8]}"

        db.create_job(
            job_id=job_uuid,
            job_type=JobType.IMATRIX.value,
            input_model=str(config.input_path),
            output_model=str(config.output_path),
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

        if not exec_res.success or not config.output_path.is_file():
            err = exec_res.error_message or "Imatrix output file was not produced."
            db.finalize_job(
                job_id=job_uuid,
                status=JobStatus.FAILED.value,
                return_code=exec_res.return_code,
                error_message=err,
                full_logs=exec_res.stderr + "\n" + exec_res.stdout,
            )
            return ImatrixResult(
                job_id=job_uuid,
                output_path=config.output_path,
                dataset_path=config.data_path,
                success=False,
                duration_seconds=exec_res.duration_seconds,
                file_size_bytes=0,
                chunks_processed=0,
                error_message=err,
                stdout=exec_res.stdout,
                stderr=exec_res.stderr,
            )

        file_size = config.output_path.stat().st_size
        metrics = {
            "output_size_bytes": file_size,
            "chunks_processed": config.chunks,
            "duration_seconds": exec_res.duration_seconds,
        }

        db.finalize_job(
            job_id=job_uuid,
            status=JobStatus.COMPLETED.value,
            return_code=0,
            metrics=metrics,
            full_logs=exec_res.stderr + "\n" + exec_res.stdout,
        )

        return ImatrixResult(
            job_id=job_uuid,
            output_path=config.output_path,
            dataset_path=config.data_path,
            success=True,
            duration_seconds=exec_res.duration_seconds,
            file_size_bytes=file_size,
            chunks_processed=config.chunks,
            stdout=exec_res.stdout,
            stderr=exec_res.stderr,
        )
