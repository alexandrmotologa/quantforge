"""Perplexity evaluation engine wrapping llama-perplexity."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional
import uuid

from quantforge.core.binary_manager import BinaryManager
from quantforge.core.config import JobStatus, JobType, PerplexityConfig, settings
from quantforge.core.db import db
from quantforge.core.executor import (
    ExecutionProgress,
    ExecutionResult,
    ProgressCallback,
    SubprocessExecutor,
)


@dataclass
class PerplexityResult:
    """Outcome of a perplexity evaluation run."""

    job_id: str
    model_path: Path
    dataset_path: Path
    success: bool
    perplexity: Optional[float] = None
    perplexity_stderr: Optional[float] = None
    delta_perplexity: Optional[float] = None
    degradation_pct: Optional[float] = None
    duration_seconds: float = 0.0
    error_message: Optional[str] = None
    stdout: str = ""
    stderr: str = ""


class PerplexityEngine:
    """Manages perplexity measurement with llama-perplexity."""

    def __init__(self, binary_manager: Optional[BinaryManager] = None) -> None:
        self.binary_manager = binary_manager or BinaryManager()
        self.executor = SubprocessExecutor()

    def build_command(self, config: PerplexityConfig) -> List[str]:
        """Constructs llama-perplexity invocation command."""
        exe = self.binary_manager.require_binary("perplexity")
        cmd = [
            str(exe),
            "-m", str(config.model_path),
            "-f", str(config.data_path),
            "-c", str(config.ctx_size),
            "-b", str(config.batch_size),
        ]

        if config.n_gpu_layers > 0:
            cmd.extend(["-ngl", str(config.n_gpu_layers)])

        threads = config.threads or settings.default_threads
        if threads:
            cmd.extend(["-t", str(threads)])

        if config.extra_args:
            cmd.extend(config.extra_args)

        return cmd

    async def evaluate(
        self,
        config: PerplexityConfig,
        baseline_ppl: Optional[float] = None,
        job_id: Optional[str] = None,
        on_progress: Optional[ProgressCallback] = None,
    ) -> PerplexityResult:
        """Executes perplexity evaluation and computes degradation relative to baseline."""
        if not config.model_path.is_file():
            raise FileNotFoundError(f"Model not found: {config.model_path}")
        if not config.data_path.is_file():
            raise FileNotFoundError(f"Evaluation dataset not found: {config.data_path}")

        job_uuid = job_id or f"job-ppl-{uuid.uuid4().hex[:8]}"

        db.create_job(
            job_id=job_uuid,
            job_type=JobType.PERPLEXITY.value,
            input_model=str(config.model_path),
        )

        cmd = self.build_command(config)

        async def _internal_progress(p: ExecutionProgress) -> None:
            db.update_job_progress(
                job_id=job_uuid,
                progress_pct=p.percent,
                current_step=f"PPL: {p.metric_value}" if p.metric_value else p.current_item,
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
            err = exec_res.error_message or "Perplexity calculation failed."
            db.finalize_job(
                job_id=job_uuid,
                status=JobStatus.FAILED.value,
                return_code=exec_res.return_code,
                error_message=err,
                full_logs=exec_res.stderr + "\n" + exec_res.stdout,
            )
            return PerplexityResult(
                job_id=job_uuid,
                model_path=config.model_path,
                dataset_path=config.data_path,
                success=False,
                duration_seconds=exec_res.duration_seconds,
                error_message=err,
                stdout=exec_res.stdout,
                stderr=exec_res.stderr,
            )

        final_ppl = exec_res.parsed_metrics.get("final_ppl")
        final_stderr = exec_res.parsed_metrics.get("final_ppl_stderr")

        delta_ppl = None
        degrad_pct = None
        if final_ppl is not None and baseline_ppl is not None and baseline_ppl > 0:
            delta_ppl = round(final_ppl - baseline_ppl, 4)
            degrad_pct = round((delta_ppl / baseline_ppl) * 100.0, 2)

        metrics = {
            "perplexity": final_ppl,
            "perplexity_stderr": final_stderr,
            "delta_perplexity": delta_ppl,
            "degradation_percent": degrad_pct,
            "duration_seconds": exec_res.duration_seconds,
        }

        db.finalize_job(
            job_id=job_uuid,
            status=JobStatus.COMPLETED.value,
            return_code=0,
            metrics=metrics,
            full_logs=exec_res.stderr + "\n" + exec_res.stdout,
        )

        return PerplexityResult(
            job_id=job_uuid,
            model_path=config.model_path,
            dataset_path=config.data_path,
            success=True,
            perplexity=final_ppl,
            perplexity_stderr=final_stderr,
            delta_perplexity=delta_ppl,
            degradation_pct=degrad_pct,
            duration_seconds=exec_res.duration_seconds,
            stdout=exec_res.stdout,
            stderr=exec_res.stderr,
        )
