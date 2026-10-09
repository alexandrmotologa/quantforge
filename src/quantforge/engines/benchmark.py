"""Inference speed and throughput benchmarking engine wrapping llama-bench."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional
import uuid

from quantforge.core.binary_manager import BinaryManager
from quantforge.core.config import BenchmarkConfig, JobStatus, JobType, settings
from quantforge.core.db import db
from quantforge.core.executor import ExecutionResult, SubprocessExecutor


@dataclass
class BenchmarkResult:
    """Speed profiling outcome from llama-bench."""

    job_id: str
    model_path: Path
    success: bool
    prompt_processing_tok_s: Optional[float] = None
    text_generation_tok_s: Optional[float] = None
    duration_seconds: float = 0.0
    error_message: Optional[str] = None
    stdout: str = ""
    stderr: str = ""


class BenchmarkEngine:
    """Manages inference speed profiling with llama-bench."""

    # Matches table row: | model | ... | pp512 | tg128 | ... |
    RE_SPEED = re.compile(
        r"\|\s*(?P<pp>[0-9\.]+)\s*\+/-\s*[0-9\.]+\s*\|\s*(?P<tg>[0-9\.]+)\s*\+/-",
        re.IGNORECASE,
    )

    def __init__(self, binary_manager: Optional[BinaryManager] = None) -> None:
        self.binary_manager = binary_manager or BinaryManager()
        self.executor = SubprocessExecutor()

    def build_command(self, config: BenchmarkConfig) -> List[str]:
        """Constructs llama-bench invocation command."""
        exe = self.binary_manager.require_binary("bench")
        cmd = [
            str(exe),
            "-m", str(config.model_path),
            "-p", str(config.prompt_tokens),
            "-n", str(config.gen_tokens),
            "-r", str(config.repetitions),
        ]

        if config.n_gpu_layers > 0:
            cmd.extend(["-ngl", str(config.n_gpu_layers)])

        threads = config.threads or settings.default_threads
        if threads:
            cmd.extend(["-t", str(threads)])

        return cmd

    async def benchmark(
        self,
        config: BenchmarkConfig,
        job_id: Optional[str] = None,
    ) -> BenchmarkResult:
        """Executes throughput benchmarking."""
        if not config.model_path.is_file():
            raise FileNotFoundError(f"Model not found: {config.model_path}")

        job_uuid = job_id or f"job-bench-{uuid.uuid4().hex[:8]}"

        db.create_job(
            job_id=job_uuid,
            job_type=JobType.BENCHMARK.value,
            input_model=str(config.model_path),
        )

        cmd = self.build_command(config)
        exec_res = await self.executor.execute(command=cmd)

        if not exec_res.success:
            err = exec_res.error_message or "Benchmark execution failed."
            db.finalize_job(
                job_id=job_uuid,
                status=JobStatus.FAILED.value,
                return_code=exec_res.return_code,
                error_message=err,
                full_logs=exec_res.stderr + "\n" + exec_res.stdout,
            )
            return BenchmarkResult(
                job_id=job_uuid,
                model_path=config.model_path,
                success=False,
                duration_seconds=exec_res.duration_seconds,
                error_message=err,
                stdout=exec_res.stdout,
                stderr=exec_res.stderr,
            )

        # Parse speed from stdout table
        pp_speed = None
        tg_speed = None
        for line in exec_res.stdout.splitlines():
            m = self.RE_SPEED.search(line)
            if m:
                pp_speed = float(m.group("pp"))
                tg_speed = float(m.group("tg"))
                break

        metrics = {
            "prompt_processing_tok_s": pp_speed,
            "text_generation_tok_s": tg_speed,
            "duration_seconds": exec_res.duration_seconds,
        }

        db.finalize_job(
            job_id=job_uuid,
            status=JobStatus.COMPLETED.value,
            return_code=0,
            metrics=metrics,
            full_logs=exec_res.stderr + "\n" + exec_res.stdout,
        )

        return BenchmarkResult(
            job_id=job_uuid,
            model_path=config.model_path,
            success=True,
            prompt_processing_tok_s=pp_speed,
            text_generation_tok_s=tg_speed,
            duration_seconds=exec_res.duration_seconds,
            stdout=exec_res.stdout,
            stderr=exec_res.stderr,
        )
