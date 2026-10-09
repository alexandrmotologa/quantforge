"""Subprocess streaming executor with real-time log parsing and metric extraction."""

from __future__ import annotations

import asyncio
import os
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Coroutine, Dict, List, Optional, Union


@dataclass
class ExecutionProgress:
    """Parsed progress snapshot from a live process."""

    step: str = ""
    percent: float = 0.0
    current_index: int = 0
    total_count: int = 0
    current_item: str = ""
    metric_value: Optional[float] = None
    metric_label: str = ""
    raw_line: str = ""


@dataclass
class ExecutionResult:
    """Final outcome of an executed command."""

    command: List[str]
    return_code: int
    success: bool
    stdout: str
    stderr: str
    duration_seconds: float
    parsed_metrics: Dict[str, Any] = field(default_factory=dict)
    error_message: Optional[str] = None


ProgressCallback = Callable[[ExecutionProgress], Union[None, Coroutine[Any, Any, None]]]
LineCallback = Callable[[str], Union[None, Coroutine[Any, Any, None]]]


class SubprocessExecutor:
    """Runs commands asynchronously with live line streaming and pattern parsing."""

    # Regex patterns for llama.cpp tools
    RE_QUANT_PROGRESS = re.compile(
        r"\[\s*(?P<curr>\d+)\s*/\s*(?P<total>\d+)\s*\]\s*-\s*(?P<tensor>[^:]+):\s*(?P<type>[^,]+)"
    )
    RE_PPL_CHUNK = re.compile(
        r"\[(?P<chunk>\d+)\]\s*(?P<ppl>[0-9\.]+)"
    )
    RE_PPL_FINAL = re.compile(
        r"final result:\s*PPL\s*=\s*(?P<ppl>[0-9\.]+)\s*\+/-\s*(?P<stderr>[0-9\.]+)"
    )
    RE_IMATRIX_CHUNK = re.compile(
        r"computing importance matrix.*?chunk\s*(?P<curr>\d+)\s*/\s*(?P<total>\d+)",
        re.IGNORECASE,
    )

    def __init__(self) -> None:
        self._process: Optional[asyncio.subprocess.Process] = None
        self._cancelled: bool = False

    def cancel(self) -> None:
        """Signals running process to terminate."""
        self._cancelled = True
        if self._process:
            try:
                self._process.terminate()
            except ProcessLookupError:
                pass

    def parse_line(self, line: str) -> Optional[ExecutionProgress]:
        """Inspects line for known progress signatures across tools."""
        clean = line.strip()

        # 1. llama-quantize: [ 42/291] - blk.0.attn_q.weight: Q4_K_M
        m_quant = self.RE_QUANT_PROGRESS.search(clean)
        if m_quant:
            curr = int(m_quant.group("curr"))
            total = int(m_quant.group("total"))
            pct = (curr / total) * 100.0 if total > 0 else 0.0
            return ExecutionProgress(
                step="quantize",
                percent=round(pct, 2),
                current_index=curr,
                total_count=total,
                current_item=m_quant.group("tensor").strip(),
                raw_line=clean,
            )

        # 2. llama-imatrix: computing importance matrix ... chunk 4/64
        m_imatrix = self.RE_IMATRIX_CHUNK.search(clean)
        if m_imatrix:
            curr = int(m_imatrix.group("curr"))
            total = int(m_imatrix.group("total"))
            pct = (curr / total) * 100.0 if total > 0 else 0.0
            return ExecutionProgress(
                step="imatrix",
                percent=round(pct, 2),
                current_index=curr,
                total_count=total,
                current_item=f"chunk {curr}/{total}",
                raw_line=clean,
            )

        # 3. llama-perplexity: [4] 6.1243
        m_ppl = self.RE_PPL_CHUNK.search(clean)
        if m_ppl:
            chunk = int(m_ppl.group("chunk"))
            val = float(m_ppl.group("ppl"))
            return ExecutionProgress(
                step="perplexity",
                percent=0.0,
                current_index=chunk,
                metric_value=val,
                metric_label="interim_ppl",
                raw_line=clean,
            )

        # 4. llama-perplexity final: final result: PPL = 5.234 +/- 0.041
        m_final = self.RE_PPL_FINAL.search(clean)
        if m_final:
            val = float(m_final.group("ppl"))
            return ExecutionProgress(
                step="perplexity_final",
                percent=100.0,
                metric_value=val,
                metric_label="final_ppl",
                raw_line=clean,
            )

        return None

    async def execute(
        self,
        command: List[str],
        cwd: Optional[Path] = None,
        env: Optional[Dict[str, str]] = None,
        timeout: Optional[float] = None,
        on_line: Optional[LineCallback] = None,
        on_progress: Optional[ProgressCallback] = None,
    ) -> ExecutionResult:
        """Executes binary asynchronously, streaming output lines and parsing progress."""
        self._cancelled = False
        start_time = time.monotonic()
        stdout_lines: List[str] = []
        stderr_lines: List[str] = []
        parsed_metrics: Dict[str, Any] = {}

        process_env = os.environ.copy()
        if env:
            process_env.update(env)

        # Launch process
        self._process = await asyncio.create_subprocess_exec(
            *command,
            cwd=str(cwd) if cwd else None,
            env=process_env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        async def read_stream(
            stream: asyncio.StreamReader,
            sink: List[str],
            is_stderr: bool = False,
        ) -> None:
            while True:
                line_bytes = await stream.readline()
                if not line_bytes:
                    break
                line = line_bytes.decode(errors="replace").rstrip("\r\n")
                sink.append(line)

                # Parse metrics from both streams (llama.cpp prints progress to stderr)
                prog = self.parse_line(line)
                if prog:
                    if prog.metric_label and prog.metric_value is not None:
                        parsed_metrics[prog.metric_label] = prog.metric_value
                    if on_progress:
                        res = on_progress(prog)
                        if asyncio.iscoroutine(res):
                            await res

                if on_line:
                    res = on_line(line)
                    if asyncio.iscoroutine(res):
                        await res

        try:
            if timeout:
                await asyncio.wait_for(
                    asyncio.gather(
                        read_stream(self._process.stdout, stdout_lines, False),  # type: ignore
                        read_stream(self._process.stderr, stderr_lines, True),   # type: ignore
                        self._process.wait(),
                    ),
                    timeout=timeout,
                )
            else:
                await asyncio.gather(
                    read_stream(self._process.stdout, stdout_lines, False),      # type: ignore
                    read_stream(self._process.stderr, stderr_lines, True),       # type: ignore
                    self._process.wait(),
                )
        except asyncio.TimeoutError:
            self.cancel()
            raise TimeoutError(f"Process timed out after {timeout} seconds: {' '.join(command)}")
        except asyncio.CancelledError:
            self.cancel()
            raise

        duration = time.monotonic() - start_time
        return_code = self._process.returncode if self._process.returncode is not None else -1

        # Check for final perplexity in full stream if not yet captured
        if "final_ppl" not in parsed_metrics:
            combined = "\n".join(stdout_lines + stderr_lines)
            m_final = self.RE_PPL_FINAL.search(combined)
            if m_final:
                parsed_metrics["final_ppl"] = float(m_final.group("ppl"))
                parsed_metrics["final_ppl_stderr"] = float(m_final.group("stderr"))

        err_msg = None
        if return_code != 0 and not self._cancelled:
            # Capture last few lines of stderr/stdout for context
            last_err = [l for l in (stderr_lines + stdout_lines) if l.strip()]
            err_msg = "\n".join(last_err[-6:]) if last_err else f"Process exited with code {return_code}"

        return ExecutionResult(
            command=command,
            return_code=return_code,
            success=(return_code == 0),
            stdout="\n".join(stdout_lines),
            stderr="\n".join(stderr_lines),
            duration_seconds=round(duration, 3),
            parsed_metrics=parsed_metrics,
            error_message=err_msg,
        )
