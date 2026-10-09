"""Tests for SubprocessExecutor with live output streaming and parsing."""

import asyncio
import sys
import pytest

from quantforge.core.executor import ExecutionProgress, SubprocessExecutor


def test_parse_quantize_line():
    executor = SubprocessExecutor()
    line = "[  42/ 291] - blk.0.attn_q.weight: Q4_K_M, size = 12.50 MB -> 3.20 MB"
    prog = executor.parse_line(line)
    assert prog is not None
    assert prog.step == "quantize"
    assert prog.current_index == 42
    assert prog.total_count == 291
    assert prog.current_item == "blk.0.attn_q.weight"
    assert round(prog.percent, 1) == 14.4


def test_parse_imatrix_line():
    executor = SubprocessExecutor()
    line = "computing importance matrix over 64 chunks ... chunk 12/64"
    prog = executor.parse_line(line)
    assert prog is not None
    assert prog.step == "imatrix"
    assert prog.current_index == 12
    assert prog.total_count == 64
    assert prog.percent == 18.75


def test_parse_perplexity_lines():
    executor = SubprocessExecutor()
    line_chunk = "[4] 7.1234"
    prog_chunk = executor.parse_line(line_chunk)
    assert prog_chunk is not None
    assert prog_chunk.step == "perplexity"
    assert prog_chunk.metric_value == 7.1234
    assert prog_chunk.metric_label == "interim_ppl"

    line_final = "final result: PPL = 5.2341 +/- 0.0412"
    prog_final = executor.parse_line(line_final)
    assert prog_final is not None
    assert prog_final.step == "perplexity_final"
    assert prog_final.metric_value == 5.2341
    assert prog_final.metric_label == "final_ppl"


@pytest.mark.asyncio
async def test_execute_real_command():
    executor = SubprocessExecutor()
    captured_lines = []

    def on_line(line: str):
        captured_lines.append(line)

    cmd = [sys.executable, "-c", "import sys; print('Line 1'); print('Line 2'); sys.exit(0)"]
    res = await executor.execute(command=cmd, on_line=on_line)

    assert res.success is True
    assert res.return_code == 0
    assert len(captured_lines) >= 2
    assert "Line 1" in res.stdout
    assert "Line 2" in res.stdout


@pytest.mark.asyncio
async def test_execute_command_error():
    executor = SubprocessExecutor()
    cmd = [sys.executable, "-c", "import sys; sys.stderr.write('Fatal crash\\n'); sys.exit(2)"]
    res = await executor.execute(command=cmd)

    assert res.success is False
    assert res.return_code == 2
    assert "Fatal crash" in res.stderr
    assert "Fatal crash" in res.error_message
