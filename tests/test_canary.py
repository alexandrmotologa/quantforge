"""Tests for Canary quality degeneration benchmark engine."""

from pathlib import Path
import pytest
from unittest.mock import AsyncMock, MagicMock

from quantforge.core.binary_manager import BinaryInfo, BinaryManager
from quantforge.core.executor import ExecutionResult, SubprocessExecutor
from quantforge.engines.canary import (
    CanaryEngine,
    STANDARD_CANARY_TESTS,
    _validate_json,
    _validate_python_code,
    _validate_math_logic,
    _calculate_4gram_diversity,
)


def test_canary_validators():
    assert _validate_json('{"status": "ok", "code": 200, "tags": ["ai"]}') is True
    assert _validate_json("Sorry, I cannot do that.") is False

    valid_code = "```python\ndef fibonacci(n):\n    return n if n <= 1 else fibonacci(n-1) + fibonacci(n-2)\n```"
    assert _validate_python_code(valid_code) is True
    assert _validate_python_code("def broken(n: return 0") is False

    assert _validate_math_logic("The farmer still has 8 sheep remaining.") is True
    assert _validate_math_logic("There are 7 sheep left.") is False

    diverse_text = "Photosynthesis is the biochemical pathway where green plants synthesize sugars using sunlight and chlorophyll."
    repetitive_text = "the same word the same word the same word the same word the same word"
    assert _calculate_4gram_diversity(diverse_text) > _calculate_4gram_diversity(repetitive_text)


@pytest.mark.asyncio
async def test_canary_benchmark_run(tmp_path: Path):
    mock_bin_mgr = MagicMock(spec=BinaryManager)
    mock_bin_mgr.find_binary.return_value = BinaryInfo(
        name="cli",
        found=True,
        path=Path("/bin/llama-cli"),
        is_executable=True,
    )

    mock_executor = MagicMock(spec=SubprocessExecutor)

    async def fake_run(cmd, *args, **kwargs):
        prompt = cmd[cmd.index("-p") + 1]
        if "JSON" in prompt:
            out = '{"status": "ok", "code": 200, "tags": ["ai", "quant"]}'
        elif "fibonacci" in prompt:
            out = "```python\ndef fibonacci(n):\n    return n\n```"
        elif "sheep" in prompt:
            out = "There are 8 sheep left."
        else:
            out = "Plants convert sunlight into chemical energy via chlorophyll."

        return ExecutionResult(
            command=cmd,
            return_code=0,
            success=True,
            stdout=out,
            stderr="",
            duration_seconds=0.1,
        )

    mock_executor.run = AsyncMock(side_effect=fake_run)

    dummy_model = tmp_path / "model.gguf"
    dummy_model.touch()

    engine = CanaryEngine(binary_manager=mock_bin_mgr, executor=mock_executor)
    report = await engine.run_benchmark(dummy_model)

    assert report.total_tests == len(STANDARD_CANARY_TESTS)
    assert report.passed_tests == len(STANDARD_CANARY_TESTS)
    assert report.pass_rate_pct == 100.0
    assert report.repetition_ratio > 0.5
    assert len(report.summary()) > 0
