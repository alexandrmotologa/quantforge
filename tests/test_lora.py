"""Tests for LoRA merge engine."""

from pathlib import Path
import pytest
from unittest.mock import AsyncMock, MagicMock

from quantforge.core.binary_manager import BinaryInfo, BinaryManager
from quantforge.core.executor import ExecutionResult, SubprocessExecutor
from quantforge.engines.lora import LoRAAdapter, LoRAMergeEngine


def test_build_command_with_scaled_adapters(tmp_path: Path):
    mock_bin_mgr = MagicMock(spec=BinaryManager)
    mock_bin_mgr.find_binary.return_value = BinaryInfo(
        name="lora",
        found=True,
        path=Path("/bin/llama-export-lora"),
        is_executable=True,
    )

    engine = LoRAMergeEngine(binary_manager=mock_bin_mgr)
    base_model = tmp_path / "base.gguf"
    base_model.touch()
    out_model = tmp_path / "merged.gguf"
    adapter1 = tmp_path / "adapter1.bin"
    adapter1.touch()
    adapter2 = tmp_path / "adapter2.bin"
    adapter2.touch()

    cmd = engine.build_command(
        base_model=base_model,
        output_path=out_model,
        adapters=[
            LoRAAdapter(path=adapter1, scale=1.0),
            LoRAAdapter(path=adapter2, scale=0.75),
        ],
        threads=4,
    )

    assert cmd[0] == str(Path("/bin/llama-export-lora"))
    assert "-m" in cmd
    assert "-o" in cmd
    assert "--lora" in cmd
    assert "--lora-scaled" in cmd
    assert "0.75" in cmd
    assert "-t" in cmd
    assert "4" in cmd


@pytest.mark.asyncio
async def test_lora_merge_execution(tmp_path: Path):
    mock_bin_mgr = MagicMock(spec=BinaryManager)
    mock_bin_mgr.find_binary.return_value = BinaryInfo(
        name="lora",
        found=True,
        path=Path("/bin/llama-export-lora"),
        is_executable=True,
    )

    mock_executor = MagicMock(spec=SubprocessExecutor)
    out_model = tmp_path / "merged.gguf"

    async def fake_run(*args, **kwargs):
        out_model.touch()
        return ExecutionResult(
            command=["llama-export-lora"],
            return_code=0,
            success=True,
            stdout="Merged successfully",
            stderr="",
            duration_seconds=0.5,
        )

    mock_executor.run = AsyncMock(side_effect=fake_run)

    base_model = tmp_path / "base.gguf"
    base_model.touch()
    adapter = tmp_path / "adapter.bin"
    adapter.touch()

    engine = LoRAMergeEngine(binary_manager=mock_bin_mgr, executor=mock_executor)
    result = await engine.merge(
        base_model=base_model,
        output_path=out_model,
        adapters=[LoRAAdapter(path=adapter)],
    )

    assert result.success is True
    assert result.output_path == out_model
    assert result.duration_seconds >= 0
