"""Tests for GGUF model splitter and merger."""

from pathlib import Path
import pytest
from unittest.mock import AsyncMock, MagicMock

from quantforge.core.binary_manager import BinaryInfo, BinaryManager
from quantforge.core.executor import ExecutionResult, SubprocessExecutor
from quantforge.formats.splitter import GGUFSplitter


def test_build_split_and_merge_commands(tmp_path: Path):
    mock_bin_mgr = MagicMock(spec=BinaryManager)
    mock_bin_mgr.find_binary.return_value = BinaryInfo(
        name="split",
        found=True,
        path=Path("/bin/llama-gguf-split"),
        is_executable=True,
    )

    splitter = GGUFSplitter(binary_manager=mock_bin_mgr)
    in_file = tmp_path / "model.gguf"
    out_prefix = tmp_path / "shards" / "model"
    out_merged = tmp_path / "model_merged.gguf"

    cmd_split = splitter.build_split_command(in_file, out_prefix, max_size="2G", dry_run=True)
    assert cmd_split[0] == str(Path("/bin/llama-gguf-split"))
    assert "--split" in cmd_split
    assert "--split-max-size" in cmd_split
    assert "2G" in cmd_split
    assert "--dry-run" in cmd_split

    first_shard = tmp_path / "shards" / "model-00001-of-00002.gguf"
    cmd_merge = splitter.build_merge_command(first_shard, out_merged, delete_splits=True)
    assert cmd_merge[0] == str(Path("/bin/llama-gguf-split"))
    assert "--merge" in cmd_merge
    assert "--delete-splits" in cmd_merge


@pytest.mark.asyncio
async def test_split_and_merge_execution(tmp_path: Path):
    mock_bin_mgr = MagicMock(spec=BinaryManager)
    mock_bin_mgr.find_binary.return_value = BinaryInfo(
        name="split",
        found=True,
        path=Path("/bin/llama-gguf-split"),
        is_executable=True,
    )

    mock_executor = MagicMock(spec=SubprocessExecutor)
    shard_dir = tmp_path / "shards"

    async def fake_split(cmd, *args, **kwargs):
        shard_dir.mkdir(parents=True, exist_ok=True)
        (shard_dir / "model-00001-of-00002.gguf").touch()
        (shard_dir / "model-00002-of-00002.gguf").touch()
        return ExecutionResult(command=cmd, return_code=0, success=True, stdout="", stderr="", duration_seconds=0.2)

    mock_executor.run = AsyncMock(side_effect=fake_split)

    in_file = tmp_path / "model.gguf"
    in_file.touch()

    splitter = GGUFSplitter(binary_manager=mock_bin_mgr, executor=mock_executor)
    split_res = await splitter.split(in_file, shard_dir / "model", max_size="2G")

    assert split_res.success is True
    assert len(split_res.split_files) == 2

    # Now test merge
    out_merged = tmp_path / "model_merged.gguf"

    async def fake_merge(cmd, *args, **kwargs):
        out_merged.touch()
        return ExecutionResult(command=cmd, return_code=0, success=True, stdout="", stderr="", duration_seconds=0.2)

    mock_executor.run = AsyncMock(side_effect=fake_merge)
    merge_res = await splitter.merge(split_res.split_files[0], out_merged)

    assert merge_res.success is True
    assert merge_res.output_file == out_merged
