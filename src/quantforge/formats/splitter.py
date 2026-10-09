"""GGUF model sharding and merging engine for large multi-gigabyte checkpoints."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional, Union

from quantforge.core.binary_manager import BinaryManager
from quantforge.core.executor import SubprocessExecutor

logger = logging.getLogger(__name__)


@dataclass
class SplitResult:
    """Outcome of a GGUF sharding operation."""

    success: bool
    input_file: Path
    output_prefix: Path
    split_files: List[Path] = field(default_factory=list)
    duration_seconds: float = 0.0
    error_message: Optional[str] = None


@dataclass
class MergeResult:
    """Outcome of a GGUF shard reassembly operation."""

    success: bool
    output_file: Path
    input_splits: List[Path] = field(default_factory=list)
    duration_seconds: float = 0.0
    error_message: Optional[str] = None


class GGUFSplitter:
    """Automates partitioning large GGUF models into volume shards and reassembling them."""

    def __init__(
        self,
        binary_manager: Optional[BinaryManager] = None,
        executor: Optional[SubprocessExecutor] = None,
    ) -> None:
        self.binary_manager = binary_manager or BinaryManager()
        self.executor = executor or SubprocessExecutor()

    def _resolve_binary(self) -> Path:
        """Finds llama-gguf-split executable."""
        binary_info = self.binary_manager.find_binary("split")
        if not binary_info.found or not binary_info.path:
            raise FileNotFoundError(
                "Executable 'llama-gguf-split' was not found in your environment or llama.cpp path."
            )
        return binary_info.path

    def build_split_command(
        self,
        input_file: Union[str, Path],
        output_prefix: Union[str, Path],
        max_size: str = "4G",
        dry_run: bool = False,
    ) -> List[str]:
        """Constructs split command arguments."""
        bin_path = self._resolve_binary()
        cmd = [
            str(bin_path),
            "--split",
            "--split-max-size",
            max_size,
        ]
        if dry_run:
            cmd.append("--dry-run")

        cmd.extend([
            str(Path(input_file).resolve()),
            str(Path(output_prefix).resolve()),
        ])
        return cmd

    def build_merge_command(
        self,
        first_split: Union[str, Path],
        output_file: Union[str, Path],
        delete_splits: bool = False,
    ) -> List[str]:
        """Constructs merge command arguments."""
        bin_path = self._resolve_binary()
        cmd = [
            str(bin_path),
            "--merge",
        ]
        if delete_splits:
            cmd.append("--delete-splits")

        cmd.extend([
            str(Path(first_split).resolve()),
            str(Path(output_file).resolve()),
        ])
        return cmd

    async def split(
        self,
        input_file: Union[str, Path],
        output_prefix: Union[str, Path],
        max_size: str = "4G",
        dry_run: bool = False,
        on_line: Optional[Callable[[str], None]] = None,
    ) -> SplitResult:
        """Splits a GGUF file into shards based on maximum file size."""
        in_path = Path(input_file).resolve()
        out_prefix = Path(output_prefix).resolve()

        if not in_path.is_file():
            raise FileNotFoundError(f"Input GGUF file not found: {in_path}")

        out_prefix.parent.mkdir(parents=True, exist_ok=True)
        cmd = self.build_split_command(
            input_file=in_path,
            output_prefix=out_prefix,
            max_size=max_size,
            dry_run=dry_run,
        )

        start_time = time.time()
        proc = await self.executor.run(cmd, on_line=on_line)
        duration = time.time() - start_time

        # Identify generated shards: {prefix}-00001-of-0000N.gguf or {prefix}.000.gguf
        parent_dir = out_prefix.parent
        stem = out_prefix.name
        split_files = sorted(list(parent_dir.glob(f"{stem}*.gguf")))

        success = proc.return_code == 0 and (dry_run or len(split_files) > 0)
        error_msg = None if success else (proc.stderr or "Split operation failed.")

        return SplitResult(
            success=success,
            input_file=in_path,
            output_prefix=out_prefix,
            split_files=split_files,
            duration_seconds=duration,
            error_message=error_msg,
        )

    async def merge(
        self,
        first_split: Union[str, Path],
        output_file: Union[str, Path],
        delete_splits: bool = False,
        on_line: Optional[Callable[[str], None]] = None,
    ) -> MergeResult:
        """Reassembles volume shards back into a single unified GGUF model."""
        first_path = Path(first_split).resolve()
        out_path = Path(output_file).resolve()

        if not first_path.is_file():
            raise FileNotFoundError(f"First split file not found: {first_path}")

        out_path.parent.mkdir(parents=True, exist_ok=True)
        cmd = self.build_merge_command(
            first_split=first_path,
            output_file=out_path,
            delete_splits=delete_splits,
        )

        start_time = time.time()
        proc = await self.executor.run(cmd, on_line=on_line)
        duration = time.time() - start_time

        success = proc.return_code == 0 and out_path.is_file()
        error_msg = None if success else (proc.stderr or "Merge operation failed.")

        # Find all sibling splits
        stem_base = first_path.name.split("-")[0]
        splits = sorted(list(first_path.parent.glob(f"{stem_base}*.gguf")))

        return MergeResult(
            success=success,
            output_file=out_path,
            input_splits=splits,
            duration_seconds=duration,
            error_message=error_msg,
        )
