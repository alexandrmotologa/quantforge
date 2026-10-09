"""LoRA adapter merge engine integrating base GGUF models with LoRA checkpoints."""

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
class LoRAAdapter:
    """Specification of an individual LoRA adapter with optional scaling weight."""

    path: Path
    scale: float = 1.0


@dataclass
class LoRAMergeResult:
    """Outcome of a LoRA merge operation."""

    success: bool
    base_model: Path
    output_path: Path
    adapters: List[LoRAAdapter]
    duration_seconds: float
    error_message: Optional[str] = None


class LoRAMergeEngine:
    """Merges one or more LoRA adapters into an unquantized or base GGUF model."""

    def __init__(
        self,
        binary_manager: Optional[BinaryManager] = None,
        executor: Optional[SubprocessExecutor] = None,
    ) -> None:
        self.binary_manager = binary_manager or BinaryManager()
        self.executor = executor or SubprocessExecutor()

    def build_command(
        self,
        base_model: Union[str, Path],
        output_path: Union[str, Path],
        adapters: List[LoRAAdapter],
        threads: Optional[int] = None,
    ) -> List[str]:
        """Constructs CLI arguments for llama-export-lora."""
        binary_info = self.binary_manager.find_binary("lora")
        if not binary_info.found or not binary_info.path:
            raise FileNotFoundError(
                "Executable 'llama-export-lora' was not found in your environment or llama.cpp path."
            )

        cmd = [
            str(binary_info.path),
            "-m",
            str(Path(base_model).resolve()),
            "-o",
            str(Path(output_path).resolve()),
        ]

        for adapter in adapters:
            a_path = str(Path(adapter.path).resolve())
            if adapter.scale != 1.0:
                cmd.extend(["--lora-scaled", a_path, str(adapter.scale)])
            else:
                cmd.extend(["--lora", a_path])

        if threads and threads > 0:
            cmd.extend(["-t", str(threads)])

        return cmd

    async def merge(
        self,
        base_model: Union[str, Path],
        output_path: Union[str, Path],
        adapters: List[LoRAAdapter],
        threads: Optional[int] = None,
        on_line: Optional[Callable[[str], None]] = None,
    ) -> LoRAMergeResult:
        """Executes the LoRA merge process asynchronously."""
        base_path = Path(base_model).resolve()
        out_path = Path(output_path).resolve()

        if not base_path.is_file():
            raise FileNotFoundError(f"Base model not found: {base_path}")

        if not adapters:
            raise ValueError("At least one LoRA adapter must be specified for merge.")

        for ad in adapters:
            if not Path(ad.path).is_file():
                raise FileNotFoundError(f"LoRA adapter file not found: {ad.path}")

        cmd = self.build_command(
            base_model=base_path,
            output_path=out_path,
            adapters=adapters,
            threads=threads,
        )

        out_path.parent.mkdir(parents=True, exist_ok=True)
        start_time = time.time()

        proc_result = await self.executor.run(cmd, on_line=on_line)
        duration = time.time() - start_time

        success = proc_result.return_code == 0 and out_path.exists()
        error_msg = None if success else (proc_result.stderr or "Process failed to produce merged model.")

        return LoRAMergeResult(
            success=success,
            base_model=base_path,
            output_path=out_path,
            adapters=adapters,
            duration_seconds=duration,
            error_message=error_msg,
        )
