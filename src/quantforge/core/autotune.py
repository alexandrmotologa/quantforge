"""Hardware profiling and parameter auto-tuning for inference and quantization."""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass
class HardwareProfile:
    """Detected host system hardware specifications."""

    cpu_logical: int
    cpu_physical: int
    optimal_threads: int
    os_name: str
    arch: str
    gpu_available: bool = False
    gpu_name: Optional[str] = None
    gpu_vram_gb: Optional[float] = None
    gpu_backend: str = "cpu"


class HardwareAutotuner:
    """Profiles hardware and calculates optimal quantization and inference hyperparameters."""

    @staticmethod
    def profile() -> HardwareProfile:
        """Inspects CPU and GPU resources on the current system."""
        logical = os.cpu_count() or 4
        # Rough heuristic for physical cores if psutil not installed
        physical = max(1, logical // 2) if logical > 2 else logical
        optimal_threads = max(1, physical)

        system = platform.system().lower()
        arch = platform.machine()

        gpu_found = False
        gpu_name = None
        gpu_vram = None
        gpu_backend = "cpu"

        # Check for NVIDIA GPU via nvidia-smi
        if shutil.which("nvidia-smi"):
            try:
                res = subprocess.run(
                    ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
                    capture_output=True,
                    text=True,
                    timeout=3,
                    check=False,
                )
                if res.returncode == 0 and res.stdout.strip():
                    line = res.stdout.strip().splitlines()[0]
                    parts = line.split(",")
                    if len(parts) >= 2:
                        gpu_name = parts[0].strip()
                        gpu_vram = round(float(parts[1].strip()) / 1024.0, 1)
                        gpu_found = True
                        gpu_backend = "cuda"
            except Exception:
                pass

        return HardwareProfile(
            cpu_logical=logical,
            cpu_physical=physical,
            optimal_threads=optimal_threads,
            os_name=system,
            arch=arch,
            gpu_available=gpu_found,
            gpu_name=gpu_name,
            gpu_vram_gb=gpu_vram,
            gpu_backend=gpu_backend,
        )

    def recommend(
        self,
        model_size_gb: Optional[float] = None,
        context_length: int = 4096,
        layer_count: int = 32,
    ) -> Dict[str, Any]:
        """Calculates optimal threads and GPU offload layers."""
        profile = self.profile()

        # Threads recommendation: physical cores avoiding hyperthreading thread fighting
        rec_threads = profile.optimal_threads

        # GPU offload recommendation
        rec_layers = 0
        fits_fully = False

        if profile.gpu_available and profile.gpu_vram_gb and model_size_gb:
            # Est. VRAM required = weights + KV cache overhead (~0.8GB per 4k tokens for 7B)
            kv_overhead = (context_length / 4096.0) * 0.8
            total_req = model_size_gb + kv_overhead + 0.5  # Context + scratch buffers

            if profile.gpu_vram_gb >= total_req:
                rec_layers = layer_count + 1  # Full offload
                fits_fully = True
            else:
                # Partial layer offload proportional to available VRAM
                ratio = max(0.0, (profile.gpu_vram_gb - kv_overhead - 0.5) / model_size_gb)
                rec_layers = max(0, int(layer_count * ratio))

        return {
            "optimal_threads": rec_threads,
            "recommended_n_gpu_layers": rec_layers,
            "gpu_detected": profile.gpu_available,
            "gpu_name": profile.gpu_name,
            "gpu_vram_gb": profile.gpu_vram_gb,
            "gpu_backend": profile.gpu_backend,
            "fits_fully_in_vram": fits_fully,
        }
