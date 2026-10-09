"""Tests for HardwareAutotuner."""

from pathlib import Path
from quantforge.core.autotune import HardwareAutotuner


def test_hardware_profiler():
    tuner = HardwareAutotuner()
    profile = tuner.profile()

    assert profile.cpu_logical >= 1
    assert profile.cpu_physical >= 1
    assert profile.optimal_threads >= 1
    assert profile.os_name in ["windows", "linux", "darwin"]


def test_hardware_recommendations():
    tuner = HardwareAutotuner()
    rec = tuner.recommend(model_size_gb=4.0, context_length=2048, layer_count=32)

    assert "optimal_threads" in rec
    assert rec["optimal_threads"] >= 1
    assert "recommended_n_gpu_layers" in rec
    assert isinstance(rec["recommended_n_gpu_layers"], int)
