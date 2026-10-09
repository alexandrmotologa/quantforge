"""Built-in recipe templates for common quantization scenarios."""

from __future__ import annotations

import yaml
from pathlib import Path
from typing import Any, Dict


BALANCED_RECIPE = {
    "name": "balanced-production",
    "description": "Standard production set with high-speed 4-bit, 5-bit, and 8-bit reference quants.",
    "calibration": {
        "ctx_size": 2048,
        "chunks": 64,
        "n_gpu_layers": 0,
    },
    "quants": [
        "Q4_K_M",
        "Q5_K_M",
        "Q6_K",
        "Q8_0",
    ],
    "evaluation": {
        "ctx_size": 2048,
        "calc_pareto": True,
    },
    "export_model_card": True,
}

EXTREME_COMPRESSION_RECIPE = {
    "name": "extreme-compression",
    "description": "Sub-4-bit importance matrix quants for edge and memory-constrained deployments.",
    "calibration": {
        "ctx_size": 2048,
        "chunks": 96,
        "n_gpu_layers": 0,
    },
    "quants": [
        "IQ2_M",
        "IQ3_XXS",
        "IQ3_M",
        "IQ4_XS",
        "Q4_K_S",
    ],
    "evaluation": {
        "ctx_size": 2048,
        "calc_pareto": True,
    },
    "export_model_card": True,
}

FULL_MATRIX_RECIPE = {
    "name": "full-distribution-matrix",
    "description": "Comprehensive suite spanning 2-bit to 8-bit for model distribution.",
    "calibration": {
        "ctx_size": 2048,
        "chunks": 64,
        "n_gpu_layers": 0,
    },
    "quants": [
        "IQ2_M",
        "IQ3_M",
        "Q4_K_S",
        "Q4_K_M",
        "IQ4_XS",
        "Q5_K_M",
        "Q6_K",
        "Q8_0",
    ],
    "evaluation": {
        "ctx_size": 2048,
        "calc_pareto": True,
    },
    "export_model_card": True,
}


def dump_recipe_to_yaml(recipe_dict: Dict[str, Any], target_path: Path) -> Path:
    """Writes a recipe dict to a YAML file."""
    target_path.parent.mkdir(parents=True, exist_ok=True)
    with open(target_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(recipe_dict, f, sort_keys=False)
    return target_path
