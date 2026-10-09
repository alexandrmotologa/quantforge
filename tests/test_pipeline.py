"""Tests for PipelineRunner and recipe execution."""

from pathlib import Path
import pytest
import yaml

from quantforge.pipelines.runner import PipelineProgress, PipelineRunner
from quantforge.pipelines.templates import BALANCED_RECIPE, dump_recipe_to_yaml
from .helpers import write_synthetic_gguf


def test_pipeline_progress_dataclass():
    prog = PipelineProgress(stage="quantize", current_step=2, total_steps=4, detail="Quantizing Q4_K_M")
    assert prog.stage == "quantize"
    assert prog.current_step == 2
    assert prog.total_steps == 4


def test_dump_recipe_to_yaml(tmp_path: Path):
    target = tmp_path / "recipe.yaml"
    res = dump_recipe_to_yaml(BALANCED_RECIPE, target)
    assert res.is_file()

    with open(res, "r", encoding="utf-8") as f:
        loaded = yaml.safe_load(f)
    assert loaded["name"] == "balanced-production"
    assert "Q4_K_M" in loaded["quants"]


@pytest.mark.asyncio
async def test_pipeline_runner_missing_recipe():
    runner = PipelineRunner()
    with pytest.raises(FileNotFoundError):
        await runner.run_recipe_file(Path("nonexistent_recipe.yaml"))


@pytest.mark.asyncio
async def test_pipeline_runner_missing_model(tmp_path: Path):
    recipe = {
        "name": "test-pipeline",
        "input_model": str(tmp_path / "missing.gguf"),
        "output_dir": str(tmp_path / "out"),
    }
    recipe_file = tmp_path / "rec.yaml"
    dump_recipe_to_yaml(recipe, recipe_file)

    runner = PipelineRunner()
    with pytest.raises(FileNotFoundError):
        await runner.run_recipe_file(recipe_file)
