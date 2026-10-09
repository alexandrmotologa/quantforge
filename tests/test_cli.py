"""Tests for Typer CLI commands."""

from typer.testing import CliRunner
from pathlib import Path
from quantforge.cli.app import cli
from .helpers import write_synthetic_gguf

runner = CliRunner()


def test_cli_version():
    result = runner.invoke(cli, ["--version"])
    assert result.exit_code == 0
    assert "QuantForge version" in result.stdout


def test_cli_bin_paths():
    result = runner.invoke(cli, ["bin", "paths"])
    assert result.exit_code == 0
    assert "Configured Search Directories:" in result.stdout


def test_cli_bin_check():
    result = runner.invoke(cli, ["bin", "check"])
    assert result.exit_code == 0
    assert "llama.cpp Binary Status" in result.stdout


def test_cli_pipeline_init(tmp_path: Path):
    out_recipe = tmp_path / "test_recipe.yaml"
    result = runner.invoke(cli, ["pipeline", "init", "--output", str(out_recipe)])
    assert result.exit_code == 0
    assert out_recipe.is_file()
    assert "Recipe initialized at:" in result.stdout


def test_cli_inspect(tmp_path: Path):
    model = tmp_path / "model.gguf"
    write_synthetic_gguf(model)

    result = runner.invoke(cli, ["inspect", str(model)])
    assert result.exit_code == 0
    assert "GGUF Model Info:" in result.stdout
    assert "Architecture" in result.stdout
    assert "llama" in result.stdout

    # Test json output
    json_res = runner.invoke(cli, ["inspect", str(model), "--json"])
    assert json_res.exit_code == 0
    assert '"architecture": "llama"' in json_res.stdout
